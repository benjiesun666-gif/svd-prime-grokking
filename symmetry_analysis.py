"""
对称性分析（第一层具体数值）
支持标准 Transformer 和 IHC 架构。
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==================== 配置（请根据您的实际路径修改）====================
MODEL_CONFIGS = [
    {
        "name": "Plan A",
        "path": r"D:\illtenligence\Plan A.pt",
        "type": "standard",
        "learnable": True,
    },
    {
        "name": "Plan B",
        "path": r"D:\illtenligence\Plan B.pt",
        "type": "standard",
        "learnable": True,
    },
    {
        "name": "Plan C",
        "path": r"D:\illtenligence\Plan C.pt",
        "type": "standard",
        "learnable": False,
    },
    {
        "name": "Chaos Success",
        "path": r"D:\pythonstudy\python_task\改进\chaos_plan_A_results\chaotic_success.pt",
        "type": "standard",
        "learnable": True,
    },
    {
        "name": "Chaos Fail",
        "path": r"D:\pythonstudy\python_task\改进\chaos_plan_C_results\chaotic_fail.pt",
        "type": "standard",
        "learnable": False,
    },
    # IHC 模型（用于表4，如果需要）
    {
        "name": "IHC Prime",
        "path": r"D:\pythonstudy\python_task\IHC\riemann_pure_emergence_results\IHC_prime.pt",
        "type": "ihc",
        "learnable": True,
    },
    {
        "name": "IHC Chaos",
        "path": r"D:\pythonstudy\python_task\IHC\chaos_ihc_results\IHC_chaos.pt",
        "type": "ihc",
        "learnable": True,
    },
]

# 模型参数（与训练时一致）
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
NUM_FLOWS = 4          # IHC 专用

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 基础模块（与训练时一致）====================
class RiemannEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000000, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.embedding = nn.Embedding(max_len, d_model)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
    def forward(self, x):
        if self.learnable:
            return self.embedding(x)
        else:
            return self.pe[x]

class PrimeGapPredictor(nn.Module):
    """标准 Transformer 模型"""
    def __init__(self, learnable_embedding=True):
        super().__init__()
        self.riemann_embedding = RiemannEmbedding(D_MODEL, learnable=learnable_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL*4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.output = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL//2, D_MODEL//4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL//4, 1)
        )
    def forward(self, x):
        emb = self.riemann_embedding(x).unsqueeze(1)
        out = self.transformer(emb)
        return self.output(out.squeeze(1))

class IHCTransformerLayer(nn.Module):
    def __init__(self, num_flows, dim, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.num_flows = num_flows
        self.dim = dim
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.phi = nn.Sequential(
            nn.Linear(num_flows * dim, num_flows * 2),
            nn.GELU(),
            nn.Linear(num_flows * 2, num_flows * 2)
        )
    def forward(self, x_flows):
        batch, num_flows, dim = x_flows.shape
        phi_in = x_flows.reshape(batch, -1)
        phi_out = self.phi(phi_in)
        h_pre = F.softmax(phi_out[:, :num_flows], dim=-1)
        h_post = F.softmax(phi_out[:, num_flows:], dim=-1)
        agg = torch.einsum('bf,bfd->bd', h_pre, x_flows)
        agg2 = self.norm1(agg + self._sa_block(agg))
        agg3 = self.norm2(agg2 + self._ffn_block(agg2))
        new_flows = x_flows + h_post.unsqueeze(-1) * agg3.unsqueeze(1)
        return new_flows
    def _sa_block(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.self_attn(x, x, x)
        return attn_out.squeeze(1)
    def _ffn_block(self, x):
        return self.ffn(x)

class IHCPrimeGapPredictor(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout, learnable_embedding, num_flows):
        super().__init__()
        assert d_model % num_flows == 0
        self.d_model = d_model
        self.num_flows = num_flows
        self.flow_dim = d_model // num_flows
        self.riemann_embedding = RiemannEmbedding(d_model, learnable=learnable_embedding)
        self.layers = nn.ModuleList([
            IHCTransformerLayer(num_flows, self.flow_dim, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(self.flow_dim, self.flow_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.flow_dim // 2, self.flow_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.flow_dim // 4, 1)
        )
    def forward(self, x):
        emb = self.riemann_embedding(x)
        x_flows = emb.view(-1, self.num_flows, self.flow_dim)
        for layer in self.layers:
            x_flows = layer(x_flows)
        pooled = x_flows.mean(dim=1)
        return self.output_head(pooled)

# ==================== 加载模型 ====================
def load_model(cfg):
    if cfg["type"] == "standard":
        model = PrimeGapPredictor(learnable_embedding=cfg["learnable"]).to(device)
    elif cfg["type"] == "ihc":
        model = IHCPrimeGapPredictor(
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            dropout=DROPOUT,
            learnable_embedding=cfg["learnable"],
            num_flows=NUM_FLOWS
        ).to(device)
    else:
        raise ValueError(f"Unknown type: {cfg['type']}")
    checkpoint = torch.load(cfg["path"], map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ==================== 提取第一层 Q 矩阵 ====================
def get_first_layer_q(model, model_type):
    if model_type == "standard":
        # 假设第一层的 in_proj_weight 存在于 transformer.layers[0].self_attn.in_proj_weight
        layer = model.transformer.layers[0]
        W = layer.self_attn.in_proj_weight.detach().cpu().numpy()  # shape (3*D_MODEL, D_MODEL)
        q_part = W[:D_MODEL, :]  # 取前 D_MODEL 行作为 Q 矩阵
    elif model_type == "ihc":
        layer = model.layers[0]
        W = layer.self_attn.in_proj_weight.detach().cpu().numpy()  # shape (3*flow_dim, flow_dim)
        q_part = W[:layer.dim, :]  # 取前 flow_dim 行作为 Q 矩阵
    else:
        raise ValueError(f"Unknown type: {model_type}")
    return q_part

# ==================== 计算对称性指标 ====================
def compute_symmetry_metrics(q_matrix):
    """
    输入 Q 矩阵 (n_rows, n_cols)
    返回 inter_dist, balance, sym_quality
    """
    pca = PCA(n_components=2)
    rows_pca = pca.fit_transform(q_matrix)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rows_pca)
    centers = kmeans.cluster_centers_

    inter_dist = np.linalg.norm(centers[0] - centers[1])

    size0 = np.sum(labels == 0)
    size1 = np.sum(labels == 1)
    balance = 1 - abs(size0 - size1) / (size0 + size1 + 1e-8)

    intra0 = np.mean([np.linalg.norm(rows_pca[i] - centers[0]) for i in range(len(rows_pca)) if labels[i] == 0])
    intra1 = np.mean([np.linalg.norm(rows_pca[i] - centers[1]) for i in range(len(rows_pca)) if labels[i] == 1])
    intra_avg = (intra0 + intra1) / 2
    sym_quality = inter_dist / (intra_avg + 1e-8)

    return inter_dist, balance, sym_quality

# ==================== 主程序 ====================
def main():
    print("第一层对称性指标重新计算（用于表3/4）")
    print("="*60)
    for cfg in MODEL_CONFIGS:
        print(f"\n模型: {cfg['name']}")
        if not os.path.exists(cfg["path"]):
            print(f"  文件不存在: {cfg['path']}")
            continue
        model = load_model(cfg)
        q_mat = get_first_layer_q(model, cfg["type"])
        inter_dist, balance, sym_quality = compute_symmetry_metrics(q_mat)
        print(f"  inter‑cluster distance: {inter_dist:.4f}")
        print(f"  cluster balance: {balance:.4f}")
        print(f"  symmetry quality: {sym_quality:.4f}")
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
