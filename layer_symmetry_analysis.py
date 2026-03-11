"""
计算所有模型每层的对称质量并绘制层间变化曲线
支持标准 Transformer 和 IHC 架构
图片保存在运行目录下，同时打印各层对称质量到终端。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# ==================== 配置（请根据你的实际路径修改）====================
MODEL_CONFIGS = [
    {
        "name": "Plan A (Prime)",
        "path": r"D:\pythonstudy\python_task\权重分析\Plan A.pt",
        "type": "standard",
        "learnable": True,
        "color": "#E74C3C"
    },
    {
        "name": "Plan B (Prime)",
        "path": r"D:\pythonstudy\python_task\权重分析\Plan B.pt",
        "type": "standard",
        "learnable": True,
        "color": "#3498DB"
    },
    {
        "name": "Plan C (Prime)",
        "path": r"D:\pythonstudy\python_task\权重分析\Plan C.pt",
        "type": "standard",
        "learnable": False,
        "color": "#2ECC71"
    },
    {
        "name": "Chaos Success",
        "path": r"D:\pythonstudy\python_task\改进\chaos_plan_A_results\chaotic_success.pt",
        "type": "standard",
        "learnable": True,
        "color": "#9B59B6"
    },
    {
        "name": "Chaos Fail",
        "path": r"D:\pythonstudy\python_task\改进\chaos_plan_C_results\chaotic_fail.pt",
        "type": "standard",
        "learnable": False,
        "color": "#FFA500"
    },
    {
        "name": "IHC Prime",
        "path": r"D:\pythonstudy\python_task\IHC\riemann_pure_emergence_results\best_model.pt",
        "type": "ihc",
        "learnable": True,
        "color": "#1ABC9C"
    },
    # 预留 IHC Chaos（如果还没跑，可以注释掉，或等跑完再放开）
    {
        "name": "IHC Chaos",
        "path": r"D:\pythonstudy\python_task\IHC\chaos_ihc_results\best_model.pt",
        "type": "ihc",
        "learnable": True,
        "color": "#34495E"
     }
]

OUTPUT_FILE = "layer_symmetry_curves.png"

D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
NUM_FLOWS = 4          # IHC 专用

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}\n")

# ==================== 模型定义（与之前完全一致）====================
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
        raise ValueError(f"Unknown model type: {cfg['type']}")

    checkpoint = torch.load(cfg["path"], map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ==================== 计算单层对称质量 ====================
def symmetry_quality_for_matrix(q_mat):
    pca = PCA(n_components=2)
    rows_pca = pca.fit_transform(q_mat)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rows_pca)
    centers = kmeans.cluster_centers_
    inter_dist = np.linalg.norm(centers[0] - centers[1])
    intra0 = np.mean([np.linalg.norm(rows_pca[i] - centers[0]) for i in range(len(rows_pca)) if labels[i]==0])
    intra1 = np.mean([np.linalg.norm(rows_pca[i] - centers[1]) for i in range(len(rows_pca)) if labels[i]==1])
    intra_avg = (intra0 + intra1) / 2
    sym_quality = inter_dist / (intra_avg + 1e-8)
    return sym_quality

def compute_layer_symmetries(model, model_type):
    qualities = []
    if model_type == "standard":
        layers = model.transformer.layers
        for layer in layers:
            if hasattr(layer.self_attn, 'in_proj_weight'):
                W = layer.self_attn.in_proj_weight.detach().cpu().numpy()
                q_part = W[:D_MODEL, :]
                q = symmetry_quality_for_matrix(q_part)
                qualities.append(q)
            else:
                qualities.append(np.nan)
    elif model_type == "ihc":
        for layer in model.layers:
            if hasattr(layer.self_attn, 'in_proj_weight'):
                W = layer.self_attn.in_proj_weight.detach().cpu().numpy()
                q_part = W[:layer.dim, :]
                q = symmetry_quality_for_matrix(q_part)
                qualities.append(q)
            else:
                qualities.append(np.nan)
    return qualities

# ==================== 主程序 ====================
def main():
    plt.figure(figsize=(10, 6))

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"⚠️ 模型文件不存在，跳过: {path}")
            continue

        print(f"\n📦 处理 {name} ...")
        model = load_model(cfg)
        qualities = compute_layer_symmetries(model, cfg["type"])
        layers = list(range(len(qualities)))

        # 打印各层对称质量
        print(f"  各层对称质量（层索引 : 值）:")
        for i, q in enumerate(qualities):
            print(f"    层 {i}: {q:.4f}")

        plt.plot(layers, qualities, marker='o', linestyle='-', color=cfg["color"], label=name)

        del model
        torch.cuda.empty_cache()

    plt.xlabel("Layer Index")
    plt.ylabel("Symmetry Quality")
    plt.title("Layer-wise Symmetry Quality across Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.close()
    print(f"\n✅ 曲线图已保存至当前目录: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()