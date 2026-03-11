"""
计算 IHC 架构随机初始化模型的 SVD 指标（有效秩、最大比、KS距离）
作为架构本征压缩的基线参考。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
import os

# 模型参数（必须与训练时一致）
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
LEARNABLE_EMBEDDING = True  # IHC 使用可学习嵌入
NUM_FLOWS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}\n")

# ==================== 基础模块（与训练脚本一致）====================
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

# ==================== SVD 分析函数（复用之前的逻辑）====================
def analyze_svd_properties(model, label="IHC Random Baseline"):
    print(f"  🔬 分析SVD谱...")
    weights = []
    for name, param in model.named_parameters():
        if 'in_proj_weight' in name:
            weights.append(param.detach().cpu().numpy())
    if not weights:
        for name, param in model.named_parameters():
            if len(param.shape) == 2 and param.shape[0] == param.shape[1]:
                weights.append(param.detach().cpu().numpy())
    if not weights:
        print("    ⚠️ 未找到可用权重矩阵")
        return None
    W_huge = np.concatenate(weights, axis=0)
    n = min(2048, W_huge.shape[0], W_huge.shape[1])
    W = W_huge[:n, :n]
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    total = np.sum(S)
    p = S / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(entropy)
    max_s = S[0]
    mean_s = np.mean(S)
    max_ratio = max_s / mean_s
    np.random.seed(42)
    W_random = np.random.randn(n, n)
    U_rand, S_rand, Vh_rand = np.linalg.svd(W_random, full_matrices=False)
    ks_random = stats.ks_2samp(S, S_rand).statistic
    print(f"    ✅ 有效秩 = {eff_rank:.2f}")
    print(f"      最大/平均奇异值比 = {max_ratio:.2f}")
    print(f"      KS距离 vs 随机基线 = {ks_random:.4f}")
    return {
        "eff_rank": eff_rank,
        "max_ratio": max_ratio,
        "ks_random": ks_random
    }

# ==================== 主程序 ====================
def main():
    # 创建未训练的 IHC 模型（随机初始化）
    model = IHCPrimeGapPredictor(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        learnable_embedding=LEARNABLE_EMBEDDING,
        num_flows=NUM_FLOWS
    ).to(device)
    # 不加载任何权重，保持随机初始化
    model.eval()
    print("📦 分析 IHC 随机初始化模型（未训练）...")
    svd_res = analyze_svd_properties(model, label="IHC Random Baseline")
    if svd_res:
        print("\n📊 IHC 随机基线结果:")
        print(f"  有效秩 = {svd_res['eff_rank']:.2f}")
        print(f"  最大/平均比 = {svd_res['max_ratio']:.2f}")
        print(f"  KS距离 = {svd_res['ks_random']:.4f}")

if __name__ == "__main__":
    main()