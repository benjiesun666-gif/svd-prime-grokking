"""
随机初始化基线实验（标准 Transformer）
- 创建与主实验相同架构的 Transformer 模型（未训练）
- 提取注意力权重矩阵，进行 SVD 分析
- 计算有效秩、最大/平均奇异值比、KS 距离（与随机高斯矩阵）
- 输出多次实例的统计值
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 模型定义（与主实验一致）----------
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
    def __init__(self, d_model=256, n_layers=6, n_heads=8, dropout=0.1, learnable_embedding=True):
        super().__init__()
        self.d_model = d_model
        self.riemann_embedding = RiemannEmbedding(d_model, learnable=learnable_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

    def get_attention_weights(self):
        """提取所有注意力层的 in_proj_weight 矩阵（按行拼接）"""
        weights = []
        for layer in self.transformer.layers:
            attn_weights = layer.self_attn.in_proj_weight  # shape (3*d_model, d_model)
            weights.append(attn_weights.detach().cpu())
        return torch.cat(weights, dim=0)  # shape (n_layers * 3*d_model, d_model)


# ---------- SVD 分析函数（与主实验一致）----------
def svd_analysis(W):
    """
    对矩阵 W (n x n) 进行 SVD，返回有效秩、最大/平均奇异值比、KS 距离
    """
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    S = S[S > 1e-12]  # 去除极小的奇异值（数值零）
    n = len(S)

    # 有效秩
    total = np.sum(S)
    p = S / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(entropy)

    # 最大/平均比
    max_s = S[0]
    mean_s = np.mean(S)
    max_ratio = max_s / mean_s

    # 随机高斯基线
    np.random.seed(42)  # 固定种子使结果可复现（但对每个实例独立？这里为简化，每次调用都会重置，可接受）
    W_rand = np.random.randn(n, n)
    S_rand = np.linalg.svd(W_rand, compute_uv=False)
    ks = stats.ks_2samp(S, S_rand).statistic

    return eff_rank, max_ratio, ks


def analyze_random_model(model):
    """提取模型权重，拼接并取前256×256子矩阵，进行 SVD 分析"""
    W_huge = model.get_attention_weights().numpy()  # (4608, 256) 或类似
    # 确保至少有256行和256列
    if W_huge.shape[0] < 256 or W_huge.shape[1] < 256:
        raise ValueError("矩阵维度不足256")
    W = W_huge[:256, :256]  # 取前256行前256列（与主实验一致）
    return svd_analysis(W)


# ---------- 主程序 ----------
if __name__ == "__main__":
    # 设置随机种子以获得可复现的结果（但每次实例都会重新初始化，所以全局种子影响初始化顺序）
    torch.manual_seed(42)
    np.random.seed(42)

    n_instances = 3  # 随机实例个数
    eff_ranks = []
    max_ratios = []
    ks_vals = []

    print("Standard Transformer random baseline SVD analysis")
    print("=" * 60)

    for i in range(n_instances):
        # 每次创建一个新模型（随机初始化）
        model = PrimeGapPredictor(
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            learnable_embedding=True  # 与 Plan A/B 一致
        )
        model.eval()  # 无需训练

        eff, maxr, ks = analyze_random_model(model)
        eff_ranks.append(eff)
        max_ratios.append(maxr)
        ks_vals.append(ks)

        print(f"Instance {i+1}: eff_rank = {eff:.2f}, max/mean = {maxr:.2f}, KS = {ks:.4f}")

    # 计算统计值
    mean_eff = np.mean(eff_ranks)
    std_eff = np.std(eff_ranks)
    mean_max = np.mean(max_ratios)
    std_max = np.std(max_ratios)
    mean_ks = np.mean(ks_vals)
    std_ks = np.std(ks_vals)

    print("\n" + "=" * 60)
    print("Summary (mean ± std):")
    print(f"Effective rank      = {mean_eff:.2f} ± {std_eff:.2f}")
    print(f"Max/mean ratio      = {mean_max:.2f} ± {std_max:.2f}")
    print(f"KS distance         = {mean_ks:.4f} ± {std_ks:.4f}")

    # 输出表格（Markdown格式）
    print("\n**Table: SVD metrics for randomly initialized standard Transformer models.**")
    print("| Instance | Effective Rank | Max/Mean Ratio | KS Distance |")
    print("|----------|----------------|----------------|-------------|")
    for i, (e, m, k) in enumerate(zip(eff_ranks, max_ratios, ks_vals), 1):
        print(f"| Random {i} | {e:.2f} | {m:.2f} | {k:.4f} |")
    print(f"| Mean ± std | {mean_eff:.2f} ± {std_eff:.2f} | {mean_max:.2f} ± {std_max:.2f} | {mean_ks:.4f} ± {std_ks:.4f} |")

