"""
随机初始化基线实验
- 创建与主实验相同架构的 Transformer 模型（未训练）
- 提取注意力权重矩阵，进行谱分析
- 输出 KS(GOE)、KS(Poisson) 和判决
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats


# ---------- 模型定义（复用主实验代码）----------
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
        """提取所有注意力层的 in_proj_weight 矩阵"""
        weights = []
        for layer in self.transformer.layers:
            # 每个 encoder 层中的 self_attn 的 in_proj_weight
            attn_weights = layer.self_attn.in_proj_weight
            weights.append(attn_weights.detach().cpu())
        return torch.cat(weights, dim=0)


# ---------- 谱分析函数（与主实验一致）----------
def goe_cdf(x):
    return 1 - np.exp(-np.pi * x ** 2 / 4)


def poisson_cdf(x):
    return 1 - np.exp(-x)


def analyze_matrix(W):
    """输入一个方阵 W，返回 KS(GOE), KS(Poisson), verdict"""
    H = (W + W.T) / 2
    eigvals = np.linalg.eigvalsh(H)
    eigvals = np.sort(eigvals)
    n = len(eigvals)
    low = int(n * 0.15)
    high = int(n * 0.85)
    eigvals = eigvals[low:high]
    spacings = np.diff(eigvals)
    if np.mean(spacings) == 0:
        return 1.0, 1.0, "Poisson"
    s = spacings / np.mean(spacings)
    s = s[s <= 4]  # 截断
    ks_goe = stats.kstest(s, goe_cdf).statistic
    ks_poisson = stats.kstest(s, poisson_cdf).statistic
    verdict = "GOE" if ks_goe < ks_poisson else "Poisson"
    return ks_goe, ks_poisson, verdict


def analyze_random_model(model):
    """提取模型的所有 in_proj_weight，拼接，截取 256x256，返回谱分析结果"""
    W_huge = model.get_attention_weights().numpy()  # (4608, 256) 或类似
    # 确保至少有256行和256列
    if W_huge.shape[0] < 256 or W_huge.shape[1] < 256:
        raise ValueError("矩阵维度不足256")
    W = W_huge[:256, :256]
    return analyze_matrix(W)


# ---------- 主程序 ----------
if __name__ == "__main__":
    # 设置随机种子以获得可复现的结果（可选）
    torch.manual_seed(42)
    np.random.seed(42)

    n_instances = 3  # 随机实例个数
    results = []

    for i in range(n_instances):
        # 每次创建一个新模型（随机初始化）
        model = PrimeGapPredictor(
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            learnable_embedding=True  # 与 Plan A/B 一致，也可设为 False 但影响不大
        )
        model.eval()  # 无需训练，直接进入评估模式
        ks_goe, ks_poisson, verdict = analyze_random_model(model)
        results.append((ks_goe, ks_poisson, verdict))
        print(f"Random instance {i + 1}: KS(GOE)={ks_goe:.4f}, KS(Poisson)={ks_poisson:.4f}, Verdict={verdict}")

    # 输出表格（Markdown格式）
    print("\n**Table X:** Spectral statistics of randomly initialized (untrained) Transformer models.")
    print("| Instance | KS(GOE) | KS(Poisson) | Verdict |")
    print("|----------|---------|-------------|---------|")
    for i, (g, p, v) in enumerate(results, 1):
        print(f"| Random {i} | {g:.4f} | {p:.4f} | {v} |")