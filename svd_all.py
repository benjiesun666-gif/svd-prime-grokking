"""
一键生成所有模型的预测精度和 SVD 谱图
支持标准 Transformer 和 IHC 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sympy import primerange
import os
from pathlib import Path

# ==================== 配置（请根据你的实际路径修改）====================
MODEL_CONFIGS = [
    # 素数原始模型
    {
        "name": "Plan_A",
        "path": r"D:\pythonstudy\python_task\权重分析\Plan A.pt",
        "type": "standard",
        "task": "prime",
        "learnable": True,
        "color": "#E74C3C",
        "label": "Plan A (Prime)"
    },
    {
        "name": "Plan_B",
        "path": r"D:\pythonstudy\python_task\权重分析\Plan B.pt",
        "type": "standard",
        "task": "prime",
        "learnable": True,
        "color": "#3498DB",
        "label": "Plan B (Prime)"
    },
    {
        "name": "Plan_C",
        "path": r"D:\pythonstudy\python_task\权重分析\Plan C.pt",
        "type": "standard",
        "task": "prime",
        "learnable": False,
        "color": "#2ECC71",
        "label": "Plan C (Prime)"
    },
    # 混沌原始模型
    {
        "name": "Chaos_Success",
        "path": r"D:\pythonstudy\python_task\改进\chaos_plan_A_results\chaotic_success.pt",
        "type": "standard",
        "task": "chaos",
        "learnable": True,
        "color": "#9B59B6",
        "label": "Chaos Success"
    },
    {
        "name": "Chaos_Fail",
        "path": r"D:\pythonstudy\python_task\改进\chaos_plan_C_results\chaotic_fail.pt",
        "type": "standard",
        "task": "chaos",
        "learnable": False,
        "color": "#FFA500",
        "label": "Chaos Fail"
    },
    # IHC 模型
    {
        "name": "IHC_Prime",
        "path": r"D:\pythonstudy\python_task\IHC\riemann_pure_emergence_results\best_model.pt",
        "type": "ihc",
        "task": "prime",
        "learnable": True,
        "color": "#1ABC9C",
        "label": "IHC Prime"
    },
    {
        "name": "IHC_Chaos",
        "path": r"D:\pythonstudy\python_task\IHC\chaos_ihc_results\best_model.pt",  # 请根据实际路径修改
        "type": "ihc",
        "task": "chaos",
        "learnable": True,
        "color": "#34495E",
        "label": "IHC Chaos"
    }
]

# 输出目录
OUTPUT_DIR = Path(r"D:\pythonstudy\python_task\IHC")
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型通用参数
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
NUM_PRIMES = 1000000
NUM_FLOWS = 4          # IHC 专用

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}\n")

# ==================== 基础嵌入层（所有模型共用）====================
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

# ==================== 标准 Transformer 模型 ====================
class PrimeGapPredictor(nn.Module):
    def __init__(self, learnable_embedding=True):
        super().__init__()
        self.riemann_embedding = RiemannEmbedding(D_MODEL, NUM_PRIMES, learnable=learnable_embedding)
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
        embedded = self.riemann_embedding(x).unsqueeze(1)
        transformed = self.transformer(embedded)
        return self.output(transformed.squeeze(1))

# ==================== IHC 模型定义 ====================
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
        emb = self.riemann_embedding(x)                         # [batch, d_model]
        x_flows = emb.view(-1, self.num_flows, self.flow_dim)   # [batch, num_flows, flow_dim]
        for layer in self.layers:
            x_flows = layer(x_flows)
        pooled = x_flows.mean(dim=1)                             # [batch, flow_dim]
        return self.output_head(pooled)

# ==================== 数据生成函数 ====================
def generate_prime_data():
    all_primes = list(primerange(1, 18500000))[:NUM_PRIMES]
    all_gaps = np.diff(all_primes)
    global_mean = np.mean(all_gaps)
    global_std = np.std(all_gaps)
    return all_gaps, global_mean, global_std

def generate_chaos_data():
    mu = 3.8
    x0 = 0.5
    seq = [x0]
    for _ in range(1, NUM_PRIMES):
        seq.append(mu * seq[-1] * (1 - seq[-1]))
    arr = np.array(seq)
    gaps = np.diff(arr)  # 长度 NUM_PRIMES-1
    global_mean = np.mean(gaps)
    global_std = np.std(gaps)
    return gaps, global_mean, global_std

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

# ==================== 预测精度分析（与原脚本一致）====================
def analyze_prediction_accuracy(model, cfg, gaps, mean, std):
    print(f"  🔍 分析预测精度...")

    total_gaps = len(gaps)
    target_len = 1000
    start_idx = total_gaps - target_len
    end_idx = total_gaps
    target_gaps = gaps[start_idx:end_idx]

    indices = torch.arange(start_idx, end_idx, device=device)
    with torch.no_grad():
        preds_norm = model(indices).squeeze().cpu().numpy()

    preds_real = (preds_norm * std) + mean
    mae = np.mean(np.abs(preds_real - target_gaps))

    # 绘图
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(target_gaps[:200], color='black', alpha=0.6, label='Real Truth', linewidth=2)
    plt.plot(preds_real[:200], color='red', alpha=0.8, linestyle='--', label='AI Prediction', linewidth=1.5)
    plt.title(f'{cfg["label"]} - Micro View')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(target_gaps, preds_real, alpha=0.5, s=10, c='blue')
    min_v = min(target_gaps.min(), preds_real.min())
    max_v = max(target_gaps.max(), preds_real.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect Fit')
    plt.title(f'Correlation (MAE={mae:.4f})')
    plt.xlabel('Real Gap')
    plt.ylabel('Predicted Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{cfg["name"]}_prediction.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ MAE = {mae:.4f}, 图片已保存")
    return mae

# ==================== SVD谱分析（与原脚本一致）====================
def analyze_svd_properties(model, cfg):
    print(f"  🔬 分析SVD谱...")

    # 提取所有权重矩阵（优先找 in_proj_weight）
    weights = []
    for name, param in model.named_parameters():
        if 'in_proj_weight' in name:
            weights.append(param.detach().cpu().numpy())

    if not weights:
        # 如果没有 in_proj_weight，取所有方阵
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

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(S, 'o-', markersize=3, linewidth=1, color=cfg['color'],
                label=f'{cfg["label"]} singular values')
    ax.semilogy(S_rand, '--', linewidth=2, color='gray', alpha=0.7, label='Random baseline')
    ax.set_title(f'{cfg["label"]}: Singular Value Spectrum', fontsize=14, fontweight='bold')
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Singular Value (log scale)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f'{cfg["name"]}_svd_spectrum.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    print("="*60)
    print("🚀 一键生成所有模型对比图")
    print("="*60)

    # 预先计算数据
    prime_gaps, prime_mean, prime_std = generate_prime_data()
    chaos_gaps, chaos_mean, chaos_std = generate_chaos_data()

    results = {}

    for cfg in MODEL_CONFIGS:
        print(f"\n📦 处理 {cfg['name']}...")
        if not os.path.exists(cfg["path"]):
            print(f"  ⚠️ 权重文件不存在，跳过: {cfg['path']}")
            continue

        print(f"  ⏳ 加载权重...")
        model = load_model(cfg)

        # 选择对应的数据
        if cfg["task"] == "prime":
            gaps, mean, std = prime_gaps, prime_mean, prime_std
        else:
            gaps, mean, std = chaos_gaps, chaos_mean, chaos_std

        mae = analyze_prediction_accuracy(model, cfg, gaps, mean, std)
        svd_res = analyze_svd_properties(model, cfg)

        results[cfg["name"]] = {
            "mae": mae,
            "svd": svd_res
        }

        del model
        torch.cuda.empty_cache()

    # 打印汇总
    print("\n" + "="*60)
    print("📊 实验结果汇总")
    print("="*60)
    for name, res in results.items():
        print(f"\n【{name}】")
        print(f"  MAE = {res['mae']:.4f}")
        if res['svd']:
            print(f"  有效秩 = {res['svd']['eff_rank']:.2f}")
            print(f"  最大/平均比 = {res['svd']['max_ratio']:.2f}")
            print(f"  KS距离 = {res['svd']['ks_random']:.4f}")

    print("\n" + "="*60)
    print(f"✅ 所有图片已保存至: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()