"""
混沌映射 SVD 分析脚本
加载混沌映射的成功和失败模型，计算有效秩、最大比、KS距离，绘制SVD谱图。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# ========== 配置（请根据实际路径修改）==========
CHAOS_A_PATH = r"D:\pythonstudy\python_task\改进\chaos_plan_A_results\best_model.pt"
CHAOS_C_PATH = r"D:\pythonstudy\python_task\改进\chaos_plan_C_results\best_model.pt"
OUTPUT_DIR = "论文图片"
os.makedirs(OUTPUT_DIR, exist_ok=True)

D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 模型定义 ==========
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

class GapPredictor(nn.Module):
    def __init__(self, learnable_embedding=True):
        super().__init__()
        self.riemann_embedding = RiemannEmbedding(D_MODEL, learnable=learnable_embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL*4,
            dropout=DROPOUT, batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.output = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL//2, D_MODEL//4), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL//4, 1)
        )
    def forward(self, x):
        emb = self.riemann_embedding(x).unsqueeze(1)
        out = self.transformer(emb)
        return self.output(out.squeeze(1))
    def get_attention_weights(self):
        weights = []
        for layer in self.transformer.layers:
            weights.append(layer.self_attn.in_proj_weight.detach().cpu())
        return torch.cat(weights, dim=0)

def load_model(path, learnable):
    model = GapPredictor(learnable_embedding=learnable).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def svd_analysis(model):
    weights = model.get_attention_weights().numpy()
    n = min(weights.shape[0], weights.shape[1], 256)
    W = weights[:n, :n]
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    # 有效秩
    total = np.sum(S)
    p = S / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(entropy)
    # 最大比
    max_ratio = S[0] / np.mean(S)
    # 随机基线
    np.random.seed(42)
    W_rand = np.random.randn(n, n)
    S_rand = np.linalg.svd(W_rand, compute_uv=False)
    ks = stats.ks_2samp(S, S_rand).statistic
    return eff_rank, max_ratio, ks, S, S_rand

def plot_svd_spectra():
    models = {'success': (CHAOS_A_PATH, True), 'fail': (CHAOS_C_PATH, False)}
    colors = {'success': 'blue', 'fail': 'orange'}
    plt.figure(figsize=(8,5))
    svd_results = {}
    for key, (path, learnable) in models.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {key}.")
            continue
        model = load_model(path, learnable)
        eff, maxr, ks, S, S_rand = svd_analysis(model)
        svd_results[key] = {'eff_rank': eff, 'max_ratio': maxr, 'ks': ks}
        plt.semilogy(S, 'o-', markersize=3, color=colors[key], label=f'Chaotic {key} (eff_rank={eff:.1f}, max_ratio={maxr:.1f})')
        # 释放显存
        del model
        torch.cuda.empty_cache()
    # 绘制随机基线（用最后一次的S_rand）
    plt.semilogy(S_rand, '--', color='gray', alpha=0.7, label='Random baseline')
    plt.xlabel('Index')
    plt.ylabel('Singular value (log scale)')
    plt.title('Chaotic map SVD spectra')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chaos_svd_spectra.png'), dpi=300)
    plt.close()
    print("混沌SVD谱图已保存:", os.path.join(OUTPUT_DIR, 'chaos_svd_spectra.png'))
    return svd_results

if __name__ == "__main__":
    plot_svd_spectra()