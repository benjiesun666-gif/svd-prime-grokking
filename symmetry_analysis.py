"""
重新生成 PCA 行向量投影图（用于 Figure 4）
基于最新模型权重，输出与表3一致的指标，并保存拼接图片。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# ========== 配置路径（请根据您的实际路径修改）==========
PLOT_MODELS = [
    {"key": "prime_A", "path": r"D:\illtenligence\Plan A.pt", "learnable": True},
    {"key": "prime_B", "path": r"D:\illtenligence\Plan B.pt", "learnable": True},
    {"key": "prime_C", "path": r"D:\illtenligence\Plan C.pt", "learnable": False},
    {"key": "chaos_A", "path": r"D:\pythonstudy\python_task\改进\chaos_plan_A_results\chaotic_success.pt", "learnable": True},
    {"key": "chaos_C", "path": r"D:\pythonstudy\python_task\改进\chaos_plan_C_results\chaotic_fail.pt", "learnable": False},
]

OUTPUT_DIR = "论文图片"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型参数
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 模型定义（与训练时一致）==========
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
    model = PrimeGapPredictor(learnable_embedding=learnable).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def plot_pca_clusters():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, cfg in enumerate(PLOT_MODELS):
        ax = axes[idx]
        path = cfg["path"]
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f'{cfg["key"]} not found', ha='center', va='center')
            ax.set_title(cfg["key"])
            continue

        print(f"Processing {cfg['key']}...")
        model = load_model(path, cfg["learnable"])
        weights = model.get_attention_weights().numpy()
        q_part = weights[:D_MODEL, :]  # (256,256)

        # PCA
        pca = PCA(n_components=2)
        rows_pca = pca.fit_transform(q_part)

        # KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rows_pca)
        centers = kmeans.cluster_centers_

        # 计算指标
        inter_dist = np.linalg.norm(centers[0] - centers[1])
        size0 = np.sum(labels == 0)
        size1 = np.sum(labels == 1)
        balance = 1 - abs(size0 - size1) / (size0 + size1)
        intra0 = np.mean([np.linalg.norm(rows_pca[i] - centers[0]) for i in range(len(rows_pca)) if labels[i] == 0])
        intra1 = np.mean([np.linalg.norm(rows_pca[i] - centers[1]) for i in range(len(rows_pca)) if labels[i] == 1])
        intra_avg = (intra0 + intra1) / 2
        sym_quality = inter_dist / (intra_avg + 1e-8)

        print(f"  inter={inter_dist:.4f}, balance={balance:.4f}, qual={sym_quality:.4f}")

        # 绘图
        colors = ['red', 'blue']
        for i in range(2):
            ax.scatter(rows_pca[labels == i, 0], rows_pca[labels == i, 1],
                       c=colors[i], alpha=0.6, s=20, label=f'Cluster {i}')
        ax.set_title(f'{cfg["key"]}\ninter={inter_dist:.2f}, balance={balance:.3f}, qual={sym_quality:.2f}')
        ax.legend()

        del model
        torch.cuda.empty_cache()

    # 隐藏多余的子图
    for j in range(len(PLOT_MODELS), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'pca_clusters_all.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n图片已保存至: {out_path}")

if __name__ == "__main__":
    plot_pca_clusters()
