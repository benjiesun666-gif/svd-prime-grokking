"""
对称性分析脚本
对每个模型的第一层注意力权重的Q部分进行PCA降维、KMeans聚类，计算对称性指标并绘图。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import sys

# ========== 配置路径（请根据实际路径修改）==========
PATHS = {
    'prime_A': r"D:\illtenligence\Plan A.pt",
    'prime_B': r"D:\illtenligence\Plan B.pt",
    'prime_C': r"D:\illtenligence\Plan C.pt",
    'chaos_A': r"D:\pythonstudy\python_task\改进\chaos_plan_A_results\best_model.pt",
    'chaos_C': r"D:\pythonstudy\python_task\改进\chaos_plan_C_results\best_model.pt",
}
LEARNABLE = {
    'prime_A': True,
    'prime_B': True,
    'prime_C': False,
    'chaos_A': True,
    'chaos_C': False,
}
OUTPUT_DIR = "论文图片"
os.makedirs(OUTPUT_DIR, exist_ok=True)

D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 模型定义（同之前）==========
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

def symmetry_analysis(model):
    """对第一层注意力权重的Q部分进行PCA+KMeans，返回指标和PCA数据"""
    weights = model.get_attention_weights().numpy()
    # 取第一层的Q部分（前D_MODEL行）
    q_part = weights[:D_MODEL, :]  # (256,256)
    # PCA降维到2维
    pca = PCA(n_components=2)
    rows_pca = pca.fit_transform(q_part)
    # KMeans聚类
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rows_pca)
    centers = kmeans.cluster_centers_  # (2,2)
    # 簇间距离
    inter_dist = np.linalg.norm(centers[0] - centers[1])
    # 簇大小均衡度
    size0 = np.sum(labels == 0)
    size1 = np.sum(labels == 1)
    balance = 1 - abs(size0 - size1) / (size0 + size1)
    # 簇内平均距离
    intra_dist0 = np.mean([np.linalg.norm(rows_pca[i] - centers[0]) for i in range(len(rows_pca)) if labels[i]==0])
    intra_dist1 = np.mean([np.linalg.norm(rows_pca[i] - centers[1]) for i in range(len(rows_pca)) if labels[i]==1])
    intra_avg = (intra_dist0 + intra_dist1) / 2
    sym_quality = inter_dist / (intra_avg + 1e-8)
    return inter_dist, balance, sym_quality, rows_pca, labels, pca

def plot_all_symmetry():
    models_to_plot = ['prime_A', 'prime_B', 'prime_C', 'chaos_A', 'chaos_C']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    results = {}
    for idx, key in enumerate(models_to_plot):
        if idx >= len(axes):
            break
        path = PATHS.get(key)
        if not path or not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {key}.")
            axes[idx].text(0.5, 0.5, f'{key} not found', ha='center', va='center')
            axes[idx].set_title(key)
            continue
        model = load_model(path, LEARNABLE[key])
        inter_dist, balance, sym_quality, rows_pca, labels, pca = symmetry_analysis(model)
        results[key] = {
            'inter_dist': inter_dist,
            'balance': balance,
            'sym_quality': sym_quality,
            'pca_explained': pca.explained_variance_ratio_
        }
        # 绘制散点图，用不同颜色标记簇
        for i in range(2):
            axes[idx].scatter(rows_pca[labels==i,0], rows_pca[labels==i,1],
                              c=['red','blue'][i], alpha=0.6, s=20, label=f'Cluster {i}')
        axes[idx].set_title(f'{key}\ninter={inter_dist:.2f}, balance={balance:.3f}, qual={sym_quality:.2f}')
        axes[idx].legend()
        # 释放模型
        del model
        torch.cuda.empty_cache()
    # 隐藏多余的子图
    for j in range(len(models_to_plot), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_clusters_all.png'), dpi=300)
    plt.close()
    print("对称性PCA图已保存:", os.path.join(OUTPUT_DIR, 'pca_clusters_all.png'))
    return results

if __name__ == "__main__":
    res = plot_all_symmetry()
    # 打印汇总结果
    print("\n对称性指标汇总:")
    for key, val in res.items():
        print(f"{key}: inter={val['inter_dist']:.2f}, balance={val['balance']:.3f}, quality={val['sym_quality']:.2f}")