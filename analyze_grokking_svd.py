import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob

# 导入模型定义（确保 base.py 在同一目录）
import base

# ==================== SVD 指标计算函数 ====================
def compute_svd_metrics(mat):
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()
    if mat.shape[0] < 2 or mat.shape[1] < 2:
        return None, None, None
    n = min(mat.shape[0], mat.shape[1], 256)
    mat = mat[:n, :n]
    try:
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None
    p = s / (np.sum(s) + 1e-12)
    H = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(H)
    max_ratio = s[0] / (np.mean(s) + 1e-12)
    np.random.seed(42)
    rand_mat = np.random.randn(n, n)
    u_rand, s_rand, vh_rand = np.linalg.svd(rand_mat, full_matrices=False)
    ks = stats.ks_2samp(s, s_rand).statistic
    return eff_rank, max_ratio, ks

def extract_weight_matrices(model):
    matrices = []
    for block in model.blocks:
        attn = block.attn
        d_model = attn.W_Q.shape[0] * attn.W_Q.shape[1]
        W_Q = attn.W_Q.detach().reshape(d_model, -1).cpu().numpy()
        W_K = attn.W_K.detach().reshape(d_model, -1).cpu().numpy()
        W_V = attn.W_V.detach().reshape(d_model, -1).cpu().numpy()
        W_O = attn.W_O.detach().reshape(d_model, -1).cpu().numpy()
        matrices.extend([W_Q, W_K, W_V, W_O])
    return matrices

def avg_svd_metrics_for_model(model):
    mats = extract_weight_matrices(model)
    effs, maxs, kss = [], [], []
    for mat in mats:
        eff, mr, ks = compute_svd_metrics(mat)
        if eff is not None:
            effs.append(eff)
            maxs.append(mr)
            kss.append(ks)
    if not effs:
        return None, None, None
    return np.mean(effs), np.mean(maxs), np.mean(kss)

def get_random_baseline(config, num_samples=3):
    effs, maxs, kss = [], [], []
    for seed in range(42, 42 + num_samples):
        torch.manual_seed(seed)
        model = base.Transformer(config).eval()
        eff, mr, ks = avg_svd_metrics_for_model(model)
        if eff is not None:
            effs.append(eff)
            maxs.append(mr)
            kss.append(ks)
    return {
        'eff_rank': np.mean(effs),
        'max_ratio': np.mean(maxs),
        'ks_random': np.mean(kss)
    }

def main():
    config = base.Config(
        p=97,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_mlp=512,
        n_ctx=3,
        bol_pos=True,
        act_type='ReLU',
        device=torch.device('cpu')
    )

    checkpoint_files = glob.glob("checkpoint_epoch_*.pt")
    def extract_epoch(fname):
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', fname)
        return int(match.group(1)) if match else 0
    checkpoint_files.sort(key=extract_epoch)

    if not checkpoint_files:
        print("未找到检查点文件")
        return

    print("正在计算随机基线...")
    baseline = get_random_baseline(config)

    epochs = []
    eff_ranks = []
    max_ratios = []
    ks_dists = []

    for fname in checkpoint_files:
        epoch = extract_epoch(fname)
        print(f"处理 epoch {epoch} ...")
        model = base.Transformer(config)
        state_dict = torch.load(fname, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        eff, mr, ks = avg_svd_metrics_for_model(model)
        if eff is not None:
            epochs.append(epoch)
            eff_ranks.append(eff)
            max_ratios.append(mr)
            ks_dists.append(ks)
        else:
            print(f"  警告：epoch {epoch} 指标无效，跳过")

    if len(epochs) == 0:
        print("没有有效指标")
        return

    # 保存数据到 CSV
    df = pd.DataFrame({
        'epoch': epochs,
        'effective_rank': eff_ranks,
        'max_mean_ratio': max_ratios,
        'ks_distance': ks_dists
    })
    df.to_csv('grokking_svd_data.csv', index=False)
    print("数据已保存到 grokking_svd_data.csv")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(epochs, eff_ranks, 'o-', linewidth=2, markersize=4, color='blue')
    axes[0].axhline(y=baseline['eff_rank'], color='gray', linestyle='--', label=f"Random baseline ({baseline['eff_rank']:.2f})")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title('Effective Rank')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, max_ratios, 'o-', color='red', linewidth=2, markersize=4)
    axes[1].axhline(y=baseline['max_ratio'], color='gray', linestyle='--', label=f"Random baseline ({baseline['max_ratio']:.2f})")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Max/Mean Ratio')
    axes[1].set_title('Max/Mean Ratio')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, ks_dists, 'o-', color='green', linewidth=2, markersize=4)
    axes[2].axhline(y=baseline['ks_random'], color='gray', linestyle='--', label=f"Random baseline ({baseline['ks_random']:.2f})")
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KS Distance')
    axes[2].set_title('KS Distance to Random')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('grokking_svd_analysis.png', dpi=300)
    plt.savefig('grokking_svd_analysis.pdf')
    plt.show()
    print("✅ 图表已保存为 grokking_svd_analysis.png 和 .pdf")

if __name__ == "__main__":
    main()