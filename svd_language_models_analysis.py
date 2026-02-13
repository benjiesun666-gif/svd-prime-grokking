import os

# 🔥 关键：必须在导入 huggingface_hub 之前设置环境变量！
HF_CACHE_DIR = r"D:\pythonstudy\huggingface_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')
os.environ['HF_HUB_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime

print(f"💾 Hugging Face 缓存路径: {HF_CACHE_DIR}")
print(f"   HF_HOME: {os.environ['HF_HOME']}")
print(f"   HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
print(f"   (避免 C 盘空间不足)\n")


# ==================== SVD 分析函数（与主实验一致）====================
def analyze_svd(W):
    """
    输入方阵 W，计算 SVD 相关指标：
    - 有效秩 (effective rank)
    - 最大奇异值与平均奇异值之比
    - 与同尺寸随机高斯矩阵的 KS 距离
    """
    U, S, Vh = np.linalg.svd(W, full_matrices=False)

    # 有效秩
    total = np.sum(S)
    p = S / total
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(entropy)

    # 最大/平均比
    max_s = S[0]
    mean_s = np.mean(S)
    max_ratio = max_s / mean_s

    # 与随机基线的 KS 距离（固定种子保证可复现）
    np.random.seed(42)
    W_rand = np.random.randn(W.shape[0], W.shape[1])
    U_rand, S_rand, Vh_rand = np.linalg.svd(W_rand, full_matrices=False)
    ks_random = stats.ks_2samp(S, S_rand).statistic

    return eff_rank, max_ratio, ks_random, S


def plot_comparison(all_results, save_path="batch_analysis_summary.png"):
    """批量结果可视化（基于 SVD 指标）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = []
    eff_ranks = []
    max_ratios = []
    ks_randoms = []

    for model_name, res in all_results.items():
        models.append(model_name)
        eff_ranks.append(res['eff_rank'])
        max_ratios.append(res['max_ratio'])
        ks_randoms.append(res['ks_random'])

    x = np.arange(len(models))

    # 图1：有效秩
    ax1 = axes[0, 0]
    ax1.bar(x, eff_ranks, color='blue', alpha=0.7)
    ax1.set_ylabel('Effective Rank')
    ax1.set_title('Effective Rank (lower = more structured)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=256, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Full rank')
    ax1.legend()

    # 图2：最大/平均奇异值比
    ax2 = axes[0, 1]
    ax2.bar(x, max_ratios, color='red', alpha=0.7)
    ax2.set_ylabel('Max/Mean Ratio')
    ax2.set_title('Max/Mean Singular Value Ratio (higher = more structured)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(y=3, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Random baseline (~2-3)')
    ax2.legend()

    # 图3：与随机基线的 KS 距离
    ax3 = axes[1, 0]
    ax3.bar(x, ks_randoms, color='green', alpha=0.7)
    ax3.set_ylabel('KS Distance to Random')
    ax3.set_title('KS Distance from Random SVD (higher = more structured)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax3.grid(alpha=0.3, axis='y')
    ax3.axhline(y=0.05, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Random similarity')
    ax3.legend()

    # 图4：散点图（有效秩 vs 最大比）
    ax4 = axes[1, 1]
    ax4.scatter(eff_ranks, max_ratios, alpha=0.7, s=100)
    for i, model in enumerate(models):
        ax4.annotate(model, (eff_ranks[i], max_ratios[i]), fontsize=6, alpha=0.7)
    ax4.set_xlabel('Effective Rank')
    ax4.set_ylabel('Max/Mean Ratio')
    ax4.set_title('Effective Rank vs Max/Mean Ratio')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 汇总图已保存: {save_path}")
    plt.show()


# 辅助函数：将 numpy 类型转换为 Python 原生类型以便 JSON 序列化
def convert_numpy(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj


# ==================== 批量模型分析 ====================
MODELS_TO_ANALYZE = [
    ("Qwen/Qwen2.5-0.5B", "model.safetensors"),
    ("Qwen/Qwen2.5-1.5B", "model.safetensors"),
    ("Qwen/Qwen2.5-3B", "model-00001-of-00002.safetensors"),
    ("openai-community/gpt2", "model.safetensors"),
    ("openai-community/gpt2-medium", "model.safetensors"),
    ("openai-community/gpt2-large", "model.safetensors"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.safetensors"),
    ("allenai/OLMo-1B-hf", "model.safetensors"),
    ("EleutherAI/pythia-410m", "model.safetensors"),
    ("EleutherAI/pythia-1b", "model.safetensors"),
    ("bigscience/bloom-560m", "model.safetensors"),
    # OPT 系列文件可能不是 safetensors，暂时跳过
    # ("facebook/opt-350m", "model.safetensors"),
    # ("facebook/opt-1.3b", "model.safetensors"),
]

print("🚀 大规模 Transformer 架构 SVD 分析（与主实验一致的方法）")
print("=" * 70)
print(f"目标模型数: {len(MODELS_TO_ANALYZE)}")
print("=" * 70)

all_results = {}
total_analyzed = 0
total_errors = 0

for repo_id, filename in MODELS_TO_ANALYZE:
    model_name = repo_id.split('/')[-1]
    print(f"\n{'=' * 70}")
    print(f"📦 模型: {model_name}")
    print(f"{'=' * 70}")

    try:
        print(f"📥 下载/加载权重...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"   本地路径: {file_path}")

        weights = load_file(file_path)

        # 查找所有注意力层权重矩阵（多种命名规范）
        attn_patterns = [
            'self_attn.q_proj.weight',
            'attn.q_proj.weight',
            'attn.c_attn.weight',
            'attention.query_key_value.weight',
            'self_attention.query_key_value.weight'
        ]
        weight_matrices = []
        for pattern in attn_patterns:
            found = [weights[k] for k in weights.keys() if pattern in k]
            if found:
                weight_matrices.extend(found)
                break  # 假设只匹配一种模式，所有层使用相同命名
        print(f"✅ 收集到 {len(weight_matrices)} 个注意力层权重矩阵")

        if len(weight_matrices) == 0:
            print("  ⚠️ 未找到注意力层权重，跳过")
            continue

        # 将所有矩阵转为 numpy 并检查维度
        np_matrices = []
        for w in weight_matrices:
            w_np = w.float().cpu().numpy()
            # 确保是二维且至少有一维 >=256（便于截取）
            if w_np.ndim == 2 and w_np.shape[0] >= 256 and w_np.shape[1] >= 256:
                np_matrices.append(w_np)
            else:
                print(f"  ⚠️ 跳过形状 {w_np.shape} 的矩阵（至少需要256维）")

        if len(np_matrices) == 0:
            print("  ⚠️ 没有满足维度要求的矩阵，跳过")
            continue

        # 拼接所有矩阵（沿第一维）
        W_huge = np.concatenate(np_matrices, axis=0)

        # 确保至少有256行和256列
        if W_huge.shape[0] < 256 or W_huge.shape[1] < 256:
            print(f"  ⚠️ 拼接后矩阵形状 {W_huge.shape} 小于256，无法分析")
            continue

        # 截取前256行和前256列
        W = W_huge[:256, :256]

        # 执行 SVD 分析
        eff_rank, max_ratio, ks_random, S = analyze_svd(W)

        all_results[model_name] = {
            'eff_rank': eff_rank,
            'max_ratio': max_ratio,
            'ks_random': ks_random
        }
        print(f"  ✅ 有效秩 = {eff_rank:.2f}, 最大/平均比 = {max_ratio:.2f}, KS随机距离 = {ks_random:.4f}")
        total_analyzed += 1

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        total_errors += 1

# ==================== 统计汇总 ====================
print(f"\n\n{'=' * 70}")
print("📊 最终统计")
print(f"{'=' * 70}")

eff_ranks = []
max_ratios = []
ks_randoms = []

for model_name, res in all_results.items():
    print(f"\n【{model_name}】")
    print(f"  有效秩 = {res['eff_rank']:.2f}")
    print(f"  最大/平均奇异值比 = {res['max_ratio']:.2f}")
    print(f"  KS距离 vs 随机基线 = {res['ks_random']:.4f}")
    eff_ranks.append(res['eff_rank'])
    max_ratios.append(res['max_ratio'])
    ks_randoms.append(res['ks_random'])

print(f"\n✅ 成功分析: {total_analyzed} 个模型")
print(f"❌ 失败: {total_errors} 个模型")

print("\n📈 统计摘要:")
print(f"  有效秩: 均值={np.mean(eff_ranks):.2f}, 标准差={np.std(eff_ranks):.2f}")
print(f"  最大/平均比: 均值={np.mean(max_ratios):.2f}, 标准差={np.std(max_ratios):.2f}")
print(f"  KS随机距离: 均值={np.mean(ks_randoms):.4f}, 标准差={np.std(ks_randoms):.4f}")

# 保存结果到 JSON（先转换 numpy 类型）
results_json = {
    'timestamp': datetime.now().isoformat(),
    'total_models': total_analyzed,
    'statistics': {
        'eff_rank_mean': float(np.mean(eff_ranks)),
        'eff_rank_std': float(np.std(eff_ranks)),
        'max_ratio_mean': float(np.mean(max_ratios)),
        'max_ratio_std': float(np.std(max_ratios)),
        'ks_random_mean': float(np.mean(ks_randoms)),
        'ks_random_std': float(np.std(ks_randoms)),
    },
    'detailed_results': all_results
}

# 转换 numpy 类型
results_json = convert_numpy(results_json)

with open('transformer_svd_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print(f"\n💾 详细结果已保存: transformer_svd_analysis.json")

# 生成汇总图
plot_comparison(all_results, save_path="transformer_batch_analysis.png")

print("\n🎯 最终结论")
print("分析完成。可将这些 SVD 指标与素数模型（Plan A/B/C）的结果进行对比。")
print(f"\n✅ 分析完成！共处理 {total_analyzed} 个样本")