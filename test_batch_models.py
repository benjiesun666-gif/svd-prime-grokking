import os
# 设置缓存目录（必须在导入 huggingface_hub 之前）
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

# ==================== SVD 指标计算函数 ====================
def compute_svd_metrics(W):
    """
    输入：二维 numpy 数组 W（权重矩阵）
    返回：有效秩、最大/平均比、KS距离（与随机高斯矩阵比较）
    """
    if W.ndim != 2 or W.shape[0] < 2 or W.shape[1] < 2:
        return None, None, None
    # 截取方阵，最多 256 维（与论文一致）
    n = min(W.shape[0], W.shape[1], 256)
    W_square = W[:n, :n]
    try:
        U, S, Vh = np.linalg.svd(W_square, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None, None

    # 有效秩
    total = np.sum(S)
    p = S / total
    H = -np.sum(p * np.log(p + 1e-12))
    eff_rank = np.exp(H)

    # 最大/平均比
    max_ratio = S[0] / np.mean(S)

    # KS 距离（与固定随机种子生成的随机矩阵比较）
    np.random.seed(42)
    rand_mat = np.random.randn(n, n)
    U_rand, S_rand, Vh_rand = np.linalg.svd(rand_mat, full_matrices=False)
    ks = stats.ks_2samp(S, S_rand).statistic

    return eff_rank, max_ratio, ks

def convert_numpy(obj):
    """递归将 NumPy 类型转换为 Python 原生类型（用于 JSON 序列化）"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

# ==================== 绘图函数（生成 PDF）====================
def plot_svd_comparison(model_summary, save_path="svd_language_models.pdf"):
    """
    model_summary: dict {model_name: (eff_rank, max_ratio, ks)}
    """
    models = list(model_summary.keys())
    effs = [model_summary[m][0] for m in models]
    maxs = [model_summary[m][1] for m in models]
    kss = [model_summary[m][2] for m in models]

    # 计算随机基线（使用同尺寸高斯矩阵）
    np.random.seed(42)
    baseline_effs, baseline_maxs, baseline_kss = [], [], []
    for _ in range(3):
        rand_mat = np.random.randn(256, 256)
        eff, mr, ks = compute_svd_metrics(rand_mat)
        if eff is not None:
            baseline_effs.append(eff)
            baseline_maxs.append(mr)
            baseline_kss.append(ks)
    baseline_eff = np.mean(baseline_effs) if baseline_effs else 0
    baseline_max = np.mean(baseline_maxs) if baseline_maxs else 0
    baseline_ks = np.mean(baseline_kss) if baseline_kss else 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(models))

    # 有效秩
    axes[0].bar(x, effs, color='steelblue', alpha=0.8)
    axes[0].axhline(y=baseline_eff, color='gray', linestyle='--', label=f'Random baseline ({baseline_eff:.1f})')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title('Effective Rank (lower = more structured)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()

    # 最大/平均比
    axes[1].bar(x, maxs, color='coral', alpha=0.8)
    axes[1].axhline(y=baseline_max, color='gray', linestyle='--', label=f'Random baseline ({baseline_max:.2f})')
    axes[1].set_ylabel('Max/Mean Ratio')
    axes[1].set_title('Max/Mean Ratio (higher = more structured)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()

    # KS 距离
    axes[2].bar(x, kss, color='mediumseagreen', alpha=0.8)
    axes[2].axhline(y=baseline_ks, color='gray', linestyle='--', label=f'Random baseline ({baseline_ks:.3f})')
    axes[2].set_ylabel('KS Distance')
    axes[2].set_title('KS Distance to Random')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 图表已保存: {save_path}")
    plt.show()

# ==================== 批量模型分析 ====================
MODELS_TO_ANALYZE = [
    ("Qwen/Qwen2.5-0.5B", "model.safetensors", [0, 5, 11]),
    ("Qwen/Qwen2.5-1.5B", "model.safetensors", [0, 9, 17]),
    ("Qwen/Qwen2.5-3B", "model-00001-of-00002.safetensors", [0, 12, 23]),
    ("openai-community/gpt2", "model.safetensors", [0, 5, 11]),
    ("openai-community/gpt2-medium", "model.safetensors", [0, 11, 23]),
    ("openai-community/gpt2-large", "model.safetensors", [0, 17, 35]),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.safetensors", [0, 10, 21]),
    ("allenai/OLMo-1B-hf", "model.safetensors", [0, 7, 15]),
    ("EleutherAI/pythia-410m", "model.safetensors", [0, 11, 23]),
    ("EleutherAI/pythia-1b", "model.safetensors", [0, 7, 15]),
    ("bigscience/bloom-560m", "model.safetensors", [0, 11, 23]),
    # 以下两个模型可能不是 safetensors 格式，可暂时注释或保留（如果出错会捕获异常）
    # ("facebook/opt-350m", "model.safetensors", [0, 11, 23]),
    # ("facebook/opt-1.3b", "model.safetensors", [0, 11, 23]),
]

print("🚀 语言模型 SVD 分析（与主实验一致的方法）")
print("=" * 70)
print(f"目标模型数: {len(MODELS_TO_ANALYZE)}")
print(f"预计总样本数: {sum(len(layers) for _, _, layers in MODELS_TO_ANALYZE)} (每模型3层)")
print("=" * 70)

all_results = {}          # 存放各层详细结果
model_averages = {}       # 存放每个模型的平均指标
total_analyzed = 0
total_errors = 0

for repo_id, filename, layer_indices in MODELS_TO_ANALYZE:
    model_name = repo_id.split('/')[-1]
    print(f"\n{'='*70}")
    print(f"📦 模型: {model_name}")
    print(f"{'='*70}")

    try:
        print(f"📥 加载权重（使用缓存）...")
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            resume_download=True,
            local_dir_use_symlinks=False,
            # 如果希望强制只使用本地缓存，可取消下一行的注释
            # local_files_only=True
        )
        print(f"   本地路径: {file_path}")

        weights = load_file(file_path)

        # 查找注意力层权重（支持多种命名规范）
        attn_patterns = [
            'self_attn.q_proj.weight',
            'attn.q_proj.weight',
            'attn.c_attn.weight',
            'attention.query_key_value.weight',
            'self_attention.query_key_value.weight'
        ]
        attn_keys = []
        for pattern in attn_patterns:
            found = [k for k in weights.keys() if pattern in k]
            if found:
                attn_keys = found
                break
        print(f"✅ 找到 {len(attn_keys)} 个注意力层")

        if model_name not in all_results:
            all_results[model_name] = {}

        layer_effs, layer_maxs, layer_kss = [], [], []

        for layer_idx in layer_indices:
            if layer_idx < len(attn_keys):
                key = attn_keys[layer_idx]
                layer_name = f"Layer {layer_idx}"
                print(f"  🔬 分析 {layer_name}...", end=' ')

                W = weights[key].float().numpy()
                eff, mr, ks = compute_svd_metrics(W)

                if eff is not None:
                    all_results[model_name][layer_name] = {
                        'eff_rank': eff,
                        'max_ratio': mr,
                        'ks_random': ks
                    }
                    layer_effs.append(eff)
                    layer_maxs.append(mr)
                    layer_kss.append(ks)
                    print(f"✅ eff={eff:.2f}, mr={mr:.2f}, ks={ks:.4f}")
                    total_analyzed += 1
                else:
                    print(f"❌ 指标计算失败")
            else:
                print(f"  ⚠️ Layer {layer_idx} 不存在")

        if layer_effs:
            model_averages[model_name] = (
                np.mean(layer_effs),
                np.mean(layer_maxs),
                np.mean(layer_kss)
            )

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        total_errors += 1

# ==================== 统计汇总 ====================
print(f"\n\n{'='*70}")
print("📊 最终统计")
print(f"{'='*70}")

for model_name, avg in model_averages.items():
    print(f"\n【{model_name}】")
    print(f"  平均有效秩 = {avg[0]:.2f}")
    print(f"  平均最大比 = {avg[1]:.2f}")
    print(f"  平均KS距离 = {avg[2]:.4f}")

print(f"\n✅ 成功分析: {total_analyzed} 个样本")
print(f"❌ 失败: {total_errors} 个模型")

# ==================== 保存结果到 JSON ====================
results_json = {
    'timestamp': datetime.now().isoformat(),
    'total_samples': total_analyzed,
    'total_models': len(model_averages),
    'detailed_results': all_results,
    'model_averages': {
        model: {'eff_rank': avg[0], 'max_ratio': avg[1], 'ks_random': avg[2]}
        for model, avg in model_averages.items()
    }
}

results_json = convert_numpy(results_json)

with open('svd_language_models_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print(f"\n💾 详细结果已保存: svd_language_models_analysis.json")

# ==================== 生成汇总图（PDF）====================
if model_averages:
    plot_svd_comparison(model_averages, save_path="svd_language_models.pdf")
else:
    print("⚠️ 没有有效模型结果，无法绘图")

print(f"\n✅ 分析完成！")
