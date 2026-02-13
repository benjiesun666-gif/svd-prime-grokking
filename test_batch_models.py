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

# ==================== 与主实验完全一致的 GOE 分析函数 ====================
def analyze_layer_weights(W):
    """
    严格按照主实验（素数模型）的方法分析单层权重矩阵
    返回 verdict, ks_goe, ks_poisson
    """
    # 1. 确保方阵：取前 min(rows,cols) 行和列
    rows, cols = W.shape
    n = min(rows, cols)
    W_square = W[:n, :n]

    # 2. 对称化
    H = (W_square + W_square.T) / 2

    # 3. 特征值（实数）
    eigvals = np.linalg.eigvalsh(H)
    eigvals = np.sort(eigvals)

    # 4. 取中间 70% 的特征值（去掉两端各 15%）
    low = int(len(eigvals) * 0.15)
    high = int(len(eigvals) * 0.85)
    eigvals = eigvals[low:high]

    # 5. 能级间距 & 归一化
    spacings = np.diff(eigvals)
    if np.mean(spacings) == 0:
        return "Poisson", 1.0, 1.0  # 异常情况
    s = spacings / np.mean(spacings)

    # 6. 截断到 [0,4]（与主实验一致，避免极端值影响 KS）
    s = s[s <= 4]

    # 7. GOE 和 Poisson 的 CDF
    def goe_cdf(x):
        return 1 - np.exp(-np.pi * x ** 2 / 4)

    def poisson_cdf(x):
        return 1 - np.exp(-x)

    # 8. KS 距离
    ks_goe = stats.kstest(s, goe_cdf).statistic
    ks_poisson = stats.kstest(s, poisson_cdf).statistic

    # 9. 判决：直接比较 KS 值（与主实验判决逻辑一致）
    verdict = "GOE" if ks_goe < ks_poisson else "Poisson"

    return verdict, ks_goe, ks_poisson


def plot_comparison(all_results, save_path="batch_analysis_summary.png"):
    """批量结果可视化（与原来一致，仅需 KS 值）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = []
    ks_goe_list = []
    ks_poisson_list = []
    verdicts = []

    for model_name, layers in all_results.items():
        for layer_name, res in layers.items():
            if layer_name == '_metadata' or not isinstance(res, dict) or 'ks_goe' not in res:
                continue
            models.append(f"{model_name}\n{layer_name}")
            ks_goe_list.append(res['ks_goe'])
            ks_poisson_list.append(res['ks_poisson'])
            verdicts.append(res['verdict'])

    # 图1：KS距离对比（条形图）
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width / 2, ks_goe_list, width, label='KS(GOE)', alpha=0.8, color='red')
    ax1.bar(x + width / 2, ks_poisson_list, width, label='KS(Poisson)', alpha=0.8, color='green')
    ax1.set_ylabel('KS Distance')
    ax1.set_title('KS统计量对比（越小越好）')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0.05, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # 图2：判决统计（饼图）
    ax2 = axes[0, 1]
    verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}
    colors = {'GOE': 'red', 'Poisson': 'green'}
    ax2.pie(verdict_counts.values(), labels=verdict_counts.keys(), autopct='%1.1f%%',
            colors=[colors.get(k, 'gray') for k in verdict_counts.keys()],
            startangle=90)
    ax2.set_title(f'所有样本判决分布 (n={len(models)})')

    # 图3：KS差值
    ax3 = axes[1, 0]
    ks_diff = np.array(ks_goe_list) - np.array(ks_poisson_list)
    colors_bar = ['green' if d > 0 else 'red' for d in ks_diff]
    ax3.barh(models, ks_diff, color=colors_bar, alpha=0.7)
    ax3.set_xlabel('KS(GOE) - KS(Poisson)')
    ax3.set_title('Poisson优势度（正值=支持Poisson）')
    ax3.axvline(x=0, color='black', linewidth=1.5)
    ax3.grid(alpha=0.3, axis='x')
    ax3.tick_params(axis='y', labelsize=8)

    # 图4：散点图
    ax4 = axes[1, 1]
    for v in set(verdicts):
        mask = [ver == v for ver in verdicts]
        color = 'red' if v == 'GOE' else 'green'
        ax4.scatter(np.array(ks_goe_list)[mask], np.array(ks_poisson_list)[mask],
                    label=v, alpha=0.7, s=100, color=color)
    ax4.plot([0, max(ks_goe_list + ks_poisson_list)],
             [0, max(ks_goe_list + ks_poisson_list)],
             'k--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('KS(GOE)')
    ax4.set_ylabel('KS(Poisson)')
    ax4.set_title('KS距离散点图')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 汇总图已保存: {save_path}")
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
    ("facebook/opt-350m", "model.safetensors", [0, 11, 23]),
    ("facebook/opt-1.3b", "model.safetensors", [0, 11, 23]),
]

print("🚀 大规模 Transformer 架构谱分析（与主实验一致的方法）")
print("=" * 70)
print(f"目标模型数: {len(MODELS_TO_ANALYZE)}")
print(f"预计总样本数: {sum(len(layers) for _, _, layers in MODELS_TO_ANALYZE)} (每模型3层)")
print("=" * 70)

all_results = {}
total_analyzed = 0
total_errors = 0

for repo_id, filename, layer_indices in MODELS_TO_ANALYZE:
    model_name = repo_id.split('/')[-1]
    print(f"\n{'='*70}")
    print(f"📦 模型: {model_name}")
    print(f"{'='*70}")

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

        # 查找注意力层权重（多种命名规范）
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
            all_results[model_name] = {
                '_metadata': {
                    'repo_id': repo_id,
                    'filename': filename,
                    'total_layers': len(attn_keys)
                }
            }

        # 分析指定层
        for layer_idx in layer_indices:
            if layer_idx < len(attn_keys):
                key = attn_keys[layer_idx]
                layer_name = f"Layer {layer_idx}"
                print(f"  🔬 分析 {layer_name}...", end=' ')

                W = weights[key].float().numpy()
                verdict, ks_goe, ks_poisson = analyze_layer_weights(W)

                all_results[model_name][layer_name] = {
                    'verdict': verdict,
                    'ks_goe': ks_goe,
                    'ks_poisson': ks_poisson
                }

                print(f"✅ {verdict} (KS_GOE={ks_goe:.4f}, KS_Poisson={ks_poisson:.4f})")
                total_analyzed += 1
            else:
                print(f"  ⚠️ Layer {layer_idx} 不存在")

    except Exception as e:
        print(f"  ❌ 错误: {e}")
        total_errors += 1

# ==================== 统计汇总 ====================
print(f"\n\n{'='*70}")
print("📊 最终统计")
print(f"{'='*70}")

all_verdicts = []
all_ks_goe = []
all_ks_poisson = []

for model_name, layers in all_results.items():
    print(f"\n【{model_name}】")
    for layer_name, res in layers.items():
        if layer_name == '_metadata':
            continue
        if not isinstance(res, dict) or 'verdict' not in res:
            continue
        print(f"  {layer_name}: {res['verdict']} (KS_GOE={res['ks_goe']:.4f}, KS_Poisson={res['ks_poisson']:.4f})")
        all_verdicts.append(res['verdict'])
        all_ks_goe.append(res['ks_goe'])
        all_ks_poisson.append(res['ks_poisson'])

print(f"\n✅ 成功分析: {total_analyzed} 个样本")
print(f"❌ 失败: {total_errors} 个模型")

# 总体判决分布
verdict_counts = {}
for v in all_verdicts:
    verdict_counts[v] = verdict_counts.get(v, 0) + 1
print("\n📈 总体判决分布:")
for verdict, count in verdict_counts.items():
    print(f"   {verdict}: {count} ({count/len(all_verdicts)*100:.1f}%)")

print("\n📉 KS距离统计:")
print(f"   GOE:     均值={np.mean(all_ks_goe):.4f}, 标准差={np.std(all_ks_goe):.4f}")
print(f"   Poisson: 均值={np.mean(all_ks_poisson):.4f}, 标准差={np.std(all_ks_poisson):.4f}")

# 配对t检验
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(all_ks_goe, all_ks_poisson)
print(f"\n🔬 配对t检验 (GOE vs Poisson): t={t_stat:.4f}, p={p_value:.4e}")
if p_value < 0.001:
    winner = "Poisson" if t_stat > 0 else "GOE"
    print(f"   ✅ {winner} 显著更优 (p<0.001)")

# 保存结果
results_json = {
    'timestamp': datetime.now().isoformat(),
    'total_samples': total_analyzed,
    'verdict_distribution': verdict_counts,
    'ks_statistics': {
        'goe_mean': float(np.mean(all_ks_goe)),
        'poisson_mean': float(np.mean(all_ks_poisson))
    },
    'ttest': {'t_statistic': float(t_stat), 'p_value': float(p_value)},
    'detailed_results': {
        model: {layer: res for layer, res in layers.items() if layer != '_metadata'}
        for model, layers in all_results.items()
    }
}
with open('transformer_spectra_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print(f"\n💾 详细结果已保存: transformer_spectra_analysis.json")

# 生成汇总图
plot_comparison(all_results, save_path="transformer_batch_analysis.png")

print("\n🎯 最终结论")
if verdict_counts.get('Poisson', 0) > total_analyzed * 0.7:
    print("✅ Transformer 架构整体展现 Poisson 分布特征")
    print("   这意味着你的素数模型的 GOE 特征很可能是素数特有的！")
elif verdict_counts.get('GOE', 0) > total_analyzed * 0.7:
    print("⚠️ Transformer 架构整体展现 GOE 分布特征")
    print("   这会削弱素数模型 GOE 特征的独特性，需进一步检查")
else:
    print("⚠️ 结果混合，需进一步分析")

print(f"\n✅ 分析完成！共处理 {total_analyzed} 个样本")
