import os

# 🔥 关键：必须在导入 huggingface_hub 之前设置环境变量！
HF_CACHE_DIR = r"D:\pythonstudy\huggingface_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')
os.environ['HF_HUB_CACHE'] = os.path.join(HF_CACHE_DIR, 'hub')
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

# 现在才导入 huggingface_hub
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

# ==================== GOE 混沌分析核心函数 ====================
def hermitize(W):
    """厄米化：(W + W^T) / 2，确保实特征值
    
    如果 W 不是方阵（如 GPT-2 的 c_attn 包含 Q+K+V），
    先转换为方阵：W @ W.T 或 W.T @ W（取较小的那个）
    """
    if W.shape[0] != W.shape[1]:
        # 非方阵：选择较小维度构造方阵
        if W.shape[0] < W.shape[1]:
            W = W @ W.T  # (m, n) @ (n, m) = (m, m)
        else:
            W = W.T @ W  # (n, m) @ (m, n) = (n, n)
    
    return (W + W.T) / 2

def analyze_spectra_goe_silent(W_matrix, title="Layer"):
    """
    静默版 GOE 分析（不打印，只返回结果）
    注意：使用 GOE 作为参考分布，因为神经网络权重是实数矩阵
    """
    # 1. 厄米化
    H = hermitize(W_matrix)
    
    # 2. 计算特征值（实数）
    eigenvalues = np.linalg.eigvalsh(H)
    eigenvalues = np.sort(eigenvalues)
    
    # 3. 计算相邻能级间隙
    spacings = np.diff(eigenvalues)
    
    # 4. 归一化：s = Δλ / <Δλ>
    mean_spacing = np.mean(spacings)
    s = spacings / mean_spacing
    
    # 5. 统计检验（使用 GOE 作为参考，因为权重是实数矩阵）
    def goe_cdf_approx(x):
        # GOE: P(s) = (π/2) * s * exp(-π*s²/4)
        # CDF: 1 - exp(-π*s²/4)
        return 1 - np.exp(-np.pi * x**2 / 4)
    
    ks_goe, p_goe = stats.kstest(s, goe_cdf_approx)
    ks_poisson, p_poisson = stats.kstest(s, lambda x: 1 - np.exp(-x))
    
    # 6. 自动判决
    if ks_goe < ks_poisson * 0.7:
        verdict = "GOE"
        confidence = ks_poisson / ks_goe
    elif ks_poisson < ks_goe * 0.7:
        verdict = "Poisson"
        confidence = ks_goe / ks_poisson
    else:
        verdict = "Mixed"
        confidence = abs(ks_goe - ks_poisson) / min(ks_goe, ks_poisson)
    
    return {
        'eigenvalues': eigenvalues,
        'spacings': spacings,
        's_normalized': s,
        'ks_goe': ks_goe,
        'ks_poisson': ks_poisson,
        'p_goe': p_goe,
        'p_poisson': p_poisson,
        'verdict': verdict,
        'confidence': confidence,
        'matrix_shape': W_matrix.shape
    }

def plot_comparison(all_results, save_path="batch_analysis_summary.png"):
    """批量结果可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 提取数据
    models = []
    ks_goe_list = []
    ks_poisson_list = []
    verdicts = []
    
    for model_name, layers in all_results.items():
        for layer_name, result in layers.items():
            # 跳过元数据
            if layer_name == '_metadata' or not isinstance(result, dict) or 'verdict' not in result:
                continue
            
            models.append(f"{model_name}\n{layer_name}")
            ks_goe_list.append(result['ks_goe'])
            ks_poisson_list.append(result['ks_poisson'])
            verdicts.append(result['verdict'])
    
    # 图1：KS距离对比（条形图）
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, ks_goe_list, width, label='KS(GOE)', alpha=0.8, color='red')
    ax1.bar(x + width/2, ks_poisson_list, width, label='KS(Poisson)', alpha=0.8, color='green')
    ax1.set_ylabel('KS Distance', fontsize=11)
    ax1.set_title('KS统计量对比（越小越好）', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0.05, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='显著性阈值')
    
    # 图2：判决统计（饼图）
    ax2 = axes[0, 1]
    verdict_counts = {}
    for v in verdicts:
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    colors = {'GOE': 'red', 'Poisson': 'green', 'Mixed': 'orange'}
    ax2.pie(verdict_counts.values(), labels=verdict_counts.keys(), autopct='%1.1f%%',
            colors=[colors.get(k, 'gray') for k in verdict_counts.keys()],
            startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'所有样本判决分布 (n={len(models)})', fontsize=12, fontweight='bold')
    
    # 图3：KS差值（Poisson优势度）
    ax3 = axes[1, 0]
    ks_diff = np.array(ks_goe_list) - np.array(ks_poisson_list)
    colors_bar = ['green' if d > 0 else 'red' for d in ks_diff]
    ax3.barh(models, ks_diff, color=colors_bar, alpha=0.7)
    ax3.set_xlabel('KS(GOE) - KS(Poisson)', fontsize=11)
    ax3.set_title('Poisson优势度（正值=支持Poisson）', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='black', linewidth=1.5)
    ax3.grid(alpha=0.3, axis='x')
    ax3.tick_params(axis='y', labelsize=8)
    
    # 图4：散点图（GOE vs Poisson）
    ax4 = axes[1, 1]
    verdict_colors = {'GOE': 'red', 'Poisson': 'green', 'Mixed': 'orange'}
    for v in set(verdicts):
        mask = [verdict == v for verdict in verdicts]
        ax4.scatter(np.array(ks_goe_list)[mask], np.array(ks_poisson_list)[mask],
                   label=v, alpha=0.7, s=100, color=verdict_colors.get(v, 'gray'))
    ax4.plot([0, max(max(ks_goe_list), max(ks_poisson_list))],
             [0, max(max(ks_goe_list), max(ks_poisson_list))],
             'k--', linewidth=1, alpha=0.5, label='对角线')
    ax4.set_xlabel('KS(GOE)', fontsize=11)
    ax4.set_ylabel('KS(Poisson)', fontsize=11)
    ax4.set_title('KS距离散点图（越靠近左下=越符合理论）', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 汇总图已保存: {save_path}")
    plt.show()

# ==================== 批量模型分析 ====================
MODELS_TO_ANALYZE = [
    # ============ Qwen 系列（阿里，无需认证）============
    ("Qwen/Qwen2.5-0.5B", "model.safetensors", [0, 5, 11]),  # 12层，988MB
    ("Qwen/Qwen2.5-1.5B", "model.safetensors", [0, 9, 17]),  # 28层，3.09GB ✅已下载
    ("Qwen/Qwen2.5-3B", "model-00001-of-00002.safetensors", [0, 12, 23]),  # 36层，6.54GB
    
    # ============ GPT-2 系列（OpenAI 开源，经典架构）============
    ("openai-community/gpt2", "model.safetensors", [0, 5, 11]),  # 12层，548MB
    ("openai-community/gpt2-medium", "model.safetensors", [0, 11, 23]),  # 24层，1.52GB
    ("openai-community/gpt2-large", "model.safetensors", [0, 17, 35]),  # 36层，3.25GB
    
    # ============ TinyLlama（最小 Llama，无需认证）============
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model.safetensors", [0, 10, 21]),  # 22层，2.2GB
    
    # ============ OLMo 系列（AI2，完全开源）============
    ("allenai/OLMo-1B-hf", "model.safetensors", [0, 7, 15]),  # 16层，2.46GB
    
    # ============ Pythia 系列（EleutherAI，研究用）============
    ("EleutherAI/pythia-410m", "model.safetensors", [0, 11, 23]),  # 24层，821MB
    ("EleutherAI/pythia-1b", "model.safetensors", [0, 7, 15]),  # 16层，2.05GB
    
    # ============ BLOOM 系列（BigScience，多语言）============
    ("bigscience/bloom-560m", "model.safetensors", [0, 11, 23]),  # 24层，1.12GB
    
    # ============ OPT 系列（Meta，GPT-3 开源复现）============
    ("facebook/opt-350m", "model.safetensors", [0, 11, 23]),  # 24层，700MB
    ("facebook/opt-1.3b", "model.safetensors", [0, 11, 23]),  # 24层，2.63GB
]

print("🚀 大规模 Transformer 架构谱分析（优化版）")
print("=" * 70)
print("📋 覆盖范围（仅无需认证 + 文件路径正确的模型）:")
print("   ✅ Qwen 系列（3个）- 0.5B, 1.5B, 3B（阿里）")
print("   ✅ GPT-2 系列（3个）- GPT-2, Medium, Large（OpenAI 开源）")
print("   ✅ TinyLlama（1个）- 1.1B（最小 Llama 变体）")
print("   ✅ OLMo 系列（1个）- 1B（AI2 完全开源）")
print("   ✅ Pythia 系列（2个）- 410M, 1B（EleutherAI 研究）")
print("   ✅ BLOOM 系列（1个）- 560M（BigScience 多语言）")
print("   ✅ OPT 系列（2个）- 350M, 1.3B（Meta GPT-3 复现）")
print()
print(f"💾 缓存位置: {HF_CACHE_DIR} (避免 C 盘空间不足)")
print(f"\n目标模型数: {len(MODELS_TO_ANALYZE)}")
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
        # 下载模型文件（带来源追踪和进度）
        print(f"📥 下载中...")
        print(f"   来源: https://huggingface.co/{repo_id}")
        print(f"   文件: {filename}")
        print(f"   💡 提示: 大文件下载需要时间，请耐心等待...")
        print(f"   ⏳ 正在连接服务器并下载...\n")
        
        # resume_download=True 显示进度条，force_download=False 使用缓存
        file_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename,
            resume_download=True,  # 支持断点续传
            local_dir_use_symlinks=False  # 避免符号链接警告
        )
        
        print(f"\n   ✅ 下载完成！")
        
        # 验证文件完整性
        import hashlib
        print(f"   本地路径: {file_path}")
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"   SHA256: {file_hash[:16]}...{file_hash[-16:]}")
        
        # 加载权重
        print(f"📂 加载权重...")
        weights = load_file(file_path)
        
        # 查找所有注意力层的权重（支持多种命名规范）
        attn_patterns = [
            'self_attn.q_proj.weight',     # Llama, Qwen, Mistral
            'attn.q_proj.weight',           # 某些变体
            'attn.c_attn.weight',           # GPT-2
            'attention.query_key_value.weight',  # Pythia (GPT-NeoX)
            'self_attention.query_key_value.weight'  # BLOOM
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
                    'file_path': file_path,
                    'sha256': file_hash,
                    'huggingface_url': f"https://huggingface.co/{repo_id}",
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
                result = analyze_spectra_goe_silent(W, title=layer_name)
                
                all_results[model_name][layer_name] = result
                
                # 简洁输出
                print(f"✅ {result['verdict']} (KS_GOE={result['ks_goe']:.4f}, KS_Poisson={result['ks_poisson']:.4f})")
                total_analyzed += 1
            else:
                print(f"  ⚠️ Layer {layer_idx} 不存在（只有 {len(attn_keys)} 层）")
        
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        total_errors += 1
        continue

# ==================== 统计汇总 ====================
print(f"\n\n{'='*70}")
print("📊 最终统计")
print(f"{'='*70}")

all_verdicts = []
all_ks_goe = []
all_ks_poisson = []

for model_name, layers in all_results.items():
    print(f"\n【{model_name}】")
    for layer_name, result in layers.items():
        # 跳过元数据
        if layer_name == '_metadata':
            continue
        
        # 确保 result 是有效的分析结果
        if not isinstance(result, dict) or 'verdict' not in result:
            print(f"  {layer_name}: ⚠️ 分析失败或数据不完整")
            continue
        
        print(f"  {layer_name}: {result['verdict']} (置信度={result['confidence']:.2f})")
        all_verdicts.append(result['verdict'])
        all_ks_goe.append(result['ks_goe'])
        all_ks_poisson.append(result['ks_poisson'])

print(f"\n{'='*70}")
print(f"✅ 成功分析: {total_analyzed} 个样本")
print(f"❌ 失败: {total_errors} 个模型")

# 总体判决
verdict_counts = {}
for v in all_verdicts:
    verdict_counts[v] = verdict_counts.get(v, 0) + 1

print(f"\n📈 总体判决分布:")
for verdict, count in verdict_counts.items():
    print(f"   {verdict}: {count} ({count/len(all_verdicts)*100:.1f}%)")

print(f"\n📉 KS距离统计:")
print(f"   GOE:     均值={np.mean(all_ks_goe):.4f}, 标准差={np.std(all_ks_goe):.4f}")
print(f"   Poisson: 均值={np.mean(all_ks_poisson):.4f}, 标准差={np.std(all_ks_poisson):.4f}")

# 配对t检验：GOE vs Poisson
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(all_ks_goe, all_ks_poisson)
print(f"\n🔬 配对t检验 (GOE vs Poisson):")
print(f"   t统计量 = {t_stat:.4f}")
print(f"   p值 = {p_value:.4e}")
if p_value < 0.001:
    winner = "Poisson" if t_stat > 0 else "GOE"
    print(f"   ✅ 结论: {winner} 显著更优 (p < 0.001)")
else:
    print(f"   ⚠️ 结论: 无显著差异")

# 保存结果到JSON
results_json = {
    'timestamp': datetime.now().isoformat(),
    'total_models': len(MODELS_TO_ANALYZE),
    'total_samples': total_analyzed,
    'verdict_distribution': verdict_counts,
    'ks_statistics': {
        'goe_mean': float(np.mean(all_ks_goe)),
        'goe_std': float(np.std(all_ks_goe)),
        'poisson_mean': float(np.mean(all_ks_poisson)),
        'poisson_std': float(np.std(all_ks_poisson)),
    },
    'ttest': {
        't_statistic': float(t_stat),
        'p_value': float(p_value)
    },
    'detailed_results': {
        model: {
            layer: {
                'verdict': res['verdict'],
                'ks_goe': float(res['ks_goe']),
                'ks_poisson': float(res['ks_poisson']),
                'confidence': float(res['confidence'])
            }
            for layer, res in layers.items()
            if layer != '_metadata' and isinstance(res, dict) and 'verdict' in res
        }
        for model, layers in all_results.items()
    }
}

with open('transformer_spectra_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)

print(f"\n💾 详细结果已保存: transformer_spectra_analysis.json")

# 生成汇总图
plot_comparison(all_results, save_path="transformer_batch_analysis.png")

print(f"\n{'='*70}")
print("🎯 最终结论")
print(f"{'='*70}")

if verdict_counts.get('Poisson', 0) > total_analyzed * 0.7:
    print("✅ Transformer 架构整体展现 Poisson 分布特征")
    print("💡 这意味着你的素数模型的 GOE 特征很可能是素数特有的！")
elif verdict_counts.get('GOE', 0) > total_analyzed * 0.7:
    print("⚠️ Transformer 架构整体展现 GOE 分布特征")
    print("💡 这意味着架构本身可能有时间反演对称混沌倾向")
    print("🚨 警告：这会削弱素数模型 GOE 特征的独特性！")
else:
    print("⚠️ 结果混合，需要进一步分析")

print(f"\n✅ 分析完成！共处理 {total_analyzed} 个样本")
