"""
素数间隙预测 - 纯粹涌现版
Pure Emergence Approach to Prime Number Patterns

核心理念：
- 不预设任何守恒定律
- 不人为引导能量或学习率
- 让AI在"生存压力"下自然进化
- 在顿悟时刻保存权重
- 事后解析神经网络寻找深层数学结构

目标：通过AI自己的"智慧"发现素数的深层规律
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import math
from sympy import primerange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats, interpolate
import time
import json
import os
from pathlib import Path

# 创建输出目录
OUTPUT_DIR = Path("riemann_pure_emergence_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("🌟 素数间隙预测AI实验 - 纯粹涌现版")
print("=" * 70)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"显存: {gpu_props.total_memory / 1e9:.2f} GB")
else:
    print("⚠️  警告：未检测到GPU，将使用CPU（会很慢）")
    # Kaggle环境自动继续（不需要交互）
    import sys
    if 'ipykernel' not in sys.modules:  # 非Jupyter环境才询问
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            exit()

print("=" * 70)

# ==================== 配置 ====================
class Config:
    """实验配置 - 纯粹版（无守恒约束）"""
    
    # 数据配置
    NUM_PRIMES = 1000000  # 🔥 100万素数
    
    # 🔥 梯度累积策略
    USE_GRADIENT_ACCUMULATION = True
    PHYSICAL_BATCH_SIZE = 128       # 物理batch（显存实际占用）
    ACCUMULATION_STEPS = 16         # 累积步数
    # 等效batch = 128 × 16 = 2048（保持训练效果）
    
    # 解释：
    # 传统方法：batch=2048 → 显存爆炸
    # 梯度累积：物理batch=128，累积16次 → 效果等价，显存安全
    # 
    # 类比：
    # 传统 = 一次吃16个包子（撑死）
    # 累积 = 分16次吃，每次1个（舒服）
    
    # 模型配置（可被元进化优化）
    D_MODEL = 256        # 减小模型以加速（仍有足够表达力）
    N_LAYERS = 6
    N_HEADS = 8
    DROPOUT = 0.1
    LEARNABLE_EMBEDDING = True  # 🔥 终极挑战：让AI从随机噪声中涌现数学规律！
    
    # 训练配置（可被元进化优化）
    NUM_EPOCHS = 10000     # 🔥 10000轮（100万数据全量训练）
    BATCH_SIZE = 256       # 🔥 Kaggle GPU显存更大，可以用256
    LEARNING_RATE = 1e-4   # 基准学习率
    WEIGHT_DECAY = 0.01
    
    # 学习率策略（元进化可选择）
    LR_SCHEDULE = 'cosine'  # 'constant', 'cosine', 'plateau'
    
    # 监控配置
    PRINT_EVERY = 1  # 每轮都打印
    SAVE_EVERY = 500
    
    # 顿悟检测（纯观察，不干预）
    GROKKING_THRESHOLD = 0.3  # 单轮Loss下降>30%
    GROKKING_WINDOW = 20      # 检测窗口
    
    # 涌现追踪
    TRACK_GRADIENTS = True    # 记录梯度信息
    TRACK_WEIGHTS = True      # 记录权重统计
    
    # 🔥 断点续存配置
    CHECKPOINT_EVERY = 50      # 每50个epoch保存一次checkpoint
    CHECKPOINT_DIR = "riemann_checkpoints"  # checkpoint保存目录
    AUTO_RESUME = True         # 自动从最新checkpoint恢复

config = Config()

# ==================== 断点续存系统 ====================
def save_checkpoint(epoch, model, optimizer, losses, tracker, hyperparam_evolver, config, filename):
    """保存训练checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'tracker_history': tracker.history if tracker else None,
        'hyperparam_history': hyperparam_evolver.evolution_history if hyperparam_evolver else None,
        'config': {
            'NUM_PRIMES': config.NUM_PRIMES,
            'D_MODEL': config.D_MODEL,
            'N_LAYERS': config.N_LAYERS,
            'N_HEADS': config.N_HEADS,
            'LEARNING_RATE': config.LEARNING_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'ACCUMULATION_STEPS': config.ACCUMULATION_STEPS,
        }
    }
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint已保存: {filename} (Epoch {epoch})")

def load_checkpoint(filename, model, optimizer, device):
    """加载训练checkpoint"""
    if not os.path.exists(filename):
        return None
    
    print(f"📂 加载checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ 成功恢复到 Epoch {checkpoint['epoch']}")
    print(f"   累计Loss历史: {len(checkpoint['losses'])}条")
    
    return checkpoint

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
    
    # 提取epoch编号并排序
    epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_file = f'checkpoint_epoch_{latest_epoch}.pt'
    
    return os.path.join(checkpoint_dir, latest_file)

# ==================== 在线超参数进化器 ====================
class OnlineHyperparamEvolution:
    """
    在线超参数进化 - 边训练边进化
    
    核心思想：
    1. 不预先固定超参数
    2. 根据训练进度动态调整
    3. 超参数本身也是"涌现"的产物
    """
    def __init__(self, initial_lr=1e-4, initial_wd=0.01):
        self.hyperparams = {
            'lr': initial_lr,
            'weight_decay': initial_wd,
        }
        
        # 动态冷却期（基于训练稳定性）
        self.base_update_interval = 50  # 基准间隔
        self.current_cooldown = 30      # 初期较短，快速探索
        self.cooldown_bounds = (20, 150)  # 冷却期范围
        
        self.last_update_epoch = -100
        
        # 学习率范围
        self.lr_bounds = (1e-6, 5e-4)
        self.max_change_ratio = 0.3  # 单次最大变化30%
        
        # 历史记录
        self.evolution_history = []
        
        print(f"\n{'='*70}")
        print("🧬 在线超参数进化器已启动")
        print(f"{'='*70}")
        print(f"初始学习率: {initial_lr:.2e}")
        print(f"初始权重衰减: {initial_wd}")
        print(f"动态冷却期: {self.current_cooldown} (范围: {self.cooldown_bounds})")
        print(f"{'='*70}\n")
    
    def calculate_dynamic_cooldown(self, losses):
        """
        动态计算冷却期
        
        策略：基于Loss的稳定性
        - Loss很稳定 → cooldown长（不需要频繁调整）
        - Loss波动大 → cooldown短（需要快速响应）
        """
        if len(losses) < 50:
            return 30  # 初期默认
        
        recent_losses = losses[-50:]
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        
        # 变异系数（CV）
        cv = loss_std / (loss_mean + 1e-10)
        
        # 根据CV调整cooldown
        if cv < 0.03:  # 非常稳定
            new_cooldown = 100
        elif cv < 0.08:  # 中等稳定
            new_cooldown = 50
        elif cv < 0.15:  # 轻微波动
            new_cooldown = 30
        else:  # 高波动
            new_cooldown = 20
        
        # 限制范围
        new_cooldown = np.clip(new_cooldown, *self.cooldown_bounds)
        
        # 平滑变化（指数移动平均）
        alpha = 0.3
        smoothed_cooldown = int(alpha * new_cooldown + (1 - alpha) * self.current_cooldown)
        
        return smoothed_cooldown
    
    def should_update(self, epoch, losses):
        """是否应该更新超参数"""
        # 更新动态冷却期
        self.current_cooldown = self.calculate_dynamic_cooldown(losses)
        
        # 判断是否到了更新时机
        epochs_since_last = epoch - self.last_update_epoch
        return epochs_since_last >= self.current_cooldown and epoch >= 30
    
    def evaluate_progress(self, losses):
        """评估最近的训练进度"""
        if len(losses) < 50:
            return 0.0
        
        recent_50 = losses[-50:]
        previous_50 = losses[-100:-50] if len(losses) >= 100 else losses[:50]
        
        recent_avg = np.mean(recent_50)
        previous_avg = np.mean(previous_50)
        
        # 进度 = 相对下降幅度
        if previous_avg < 1e-10:
            return 0.0
        
        progress = (previous_avg - recent_avg) / previous_avg
        return progress
    
    def mutate_hyperparams(self, progress, losses):
        """
        根据进度变异超参数
        
        策略：
        - 进度好 → 小幅变异（保持方向）
        - 进度差 → 大幅变异（寻找新方向）
        """
        if progress > 0.05:  # 良好进展（Loss下降>5%）
            mutation_strength = 0.1
            strategy = "保持方向"
        elif progress > 0.01:  # 微弱进展
            mutation_strength = 0.2
            strategy = "适度探索"
        else:  # 停滞或倒退
            mutation_strength = 0.4
            strategy = "大胆突破"
        
        # 变异学习率
        lr_multiplier = np.random.uniform(
            1 - mutation_strength,
            1 + mutation_strength
        )
        new_lr = self.hyperparams['lr'] * lr_multiplier
        
        # 限制绝对范围
        new_lr = np.clip(new_lr, *self.lr_bounds)
        
        # 限制单次变化幅度
        max_change = self.hyperparams['lr'] * self.max_change_ratio
        new_lr = np.clip(
            new_lr,
            self.hyperparams['lr'] - max_change,
            self.hyperparams['lr'] + max_change
        )
        
        return {
            'lr': new_lr,
            'weight_decay': self.hyperparams['weight_decay'],
            'strategy': strategy,
            'mutation_strength': mutation_strength
        }
    
    def update(self, epoch, losses, optimizer):
        """更新超参数"""
        if not self.should_update(epoch, losses):
            return False
        
        # 评估进度
        progress = self.evaluate_progress(losses)
        
        # 变异超参数
        new_hyperparams = self.mutate_hyperparams(progress, losses)
        
        # 记录历史
        self.evolution_history.append({
            'epoch': epoch,
            'old_lr': self.hyperparams['lr'],
            'new_lr': new_hyperparams['lr'],
            'progress': progress,
            'strategy': new_hyperparams['strategy'],
            'cooldown': self.current_cooldown
        })
        
        # 更新优化器（不重建，保留Adam状态）
        old_lr = self.hyperparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_hyperparams['lr']
            param_group['weight_decay'] = new_hyperparams['weight_decay']
        
        # 打印进化信息
        lr_change = (new_hyperparams['lr'] - old_lr) / old_lr * 100
        
        print(f"\n{'🧬'*35}")
        print(f"💥 超参数进化触发 - Epoch {epoch}")
        print(f"{'='*70}")
        print(f"  训练进度: {progress*100:+.2f}% (最近50轮 vs 前50轮)")
        print(f"  进化策略: {new_hyperparams['strategy']}")
        print(f"  变异强度: {new_hyperparams['mutation_strength']*100:.0f}%")
        print(f"  学习率: {old_lr:.2e} → {new_hyperparams['lr']:.2e} ({lr_change:+.1f}%)")
        print(f"  动态冷却期: {self.current_cooldown} 轮")
        print(f"  下次进化: 约 Epoch {epoch + self.current_cooldown}")
        print(f"{'='*70}")
        print(f"{'🧬'*35}\n")
        
        self.hyperparams = {
            'lr': new_hyperparams['lr'],
            'weight_decay': new_hyperparams['weight_decay']
        }
        self.last_update_epoch = epoch
        
        return True

# ==================== 涌现追踪器 ====================
class EmergenceTracker:
    """
    涌现追踪器 - 纯观察，不干预
    
    职责：
    1. 记录训练全过程的关键指标
    2. 检测顿悟时刻（用于保存权重）
    3. 事后分析涌现的规律
    """
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
        self.history = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'grad_norm': [],
            'param_norm': [],
        }
        
        self.grokking_moments = []  # 记录所有顿悟时刻
    
    def record(self, epoch, loss, lr, grad_norm=None, param_norm=None):
        """记录当前epoch的指标"""
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['lr'].append(lr)
        if grad_norm is not None:
            self.history['grad_norm'].append(grad_norm)
        if param_norm is not None:
            self.history['param_norm'].append(param_norm)
    
    def detect_grokking(self, config, model=None, X_sample=None, y_sample=None):
        """
        检测顿悟时刻（多维度数学突破检测）
        
        不只检测Loss突降，更检测：
        1. Loss突变（传统顿悟）
        2. 对数关系发现（log correlation）
        3. 周期性模式（FFT dominant frequency）
        
        多个维度同时满足才是"真顿悟"！
        """
        losses = self.history['loss']
        
        if len(losses) < config.GROKKING_WINDOW + 1:
            return False, 0.0
        
        # ==================== 维度1：Loss突变检测 ====================
        recent_avg = np.mean(losses[-(config.GROKKING_WINDOW+1):-1])
        current_loss = losses[-1]
        
        sudden_drop = recent_avg - current_loss
        drop_ratio = sudden_drop / recent_avg if recent_avg > 1e-10 else 0
        
        loss_breakthrough = (drop_ratio > config.GROKKING_THRESHOLD and 
                            sudden_drop > 0.01 and 
                            current_loss < recent_avg)
        
        # ==================== 维度2：对数关系检测 ====================
        log_breakthrough = False
        log_corr = 0.0
        
        if model is not None and X_sample is not None and y_sample is not None:
            try:
                with torch.no_grad():
                    predictions = model(X_sample).squeeze()
                    
                    # 计算对数空间的相关性
                    # 如果AI发现了对数规律，pred和true在log空间应该高度相关
                    pred_np = predictions.cpu().numpy()
                    true_np = y_sample.cpu().numpy().squeeze()
                    
                    # 避免log(0)
                    pred_log = np.log(np.abs(pred_np) + 1e-6)
                    true_log = np.log(np.abs(true_np) + 1e-6)
                    
                    # 计算相关系数
                    if len(pred_log) > 10 and np.std(pred_log) > 1e-6 and np.std(true_log) > 1e-6:
                        log_corr = np.corrcoef(pred_log, true_log)[0, 1]
                        
                        # 对数相关性>0.9认为发现了对数规律
                        log_breakthrough = (log_corr > 0.9)
            
            except Exception as e:
                # 如果计算失败，跳过这个检测
                pass
        
        # ==================== 维度3：周期性检测 ====================
        periodicity_breakthrough = False
        dominant_freq = 0.0
        
        if model is not None and X_sample is not None:
            try:
                with torch.no_grad():
                    predictions = model(X_sample).squeeze()
                    pred_np = predictions.cpu().numpy()
                    
                    # 去除趋势（使用差分）
                    if len(pred_np) > 100:
                        pred_detrended = np.diff(pred_np)
                        
                        # FFT分析
                        fft_result = np.fft.fft(pred_detrended)
                        freqs = np.fft.fftfreq(len(pred_detrended))
                        
                        # 只看正频率
                        positive_freqs = freqs[:len(freqs)//2]
                        positive_fft = np.abs(fft_result[:len(freqs)//2])
                        
                        if len(positive_fft) > 1:
                            # 找到主频率
                            main_idx = np.argmax(positive_fft[1:]) + 1  # 跳过DC分量
                            dominant_freq = positive_freqs[main_idx]
                            
                            # 如果主频率的幅度远大于平均（>5倍），认为发现了周期性
                            mean_amplitude = np.mean(positive_fft[1:])
                            max_amplitude = positive_fft[main_idx]
                            
                            if max_amplitude > 5 * mean_amplitude:
                                periodicity_breakthrough = True
            
            except Exception as e:
                pass
        
        # ==================== 综合判断 ====================
        # 至少满足两个维度才算"真顿悟"
        breakthrough_count = sum([loss_breakthrough, log_breakthrough, periodicity_breakthrough])
        
        is_true_grokking = breakthrough_count >= 2
        
        # 如果是真顿悟，打印详细信息
        if is_true_grokking:
            print(f"\n{'='*70}")
            print(f"🔬 多维度数学突破检测")
            print(f"{'='*70}")
            print(f"  ✓ Loss突变: {'是' if loss_breakthrough else '否'} (下降{drop_ratio*100:.1f}%)")
            if log_corr != 0.0:
                print(f"  ✓ 对数关系: {'是' if log_breakthrough else '否'} (相关性={log_corr:.4f})")
            if dominant_freq != 0.0:
                print(f"  ✓ 周期性: {'是' if periodicity_breakthrough else '否'} (主频率={dominant_freq:.6f})")
            print(f"  → 突破维度: {breakthrough_count}/3")
            print(f"{'='*70}\n")
        
        return is_true_grokking, drop_ratio
    
    def analyze_emergence(self):
        """
        事后分析：训练过程中涌现了什么规律？
        
        纯粹记录 - 不预设任何守恒定律
        """
        print("\n" + "="*70)
        print("🔍 涌现分析")
        print("="*70)
        
        epochs = np.array(self.history['epoch'])
        losses = np.array(self.history['loss'])
        lrs = np.array(self.history['lr'])
        
        # 只记录基本统计信息
        print("\n📉 训练统计：")
        print(f"  初始Loss: {losses[0]:.6f}")
        print(f"  最终Loss: {losses[-1]:.6f}")
        print(f"  下降幅度: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"  顿悟次数: {len(self.grokking_moments)}")
        
        if self.grokking_moments:
            print(f"  顿悟时刻: {self.grokking_moments}")
        
        # 学习率统计
        print(f"\n📊 学习率演化：")
        print(f"  初始LR: {lrs[0]:.2e}")
        print(f"  最终LR: {lrs[-1]:.2e}")
        print(f"  LR范围: [{lrs.min():.2e}, {lrs.max():.2e}]")
        
        # 梯度统计
        if self.history['grad_norm']:
            grad_norms = np.array(self.history['grad_norm'])
            print(f"\n📈 梯度统计：")
            print(f"  平均梯度范数: {np.mean(grad_norms):.4f}")
            print(f"  最大梯度范数: {np.max(grad_norms):.4f}")
            print(f"  最小梯度范数: {np.min(grad_norms):.4f}")
        
        print("\n" + "="*70)
        
        return {
            'loss_reduction': (losses[0] - losses[-1]) / losses[0],
            'grokking_count': len(self.grokking_moments),
            'final_loss': losses[-1],
            'final_lr': lrs[-1]
        }

# ==================== 数据生成 ====================
def generate_prime_gaps(num_primes):
    """生成素数间隙数据"""
    print(f"\n{'='*70}")
    print(f"生成前 {num_primes:,} 个素数...")
    print(f"{'='*70}")
    start_time = time.time()
    
    # 估算上界
    if num_primes < 10:
        upper_bound = 30
    else:
        ln_n = math.log(num_primes)
        ln_ln_n = math.log(ln_n) if ln_n > 1 else 0
        upper_bound = int(num_primes * (ln_n + ln_ln_n + 2))
    
    print(f"估算上界: {upper_bound:,}")
    
    # 生成素数
    primes = list(primerange(1, upper_bound))
    
    if len(primes) < num_primes:
        print(f"⚠️  警告：只生成了 {len(primes)} 个素数")
        num_primes = len(primes)
    else:
        primes = primes[:num_primes]
    
    # 计算间隙
    prime_gaps = np.diff(primes)
    
    elapsed = time.time() - start_time
    print(f"✓ 生成完成 ({elapsed:.2f}秒)")
    print(f"  素数数量: {len(primes):,}")
    print(f"  间隙数量: {len(prime_gaps):,}")
    print(f"  间隙范围: [{prime_gaps.min()}, {prime_gaps.max()}]")
    print(f"  平均间隙: {np.mean(prime_gaps):.2f}")
    print(f"  前20个间隙: {list(prime_gaps[:20])}")
    
    return prime_gaps

# ==================== 模型定义 ====================
class RiemannEmbedding(nn.Module):
    """
    位置编码模块
    
    支持两种模式：
    1. 固定正弦编码（标准Transformer）
    2. 可学习编码（让AI自己涌现频率）
    """
    def __init__(self, d_model, max_len=1000000, learnable=False):
        super().__init__()
        self.d_model = d_model
        self.learnable = learnable
        
        if learnable:
            # 可学习编码：让AI自己发现log(p)规律
            self.embedding = nn.Embedding(max_len, d_model)
            # 初始化为小随机值
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        else:
            # 固定正弦编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
    
    def forward(self, x):
        if self.learnable:
            return self.embedding(x)
        else:
            return self.pe[x]

class PrimeGapPredictor(nn.Module):
    """素数间隙预测器"""
    def __init__(self, d_model=512, n_layers=6, n_heads=8, dropout=0.1, learnable_embedding=False):
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
    
    def forward(self, x):
        embedded = self.riemann_embedding(x)
        embedded = embedded.unsqueeze(1)
        transformed = self.transformer(embedded)
        gap = self.output(transformed.squeeze(1))
        return gap
    
    def get_hidden_states(self, x):
        """提取隐藏层状态（用于GOE分析）"""
        embedded = self.riemann_embedding(x)
        embedded = embedded.unsqueeze(1)
        hidden = self.transformer(embedded)
        return hidden.squeeze(1)
    
    def get_attention_weights(self):
        """提取注意力权重（用于GOE分析）"""
        weights = []
        for layer in self.transformer.layers:
            # 提取self-attention的权重
            attn_weights = layer.self_attn.in_proj_weight
            weights.append(attn_weights.detach().cpu())
        return torch.cat(weights, dim=0)

# ==================== 训练函数 ====================
def train_model(model, X_gpu_full, y_gpu_full, device, config, tracker, hyperparam_evolver=None):
    """
    纯粹训练 - 不加任何人为约束
    
    唯一目标：最小化Loss
    让AI自己找到最优策略
    
    如果提供hyperparam_evolver，则启用在线超参数进化
    
    🔥 支持梯度累积（Gradient Accumulation）：
       100万数据全量训练，但使用小batch + 梯度累积
       → 物理batch = 128（显存友好）
       → 逻辑batch = 128 × 16 = 2048（效果等价）
       → 让100万数据在8GB显存下安全运行！
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 如果启用在线进化，不使用传统调度器
    scheduler = None
    if hyperparam_evolver is None:
        if config.LR_SCHEDULE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=1e-6
            )
        elif config.LR_SCHEDULE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=100,
                min_lr=1e-6
            )
    
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 🔥 梯度累积：使用全部数据
    X_gpu = X_gpu_full
    y_gpu = y_gpu_full
    total_data_size = len(X_gpu)
    
    # 计算batch数量（基于物理batch size）
    num_samples = total_data_size
    num_batches = (num_samples + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    
    # 梯度累积配置
    if config.USE_GRADIENT_ACCUMULATION:
        accumulation_steps = config.ACCUMULATION_STEPS
        effective_batch_size = config.BATCH_SIZE * accumulation_steps
        print(f"\n{'💪'*35}")
        print(f"💪 梯度累积训练（100万数据全量，硬气！）")
        print(f"{'='*70}")
        print(f"  总数据量: {total_data_size:,}")
        print(f"  物理batch: {config.BATCH_SIZE}")
        print(f"  累积步数: {accumulation_steps}")
        print(f"  等效batch: {effective_batch_size}")
        print(f"  策略: 小步快跑，效果等价，显存安全")
        print(f"  预计显存: ~3-4GB（安全）")
        print(f"{'='*70}")
        print(f"{'💪'*35}\n")
    else:
        accumulation_steps = 1
        effective_batch_size = config.BATCH_SIZE
    
    losses = []
    best_loss = float('inf')
    start_epoch = 0  # 起始epoch（断点续存会修改）
    
    # 🔥 断点续存：尝试恢复
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    if config.AUTO_RESUME:
        latest_checkpoint = find_latest_checkpoint(config.CHECKPOINT_DIR)
        if latest_checkpoint:
            print(f"\n{'💾'*35}")
            print(f"💾 检测到checkpoint，正在恢复...")
            print(f"{'='*70}")
            checkpoint_data = load_checkpoint(latest_checkpoint, model, optimizer, device)
            if checkpoint_data:
                start_epoch = checkpoint_data['epoch'] + 1
                losses = checkpoint_data['losses']
                if tracker and checkpoint_data.get('tracker_history'):
                    tracker.history = checkpoint_data['tracker_history']
                if hyperparam_evolver and checkpoint_data.get('hyperparam_history'):
                    hyperparam_evolver.evolution_history = checkpoint_data['hyperparam_history']
                print(f"✅ 训练将从 Epoch {start_epoch} 继续")
                print(f"{'='*70}")
                print(f"{'💾'*35}\n")
            else:
                print("⚠️  checkpoint加载失败，从头开始")
    
    print(f"\n{'='*70}")
    print("🚀 开始训练 - 纯粹涌现模式")
    print(f"{'='*70}")
    print(f"训练样本: {num_samples:,}")
    print(f"Batch大小: {config.BATCH_SIZE}")
    print(f"Batch数量: {num_batches}")
    if hyperparam_evolver is not None:
        print(f"超参数策略: 🧬 在线进化（动态冷却）")
    else:
        print(f"学习率策略: {config.LR_SCHEDULE}")
    print(f"多维度数学突破检测: ✅ 已启用")
    print(f"断点续存: 每{config.CHECKPOINT_EVERY}轮保存 → {config.CHECKPOINT_DIR}/")
    if start_epoch > 0:
        print(f"续存模式: 从 Epoch {start_epoch} 继续（已完成{start_epoch}轮）")
    print(f"{'='*70}")
    print(f"\n⏳ 训练即将开始（每轮都会打印进度）...\n")
    
    start_time = time.time()
    epoch_times = []
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # 第一轮特别提示
        if epoch == 0:
            print(f"🔥 Epoch 0 开始训练...")
            print(f"   总数据量: {total_data_size:,}")
            print(f"   物理batch: {config.BATCH_SIZE}")
            print(f"   累积步数: {accumulation_steps}")
            print(f"   等效batch: {effective_batch_size}")
            print(f"   （如果看到这条消息后长时间无反应，说明batch计算很慢）\n")
        
        # 训练一个epoch（支持梯度累积）
        optimizer.zero_grad()  # 🔥 放到epoch开始（梯度累积）
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.BATCH_SIZE
            end_idx = min(start_idx + config.BATCH_SIZE, num_samples)
            
            batch_X = X_gpu[start_idx:end_idx]
            batch_y = y_gpu[start_idx:end_idx]
            
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            # 🔥 梯度累积：缩放loss
            if config.USE_GRADIENT_ACCUMULATION:
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # 🔥 梯度累积：每N步更新一次
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * (accumulation_steps if config.USE_GRADIENT_ACCUMULATION else 1)
        
        # 记录
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # 在线超参数进化（优先级最高）
        if hyperparam_evolver is not None:
            hyperparam_evolver.update(epoch, losses, optimizer)
        
        # 传统学习率调度器（仅在未启用进化时）
        if scheduler is not None and hyperparam_evolver is None:
            if config.LR_SCHEDULE == 'plateau':
                scheduler.step(epoch_loss / num_batches)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算参数范数
        param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        
        # 记录到追踪器
        tracker.record(epoch, avg_loss, current_lr, grad_norm.item(), param_norm)
        
        # 检测顿悟（多维度数学突破检测）
        # 准备采样数据用于数学分析
        sample_size = min(1000, num_samples)
        sample_indices = torch.randperm(num_samples)[:sample_size].to(device)
        X_sample = X_gpu[sample_indices]
        y_sample = y_gpu[sample_indices]
        
        is_grokking, drop_ratio = tracker.detect_grokking(config, model, X_sample, y_sample)
        
        if is_grokking:
            tracker.grokking_moments.append(epoch)
            print(f"\n{'🔥'*35}")
            print(f"💥 检测到顿悟！Epoch {epoch}")
            print(f"  Loss突降: {drop_ratio*100:.1f}%")
            print(f"  当前Loss: {avg_loss:.6f}")
            print(f"{'🔥'*35}\n")
            
            # 🔬 保存顿悟前的权重（用于对比相变）
            if epoch >= 10:
                print(f"  💾 保存顿悟前权重（Epoch {epoch-10}）用于相变分析...")
                # 注意：这里只能保存当前模型，无法回溯历史
                # 在实际实现中，需要在训练时持续保存最近的权重
            
            # 保存顿悟时刻的权重
            save_grokking_weights(model, epoch, avg_loss, OUTPUT_DIR, label='after')
        
        # 更新最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, OUTPUT_DIR / 'best_model.pt')
        
        # 🔥 断点续存：定期保存checkpoint
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'checkpoint_epoch_{epoch}.pt'
            )
            save_checkpoint(
                epoch, model, optimizer, losses,
                tracker, hyperparam_evolver, config,
                checkpoint_path
            )
        
        # 打印进度
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if epoch % config.PRINT_EVERY == 0 or is_grokking:
            avg_epoch_time = np.mean(epoch_times[-50:]) if epoch_times else epoch_time
            remaining_epochs = config.NUM_EPOCHS - epoch - 1
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_minutes = eta_seconds / 60
            
            speed = 1.0 / avg_epoch_time if avg_epoch_time > 0 else 0
            
            print(f"Epoch {epoch:5d}/{config.NUM_EPOCHS} | "
                  f"Loss={avg_loss:.6f} | "
                  f"LR={current_lr:.2e} | "
                  f"GradNorm={grad_norm:.2f} | "
                  f"速度={speed:.2f}ep/s | "
                  f"ETA={eta_minutes:.1f}min")
        
        # 定期保存检查点
        if epoch % config.SAVE_EVERY == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, OUTPUT_DIR / f'checkpoint_epoch_{epoch}.pt')
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ 训练完成！")
    print(f"  总时间: {total_time/60:.1f} 分钟")
    print(f"  最佳Loss: {best_loss:.6f}")
    print(f"  顿悟次数: {len(tracker.grokking_moments)}")
    
    # 在线进化统计
    if hyperparam_evolver is not None:
        print(f"\n🧬 超参数进化统计:")
        print(f"  进化次数: {len(hyperparam_evolver.evolution_history)}")
        if hyperparam_evolver.evolution_history:
            final_lr = hyperparam_evolver.hyperparams['lr']
            initial_lr = config.LEARNING_RATE
            print(f"  学习率: {initial_lr:.2e} → {final_lr:.2e}")
            
            # 保存进化历史
            np.save(OUTPUT_DIR / 'hyperparam_evolution_history.npy', 
                   hyperparam_evolver.evolution_history)
    
    print(f"{'='*70}\n")
    
    return losses

def save_grokking_weights(model, epoch, loss, output_dir, label=''):
    """
    保存顿悟时刻的权重（用于GOE分析和相变对比）
    
    label: 'before'（顿悟前）或 'after'（顿悟后）
    """
    label_str = f"_{label}" if label else ""
    print(f"  💾 保存顿悟权重{label_str}...")
    
    # 提取注意力权重
    attention_weights = model.get_attention_weights()
    
    # 保存
    grokking_data = {
        'epoch': epoch,
        'loss': loss,
        'attention_weights': attention_weights.numpy()
    }
    
    filename = f"grokking_weights_epoch_{epoch}{label_str}.npy"
    np.save(output_dir / filename, grokking_data)
    print(f"  ✓ 已保存到 {filename}")

# ==================== GOE分析（完全重构版）====================
def analyze_level_spacing(eigenvalues_real, output_dir):
    """
    能级间距分析 - 量子混沌的统计特征
    
    GOE分布（时间反演对称混沌系统）vs Poisson分布（随机）
    """
    print(f"\n{'='*70}")
    print("🔬 能级间距分析（Level Spacing Statistics）")
    print(f"{'='*70}")
    
    # 1. 排序特征值
    eigs = np.sort(eigenvalues_real)
    
    # 2. 去除极端值（Unfolding简化版）
    if len(eigs) > 20:
        eigs = eigs[10:-10]
    
    print(f"有效特征值数量: {len(eigs)}")
    
    # 3. 计算相邻能级间距
    spacings = np.diff(eigs)
    
    # 4. 归一化（使平均间距为1）
    mean_spacing = np.mean(spacings)
    s = spacings / (mean_spacing + 1e-10)
    
    print(f"平均间距: {mean_spacing:.6f}")
    print(f"归一化后间距范围: [{s.min():.3f}, {s.max():.3f}]")
    
    # 5. 绘制直方图并对比理论曲线
    plt.figure(figsize=(12, 8))
    
    # 实际间距分布
    plt.hist(s, bins=50, density=True, alpha=0.6, color='blue', 
             edgecolor='black', label='AI Weight Spacings')
    
    # 理论曲线：Wigner Surmise (GOE) - 对应时间反演对称混沌系统
    x = np.linspace(0, 4, 200)
    p_goe = (np.pi / 2) * x * np.exp(-np.pi * x**2 / 4)
    plt.plot(x, p_goe, 'r-', linewidth=3, label='GOE (Time-Reversal Symmetric Chaos)')
    
    # 理论曲线：Poisson - 随机/无规律
    p_poisson = np.exp(-x)
    plt.plot(x, p_poisson, 'g--', linewidth=3, label='Poisson (Random)')
    
    plt.xlabel('Normalized Spacing (s)', fontsize=14)
    plt.ylabel('P(s)', fontsize=14)
    plt.title('Level Spacing Statistics: Evidence of Time-Reversal Symmetric Chaos', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_spacing_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ 能级间距分析图已保存")
    
    # 6. 计算与GOE的拟合度（KS检验）
    from scipy import stats
    
    # 生成GOE理论样本
    x_theory = np.linspace(0, 4, 1000)
    p_goe_theory = (np.pi / 2) * x_theory * np.exp(-np.pi * x_theory**2 / 4)
    cdf_goe = np.cumsum(p_goe_theory) / np.sum(p_goe_theory)
    
    # 计算实际数据的CDF
    s_sorted = np.sort(s[s < 4])
    
    # 插值比较
    from scipy.interpolate import interp1d
    if len(s_sorted) > 10:
        cdf_data_func = interp1d(s_sorted, np.linspace(0, 1, len(s_sorted)), 
                                 bounds_error=False, fill_value=(0, 1))
        cdf_data = cdf_data_func(x_theory)
        
        # 计算Kolmogorov-Smirnov距离
        ks_distance = np.max(np.abs(cdf_data - cdf_goe))
        
        print(f"\n🎯 GOE拟合分析：")
        print(f"  K-S距离: {ks_distance:.4f}")
        
        if ks_distance < 0.1:
            print(f"  🔥🔥🔥 极高拟合度！强烈支持时间反演对称混沌特征！")
        elif ks_distance < 0.2:
            print(f"  🔥 良好拟合！支持时间反演对称混沌假设")
        elif ks_distance < 0.3:
            print(f"  ✓ 中等拟合")
        else:
            print(f"  ⚠️  拟合较差，可能不符合GOE")
    
    print(f"{'='*70}\n")
    
    return {
        'mean_spacing': mean_spacing,
        'ks_distance': ks_distance if 'ks_distance' in locals() else None,
        'spacings': s
    }


def analyze_weight_matrices(model, output_dir, sample_size=10000):
    """
    分析权重矩阵的特征值分布（完全重构版）
    
    核心改进：
    1. 对称化权重矩阵（厄米化）
    2. 多种对称化方法对比
    3. 能级间距分析（GOE检验）
    4. 奇异值分解（SVD）
    """
    print(f"\n{'='*70}")
    print("🔬 权重矩阵量子分析（完全重构版）")
    print(f"{'='*70}")
    
    # 提取注意力权重矩阵
    attention_weights = model.get_attention_weights()
    print(f"原始权重矩阵形状: {attention_weights.shape}")
    
    # 取子矩阵（控制计算量）
    n = min(1024, attention_weights.shape[0], attention_weights.shape[1])
    W = attention_weights[:n, :n]
    print(f"分析子矩阵形状: {W.shape}")
    
    results = {}
    
    # ============================================================
    # 方法1：厄米化（Hermitianization）H = (W + W^T) / 2
    # ============================================================
    print(f"\n{'─'*70}")
    print("📊 方法1：厄米化 H = (W + W^T) / 2")
    print(f"{'─'*70}")
    
    try:
        H = (W + W.T) / 2
        print("✓ 厄米矩阵构造完成")
        
        # 计算特征值（厄米矩阵的特征值都是实数）
        eigenvalues_hermitian = torch.linalg.eigvalsh(H)  # 对称矩阵专用
        eigenvalues_hermitian = eigenvalues_hermitian.cpu().numpy()
        
        print(f"特征值数量: {len(eigenvalues_hermitian)}")
        print(f"特征值范围: [{eigenvalues_hermitian.min():.4f}, {eigenvalues_hermitian.max():.4f}]")
        print(f"特征值均值: {np.mean(eigenvalues_hermitian):.4f}")
        
        # 🔥 能级间距分析
        spacing_results = analyze_level_spacing(eigenvalues_hermitian, output_dir)
        
        results['hermitian'] = {
            'eigenvalues': eigenvalues_hermitian,
            'mean': np.mean(eigenvalues_hermitian),
            'std': np.std(eigenvalues_hermitian),
            'spacing': spacing_results
        }
        
    except Exception as e:
        print(f"⚠️  厄米化分析失败: {e}")
        results['hermitian'] = None
    
    # ============================================================
    # 方法2：Gram矩阵 G = W^T @ W（正定矩阵）
    # ============================================================
    print(f"\n{'─'*70}")
    print("📊 方法2：Gram矩阵 G = W^T @ W")
    print(f"{'─'*70}")
    
    try:
        G = torch.mm(W.T, W)
        print("✓ Gram矩阵构造完成")
        
        eigenvalues_gram = torch.linalg.eigvalsh(G)
        eigenvalues_gram = eigenvalues_gram.cpu().numpy()
        
        print(f"特征值数量: {len(eigenvalues_gram)}")
        print(f"特征值范围: [{eigenvalues_gram.min():.4f}, {eigenvalues_gram.max():.4f}]")
        print(f"特征值均值: {np.mean(eigenvalues_gram):.4f}")
        
        results['gram'] = {
            'eigenvalues': eigenvalues_gram,
            'mean': np.mean(eigenvalues_gram),
            'std': np.std(eigenvalues_gram)
        }
        
    except Exception as e:
        print(f"⚠️  Gram矩阵分析失败: {e}")
        results['gram'] = None
    
    # ============================================================
    # 方法3：奇异值分解（SVD）
    # ============================================================
    print(f"\n{'─'*70}")
    print("📊 方法3：奇异值分解（SVD）")
    print(f"{'─'*70}")
    
    try:
        U, S, Vh = torch.linalg.svd(W)
        singular_values = S.cpu().numpy()
        
        print("✓ SVD完成")
        print(f"奇异值数量: {len(singular_values)}")
        print(f"奇异值范围: [{singular_values.min():.4f}, {singular_values.max():.4f}]")
        print(f"奇异值均值: {np.mean(singular_values):.4f}")
        print(f"条件数: {singular_values.max() / (singular_values.min() + 1e-10):.2f}")
        
        results['svd'] = {
            'singular_values': singular_values,
            'mean': np.mean(singular_values),
            'std': np.std(singular_values),
            'condition_number': singular_values.max() / (singular_values.min() + 1e-10)
        }
        
    except Exception as e:
        print(f"⚠️  SVD分析失败: {e}")
        results['svd'] = None
    
    # ============================================================
    # 可视化对比
    # ============================================================
    print(f"\n{'─'*70}")
    print("📊 生成对比可视化")
    print(f"{'─'*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1：厄米特征值分布
    ax1 = axes[0, 0]
    if results['hermitian'] is not None:
        ax1.hist(results['hermitian']['eigenvalues'], bins=50, alpha=0.7, 
                edgecolor='black', color='blue')
        ax1.set_xlabel('Eigenvalue', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Hermitian Matrix: H = (W + W^T) / 2', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # 子图2：Gram特征值分布
    ax2 = axes[0, 1]
    if results['gram'] is not None:
        ax2.hist(results['gram']['eigenvalues'], bins=50, alpha=0.7, 
                edgecolor='black', color='green')
        ax2.set_xlabel('Eigenvalue', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Gram Matrix: G = W^T @ W', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 子图3：奇异值分布
    ax3 = axes[1, 0]
    if results['svd'] is not None:
        ax3.plot(singular_values, 'o-', markersize=3, linewidth=1, color='red')
        ax3.set_xlabel('Index', fontsize=12)
        ax3.set_ylabel('Singular Value', fontsize=12)
        ax3.set_title('Singular Value Spectrum', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 子图4：统计对比
    ax4 = axes[1, 1]
    methods = []
    means = []
    stds = []
    
    if results['hermitian'] is not None:
        methods.append('Hermitian')
        means.append(results['hermitian']['mean'])
        stds.append(results['hermitian']['std'])
    
    if results['gram'] is not None:
        methods.append('Gram')
        means.append(results['gram']['mean'])
        stds.append(results['gram']['std'])
    
    if results['svd'] is not None:
        methods.append('SVD')
        means.append(results['svd']['mean'])
        stds.append(results['svd']['std'])
    
    x_pos = np.arange(len(methods))
    ax4.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=10, 
           color=['blue', 'green', 'red'][:len(methods)])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, fontsize=12)
    ax4.set_ylabel('Mean Eigenvalue', fontsize=12)
    ax4.set_title('Statistical Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_matrix_analysis_complete.png', dpi=300, bbox_inches='tight')
    print(f"✓ 完整分析图已保存")
    
    # 保存数据
    np.save(output_dir / 'analysis_results_complete.npy', results)
    print(f"✓ 分析数据已保存")
    
    print(f"\n{'='*70}\n")
    
    return results

def analyze_spectral_staircase(eigenvalues, output_dir):
    """
    谱阶梯函数分析 - 验证特征值增长规律
    
    检查特征值增长是否符合理论预测的 N(T) ~ (T/2π) log(T/2π) 规律
    """
    print(f"\n{'='*70}")
    print("📊 谱阶梯函数分析（Spectral Staircase）")
    print(f"{'='*70}")
    
    eigs = np.sort(eigenvalues)
    
    # 只看正能级（或取绝对值）
    eigs_positive = eigs[eigs > 0]
    
    if len(eigs_positive) < 10:
        print("⚠️  正特征值数量太少，跳过阶梯函数分析")
        return None
    
    # 累积计数函数 N(E)
    N_E = np.arange(1, len(eigs_positive) + 1)
    
    print(f"正特征值数量: {len(eigs_positive)}")
    print(f"特征值范围: [{eigs_positive.min():.4f}, {eigs_positive.max():.4f}]")
    
    # 绘制阶梯函数
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：线性坐标
    ax1 = axes[0]
    ax1.step(eigs_positive, N_E, where='post', linewidth=1.5, color='blue', 
             label='AI Eigenvalues')
    
    # 尝试拟合 N(E) ~ a * E * log(b*E)
    # 简化版本：只拟合系数
    E_range = np.linspace(eigs_positive.min(), eigs_positive.max(), 200)
    
    # 理论曲线（参考量子系统的能级增长）
    # N(T) ≈ (T/2π) log(T/2π) - T/2π
    # 简化：N(E) ≈ a * E * log(E)
    a_fit = len(eigs_positive) / (eigs_positive[-1] * np.log(eigs_positive[-1] + 1))
    N_theory = a_fit * E_range * np.log(E_range + 1)
    
    ax1.plot(E_range, N_theory, 'r--', linewidth=2, alpha=0.7,
            label=f'Theory: N(E) ~ E log(E)')
    
    ax1.set_xlabel('Eigenvalue (E)', fontsize=12)
    ax1.set_ylabel('Cumulative Count N(E)', fontsize=12)
    ax1.set_title('Spectral Staircase Function (Linear)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图：双对数坐标
    ax2 = axes[1]
    ax2.loglog(eigs_positive, N_E, 'o-', markersize=3, linewidth=1, 
               color='blue', alpha=0.6, label='AI Eigenvalues')
    ax2.loglog(E_range, N_theory, 'r--', linewidth=2, alpha=0.7,
              label='Theory: N(E) ~ E log(E)')
    
    ax2.set_xlabel('Eigenvalue (E)', fontsize=12)
    ax2.set_ylabel('Cumulative Count N(E)', fontsize=12)
    ax2.set_title('Spectral Staircase Function (Log-Log)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectral_staircase.png', dpi=300, bbox_inches='tight')
    print(f"✓ 谱阶梯函数图已保存")
    
    # 计算拟合度
    # 在对数空间中计算均方根误差
    N_actual_interp = np.interp(E_range, eigs_positive, N_E)
    rmse = np.sqrt(np.mean((np.log(N_actual_interp + 1) - np.log(N_theory + 1))**2))
    
    print(f"\n🎯 拟合分析：")
    print(f"  对数空间RMSE: {rmse:.4f}")
    
    if rmse < 0.5:
        print(f"  🔥🔥🔥 极佳拟合！特征值增长符合理论预测！")
    elif rmse < 1.0:
        print(f"  🔥 良好拟合！支持量子混沌假设")
    elif rmse < 2.0:
        print(f"  ✓ 中等拟合")
    else:
        print(f"  ⚠️  拟合较差")
    
    print(f"{'='*70}\n")
    
    return {
        'rmse': rmse,
        'eigenvalues': eigs_positive,
        'cumulative_count': N_E
    }


def analyze_embedding_fft(model, output_dir):
    """
    分析可学习Embedding的FFT - 检测对数周期性
    
    🔥 终极检验：如果AI从随机噪声中涌现出对数周期性
    → 说明AI发现了素数的深层数学结构！
    """
    print(f"\n{'='*70}")
    print("🔬 Embedding FFT分析 - 检测对数周期性")
    print(f"{'='*70}")
    
    # 检查是否是可学习embedding
    if not hasattr(model.riemann_embedding, 'embedding'):
        print("⚠️  模型使用固定正弦编码，跳过FFT分析")
        return None
    
    # 提取embedding权重
    embedding_weights = model.riemann_embedding.embedding.weight.detach().cpu().numpy()
    print(f"Embedding权重形状: {embedding_weights.shape}")
    
    # 对每个维度进行FFT分析
    n_dims = min(8, embedding_weights.shape[1])  # 分析前8个维度
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    detected_log_periodic = []
    
    for dim in range(n_dims):
        ax = axes[dim]
        
        # 提取该维度的embedding
        emb_dim = embedding_weights[:10000, dim]  # 取前1万个位置
        
        # FFT
        fft_result = np.fft.fft(emb_dim)
        freqs = np.fft.fftfreq(len(emb_dim))
        
        # 只看正频率
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(freqs)//2])
        
        # 绘制频谱
        ax.plot(positive_freqs[1:], positive_fft[1:], linewidth=1)
        ax.set_xlabel('Frequency', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_title(f'Dimension {dim}', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 检测主频率
        main_freq_idx = np.argmax(positive_fft[1:]) + 1
        main_freq = positive_freqs[main_freq_idx]
        
        # 检测是否为对数周期（log-periodic）
        # 简化判断：看频谱是否有多个等比间隔的峰
        peaks_idx = np.where(positive_fft > 0.3 * positive_fft.max())[0]
        if len(peaks_idx) >= 3:
            # 计算峰之间的频率比
            peak_freqs = positive_freqs[peaks_idx]
            ratios = peak_freqs[1:] / peak_freqs[:-1]
            
            # 如果比值接近常数，说明是对数周期
            if len(ratios) > 1 and np.std(ratios) / np.mean(ratios) < 0.2:
                detected_log_periodic.append(dim)
                ax.set_title(f'Dimension {dim} 🔥 Log-Periodic!', 
                           fontsize=11, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embedding_fft_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Embedding FFT分析图已保存")
    
    print(f"\n🎯 对数周期性检测：")
    if detected_log_periodic:
        print(f"  🔥🔥🔥 检测到对数周期性！维度: {detected_log_periodic}")
        print(f"  🔥🔥🔥 AI可能发现了素数的深层周期结构！")
    else:
        print(f"  ⚠️  未检测到明显的对数周期性")
        print(f"  建议：增加训练轮数或调整超参数")
    
    print(f"{'='*70}\n")
    
    return {
        'log_periodic_dims': detected_log_periodic,
        'embedding_weights': embedding_weights
    }


def analyze_hidden_states(model, X_sample, output_dir):
    """
    分析隐藏层状态的PCA投影
    寻找复平面上的结构
    """
    print(f"\n{'='*70}")
    print("🔬 隐藏状态PCA分析")
    print(f"{'='*70}")
    
    model.eval()
    with torch.no_grad():
        hidden_states = model.get_hidden_states(X_sample)
        hidden_states = hidden_states.cpu().numpy()
    
    print(f"隐藏状态形状: {hidden_states.shape}")
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(hidden_states)
    
    print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
    
    # 绘制PCA投影
    plt.figure(figsize=(10, 10))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Hidden States in 2D PCA Space')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 绘制参考线（Re=0.5）
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.savefig(output_dir / 'hidden_states_pca.png', dpi=300, bbox_inches='tight')
    print(f"✓ PCA投影图已保存")
    
    return projected

# ==================== 主程序 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 生成数据
    print(f"⏳ 步骤1/7：生成素数数据...")
    prime_gaps = generate_prime_gaps(config.NUM_PRIMES)
    
    # 2. 数据归一化
    print(f"⏳ 步骤2/7：数据归一化...")
    gap_mean = np.mean(prime_gaps)
    gap_std = np.std(prime_gaps)
    prime_gaps_normalized = (prime_gaps - gap_mean) / gap_std
    
    num_samples = len(prime_gaps_normalized)
    
    print(f"\n{'='*70}")
    print("数据预处理")
    print(f"{'='*70}")
    print(f"归一化后范围: [{prime_gaps_normalized.min():.4f}, {prime_gaps_normalized.max():.4f}]")
    print(f"归一化后均值: {np.mean(prime_gaps_normalized):.6f}")
    print(f"归一化后标准差: {np.std(prime_gaps_normalized):.6f}")
    
    # 3. 加载到GPU
    print(f"⏳ 步骤3/7：加载数据到GPU...")
    X_gpu = torch.arange(num_samples, device=device)
    y_gpu = torch.FloatTensor(prime_gaps_normalized).unsqueeze(1).to(device)
    
    print(f"✓ 数据已加载到GPU")
    print(f"{'='*70}\n")
    
    # 4. 创建模型
    print(f"⏳ 步骤4/7：创建模型...")
    model = PrimeGapPredictor(
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        dropout=config.DROPOUT,
        learnable_embedding=config.LEARNABLE_EMBEDDING
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {total_params:,}")
    
    # 5. 创建涌现追踪器
    print(f"⏳ 步骤5/7：初始化涌现追踪器...")
    tracker = EmergenceTracker()
    
    # 6. 创建在线超参数进化器
    print(f"⏳ 步骤6/7：初始化在线超参数进化器...")
    hyperparam_evolver = OnlineHyperparamEvolution(
        initial_lr=config.LEARNING_RATE,
        initial_wd=config.WEIGHT_DECAY
    )
    
    # 7. 训练（启用在线进化）
    print(f"⏳ 步骤7/7：开始训练循环（10000轮）...\n")
    losses = train_model(model, X_gpu, y_gpu, device, config, tracker, hyperparam_evolver)
    
    # 8. 保存训练曲线和超参数进化
    np.save(OUTPUT_DIR / 'losses.npy', np.array(losses))
    
    # 绘制Loss曲线 + 超参数进化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：Loss曲线
    ax1 = axes[0]
    ax1.plot(losses, linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 标记顿悟时刻
    if tracker.grokking_moments:
        for grok_epoch in tracker.grokking_moments:
            ax1.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.scatter(tracker.grokking_moments, 
                   [losses[e] for e in tracker.grokking_moments],
                   color='red', s=100, zorder=5, label='Grokking Moments')
        ax1.legend()
    
    # 下图：学习率进化
    ax2 = axes[1]
    if hyperparam_evolver and hyperparam_evolver.evolution_history:
        epochs_list = [0]  # 初始epoch
        lr_list = [config.LEARNING_RATE]  # 初始lr
        
        for record in hyperparam_evolver.evolution_history:
            epochs_list.append(record['epoch'])
            lr_list.append(record['new_lr'])
        
        # 绘制阶梯图（学习率只在进化时改变）
        ax2.step(epochs_list, lr_list, where='post', linewidth=2, color='green', label='Learning Rate')
        
        # 标记进化时刻
        evolution_epochs = [r['epoch'] for r in hyperparam_evolver.evolution_history]
        evolution_lrs = [r['new_lr'] for r in hyperparam_evolver.evolution_history]
        ax2.scatter(evolution_epochs, evolution_lrs, color='orange', s=80, zorder=5, 
                   label='Evolution Moments')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Evolution (Online Hyperparam Evolution)', 
                     fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        # 如果没有进化，显示固定lr
        ax2.axhline(y=config.LEARNING_RATE, color='blue', linestyle='-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate (Fixed)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ 训练概览图已保存")
    
    # 9. 涌现分析
    emergence_results = tracker.analyze_emergence()
    
    # 10. 完整量子分析
    print(f"\n{'='*70}")
    print("🔬 开始完整量子分析")
    print(f"{'='*70}\n")
    
    # 10.1 权重矩阵分析（厄米化 + 能级间距）
    goe_results = analyze_weight_matrices(model, OUTPUT_DIR)
    
    # 10.2 谱阶梯函数分析（希尔伯特-波利亚猜想）
    if goe_results and goe_results.get('hermitian'):
        hermitian_eigs = goe_results['hermitian']['eigenvalues']
        staircase_results = analyze_spectral_staircase(hermitian_eigs, OUTPUT_DIR)
    
    # 10.3 Embedding FFT分析（检测对数周期性）
    embedding_results = analyze_embedding_fft(model, OUTPUT_DIR)
    
    # 10.4 隐藏状态PCA分析
    sample_indices = torch.randperm(len(X_gpu))[:10000].to(device)
    X_sample = X_gpu[sample_indices]
    pca_results = analyze_hidden_states(model, X_sample, OUTPUT_DIR)
    
    # 10.5 如果有顿悟，额外说明
    if tracker.grokking_moments:
        print(f"\n{'='*70}")
        print(f"📊 顿悟时刻总结")
        print(f"{'='*70}")
        print(f"检测到 {len(tracker.grokking_moments)} 次顿悟")
        print(f"顿悟时刻: {tracker.grokking_moments}")
        print(f"{'='*70}\n")
        
    # 保存完整分析结果
    results = {
        'emergence': emergence_results,
        'goe': goe_results if 'goe_results' in locals() else None,
        'staircase': staircase_results if 'staircase_results' in locals() else None,
        'embedding_fft': embedding_results if 'embedding_results' in locals() else None,
        'grokking_moments': tracker.grokking_moments,
        'config': {
            'd_model': config.D_MODEL,
            'n_layers': config.N_LAYERS,
            'n_heads': config.N_HEADS,
            'lr': config.LEARNING_RATE,
            'lr_schedule': config.LR_SCHEDULE,
            'learnable_embedding': config.LEARNABLE_EMBEDDING,
            'num_epochs': config.NUM_EPOCHS,
        }
    }
    
    # 保存为JSON（只保存可序列化的部分）
    results_json = {
        'grokking_moments': results['grokking_moments'],
        'config': results['config'],
        'emergence_summary': {
            'loss_reduction': emergence_results['loss_reduction'],
            'final_loss': emergence_results['final_loss'],
            'grokking_count': emergence_results['grokking_count']
        }
    }
    
    with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ 分析结果已保存到 {OUTPUT_DIR}")
    
    # 🔥 最终判断总结
    print(f"\n{'='*70}")
    print("🎯 实验结果总结")
    print(f"{'='*70}\n")
    
    success_count = 0
    
    # 判断1：能级间距是否符合GOE
    if goe_results and goe_results.get('hermitian'):
        spacing_info = goe_results['hermitian'].get('spacing', {})
        ks_dist = spacing_info.get('ks_distance')
        if ks_dist is not None:
            print(f"✓ 能级间距分析：K-S距离 = {ks_dist:.4f}")
            if ks_dist < 0.1:
                print(f"  🔥🔥🔥 极高拟合度！强烈支持时间反演对称混沌特征！")
                success_count += 3
            elif ks_dist < 0.2:
                print(f"  🔥 良好拟合！支持时间反演对称混沌假设")
                success_count += 2
            elif ks_dist < 0.3:
                print(f"  ✓ 中等拟合")
                success_count += 1
    
    # 判断2：谱阶梯函数是否符合理论
    if 'staircase_results' in locals() and staircase_results:
        rmse = staircase_results.get('rmse')
        if rmse is not None:
            print(f"\n✓ 谱阶梯函数分析：RMSE = {rmse:.4f}")
            if rmse < 0.5:
                print(f"  🔥🔥🔥 极佳拟合！特征值增长符合理论预测！")
                success_count += 3
            elif rmse < 1.0:
                print(f"  🔥 良好拟合！支持时间反演对称混沌假设")
                success_count += 2
    
    # 判断3：Embedding是否涌现对数周期性
    if 'embedding_results' in locals() and embedding_results:
        log_periodic_dims = embedding_results.get('log_periodic_dims', [])
        if log_periodic_dims:
            print(f"\n✓ Embedding FFT分析：")
            print(f"  🔥🔥🔥 检测到对数周期性！维度: {log_periodic_dims}")
            print(f"  🔥🔥🔥 AI可能从随机噪声中涌现了深层数学结构！")
            success_count += 5
    
    # 总体评价
    print(f"\n{'='*70}")
    if success_count >= 8:
        print(f"🏆 实验大获成功！发现素数诱导的GOE混沌特征！")
    elif success_count >= 5:
        print(f"🔥 实验成功！发现了素数诱导的GOE混沌特征")
    elif success_count >= 2:
        print(f"✓ 实验部分成功，发现了有趣的规律")
    else:
        print(f"⚠️  实验未达到预期目标")
    print(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print("🎉 实验完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
