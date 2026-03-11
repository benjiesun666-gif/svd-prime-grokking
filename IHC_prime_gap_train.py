"""
素数间隙预测 - 纯粹涌现版 + IHC 消融实验
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
import torch.nn.functional as F
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
    import sys
    if 'ipykernel' not in sys.modules:
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
    PHYSICAL_BATCH_SIZE = 128
    ACCUMULATION_STEPS = 16

    # 模型配置
    D_MODEL = 256
    N_LAYERS = 6
    N_HEADS = 8
    DROPOUT = 0.1
    LEARNABLE_EMBEDDING = True

    # IHC 配置（消融实验开关）
    USE_IHC = False          # 设为 True 则使用 IHC 多流架构
    NUM_FLOWS = 4            # 并行流的数量（总维度 = D_MODEL）

    # 训练配置
    NUM_EPOCHS = 10000
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    LR_SCHEDULE = 'cosine'

    # 监控配置
    PRINT_EVERY = 1
    SAVE_EVERY = 500

    # 顿悟检测（纯观察，不干预）
    GROKKING_THRESHOLD = 0.3
    GROKKING_WINDOW = 20

    # 涌现追踪
    TRACK_GRADIENTS = True
    TRACK_WEIGHTS = True

    # 🔥 断点续存配置
    CHECKPOINT_EVERY = 50
    CHECKPOINT_DIR = "riemann_checkpoints"
    AUTO_RESUME = True


config = Config()


# ==================== 断点续存系统 ====================
def save_checkpoint(epoch, model, optimizer, losses, tracker, hyperparam_evolver, config, filename):
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
    if not os.path.exists(filename):
        return None
    print(f"📂 加载checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ 成功恢复到 Epoch {checkpoint['epoch']}")
    return checkpoint


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
    epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
    latest_epoch = max(epochs)
    latest_file = f'checkpoint_epoch_{latest_epoch}.pt'
    return os.path.join(checkpoint_dir, latest_file)


# ==================== 在线超参数进化器 ====================
class OnlineHyperparamEvolution:
    def __init__(self, initial_lr=1e-4, initial_wd=0.01):
        self.hyperparams = {'lr': initial_lr, 'weight_decay': initial_wd}
        self.current_cooldown = 30
        self.cooldown_bounds = (20, 150)
        self.last_update_epoch = -100
        self.lr_bounds = (1e-6, 5e-4)
        self.max_change_ratio = 0.3
        self.evolution_history = []
        print(f"\n{'=' * 70}")
        print("🧬 在线超参数进化器已启动")
        print(f"{'=' * 70}")
        print(f"初始学习率: {initial_lr:.2e}")
        print(f"初始权重衰减: {initial_wd}")
        print(f"动态冷却期: {self.current_cooldown} (范围: {self.cooldown_bounds})")
        print(f"{'=' * 70}\n")

    def calculate_dynamic_cooldown(self, losses):
        if len(losses) < 50:
            return 30
        recent_losses = losses[-50:]
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        cv = loss_std / (loss_mean + 1e-10)
        if cv < 0.03:
            new_cooldown = 100
        elif cv < 0.08:
            new_cooldown = 50
        elif cv < 0.15:
            new_cooldown = 30
        else:
            new_cooldown = 20
        new_cooldown = np.clip(new_cooldown, *self.cooldown_bounds)
        alpha = 0.3
        smoothed_cooldown = int(alpha * new_cooldown + (1 - alpha) * self.current_cooldown)
        return smoothed_cooldown

    def should_update(self, epoch, losses):
        self.current_cooldown = self.calculate_dynamic_cooldown(losses)
        epochs_since_last = epoch - self.last_update_epoch
        return epochs_since_last >= self.current_cooldown and epoch >= 30

    def evaluate_progress(self, losses):
        if len(losses) < 50:
            return 0.0
        recent_50 = losses[-50:]
        previous_50 = losses[-100:-50] if len(losses) >= 100 else losses[:50]
        recent_avg = np.mean(recent_50)
        previous_avg = np.mean(previous_50)
        if previous_avg < 1e-10:
            return 0.0
        progress = (previous_avg - recent_avg) / previous_avg
        return progress

    def mutate_hyperparams(self, progress, losses):
        if progress > 0.05:
            mutation_strength = 0.1
            strategy = "保持方向"
        elif progress > 0.01:
            mutation_strength = 0.2
            strategy = "适度探索"
        else:
            mutation_strength = 0.4
            strategy = "大胆突破"
        lr_multiplier = np.random.uniform(1 - mutation_strength, 1 + mutation_strength)
        new_lr = self.hyperparams['lr'] * lr_multiplier
        new_lr = np.clip(new_lr, *self.lr_bounds)
        max_change = self.hyperparams['lr'] * self.max_change_ratio
        new_lr = np.clip(new_lr, self.hyperparams['lr'] - max_change, self.hyperparams['lr'] + max_change)
        return {
            'lr': new_lr,
            'weight_decay': self.hyperparams['weight_decay'],
            'strategy': strategy,
            'mutation_strength': mutation_strength
        }

    def update(self, epoch, losses, optimizer):
        if not self.should_update(epoch, losses):
            return False
        progress = self.evaluate_progress(losses)
        new_hyperparams = self.mutate_hyperparams(progress, losses)
        self.evolution_history.append({
            'epoch': epoch,
            'old_lr': self.hyperparams['lr'],
            'new_lr': new_hyperparams['lr'],
            'progress': progress,
            'strategy': new_hyperparams['strategy'],
            'cooldown': self.current_cooldown
        })
        old_lr = self.hyperparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_hyperparams['lr']
            param_group['weight_decay'] = new_hyperparams['weight_decay']
        lr_change = (new_hyperparams['lr'] - old_lr) / old_lr * 100
        print(f"\n{'🧬' * 35}")
        print(f"💥 超参数进化触发 - Epoch {epoch}")
        print(f"{'=' * 70}")
        print(f"  训练进度: {progress * 100:+.2f}% (最近50轮 vs 前50轮)")
        print(f"  进化策略: {new_hyperparams['strategy']}")
        print(f"  变异强度: {new_hyperparams['mutation_strength'] * 100:.0f}%")
        print(f"  学习率: {old_lr:.2e} → {new_hyperparams['lr']:.2e} ({lr_change:+.1f}%)")
        print(f"  动态冷却期: {self.current_cooldown} 轮")
        print(f"  下次进化: 约 Epoch {epoch + self.current_cooldown}")
        print(f"{'=' * 70}\n")
        self.hyperparams = {'lr': new_hyperparams['lr'], 'weight_decay': new_hyperparams['weight_decay']}
        self.last_update_epoch = epoch
        return True


# ==================== 涌现追踪器 ====================
class EmergenceTracker:
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.history = {'epoch': [], 'loss': [], 'lr': [], 'grad_norm': [], 'param_norm': []}
        self.grokking_moments = []

    def record(self, epoch, loss, lr, grad_norm=None, param_norm=None):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['lr'].append(lr)
        if grad_norm is not None:
            self.history['grad_norm'].append(grad_norm)
        if param_norm is not None:
            self.history['param_norm'].append(param_norm)

    def detect_grokking(self, config, model=None, X_sample=None, y_sample=None):
        losses = self.history['loss']
        if len(losses) < config.GROKKING_WINDOW + 1:
            return False, 0.0
        recent_avg = np.mean(losses[-(config.GROKKING_WINDOW + 1):-1])
        current_loss = losses[-1]
        sudden_drop = recent_avg - current_loss
        drop_ratio = sudden_drop / recent_avg if recent_avg > 1e-10 else 0
        loss_breakthrough = (drop_ratio > config.GROKKING_THRESHOLD and
                             sudden_drop > 0.01 and current_loss < recent_avg)
        log_breakthrough = False
        log_corr = 0.0
        if model is not None and X_sample is not None and y_sample is not None:
            try:
                with torch.no_grad():
                    predictions = model(X_sample).squeeze()
                pred_np = predictions.cpu().numpy()
                true_np = y_sample.cpu().numpy().squeeze()
                pred_log = np.log(np.abs(pred_np) + 1e-6)
                true_log = np.log(np.abs(true_np) + 1e-6)
                if len(pred_log) > 10 and np.std(pred_log) > 1e-6 and np.std(true_log) > 1e-6:
                    log_corr = np.corrcoef(pred_log, true_log)[0, 1]
                    log_breakthrough = (log_corr > 0.9)
            except:
                pass
        periodicity_breakthrough = False
        dominant_freq = 0.0
        if model is not None and X_sample is not None:
            try:
                with torch.no_grad():
                    predictions = model(X_sample).squeeze()
                pred_np = predictions.cpu().numpy()
                if len(pred_np) > 100:
                    pred_detrended = np.diff(pred_np)
                    fft_result = np.fft.fft(pred_detrended)
                    freqs = np.fft.fftfreq(len(pred_detrended))
                    positive_freqs = freqs[:len(freqs) // 2]
                    positive_fft = np.abs(fft_result[:len(freqs) // 2])
                    if len(positive_fft) > 1:
                        main_idx = np.argmax(positive_fft[1:]) + 1
                        dominant_freq = positive_freqs[main_idx]
                        mean_amplitude = np.mean(positive_fft[1:])
                        max_amplitude = positive_fft[main_idx]
                        if max_amplitude > 5 * mean_amplitude:
                            periodicity_breakthrough = True
            except:
                pass
        breakthrough_count = sum([loss_breakthrough, log_breakthrough, periodicity_breakthrough])
        is_true_grokking = breakthrough_count >= 2
        if is_true_grokking:
            print(f"\n{'=' * 70}")
            print(f"🔬 多维度数学突破检测")
            print(f"{'=' * 70}")
            print(f"  ✓ Loss突变: {'是' if loss_breakthrough else '否'} (下降{drop_ratio * 100:.1f}%)")
            if log_corr != 0.0:
                print(f"  ✓ 对数关系: {'是' if log_breakthrough else '否'} (相关性={log_corr:.4f})")
            if dominant_freq != 0.0:
                print(f"  ✓ 周期性: {'是' if periodicity_breakthrough else '否'} (主频率={dominant_freq:.6f})")
            print(f"  → 突破维度: {breakthrough_count}/3")
            print(f"{'=' * 70}\n")
        return is_true_grokking, drop_ratio

    def analyze_emergence(self):
        print("\n" + "=" * 70)
        print("🔍 涌现分析")
        print("=" * 70)
        epochs = np.array(self.history['epoch'])
        losses = np.array(self.history['loss'])
        lrs = np.array(self.history['lr'])
        print("\n📉 训练统计：")
        print(f"  初始Loss: {losses[0]:.6f}")
        print(f"  最终Loss: {losses[-1]:.6f}")
        print(f"  下降幅度: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"  顿悟次数: {len(self.grokking_moments)}")
        if self.grokking_moments:
            print(f"  顿悟时刻: {self.grokking_moments}")
        print(f"\n📊 学习率演化：")
        print(f"  初始LR: {lrs[0]:.2e}")
        print(f"  最终LR: {lrs[-1]:.2e}")
        print(f"  LR范围: [{lrs.min():.2e}, {lrs.max():.2e}]")
        if self.history['grad_norm']:
            grad_norms = np.array(self.history['grad_norm'])
            print(f"\n📈 梯度统计：")
            print(f"  平均梯度范数: {np.mean(grad_norms):.4f}")
            print(f"  最大梯度范数: {np.max(grad_norms):.4f}")
            print(f"  最小梯度范数: {np.min(grad_norms):.4f}")
        print("\n" + "=" * 70)
        return {
            'loss_reduction': (losses[0] - losses[-1]) / losses[0],
            'grokking_count': len(self.grokking_moments),
            'final_loss': losses[-1],
            'final_lr': lrs[-1]
        }


# ==================== 数据生成 ====================
def generate_prime_gaps(num_primes):
    print(f"\n{'=' * 70}")
    print(f"生成前 {num_primes:,} 个素数...")
    print(f"{'=' * 70}")
    start_time = time.time()
    if num_primes < 10:
        upper_bound = 30
    else:
        ln_n = math.log(num_primes)
        ln_ln_n = math.log(ln_n) if ln_n > 1 else 0
        upper_bound = int(num_primes * (ln_n + ln_ln_n + 2))
    print(f"估算上界: {upper_bound:,}")
    primes = list(primerange(1, upper_bound))
    if len(primes) < num_primes:
        print(f"⚠️  警告：只生成了 {len(primes)} 个素数")
        num_primes = len(primes)
    else:
        primes = primes[:num_primes]
    prime_gaps = np.diff(primes)
    elapsed = time.time() - start_time
    print(f"✓ 生成完成 ({elapsed:.2f}秒)")
    print(f"  素数数量: {len(primes):,}")
    print(f"  间隙数量: {len(prime_gaps):,}")
    print(f"  间隙范围: [{prime_gaps.min()}, {prime_gaps.max()}]")
    print(f"  平均间隙: {np.mean(prime_gaps):.2f}")
    return prime_gaps


# ==================== 模型定义 ====================
class RiemannEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000000, learnable=False):
        super().__init__()
        self.d_model = d_model
        self.learnable = learnable
        if learnable:
            self.embedding = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x):
        if self.learnable:
            return self.embedding(x)
        else:
            return self.pe[x]


# ==================== 原始单流 Transformer 模型 ====================
class PrimeGapPredictor(nn.Module):
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
        embedded = self.riemann_embedding(x).unsqueeze(1)
        transformed = self.transformer(embedded)
        gap = self.output(transformed.squeeze(1))
        return gap

    def get_hidden_states(self, x):
        embedded = self.riemann_embedding(x).unsqueeze(1)
        hidden = self.transformer(embedded)
        return hidden.squeeze(1)

    def get_attention_weights(self):
        weights = []
        for layer in self.transformer.layers:
            attn_weights = layer.self_attn.in_proj_weight
            weights.append(attn_weights.detach().cpu())
        return torch.cat(weights, dim=0)


# ==================== IHC 多流 Transformer 模型 ====================
class IHCTransformerLayer(nn.Module):
    """
    Identity Hyper-Connection Transformer Layer
    - num_flows: 并行流的数量
    - dim: 每流的维度（总隐层维度 = num_flows * dim）
    - 内部 Attention 和 FFN 处理聚合后的单流向量（维度 dim）
    """
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

        # ϕ 投影网络：输入为所有流的拼接 (num_flows * dim)，输出两个系数向量 h_pre, h_post
        self.phi = nn.Sequential(
            nn.Linear(num_flows * dim, num_flows * 2),
            nn.GELU(),
            nn.Linear(num_flows * 2, num_flows * 2)
        )

    def forward(self, x_flows):
        batch, num_flows, dim = x_flows.shape
        # 1. ϕ 投影生成 h_pre 和 h_post
        phi_in = x_flows.reshape(batch, -1)                     # [batch, num_flows*dim]
        phi_out = self.phi(phi_in)                               # [batch, num_flows*2]
        h_pre = F.softmax(phi_out[:, :num_flows], dim=-1)       # [batch, num_flows] 聚合权重
        h_post = F.softmax(phi_out[:, num_flows:], dim=-1)      # [batch, num_flows] 写回权重

        # 2. 用 h_pre 聚合各流信息
        agg = torch.einsum('bf,bfd->bd', h_pre, x_flows)        # [batch, dim]

        # 3. 通过 Attention（带残差和层归一化）
        agg2 = self.norm1(agg + self._sa_block(agg))

        # 4. 通过 FFN（带残差和层归一化）
        agg3 = self.norm2(agg2 + self._ffn_block(agg2))

        # 5. 用 h_post 将聚合结果写回各流，并加上残差
        new_flows = x_flows + h_post.unsqueeze(-1) * agg3.unsqueeze(1)   # [batch, num_flows, dim]
        return new_flows

    def _sa_block(self, x):
        x = x.unsqueeze(1)                # [batch, 1, dim]
        attn_out, _ = self.self_attn(x, x, x)
        return attn_out.squeeze(1)

    def _ffn_block(self, x):
        return self.ffn(x)


class IHCPrimeGapPredictor(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=8, dropout=0.1,
                 learnable_embedding=False, num_flows=4):
        super().__init__()
        assert d_model % num_flows == 0, "d_model must be divisible by num_flows"
        self.d_model = d_model
        self.num_flows = num_flows
        self.flow_dim = d_model // num_flows

        # 嵌入层不变（输出 d_model）
        self.riemann_embedding = RiemannEmbedding(d_model, learnable=learnable_embedding)

        # 堆叠 IHC 层
        self.layers = nn.ModuleList([
            IHCTransformerLayer(
                num_flows=num_flows,
                dim=self.flow_dim,
                nhead=n_heads,
                dim_feedforward=d_model * 4,   # 注意：这里保持与原始模型一致的 FFN 容量
                dropout=dropout
            ) for _ in range(n_layers)
        ])

        # 输出头：先将各流聚合（取平均），然后通过 MLP
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
        # 嵌入 -> 重塑为多流
        emb = self.riemann_embedding(x)                    # [batch, d_model]
        x_flows = emb.view(-1, self.num_flows, self.flow_dim)  # [batch, num_flows, flow_dim]

        # 逐层通过 IHC
        for layer in self.layers:
            x_flows = layer(x_flows)

        # 对各流取平均作为最终表示
        pooled = x_flows.mean(dim=1)                       # [batch, flow_dim]
        return self.output_head(pooled)

    def get_hidden_states(self, x):
        emb = self.riemann_embedding(x)
        x_flows = emb.view(-1, self.num_flows, self.flow_dim)
        for layer in self.layers:
            x_flows = layer(x_flows)
        return x_flows

    def get_attention_weights(self):
        weights = []
        for layer in self.layers:
            attn = layer.self_attn
            if hasattr(attn, 'in_proj_weight'):
                weights.append(attn.in_proj_weight.detach().cpu())
        return torch.cat(weights, dim=0) if weights else torch.tensor([])


# ==================== 训练函数 ====================
def train_model(model, X_gpu_full, y_gpu_full, device, config, tracker, hyperparam_evolver=None):
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = None
    if hyperparam_evolver is None:
        if config.LR_SCHEDULE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
        elif config.LR_SCHEDULE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                                             min_lr=1e-6)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    X_gpu = X_gpu_full
    y_gpu = y_gpu_full
    total_data_size = len(X_gpu)
    num_samples = total_data_size
    num_batches = (num_samples + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    if config.USE_GRADIENT_ACCUMULATION:
        accumulation_steps = config.ACCUMULATION_STEPS
        effective_batch_size = config.BATCH_SIZE * accumulation_steps
        print(f"\n{'💪' * 35}")
        print(f"💪 梯度累积训练（100万数据全量，硬气！）")
        print(f"{'=' * 70}")
        print(f"  总数据量: {total_data_size:,}")
        print(f"  物理batch: {config.BATCH_SIZE}")
        print(f"  累积步数: {accumulation_steps}")
        print(f"  等效batch: {effective_batch_size}")
        print(f"{'=' * 70}\n")
    else:
        accumulation_steps = 1
    losses = []
    best_loss = float('inf')
    start_epoch = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    if config.AUTO_RESUME:
        latest_checkpoint = find_latest_checkpoint(config.CHECKPOINT_DIR)
        if latest_checkpoint:
            print(f"\n{'💾' * 35}")
            print(f"💾 检测到checkpoint，正在恢复...")
            print(f"{'=' * 70}")
            checkpoint_data = load_checkpoint(latest_checkpoint, model, optimizer, device)
            if checkpoint_data:
                start_epoch = checkpoint_data['epoch'] + 1
                losses = checkpoint_data['losses']
                if tracker and checkpoint_data.get('tracker_history'):
                    tracker.history = checkpoint_data['tracker_history']
                if hyperparam_evolver and checkpoint_data.get('hyperparam_history'):
                    hyperparam_evolver.evolution_history = checkpoint_data['hyperparam_history']
                print(f"✅ 训练将从 Epoch {start_epoch} 继续")
                print(f"{'=' * 70}\n")
            else:
                print("⚠️  checkpoint加载失败，从头开始")
    print(f"\n{'=' * 70}")
    print("🚀 开始训练 - 纯粹涌现模式")
    print(f"{'=' * 70}")
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
    print(f"{'=' * 70}\n")
    start_time = time.time()
    epoch_times = []
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        if epoch == 0:
            print(f"🔥 Epoch 0 开始训练...")
            print(f"   总数据量: {total_data_size:,}")
            print(f"   物理batch: {config.BATCH_SIZE}")
            print(f"   累积步数: {accumulation_steps}")
            print(f"   （如果看到这条消息后长时间无反应，说明batch计算很慢）\n")
        optimizer.zero_grad()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.BATCH_SIZE
            end_idx = min(start_idx + config.BATCH_SIZE, num_samples)
            batch_X = X_gpu[start_idx:end_idx]
            batch_y = y_gpu[start_idx:end_idx]
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            if config.USE_GRADIENT_ACCUMULATION:
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item() * (accumulation_steps if config.USE_GRADIENT_ACCUMULATION else 1)
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        if hyperparam_evolver is not None:
            hyperparam_evolver.update(epoch, losses, optimizer)
        if scheduler is not None and hyperparam_evolver is None:
            if config.LR_SCHEDULE == 'plateau':
                scheduler.step(epoch_loss / num_batches)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        param_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        tracker.record(epoch, avg_loss, current_lr, grad_norm.item(), param_norm)
        sample_size = min(1000, num_samples)
        sample_indices = torch.randperm(num_samples)[:sample_size].to(device)
        X_sample = X_gpu[sample_indices]
        y_sample = y_gpu[sample_indices]
        is_grokking, drop_ratio = tracker.detect_grokking(config, model, X_sample, y_sample)
        if is_grokking:
            tracker.grokking_moments.append(epoch)
            print(f"\n{'🔥' * 35}")
            print(f"💥 检测到顿悟！Epoch {epoch}")
            print(f"  Loss突降: {drop_ratio * 100:.1f}%")
            print(f"  当前Loss: {avg_loss:.6f}")
            print(f"{'🔥' * 35}\n")
            save_grokking_weights(model, epoch, avg_loss, OUTPUT_DIR, label='after')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': avg_loss},
                       OUTPUT_DIR / 'IHC_prime.pt')
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(epoch, model, optimizer, losses, tracker, hyperparam_evolver, config, checkpoint_path)
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
        if epoch % config.SAVE_EVERY == 0 and epoch > 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss, 'losses': losses},
                       OUTPUT_DIR / f'checkpoint_epoch_{epoch}.pt')
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"✓ 训练完成！")
    print(f"  总时间: {total_time / 60:.1f} 分钟")
    print(f"  最佳Loss: {best_loss:.6f}")
    print(f"  顿悟次数: {len(tracker.grokking_moments)}")
    if hyperparam_evolver is not None:
        print(f"\n🧬 超参数进化统计:")
        print(f"  进化次数: {len(hyperparam_evolver.evolution_history)}")
        if hyperparam_evolver.evolution_history:
            final_lr = hyperparam_evolver.hyperparams['lr']
            initial_lr = config.LEARNING_RATE
            print(f"  学习率: {initial_lr:.2e} → {final_lr:.2e}")
            np.save(OUTPUT_DIR / 'hyperparam_evolution_history.npy', hyperparam_evolver.evolution_history)
    print(f"{'=' * 70}\n")
    return losses


def save_grokking_weights(model, epoch, loss, output_dir, label=''):
    label_str = f"_{label}" if label else ""
    print(f"  💾 保存顿悟权重{label_str}...")
    attention_weights = model.get_attention_weights()
    grokking_data = {'epoch': epoch, 'loss': loss, 'attention_weights': attention_weights.numpy()}
    filename = f"grokking_weights_epoch_{epoch}{label_str}.npy"
    np.save(output_dir / filename, grokking_data)
    print(f"  ✓ 已保存到 {filename}")


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
    print(f"\n{'=' * 70}")
    print("数据预处理")
    print(f"{'=' * 70}")
    print(f"归一化后范围: [{prime_gaps_normalized.min():.4f}, {prime_gaps_normalized.max():.4f}]")
    print(f"归一化后均值: {np.mean(prime_gaps_normalized):.6f}")
    print(f"归一化后标准差: {np.std(prime_gaps_normalized):.6f}")

    # 3. 加载到GPU
    print(f"⏳ 步骤3/7：加载数据到GPU...")
    X_gpu = torch.arange(num_samples, device=device)
    y_gpu = torch.FloatTensor(prime_gaps_normalized).unsqueeze(1).to(device)
    print(f"✓ 数据已加载到GPU")
    print(f"{'=' * 70}\n")

    # 4. 创建模型（根据配置选择架构）
    print(f"⏳ 步骤4/7：创建模型...")
    if config.USE_IHC:
        print(f"  使用 IHC 多流架构，流数 = {config.NUM_FLOWS}")
        model = IHCPrimeGapPredictor(
            d_model=config.D_MODEL,
            n_layers=config.N_LAYERS,
            n_heads=config.N_HEADS,
            dropout=config.DROPOUT,
            learnable_embedding=config.LEARNABLE_EMBEDDING,
            num_flows=config.NUM_FLOWS
        ).to(device)
    else:
        print("  使用原始单流 Transformer 架构")
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

    # 7. 训练
    print(f"⏳ 步骤7/7：开始训练循环（{config.NUM_EPOCHS} 轮）...\n")
    losses = train_model(model, X_gpu, y_gpu, device, config, tracker, hyperparam_evolver)

    # 8. 保存训练曲线和超参数进化
    np.save(OUTPUT_DIR / 'losses.npy', np.array(losses))

    # 绘制 Loss 曲线 + 超参数进化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax1 = axes[0]
    ax1.plot(losses, linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    if tracker.grokking_moments:
        for grok_epoch in tracker.grokking_moments:
            ax1.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.7)
        ax1.scatter(tracker.grokking_moments, [losses[e] for e in tracker.grokking_moments], color='red', s=100,
                    zorder=5, label='Grokking Moments')
        ax1.legend()
    ax2 = axes[1]
    if hyperparam_evolver and hyperparam_evolver.evolution_history:
        epochs_list = [0]
        lr_list = [config.LEARNING_RATE]
        for record in hyperparam_evolver.evolution_history:
            epochs_list.append(record['epoch'])
            lr_list.append(record['new_lr'])
        ax2.step(epochs_list, lr_list, where='post', linewidth=2, color='green', label='Learning Rate')
        evolution_epochs = [r['epoch'] for r in hyperparam_evolver.evolution_history]
        evolution_lrs = [r['new_lr'] for r in hyperparam_evolver.evolution_history]
        ax2.scatter(evolution_epochs, evolution_lrs, color='orange', s=80, zorder=5, label='Evolution Moments')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Evolution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.axhline(y=config.LEARNING_RATE, color='blue', linestyle='-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate (Fixed)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ 训练概览图已保存")

    # 9. 涌现分析
    emergence_results = tracker.analyze_emergence()

    # 保存基本结果
    results_json = {
        'grokking_moments': tracker.grokking_moments,
        'config': {
            'd_model': config.D_MODEL,
            'n_layers': config.N_LAYERS,
            'n_heads': config.N_HEADS,
            'lr': config.LEARNING_RATE,
            'lr_schedule': config.LR_SCHEDULE,
            'learnable_embedding': config.LEARNABLE_EMBEDDING,
            'num_epochs': config.NUM_EPOCHS,
            'use_ihc': config.USE_IHC,
            'num_flows': config.NUM_FLOWS if config.USE_IHC else None,
        },
        'emergence_summary': {
            'loss_reduction': emergence_results['loss_reduction'],
            'final_loss': emergence_results['final_loss'],
            'grokking_count': emergence_results['grokking_count']
        }
    }
    with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ 训练结果已保存到 {OUTPUT_DIR}")
    print(f"\n{'=' * 70}")
    print("🎉 训练完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()