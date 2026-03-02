"""
混沌映射（Logistic映射）间隙预测实验
支持 Plan A（可学习嵌入）和 Plan C（固定正弦编码），通过命令行参数选择。
检查点和输出目录按版本隔离。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import math
import os
import sys
from pathlib import Path

# ==================== 配置类 ====================
class Config:
    """混沌映射实验配置（基类）"""
    # 数据
    NUM_POINTS = 1_000_000          # 生成序列点数（间隙数为 NUM_POINTS-1）
    MU = 3.8                         # Logistic映射参数（混沌区）
    X0 = 0.5                          # 初始值

    # 模型架构（与素数实验一致）
    D_MODEL = 256
    N_LAYERS = 6
    N_HEADS = 8
    DROPOUT = 0.1

    # 训练配置
    NUM_EPOCHS = 300                  # 可调整，根据收敛情况
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    LR_SCHEDULE = 'cosine'            # 余弦退火

    # 梯度累积
    USE_GRADIENT_ACCUMULATION = True
    PHYSICAL_BATCH_SIZE = 128
    ACCUMULATION_STEPS = 16            # 有效batch = BATCH_SIZE * ACCUMULATION_STEPS

    # 监控与保存（将在main中根据版本动态设置）
    PRINT_EVERY = 1
    SAVE_EVERY = 50
    CHECKPOINT_EVERY = 50
    # CHECKPOINT_DIR 和 OUTPUT_DIR 将在 main 中指定
    AUTO_RESUME = True

# ==================== 数据生成 ====================
def generate_logistic_map(num_points, mu=3.8, x0=0.5):
    """
    生成 Logistic 混沌映射序列 x_{n+1} = mu * x_n * (1 - x_n)
    返回相邻差值（类似于间隙）
    """
    seq = [x0]
    for _ in range(1, num_points):
        seq.append(mu * seq[-1] * (1 - seq[-1]))
    arr = np.array(seq)
    gaps = np.diff(arr)  # 长度为 num_points-1
    return gaps

# ==================== 模型定义 ====================
class RiemannEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000000, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.embedding = nn.Embedding(max_len, d_model)
        else:
            # 固定正弦波编码（与素数实验一致）
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

class GapPredictor(nn.Module):
    """通用间隙预测模型（与素数实验架构一致）"""
    def __init__(self, d_model=256, n_layers=6, n_heads=8, dropout=0.1, learnable_embedding=True):
        super().__init__()
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
        return self.output(transformed.squeeze(1))

    def get_attention_weights(self):
        """提取所有注意力层的 in_proj_weight，用于后续 SVD 分析"""
        weights = []
        for layer in self.transformer.layers:
            attn_weights = layer.self_attn.in_proj_weight
            weights.append(attn_weights.detach().cpu())
        return torch.cat(weights, dim=0)

# ==================== 训练函数 ====================
def train_model(model, train_loader, config, device, output_dir, checkpoint_dir):
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    losses = []
    best_loss = float('inf')
    start_epoch = 0

    # 断点续存（独立于版本）
    os.makedirs(checkpoint_dir, exist_ok=True)
    if config.AUTO_RESUME:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            print(f"恢复训练，从 epoch {start_epoch} 开始")

    print(f"\n{'='*60}")
    print(f"开始训练，总 epoch: {config.NUM_EPOCHS}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(x)
                loss = criterion(outputs, y)

            # 梯度累积
            loss = loss / config.ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.ACCUMULATION_STEPS

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % config.PRINT_EVERY == 0:
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')

        # 定期保存检查点（独立目录）
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, ckpt_path)

        # 定期保存普通 checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            torch.save(model.state_dict(), output_dir / f'model_epoch_{epoch+1}.pt')

    # 保存最终模型和注意力权重
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    attn_weights = model.get_attention_weights()
    torch.save(attn_weights, output_dir / 'attention_weights.pt')
    np.save(output_dir / 'losses.npy', np.array(losses))
    print("训练完成。")
    return losses

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not files:
        return None
    epochs = [int(f.split('_')[2].split('.')[0]) for f in files]
    latest = max(epochs)
    return os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest}.pt')

# ==================== 主程序 ====================
def main():
    if len(sys.argv) < 2:
        print("请指定实验版本: python script.py A 或 python script.py C")
        sys.exit(1)
    version = sys.argv[1].upper()
    if version not in ['A', 'C']:
        print("版本必须是 A 或 C")
        sys.exit(1)

    # 根据版本设置相关目录和嵌入模式
    learnable = (version == 'A')
    output_dir = Path(f"chaos_plan_{version}_results")
    checkpoint_dir = f"chaos_checkpoints_{version}"

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    config = Config()
    print(f"=== 混沌映射 Plan {version} ===")
    print(f"可学习嵌入: {learnable}")
    print(f"输出目录: {output_dir}")
    print(f"检查点目录: {checkpoint_dir}")

    # 生成数据
    print("生成 Logistic 混沌映射序列...")
    gaps = generate_logistic_map(config.NUM_POINTS, mu=config.MU, x0=config.X0)
    print(f"序列长度: {len(gaps)}")
    print(f"范围: [{gaps.min():.4f}, {gaps.max():.4f}]")

    # 归一化
    mean = np.mean(gaps)
    std = np.std(gaps)
    gaps_norm = (gaps - mean) / std
    print(f"归一化后均值: {np.mean(gaps_norm):.6f}, 标准差: {np.std(gaps_norm):.6f}")

    # 准备数据集
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = torch.FloatTensor(data).unsqueeze(1)  # (N,1)
            self.indices = torch.arange(len(data))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.indices[idx], self.data[idx]

    dataset = SequenceDataset(gaps_norm)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = GapPredictor(
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        dropout=config.DROPOUT,
        learnable_embedding=learnable
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练（传入版本独立的目录）
    train_model(model, loader, config, device, output_dir, checkpoint_dir)

if __name__ == "__main__":
    main()