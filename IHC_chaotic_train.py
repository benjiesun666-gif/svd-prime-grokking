"""
混沌映射（Logistic映射）间隙预测实验 - IHC版本
使用 IHC (Identity Hyper-Connection) 架构，可学习嵌入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import math
import os
from pathlib import Path

# ==================== 配置 ====================
class Config:
    # 数据
    NUM_POINTS = 1_000_000          # 生成序列点数（间隙数为 NUM_POINTS-1）
    MU = 3.8                         # Logistic映射参数（混沌区）
    X0 = 0.5                          # 初始值

    # 模型架构（IHC）
    D_MODEL = 256
    N_LAYERS = 6
    N_HEADS = 8
    DROPOUT = 0.1
    NUM_FLOWS = 4                    # IHC 并行流数（必须整除 D_MODEL）

    # 训练配置
    NUM_EPOCHS = 300
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    LR_SCHEDULE = 'cosine'

    # 梯度累积
    USE_GRADIENT_ACCUMULATION = True
    PHYSICAL_BATCH_SIZE = 128
    ACCUMULATION_STEPS = 16

    # 监控与保存
    PRINT_EVERY = 1
    SAVE_EVERY = 50
    CHECKPOINT_EVERY = 50
    AUTO_RESUME = True


# ==================== 数据生成 ====================
def generate_logistic_map(num_points, mu=3.8, x0=0.5):
    seq = [x0]
    for _ in range(1, num_points):
        seq.append(mu * seq[-1] * (1 - seq[-1]))
    arr = np.array(seq)
    gaps = np.diff(arr)  # 长度为 num_points-1
    return gaps


# ==================== 基础模块 ====================
class RiemannEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000000, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.embedding = nn.Embedding(max_len, d_model)
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


# ==================== IHC 模型定义 ====================
class IHCTransformerLayer(nn.Module):
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
        self.phi = nn.Sequential(
            nn.Linear(num_flows * dim, num_flows * 2),
            nn.GELU(),
            nn.Linear(num_flows * 2, num_flows * 2)
        )

    def forward(self, x_flows):
        batch, num_flows, dim = x_flows.shape
        phi_in = x_flows.reshape(batch, -1)
        phi_out = self.phi(phi_in)
        h_pre = F.softmax(phi_out[:, :num_flows], dim=-1)
        h_post = F.softmax(phi_out[:, num_flows:], dim=-1)
        agg = torch.einsum('bf,bfd->bd', h_pre, x_flows)
        agg2 = self.norm1(agg + self._sa_block(agg))
        agg3 = self.norm2(agg2 + self._ffn_block(agg2))
        new_flows = x_flows + h_post.unsqueeze(-1) * agg3.unsqueeze(1)
        return new_flows

    def _sa_block(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.self_attn(x, x, x)
        return attn_out.squeeze(1)

    def _ffn_block(self, x):
        return self.ffn(x)


class IHCPrimeGapPredictor(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dropout, learnable_embedding, num_flows):
        super().__init__()
        assert d_model % num_flows == 0
        self.d_model = d_model
        self.num_flows = num_flows
        self.flow_dim = d_model // num_flows

        self.riemann_embedding = RiemannEmbedding(d_model, learnable=learnable_embedding)
        self.layers = nn.ModuleList([
            IHCTransformerLayer(
                num_flows=num_flows,
                dim=self.flow_dim,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
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
        emb = self.riemann_embedding(x)                         # [batch, d_model]
        x_flows = emb.view(-1, self.num_flows, self.flow_dim)   # [batch, num_flows, flow_dim]
        for layer in self.layers:
            x_flows = layer(x_flows)
        pooled = x_flows.mean(dim=1)                             # [batch, flow_dim]
        return self.output_head(pooled)

    def get_attention_weights(self):
        """提取所有注意力层的 in_proj_weight，用于后续 SVD 分析"""
        weights = []
        for layer in self.layers:
            attn = layer.self_attn
            if hasattr(attn, 'in_proj_weight'):
                weights.append(attn.in_proj_weight.detach().cpu())
        if weights:
            return torch.cat(weights, dim=0)
        else:
            return torch.tensor([])


# ==================== 训练函数 ====================
def train_model(model, train_loader, config, device, output_dir, checkpoint_dir):
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    losses = []
    best_loss = float('inf')
    start_epoch = 0

    os.makedirs(checkpoint_dir, exist_ok=True)
    if config.AUTO_RESUME:
        latest_ckpt = find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'IHC_prime.pt')

        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, ckpt_path)

        if (epoch + 1) % config.SAVE_EVERY == 0:
            torch.save(model.state_dict(), output_dir / f'model_epoch_{epoch+1}.pt')

    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    attn_weights = model.get_attention_weights()
    if attn_weights.numel() > 0:
        torch.save(attn_weights, output_dir / 'attention_weights.pt')
    np.save(output_dir / 'losses.npy', np.array(losses))
    print("训练完成。")
    return losses


def find_latest_checkpoint(checkpoint_dir):
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
    # 固定使用 IHC 架构，可学习嵌入
    output_dir = Path("chaos_ihc_results")
    checkpoint_dir = "chaos_ihc_checkpoints"
    output_dir.mkdir(exist_ok=True)

    config = Config()
    print("=== 混沌映射实验 - IHC 版本（可学习嵌入）===")
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

    # 数据集
    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = torch.FloatTensor(data).unsqueeze(1)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建 IHC 模型
    model = IHCPrimeGapPredictor(
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        dropout=config.DROPOUT,
        learnable_embedding=True,   # 可学习嵌入
        num_flows=config.NUM_FLOWS
    ).to(device)
    print("使用 IHC 架构")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    train_model(model, loader, config, device, output_dir, checkpoint_dir)


if __name__ == "__main__":
    main()