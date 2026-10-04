"""Flow Matching 训练脚本 — PhysicsInformedDiT on 2D XY Model.

使用 Optimal Transport Formulation 的 Flow Matching 框架训练
PhysicsInformedDiT 模型，学习从高斯噪声 x_0 到物理构型 x_1 的速度场 v_θ。

核心公式:
    - 插值路径:     x_t = (1 - t) * x_0 + t * x_1
    - 目标速度场:   v = x_1 - x_0  (OT 形式下的最优速度场)
    - 训练目标:     L = MSE(v_θ(x_t, t), v)

训练历史会在每个 epoch 结束后自动写入:
    - logs/training_log.json  (结构化历史，便于程序读取)
    - logs/training_log.csv   (表格历史，便于 Excel/人类查看)

运行方式:
    python src/train_flow.py
"""

import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model_dit import PhysicsInformedDiT


# ---------------------------------------------------------------------------
#  超参数配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = str(PROJECT_ROOT / 'data' / 'xy_L32_T0.89_10k.pt')
CKPT_DIR = str(PROJECT_ROOT / 'checkpoints')
CKPT_PATH = os.path.join(CKPT_DIR, 'best_dit_flow.pth')

# 训练日志路径（保存完整历史，每个 epoch 后重写）
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_JSON = str(LOG_DIR / 'training_log.json')
LOG_CSV = str(LOG_DIR / 'training_log.csv')

BATCH_SIZE = 128        # 批大小
NUM_EPOCHS = 100        # 训练轮数
LR = 1e-4               # 学习率
VAL_SIZE = 1000         # 验证集大小 (最后 1000 个样本)
NUM_WORKERS = min(4, os.cpu_count() or 1)
# Native Windows PyTorch builds commonly lack a working Triton backend.
# Keep eager mode as the reliable default; opt in on Linux/WSL only.
ENABLE_COMPILE = False


def get_device():
    """自动检测可用设备，CUDA 优先。"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def state_dict_for_save(model):
    """Return an unprefixed state dict whether or not the model is compiled."""
    return model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()


def atomic_write_text(path, text):
    """Write ``text`` atomically so a partially-written log is never visible."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp_path, path)


def save_training_history(history):
    """Persist the full training history to JSON and CSV.

    Both files are rewritten after every epoch, so consumers always observe
    the complete curve rather than an incrementally-appended file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(LOG_JSON, json.dumps(history, indent=2))
    except OSError as exc:
        print(f'[warning] Failed to write {LOG_JSON}: {exc}')

    try:
        tmp_path = LOG_CSV + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['epoch', 'train_loss', 'val_loss']
            )
            writer.writeheader()
            writer.writerows(history)
        os.replace(tmp_path, LOG_CSV)
    except OSError as exc:
        print(f'[warning] Failed to write {LOG_CSV}: {exc}')


def load_data(data_path, val_size, batch_size, device):
    """加载 XY 模型数据并构建 DataLoader。

    参数:
        data_path:  .pt 文件路径
        val_size:   验证集样本数 (取最后 val_size 个)
        batch_size: 批大小
        device:     目标设备
    返回:
        train_loader, val_loader
    """
    tensor = torch.load(data_path, weights_only=False)
    print(f'数据加载完成: {tensor.shape} ({tensor.dtype})')

    n_total = tensor.shape[0]
    n_train = n_total - val_size

    # 前 n_train 个为训练集，后 val_size 个为验证集
    train_data = tensor[:n_train]
    val_data = tensor[n_train:]
    print(f'训练集: {n_train} 样本 | 验证集: {val_data.shape[0]} 样本')

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, device):
    """执行一个 Epoch 的训练。

    Flow Matching (Optimal Transport) 训练步骤:
        1. x_1 = batch[0]           — 真实物理构型
        2. x_0 ~ N(0, I)             — 高斯白噪声
        3. t ~ U(0, 1)              — 均匀采样时间
        4. x_t = (1-t)*x_0 + t*x_1  — 线性插值
        5. target_v = x_1 - x_0      — 最优传输速度场
        6. pred_v = model(x_t, t)    — 模型预测
        7. loss = MSE(pred_v, target_v)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch in pbar:
        x1 = batch[0].to(device, non_blocking=True)     # (B, 2, 32, 32) 真实构型
        B = x1.shape[0]

        # 1. 生成高斯白噪声 x_0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 2. 采样时间 t ~ U(0, 1)
        t = torch.rand(B, device=device)                # (B,)

        # 3. reshape 为 (B, 1, 1, 1) 以支持广播
        t_b = t.view(B, 1, 1, 1)

        # 4. 构造插值状态 x_t
        x_t = (1.0 - t_b) * x0 + t_b * x1

        # 5. 目标速度场 (OT 形式)
        target_v = x1 - x0

        # bf16 uses tensor cores on Blackwell; VorticityExtractor explicitly
        # keeps its phase arithmetic in fp32.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            pred_v = model(x_t, t)
            loss = F.mse_loss(pred_v, target_v)

        # bf16 does not require GradScaler.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

    return total_loss / n_batches


@torch.no_grad()
def validate(model, val_loader, device):
    """在验证集上计算平均损失。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        x1 = batch[0].to(device, non_blocking=True)
        B = x1.shape[0]

        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=device)
        t_b = t.view(B, 1, 1, 1)

        x_t = (1.0 - t_b) * x0 + t_b * x1
        target_v = x1 - x0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            pred_v = model(x_t, t)
            loss = F.mse_loss(pred_v, target_v)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    # ---- 设备检测 ----
    device = get_device()
    print(f'设备: {device}')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    # ---- 数据加载 ----
    train_loader, val_loader = load_data(DATA_PATH, VAL_SIZE, BATCH_SIZE, device)

    # ---- 模型与优化器 ----
    model = PhysicsInformedDiT().to(device)
    if ENABLE_COMPILE and device.type == 'cuda' and os.name != 'nt':
        model = torch.compile(model, mode='max-autotune')
        print('torch.compile enabled (max-autotune)')
    elif ENABLE_COMPILE:
        print('torch.compile skipped: native Windows/Triton is unsupported.')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---- 创建 checkpoint 目录 ----
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- 训练循环 ----
    best_val_loss = float('inf')
    history = []
    print(f'\n开始训练: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}')
    print(f'训练历史将写入: {os.path.abspath(LOG_JSON)}')
    print(f'训练历史将写入: {os.path.abspath(LOG_CSV)}')
    print('=' * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # 验证
        val_loss = validate(model, val_loader, device)

        # 记录完整历史（每个 epoch 后重写 json/csv）
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        save_training_history(history)

        # 打印 epoch 结果
        print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f}', end='')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Do not save torch.compile's ``_orig_mod.`` key prefix: inference
            # notebooks instantiate the eager PhysicsInformedDiT class.
            torch.save(state_dict_for_save(model), CKPT_PATH)
            print(f' | * Saved best model')
        else:
            print()

    print('=' * 60)
    print(f'训练完成。最佳验证集 Loss: {best_val_loss:.6f}')
    print(f'模型已保存至: {CKPT_PATH}')
    print(f'训练历史已保存至: {os.path.abspath(LOG_JSON)}')
    print(f'训练历史已保存至: {os.path.abspath(LOG_CSV)}')


if __name__ == '__main__':
    main()
