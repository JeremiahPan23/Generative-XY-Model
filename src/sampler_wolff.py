"""2D XY 模型 Wolff 聚类采样器

在 L×L 方格上生成 2D XY 模型的平衡构型。采用 Wolff 聚类算法在 BKT 临界点
(beta=1.12, T≈0.89) 附近采样，最终将角度构型转化为 (cos θ, sin θ) 双通道
PyTorch 张量存储，以避免相位截断带来的不连续性。
"""

import os
import numpy as np
import torch
from numba import njit
from tqdm import tqdm


@njit
def wolff_step_xy(theta, beta, J=1.0):
    """对 2D XY 晶格执行一次 Wolff 聚类更新（周期性边界条件）。"""
    L = theta.shape[0]
    ix, iy = np.random.randint(0, L), np.random.randint(0, L)
    alpha = np.random.uniform(0, np.pi)
    proj = np.cos(theta - alpha)

    in_cluster = np.zeros((L, L), dtype=np.bool_)
    in_cluster[ix, iy] = True

    queue_x = np.zeros(L * L, dtype=np.int32)
    queue_y = np.zeros(L * L, dtype=np.int32)
    queue_x[0], queue_y[0] = ix, iy
    head, tail = 0, 1

    while head < tail:
        cx, cy = queue_x[head], queue_y[head]
        head += 1

        neighbors = [
            ((cx + 1) % L, cy), ((cx - 1 + L) % L, cy),
            (cx, (cy + 1) % L), (cx, (cy - 1 + L) % L)
        ]

        for nx, ny in neighbors:
            if not in_cluster[nx, ny]:
                coupling = beta * J * proj[cx, cy] * proj[nx, ny]
                if coupling > 0:
                    if np.random.rand() < 1.0 - np.exp(-2.0 * coupling):
                        in_cluster[nx, ny] = True
                        queue_x[tail], queue_y[tail] = nx, ny
                        tail += 1

    for i in range(L):
        for j in range(L):
            if in_cluster[i, j]:
                # 标准 O(N) Wolff 反射: 反转沿随机轴 r 的分量
                # (关于垂直于 r 的超平面的 Householder 反射)。
                # 键概率基于 cos(theta-alpha) 投影时，角度映射为
                # theta -> 2*alpha + pi - theta；注意不可遗漏 +pi，
                # 否则投影在反射下不变、细致平衡被破坏，平稳分布退化为均匀分布。
                theta[i, j] = (2 * alpha + np.pi - theta[i, j]) % (2 * np.pi)

    return theta


def main():
    # --- 物理与采样参数 ---
    L = 32                  # 晶格尺寸
    beta = 1.12             # 逆温度（BKT 临界相变点，T ≈ 1/beta ≈ 0.89）
    N_samples = 10000       # 目标样本数
    burn_in = 1000          # 热身步数（达到热力学平衡）
    sample_interval = 50    # 相邻样本间的 Wolff 步数（去关联）

    # --- 路径设置 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'xy_L32_T0.89_10k.pt')

    # --- 初始化随机角度晶格 (0, 2π) ---
    theta = np.random.uniform(0, 2 * np.pi, size=(L, L)).astype(np.float64)

    # --- Numba JIT 预编译（首次调用触发编译，避免计入采样时间）---
    print("Numba JIT 编译中 ...")
    _ = wolff_step_xy(theta.copy(), beta)

    # --- 热身（Burn-in）---
    print(f"热身: {burn_in} 步 Wolff 聚类更新 ...")
    for _ in range(burn_in):
        theta = wolff_step_xy(theta, beta)

    # --- 采样循环 ---
    samples = np.zeros((N_samples, L, L), dtype=np.float64)
    print(f"采样: {N_samples} 个构型（间隔 {sample_interval} 步）...")
    for i in tqdm(range(N_samples)):
        for _ in range(sample_interval):
            theta = wolff_step_xy(theta, beta)
        samples[i] = theta

    # --- 转化为双通道 (cos θ, sin θ) 张量 ---
    # 形状: (N_samples, 2, L, L)，类型: float32
    tensor = torch.stack([
        torch.from_numpy(np.cos(samples)).float(),
        torch.from_numpy(np.sin(samples)).float(),
    ], dim=1)

    # --- 保存 ---
    torch.save(tensor, save_path)
    print(f"已保存张量，形状 {tuple(tensor.shape)} -> {save_path}")


if __name__ == '__main__':
    main()
