"""Physics-Informed Diffusion Transformer (DiT) for Flow Matching on the 2D XY Model.

本模块实现了一个物理启发的扩散 Transformer，用于 Flow Matching 框架下的
2D XY 模型构型生成。核心创新点是在标准 DiT 架构中嵌入了可微涡旋提取器
(VorticityExtractor)，使网络能够感知拓扑缺陷 (涡旋/反涡旋) 信息。

核心组件:
    1. VorticityExtractor — 可微涡旋拓扑荷提取层 (保持通道维度)
    2. TimestepEmbedder    — 正弦连续时间编码 + MLP
    3. DiTBlock            — 带自适应层归一化 (AdaLN-Zero) 的 Transformer Block
    4. PhysicsInformedDiT  — 主网络骨架

输入: (B, 2, L, L) 的 (cos θ, sin θ) 张量 + 连续时间 t ∈ [0, 1]
输出: (B, 2, L, L) 的速度场预测 v_θ(x_t, t)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  辅助函数
# ---------------------------------------------------------------------------

def wrap(d):
    """将相位差截断到 [-π, π) 区间。

    利用模运算实现 2π 周期折叠，保证相邻自旋间的角度差
    始终落在主值区间内，消除 2π 跳变带来的虚假涡旋信号。

    参数:
        d: 任意形状的张量，表示原始相位差
    返回:
        与 d 同形状的张量，值域 [-π, π)
    """
    return (d + math.pi) % (2 * math.pi) - math.pi


def modulate(x, shift, scale):
    """自适应层归一化 (AdaLN) 调制操作: x * (1 + scale) + shift。

    在标准 LayerNorm 的基础上，通过时间嵌入预测的 scale 和 shift
    对归一化后的特征进行仿射变换，实现条件生成。

    参数:
        x:     (B, N, D) 特征序列
        shift: (B, D)   偏移参数，广播到 (B, 1, D)
        scale: (B, D)   缩放参数，广播到 (B, 1, D)
    返回:
        (B, N, D) 调制后的特征
    """
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def unpatchify(x, img_size, patch_size, out_channels):
    """将 Patch 序列还原为空间图像。

    把 (B, num_patches, patch_size² × out_channels) 的序列特征
    重新排列为 (B, out_channels, img_size, img_size) 的图像格式。

    参数:
        x:            (B, num_patches, patch_size² × out_channels)
        img_size:     原始晶格尺寸 ``L``，或 ``(height, width)``
        patch_size:   每个 Patch 的边长 (如 2)
        out_channels: 输出通道数 (如 2)
    返回:
        (B, out_channels, img_size, img_size)
    """
    B, N, _ = x.shape
    if isinstance(img_size, int):
        height = width = img_size
    else:
        height, width = img_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError('image dimensions must be divisible by patch_size')

    h, w = height // patch_size, width // patch_size
    if N != h * w:
        raise ValueError(f'Expected {h * w} patches, received {N}')
    p = patch_size
    # 重排: (B, h, w, p, p, C) → (B, C, h, p_h, w, p_w) → (B, C, H, W)
    x = x.reshape(B, h, w, p, p, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    x = x.reshape(B, out_channels, height, width)
    return x


# ---------------------------------------------------------------------------
#  1. 可微涡旋提取器
# ---------------------------------------------------------------------------

class VorticityExtractor(nn.Module):
    """可微涡旋 (Vorticity) 提取器。

    从 (cos θ, sin θ) 双通道输入中提取每个 Plaquette 的拓扑涡旋荷，
    并将其作为第三通道与原始特征拼接，增强网络对拓扑缺陷的感知能力。

    输入:  (B, 2, L, L)  — 通道 0 = cos θ, 通道 1 = sin θ
    输出:  (B, 3, L, L)  — 原始 2 通道 + 涡旋荷 1 通道

    核心算法:
        对每个 2×2 Plaquette，沿回路 (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1)→(i,j)
        计算 4 条边的相位差，截断到 [-π, π) 后求和并除以 2π。
        理论上，和为 ±2π 对应 ±1 涡旋，0 对应无拓扑缺陷。

    数值说明:
        直接由相邻自旋的叉积/点积计算每条键的主值相位差，避免先恢复
        绝对相位再相减。拓扑荷的整数跳变在 ±π branch cut 处不可避免；
        因此该映射在 branch cut 以外可微，但不可能全局平滑可微。
    """

    def forward(self, x):
        """Return ``(B, 3, L, L)``: the input plus plaquette charge.

        The topology calculation deliberately runs in fp32 even when the
        enclosing model is autocast to bf16.  Normalisation makes the bond
        angle invariant to the off-manifold radii encountered along the
        Euclidean flow path and prevents scale from affecting ``atan2``.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            spins = x.float()
            spins = spins / spins.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)

            # PBC neighbours.  A plaquette is traversed
            # s -> down -> down_right -> right -> s.
            down = torch.roll(spins, shifts=-1, dims=2)
            right = torch.roll(spins, shifts=-1, dims=3)
            down_right = torch.roll(down, shifts=-1, dims=3)

            def bond_angle(a, b):
                # arg((a_x + i a_y)^* (b_x + i b_y)) in (-π, π].
                cross = a[:, 0:1] * b[:, 1:2] - a[:, 1:2] * b[:, 0:1]
                dot = (a * b).sum(dim=1, keepdim=True)
                return torch.atan2(cross, dot)

            vort = (
                bond_angle(spins, down)
                + bond_angle(down, down_right)
                + bond_angle(down_right, right)
                + bond_angle(right, spins)
            ) / (2 * math.pi)

        # Patch embedding stays in the caller's autocast dtype.
        return torch.cat([x, vort.to(dtype=x.dtype)], dim=1)


# ---------------------------------------------------------------------------
#  2. 正弦时间编码
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """正弦时间编码器 (Sinusoidal Timestep Embedder)。

    将连续标量时间 t ∈ [0, 1] 映射为高维特征向量，
    使用类似 Transformer Positional Encoding 的正弦/余弦频率分解，
    再通过两层带 SiLU 激活的 MLP 投影到 hidden_size。

    参数:
        hidden_size:             输出特征维度 (匹配 DiT 的 hidden_size)
        frequency_embedding_size: 正弦编码的中间维度 (默认 256)
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # 两层 MLP: freq_dim → hidden → hidden
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        # Moves with model.to(device), but does not alter checkpoint format.
        self.register_buffer('frequencies', frequencies, persistent=False)

    def forward(self, t):
        """
        参数:
            t: (B,) 连续时间步，值域 [0, 1]
        返回:
            (B, hidden_size) 时间特征向量
        """
        # Scale continuous time to the conventional diffusion-time range.
        # Keep this in fp32: small Fourier frequencies would lose resolution
        # if the input time arrived as bf16 under autocast.
        args = (1000.0 * t.float())[:, None] * self.frequencies[None, :]
        # 拼接 cos 和 sin → (B, frequency_embedding_size)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


# ---------------------------------------------------------------------------
#  3. MLP 子模块
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """标准前馈网络 (Feed-Forward Network)，使用 GELU 激活。

    参数:
        in_features:    输入维度
        hidden_features: 隐藏层维度 (通常为 in_features × ratio)
    """

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
#  4. DiT Transformer Block (AdaLN-Zero)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """带自适应层归一化 (AdaLN-Zero) 的 Transformer Block。

    核心思想: 时间嵌入 c 通过线性层预测 6 组调制参数，分别控制
    Attention 和 MLP 两个子层的 LayerNorm 的仿射变换:
        - shift_msa, scale_msa: 调制 Attention 前的 LayerNorm
        - gate_msa:             Attention 输出的残差连接门控
        - shift_mlp, scale_mlp: 调制 MLP 前的 LayerNorm
        - gate_mlp:             MLP 输出的残差连接门控

    AdaLN-Zero 初始化:
        将调制层的权重和偏置初始化为 0，使 Block 在训练初期表现为
        恒等映射 (gate=0 → 无残差贡献)，大幅提升训练稳定性。

    参数:
        hidden_size: 特征维度
        num_heads:   注意力头数
        mlp_ratio:   MLP 隐藏层扩展倍数 (默认 4.0)
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # LayerNorm 不含可学习参数 (仿射变换由 AdaLN 外部调制)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))

        # AdaLN 调制层: SiLU 激活 + Linear，输出 6 × hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # AdaLN-Zero: 权重和偏置初始化为 0
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        参数:
            x: (B, N, hidden_size) Patch 序列特征
            c: (B, hidden_size)     时间嵌入
        返回:
            (B, N, hidden_size) 更新后的特征
        """
        # 从时间嵌入预测 6 组调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # ---- Self-Attention 子层 ----
        # 1. AdaLN 调制: LayerNorm(x) × (1 + scale) + shift
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        # 2. Self-Attention (Q=K=V=x_norm)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, need_weights=False
        )
        # 3. 门控残差连接
        x = x + gate_msa[:, None, :] * attn_out

        # ---- MLP 子层 ----
        # 1. AdaLN 调制
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        # 2. 前馈网络 + 门控残差连接
        x = x + gate_mlp[:, None, :] * self.mlp(x_norm)

        return x


# ---------------------------------------------------------------------------
#  5. 主网络: PhysicsInformedDiT
# ---------------------------------------------------------------------------

class PhysicsInformedDiT(nn.Module):
    """Physics-Informed Diffusion Transformer for Flow Matching.

    在标准 DiT 架构基础上，于 Patch Embedding 前插入 VorticityExtractor，
    使每个 Patch 的特征不仅包含 (cos θ, sin θ)，还包含拓扑涡旋荷信息。

    参数:
        in_channels:  输入通道数 (默认 2: cos θ, sin θ)
        out_channels: 输出通道数 (默认 2: 速度场 v_θ)
        img_size:     参考晶格尺寸 (默认 32)
        patch_size:   Patch 边长。None 时：L=32→2、L=64→4、L=128→8，
                      因而均得到 16×16=256 个 Patch。
        hidden_size:  隐藏维度 (默认 256)
        depth:        Transformer 层数 (默认 6)
        num_heads:    注意力头数 (默认 8)
        mlp_ratio:    MLP 扩展倍数 (默认 4.0)

    前向传播流:
        x(B,2,32,32)
          → VorticityExtractor → (B,3,32,32)       # 涡旋特征增强
          → Conv2d Patch Embed  → (B,256,256)      # 切 Patch + 投影
          → + 2D 插值位置编码     → (B,256,256)
          → 6× DiTBlock(t_emb)  → (B,256,256)      # 条件 Transformer
          → Final AdaLN+Linear  → (B,256,8)         # 恢复到 Patch 粒度
          → Unpatchify          → (B,2,32,32)       # 序列→图像
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        img_size=32,
        patch_size=None,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if patch_size is None:
            # Keep the dense-attention sequence at 16 x 16 tokens for the
            # intended scale-up path.  Other L values use a conservative
            # fallback that targets the same token-grid scale where possible.
            default_patch_sizes = {32: 2, 64: 4, 128: 8}
            patch_size = default_patch_sizes.get(img_size, max(1, img_size // 16))
        if img_size <= 0 or patch_size <= 0 or img_size % patch_size != 0:
            raise ValueError('img_size must be positive and divisible by patch_size')

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.base_grid_size = img_size // patch_size
        self.num_patches = self.base_grid_size ** 2

        # 1. 可微涡旋提取器: (B,2,L,L) → (B,3,L,L)
        self.vorticity_extractor = VorticityExtractor()
        vort_channels = in_channels + 1  # 3

        # 2. Patch Embedding: Conv2d 切 Patch 并投影到 hidden_size
        #    kernel_size=stride=patch_size → 无重叠切分
        self.patch_embed = nn.Conv2d(
            vort_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 3. Learnable 2D position table at the reference patch resolution.
        # It is reshaped and interpolated in forward(), so a model can accept
        # any spatial size divisible by its fixed patch_size.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. 时间编码器: 标量 t → (B, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 5. DiT Blocks: depth 层 AdaLN-Zero Transformer
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )

        # 6. 最终层: AdaLN 调制 + Linear 投影到 patch_size² × out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.final_linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels
        )
        # AdaLN-Zero 初始化: 最终层权重归零
        nn.init.zeros_(self.final_adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def interpolated_pos_embed(self, grid_height, grid_width):
        """Return a ``(1, grid_height*grid_width, D)`` 2D position encoding."""
        reference = self.pos_embed.reshape(
            1, self.base_grid_size, self.base_grid_size, self.hidden_size
        ).permute(0, 3, 1, 2)

        if (grid_height, grid_width) != (self.base_grid_size, self.base_grid_size):
            reference = F.interpolate(
                reference,
                size=(grid_height, grid_width),
                mode='bicubic',
                align_corners=False,
            )

        return reference.flatten(2).transpose(1, 2)

    def forward(self, x, t):
        """
        参数:
            x: (B, 2, 32, 32) 输入构型 (cos θ, sin θ)
            t: (B,)            连续时间步 t ∈ [0, 1]
        返回:
            (B, 2, 32, 32) 速度场预测 v_θ(x_t, t)
        """
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f'Expected x with shape (B, {self.in_channels}, H, W), got {tuple(x.shape)}'
            )
        height, width = x.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError('Input height and width must be divisible by patch_size')

        # 1. 涡旋特征增强: (B,2,H,W) → (B,3,H,W)
        x = self.vorticity_extractor(x)

        # 2. Patch Embedding: (B,3,H,W) → (B,N,hidden_size)
        x = self.patch_embed(x)
        grid_height, grid_width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)          # (B, 256, hidden)

        # 3. Inject a 2D interpolated position encoding.
        x = x + self.interpolated_pos_embed(grid_height, grid_width)

        # 4. 提取时间特征: (B,) → (B, hidden_size)
        c = self.t_embedder(t)

        # 5. 6 层 DiT Block (时间特征 c 调制每层 AdaLN)
        for block in self.blocks:
            x = block(x, c)

        # 6. 最终层: AdaLN 调制 + Linear 投影
        shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.final_linear(x)  # (B, 256, patch² × out_channels)

        # 7. Unpatchify: 序列 → 图像 (B, 2, 32, 32)
        x = unpatchify(
            x, (height, width), self.patch_size, self.out_channels
        )

        return x


# ---------------------------------------------------------------------------
#  独立运行测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = PhysicsInformedDiT()
    x = torch.randn(4, 2, 32, 32)
    t = torch.rand(4)
    out = model(x, t)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Input:    {x.shape}')
    print(f'Output:   {out.shape}')
    print(f'Params:   {n_params:,}')
