"""Conv-ViT：SCA/SAF 的 3×3 邻域残差适配器（零初始化，默认关闭）。

设计目标（对应行动指南 1. 节）：
  * 原版行为**逐位不变**：末层 1×1 卷积零初始化 ⇒ 初始残差恒为 0；
  * 起点即有真实容量：深度卷积 3×3 提供 patch 网格上的邻域聚合，1×1 提供通道混合；
  * 两条公式（论文里各一行）：

        SAF-C:  P̂ = SAF(P) + Conv1x1( GELU( DWConv3x3( LN(P) ) ) )
        SCA-C:  P̃ = P + E_pos + Conv1x1( GELU( DWConv3x3( P ) ) )

参数量：每层 1024×9（depthwise）+ 1024×1024（pointwise）≈ 1.06 M。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborResidual(nn.Module):
    """3×3 邻域残差：DWConv3×3 → GELU → Conv1×1（零初始化）。

    Args:
        dim: 通道数 C。
        kernel_size: 深度卷积核大小，默认 3（对应 3×3 邻域）。
        use_norm: 是否先做 LayerNorm（SAF-C 用 True，SCA-C 用 False）。
        zero_init: 末层是否零初始化（True ⇒ 初始输出恒为 0，行为与原版逐位一致）。
        residual: True 时返回 ``x + delta``，False 时只返回 ``delta``。
    """

    def __init__(self, dim: int, kernel_size: int = 3, use_norm: bool = False,
                 zero_init: bool = True, residual: bool = True):
        super().__init__()
        self.dim = dim
        self.residual = residual
        self.norm = nn.LayerNorm(dim) if use_norm else None
        pad = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim, bias=False)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(dim, dim, 1, bias=True)
        if zero_init:
            nn.init.zeros_(self.pw.weight)
            nn.init.zeros_(self.pw.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, C]。N 为完全平方数时按 2D 网格做 3×3 邻域，否则退化为 1D 序列。"""
        b, n, c = tokens.shape
        h = self.norm(tokens) if self.norm is not None else tokens
        side = int(round(math.sqrt(n)))
        if side * side == n:
            y = h.transpose(1, 2).reshape(b, c, side, side)
            y = self.pw(self.act(self.dw(y)))
            delta = y.reshape(b, c, n).transpose(1, 2)
        else:
            # 兜底：把序列当作 1×N 的"图像"
            y = h.transpose(1, 2).unsqueeze(2)          # [B,C,1,N]
            y = self.pw(self.act(self.dw(y)))
            delta = y.squeeze(2).transpose(1, 2)        # [B,N,C]
        return tokens + delta if self.residual else delta

    def extra_repr(self) -> str:
        zero = bool((self.pw.weight == 0).all())
        return f"dim={self.dim}, k={self.dw.kernel_size[0]}, zero_init={zero}, residual={self.residual}"


class SAFWithConv(nn.Module):
    """原 SAF（单隐层 MLP）+ 3×3 邻域残差。

    子模块命名与官方 SAF 保持一致（``mlp.0.*`` / ``mlp.3.*``），
    因此加载官方 adapter 权重时只需 ``strict=False``（多出来的 ``conv.*`` 保持零初始化）。
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None,
                 output_dim: int | None = None, dropout: float = 0.0,
                 kernel_size: int = 3, zero_init: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.conv = NeighborResidual(output_dim, kernel_size=kernel_size,
                                     use_norm=True, zero_init=zero_init, residual=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, C] 或 [B*N, C]（后者走纯 MLP，便于逐位对照原版）。"""
        if tokens.dim() == 2:
            return self.mlp(tokens)
        return self.mlp(tokens) + self.conv(tokens)

    def extra_repr(self) -> str:
        return f"kernel={self.conv.dw.kernel_size[0]}, zero_init={bool((self.conv.pw.weight == 0).all())}"


def create_saf_conv(input_dim: int, hidden_dim: int | None = None, output_dim: int | None = None,
                    dropout: float = 0.0, kernel_size: int = 3) -> SAFWithConv:
    return SAFWithConv(input_dim, hidden_dim, output_dim, dropout, kernel_size)
