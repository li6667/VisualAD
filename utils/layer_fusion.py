"""Conv-free 两件小事（不改架构）：

1) ALF — Adaptive Layer Fusion（自适应层融合权重）
   原版把 {6,12,18,24} 四层异常图**直接相加**（权重恒为 1）；ALF 让每层的权重可学习，
   并用 softmax 归一化后再乘以层数，使得 w=0 时权重恰好为 1 ⇒ **与直接求和逐位等价**。

        H = Σ_l  w_l H_l ,        w_l = L · softmax(u)_l ,   u_l(0)=0  ⇒  w_l ≡ 1

2) CLCR — Cross-Layer Consistency Regularization（跨层一致性正则）
   原版四层各训各的（逐层 Focal+Dice），层间无任何约束；CLCR 追加两两 L1 一致性损失：

        L_cl = mean_{i<j} || σ(H_i) − σ(H_j) ||_1 ,   L = L_cls + L_seg + L_ctr + λ(t)·L_cl

   λ(t) 前 200 步线性 warmup ⇒ 训练早期与原版完全一致，不会破坏已收敛的适配器。
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerFusion(nn.Module):
    """可学习的层融合权重；初值等价于"直接相加"。"""

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.logits = nn.Parameter(torch.zeros(num_layers))

    @property
    def weights(self) -> torch.Tensor:
        """归一化后的权重，和为 num_layers（因此初值全为 1）。"""
        return self.num_layers * torch.softmax(self.logits, dim=0)

    def forward(self, maps):
        """maps: List[Tensor[B,H,W]] → 加权求和 [B,H,W]。"""
        w = self.weights.to(dtype=maps[0].dtype, device=maps[0].device)
        out = maps[0] * w[0]
        for i in range(1, len(maps)):
            out = out + maps[i] * w[i]
        return out

    def extra_repr(self) -> str:
        w = self.weights.detach().cpu().tolist()
        return "num_layers=%d, weights=%s" % (self.num_layers, [round(x, 4) for x in w])


def cross_layer_consistency(maps, downsample: int = 4):
    """两两 L1 一致性（在 1/downsample 分辨率上算，省显存）。

    Args:
        maps: List[Tensor[B, H, W]] 逐层 anomaly map（未过 sigmoid 的原始 logits）
        downsample: 下采样倍数
    """
    if len(maps) < 2:
        return maps[0].new_zeros(())
    probs = []
    for m in maps:
        p = torch.sigmoid(m).unsqueeze(1)                      # [B,1,H,W]
        if downsample > 1:
            p = F.avg_pool2d(p, downsample)
        probs.append(p)
    losses = [F.l1_loss(probs[i], probs[j]) for i, j in itertools.combinations(range(len(probs)), 2)]
    return sum(losses) / len(losses)
