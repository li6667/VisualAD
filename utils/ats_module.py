"""ATS-ViT: Anisotropic Texture-Suppressed Self-Alignment Function.

在 VisualAD 的 SAF（逐层 patch 特征自对齐 MLP）之上，增加一个频域-通道联合的
各向异性纹理抑制分支（ATSGate）：

    P_hat = normalize( SAF_MLP(P_l) + alpha * (P_hf * w_l) )

其中
  * P_hf  —— 对 P_l 做 2D 正交 FFT，去掉 (0,0) 直流基底（金属表面大面积镜面高光），
             乘以可学习的各向异性频域带阻/高通权重，再 IFFT 回空域得到的高频缺陷边缘特征；
  * w_l   —— 对 P_hf 做全局 avg/max 池化后经瓶颈 MLP 得到的通道级激励权重；
  * alpha —— 可学习残差标量，初始化为 0.05，保证 warm-start 时性能几乎等同于原版 VisualAD。

设计约束（详见 docs/ATS_VIT_DESIGN.md）：
  1. `--use_ats` 关闭时，本模块完全不参与计算，行为与原版 VisualAD 逐位一致；
  2. 加载官方 adapter 权重时用 strict=False，新增参数按 alpha=0.05 的小残差起步；
  3. FFT 全程在 float32 下做，输出转回输入 dtype，兼容 fp16 推理。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _isqrt(n: int) -> int:
    return int(round(math.sqrt(n)))


class AnisotropicTextureSuppressionGate(nn.Module):
    """各向异性频空纹理抑制门控（ATSGate）。

    Args:
        embed_dim: 通道数 C（ViT-L/14 为 1024）。
        reduction: 通道激励瓶颈比例（默认 16）。
        alpha_init: 残差标量初始值（默认 0.05）。
        suppress_dc: 是否滤掉 (0,0) 直流分量（默认 True）。
        learnable_spectrum: 频域权重是否可学习（默认 True）。
        hp_ratio: 初始化时高通权重的强度，0 表示所有权重初始为 1（全通）。
        max_side: 预分配频域 mask 的最大空间边长（518/14=37，留余量到 64）。
    """

    def __init__(
        self,
        embed_dim: int,
        reduction: int = 16,
        alpha_init: float = 0.05,
        suppress_dc: bool = True,
        learnable_spectrum: bool = True,
        hp_ratio: float = 0.5,
        max_side: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.suppress_dc = suppress_dc
        self.learnable_spectrum = learnable_spectrum

        hidden = max(8, embed_dim // reduction)
        # 通道级缺陷边缘激励门控：avg + max 池化拼接 -> 瓶颈 MLP -> sigmoid
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.Sigmoid(),
        )
        # 门控末层（最后一个 Linear，索引 -2 因为最后是 Sigmoid）零初始化
        # => 初始 gate = sigmoid(0) = 0.5，配合小 alpha 保证起点温和
        last_linear = self.gate[-2]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        # 可学习的各向异性频域权重（对 rfft2 的半谱网格定义，实数）
        # 用一张足够大的 mask，forward 时按实际 side 做双线性插值取用
        self.spectrum = nn.Parameter(torch.ones(1, 1, max_side, max_side // 2 + 1))
        if hp_ratio > 0:
            with torch.no_grad():
                r = self._radial_freq(max_side, max_side // 2 + 1)
                # 低频压低、高频保留：0.6~1.0 之间渐变，作为"高通倾向"的初始化
                self.spectrum.copy_((1.0 - hp_ratio) + hp_ratio * r)
        if not learnable_spectrum:
            self.spectrum.requires_grad = False

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @staticmethod
    def _radial_freq(h: int, w: int) -> torch.Tensor:
        """归一化径向频率图（0 在直流，1 在最高频）。"""
        fy = torch.fft.fftfreq(h).abs().unsqueeze(1)  # [h,1]
        fx = torch.fft.rfftfreq((w - 1) * 2).unsqueeze(0)  # [1,w]
        r = torch.sqrt(fy ** 2 + fx ** 2)
        return (r / r.max().clamp_min(1e-6)).clamp(0, 1)

    def _spectrum_for(self, side: int) -> torch.Tensor:
        """把预分配的频域 mask 插值到当前特征分辨率。"""
        target_h, target_w = side, side // 2 + 1
        if self.spectrum.shape[-2:] == (target_h, target_w):
            return self.spectrum
        return F.interpolate(
            self.spectrum.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=torch.float32)

    def forward(self, tokens: torch.Tensor, side: int | None = None):
        """tokens: [B, C, H, W] 或 [B, N, C]（需要给 side 指定 H=W）。

        Returns:
            (P_hf, w) —— P_hf: [B, C, H, W]，w: [B, C, 1, 1]
        """
        if tokens.dim() == 3:
            b, n, c = tokens.shape
            if side is None:
                side = int(round(math.sqrt(n)))
            assert side * side == n, f"patch 数 {n} 不是完全平方数，无法reshape成 2D 网格"
            x = tokens.transpose(1, 2).reshape(b, c, side, side)
        else:
            x = tokens

        orig_dtype = x.dtype
        xf = x.float()
        spec = self._spectrum_for(x.shape[-1])

        freqs = torch.fft.rfft2(xf, norm="ortho")
        if self.suppress_dc:
            freqs = freqs.clone()
            freqs[..., 0, 0] = 0.0  # 去掉直流基底（大面积镜面高反光）
        freqs = freqs * spec
        x_hf = torch.fft.irfft2(freqs, s=xf.shape[-2:], norm="ortho")

        # 通道级激励：全局 avg + max 池化
        pooled = torch.cat(
            [x_hf.mean(dim=(2, 3)), x_hf.amax(dim=(2, 3))], dim=1
        )
        w = self.gate(pooled).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]

        return x_hf.to(orig_dtype), w.to(orig_dtype)


class ATSFeatureTransform(nn.Module):
    """原 SAF（单隐层 MLP）+ ATS 高频残差分支。

    输入既可以是被拍平的 ``[B*N, C]``（此时退化为原版 SAF，保证与官方权重 100% 兼容），
    也可以是 ``[B, N, C]`` 的 token 序列。对于 token 序列，本模块会自动识别
    「前 3 个特殊 token（anomaly/normal/class）+ 平方数个 patch」的布局：
    特殊 token 只过 MLP，patch 部分额外走 ATS 频域分支；若 patch 数不是完全平方数，
    则回退到 1D FFT（沿 token 序列方向）以避免破坏空间假设。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.0,
        reduction: int = 16,
        alpha_init: float = 0.05,
        use_spectral: bool = True,
        normalize_output: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_output = normalize_output

        # 与 VisualAD 官方 SAF 完全同构的单隐层 MLP（便于 strict=False 继承权重）
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.ats = AnisotropicTextureSuppressionGate(
            embed_dim=input_dim,
            reduction=reduction,
            alpha_init=alpha_init,
            learnable_spectrum=use_spectral,
        )
        self.use_spectral = use_spectral

    # ---------- 布局解析 ----------
    @staticmethod
    def _split_layout(n: int):
        """返回 (特殊token数, patch 数, side)；side 为 None 表示需要 1D 兜底。"""
        side = _isqrt(n)
        if side * side == n:
            return 0, n, side
        for prefix in (3, 1):  # 3 = anomaly/normal/class；1 = 仅 class
            m = n - prefix
            s = _isqrt(m)
            if s * s == m:
                return prefix, m, s
        return 0, n, None

    # ---------- 频域分支 ----------
    def _hf_2d(self, patch: torch.Tensor):
        b, m, c = patch.shape
        side = _isqrt(m)
        x = patch.transpose(1, 2).reshape(b, c, side, side)
        if not self.use_spectral:
            # 消融 (e)：只保留通道门控，不做 FFT 抑制
            xf = x.float()
            pooled = torch.cat([xf.mean(dim=(2, 3)), xf.amax(dim=(2, 3))], dim=1)
            w = self.ats.gate(pooled).unsqueeze(-1).unsqueeze(-1).to(patch.dtype)
            return x, w
        return self.ats(patch, side=side)

    def _hf_1d(self, patch: torch.Tensor):
        """1D 兜底：沿 token 序列做 rfft，去掉直流后重建高频成分。"""
        b, m, c = patch.shape
        xf = patch.float()
        freqs = torch.fft.rfft(xf, dim=1, norm="ortho")
        if self.ats.suppress_dc:
            freqs = freqs.clone()
            freqs[:, 0, :] = 0.0
        spec = self.ats.spectrum.mean(dim=(2, 3))  # [1, W']
        spec = F.interpolate(spec.view(1, 1, -1), size=freqs.shape[1], mode="linear",
                             align_corners=False).view(1, -1, 1)
        x_hf = torch.fft.irfft(freqs * spec, n=m, dim=1, norm="ortho")
        pooled = torch.cat([x_hf.mean(dim=1), x_hf.amax(dim=1)], dim=1)
        w = self.ats.gate(pooled).unsqueeze(1).to(patch.dtype)
        return x_hf.to(patch.dtype), w

    # ---------- 前向 ----------
    def forward(self, tokens: torch.Tensor):
        if tokens.dim() == 2:
            # 拍平路径：与原版 SAF 完全一致
            return self.mlp(tokens)

        b, n, c = tokens.shape
        x_saf = self.mlp(tokens.reshape(-1, c)).reshape(b, n, c)

        prefix, m, side = self._split_layout(n)
        if m == 0:
            return F.normalize(x_saf, dim=-1) if self.normalize_output else x_saf

        head, patch = tokens[:, :prefix, :], tokens[:, prefix:, :]
        head_saf = x_saf[:, :prefix, :]
        patch_saf = x_saf[:, prefix:, :]

        if side is None:
            x_hf, w = self._hf_1d(patch)
            hf = x_hf * w
        else:
            if self.use_spectral:
                x_hf, w = self.ats(patch, side=side)
                hf = (x_hf * w).flatten(2).transpose(1, 2)
            else:
                x_hf, w = self._hf_2d(patch)
                hf = (x_hf * w).flatten(2).transpose(1, 2)

        if self.normalize_output:
            fused = F.normalize(patch_saf, dim=-1) + self.ats.alpha * F.normalize(hf, dim=-1)
            fused = F.normalize(fused, dim=-1)
            out = torch.cat([F.normalize(head_saf, dim=-1), fused], dim=1) if prefix else fused
        else:
            fused = patch_saf + self.ats.alpha * hf
            out = torch.cat([head_saf, fused], dim=1) if prefix else fused
        return out

    def extra_repr(self):
        return (
            f"input_dim={self.input_dim}, alpha={float(self.ats.alpha):.4f}, "
            f"spectral={self.use_spectral}"
        )


def create_ats_transform(
    input_dim: int,
    hidden_dim: int | None = None,
    output_dim: int | None = None,
    dropout: float = 0.0,
    reduction: int = 16,
    alpha_init: float = 0.05,
    use_spectral: bool = True,
) -> ATSFeatureTransform:
    return ATSFeatureTransform(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        reduction=reduction,
        alpha_init=alpha_init,
        use_spectral=use_spectral,
    )
