"""Conv-ViT 不变量测试（验收标准第 1 条）。

核心断言：**零初始化时，SCA-C / SAF-C 的前向输出与原版逐位相等**（torch.equal，非 allclose）。
运行: python scripts/tests/test_conv_invariants.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.conv_adapter import NeighborResidual, SAFWithConv, create_saf_conv  # noqa: E402
from utils.feature_transform import FeatureTransformMLP  # noqa: E402

C = 1024
B, N = 2, 1369          # 518/14 = 37 → 37*37
failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        failures.append(name)


def main():
    torch.manual_seed(0)
    x = torch.randn(B, N, C)

    # ---------- 1) NeighborResidual 零初始化 ⇒ delta 恒为 0 ----------
    nr = NeighborResidual(C, use_norm=True, zero_init=True, residual=False).eval()
    with torch.no_grad():
        d = nr(x)
    check("零初始化时 delta 恒为 0", torch.equal(d, torch.zeros_like(d)), f"max|d|={float(d.abs().max()):.3e}")

    # ---------- 2) SAF-C 与原版 SAF 逐位一致（同一权重） ----------
    mlp = FeatureTransformMLP(C, hidden_dim=C, output_dim=C, dropout=0.0).eval()
    safc = SAFWithConv(C, hidden_dim=C, output_dim=C, dropout=0.0).eval()
    safc.mlp.load_state_dict(mlp.mlp.state_dict())   # 共享 MLP 权重
    with torch.no_grad():
        out_orig = mlp(x.reshape(-1, C)).reshape(B, N, C)
        out_new = safc(x)
    check("SAF-C 起点与原版 SAF 逐位相等", torch.equal(out_orig, out_new),
          f"max|Δ|={float((out_orig-out_new).abs().max()):.3e}")

    # ---------- 3) SAF-C 的 2D 拍平路径等同纯 MLP（兼容旧接口） ----------
    with torch.no_grad():
        flat = torch.randn(64, C)
        check("SAF-C 2D 输入等同纯 MLP", torch.equal(safc(flat), safc.mlp(flat)))

    # ---------- 4) SCA-C：零初始化 ⇒ patch 特征不变 ----------
    from utils.spatial_cross_attention import SpatialBottleneckAttention
    torch.manual_seed(1)
    base = SpatialBottleneckAttention(C, num_anchors=4, dropout=0.0, use_conv_pe=False).eval()
    torch.manual_seed(1)
    conv = SpatialBottleneckAttention(C, num_anchors=4, dropout=0.0, use_conv_pe=True).eval()
    conv.load_state_dict(base.state_dict(), strict=False)
    tok = torch.randn(B, C)
    with torch.no_grad():
        o_base = base(tok, x)
        o_conv = conv(tok, x)
    check("SCA-C 起点与原版 SCA 逐位相等", torch.equal(o_base, o_conv),
          f"max|Δ|={float((o_base-o_conv).abs().max()):.3e}")

    # ---------- 5) 参数量增量符合预期（每层约 +1.06M） ----------
    n_base = sum(p.numel() for p in base.parameters())
    n_conv = sum(p.numel() for p in conv.parameters())
    delta = n_conv - n_base
    check("SCA-C 参数量增量 ≈ 1.06M", 1.0e6 < delta < 1.2e6, f"Δ={delta/1e6:.3f}M")
    n_saf = sum(p.numel() for p in safc.parameters()) - sum(p.numel() for p in mlp.parameters())
    check("SAF-C 参数量增量 ≈ 1.06M", 1.0e6 < n_saf < 1.2e6, f"Δ={n_saf/1e6:.3f}M")

    # ---------- 6) 梯度可回传 & 训练后会离开零 ----------
    safc.train()
    x2 = torch.randn(B, N, C, requires_grad=False)
    target = torch.randn(B, N, C)
    opt = torch.optim.AdamW([p for p in safc.conv.parameters()], lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        loss = ((safc(x2) - target) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d2 = safc.conv(x2)
    check("训练后邻域残差已离开零（说明可学）", float(d2.abs().max()) > 1e-6,
          f"max|delta|={float(d2.abs().max()):.3e}")

    # ---------- 7) 非平方 patch 数走 1D 兜底 ----------
    nr2 = NeighborResidual(C, use_norm=False, zero_init=False, residual=False).eval()
    with torch.no_grad():
        o = nr2(torch.randn(1, 1000, C))
    check("非平方 patch 数 1D 兜底", tuple(o.shape) == (1, 1000, C), str(tuple(o.shape)))

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("全部不变量通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
