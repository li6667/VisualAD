"""ATS-ViT 不变量测试：确认 --use_ats 关闭时行为不变、开启时起点接近原版。

运行: python scripts/tests/test_ats_invariants.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.ats_module import AnisotropicTextureSuppressionGate, create_ats_transform  # noqa: E402

C = 1024
B = 2
failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        failures.append(name)


def main():
    # 1) 关闭 ATS 时，2D 路径与纯 MLP 逐位一致
    t = create_ats_transform(input_dim=C, hidden_dim=C, use_spectral=True).eval()
    z = torch.randn(64, C)
    check("2D 拍平路径 == 纯 MLP", torch.allclose(t(z), t.mlp(z)))

    # 2) 3D 路径输出形状与归一化
    x3 = torch.randn(B, 1369, C)
    with torch.no_grad():
        out = t(x3)
    check("3D 输出形状", tuple(out.shape) == (B, 1369, C), str(tuple(out.shape)))
    check("3D 输出已 L2 归一化", torch.allclose(out.norm(dim=-1).mean(), torch.tensor(1.0), atol=1e-4))

    # 3) 初始化：alpha=0.05、门控末层零初始化
    check("alpha 初始化 0.05", abs(float(t.ats.alpha) - 0.05) < 1e-6)
    check("门控末层零初始化", bool((t.ats.gate[-2].weight == 0).all()))

    # 4) 起点偏差：与纯 SAF 的余弦相似度应 >= 0.995
    with torch.no_grad():
        saf = F.normalize(t.mlp(x3.reshape(-1, C)).reshape(B, 1369, C), dim=-1)
        cos = float((out * saf).sum(-1).mean())
    check("起点与纯 SAF 余弦 >= 0.995", cos >= 0.995, f"cos={cos:.6f}")

    # 5) 特殊 token 布局（3 + 平方数 patch）可正确拆分
    x_mix = torch.randn(B, 1372, C)
    with torch.no_grad():
        out_mix = t(x_mix)
    check("3 特殊 token + 1369 patch 布局", tuple(out_mix.shape) == (B, 1372, C))
    check("特殊 token 部分只过 MLP",
          torch.allclose(out_mix[:, :3, :],
                         F.normalize(t.mlp(x_mix[:, :3, :].reshape(-1, C)).reshape(B, 3, C), dim=-1), atol=1e-5))

    # 6) 非平方 patch 数走 1D 回退而不报错
    with torch.no_grad():
        out_1d = t(torch.randn(1, 1000, C))
    check("非平方 patch 数 1D 回退", tuple(out_1d.shape) == (1, 1000, C))

    # 7) 梯度可回传（alpha / spectrum / gate）
    t.train()
    t(x3).sum().backward()
    check("alpha 有梯度", t.ats.alpha.grad is not None and float(t.ats.alpha.grad.abs()) > 0)
    check("spectrum 有梯度", t.ats.spectrum.grad is not None and float(t.ats.spectrum.grad.abs().sum()) > 0)

    # 8) 消融：无频域分支时也能前向
    t2 = create_ats_transform(input_dim=C, hidden_dim=C, use_spectral=False).eval()
    with torch.no_grad():
        o2 = t2(x3)
    check("消融(仅通道门控)前向", tuple(o2.shape) == (B, 1369, C))

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("全部不变量通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
