"""ALF / CLCR 不变量测试。

核心断言：ALF 在初值（logits=0）时与"直接相加"**逐位相等**，因此不改精度。
运行: python scripts/tests/test_alf_clcr.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.layer_fusion import AdaptiveLayerFusion, cross_layer_consistency  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        failures.append(name)


def main():
    torch.manual_seed(0)
    maps = [torch.randn(2, 64, 64) for _ in range(4)]

    # 1) 初值权重全为 1，融合结果与直接相加逐位相等
    alf = AdaptiveLayerFusion(4)
    check("初值权重全为 1", torch.allclose(alf.weights, torch.ones(4)),
          str([round(float(x), 6) for x in alf.weights]))
    out = alf(maps)
    ref = torch.stack(maps).sum(dim=0)
    check("ALF(初值) 与直接相加逐位相等", torch.equal(out, ref),
          f"max|Δ|={float((out - ref).abs().max()):.3e}")

    # 2) 权重可学习，训练后能偏离 1，且权重和仍为层数
    alf2 = AdaptiveLayerFusion(4)
    opt = torch.optim.AdamW([alf2.logits], lr=1e-1)
    for _ in range(50):
        opt.zero_grad()
        z = alf2([torch.zeros(1, 8, 8), torch.ones(1, 8, 8) * 3,
                  torch.randn(1, 8, 8), torch.randn(1, 8, 8)])
        loss = (z ** 2).mean()
        loss.backward()
        opt.step()
    check("权重可学习（已偏离 1）", (alf2.weights - 1).abs().max() > 1e-3,
          str([round(float(x), 4) for x in alf2.weights]))
    check("权重之和 = 层数", abs(float(alf2.weights.sum()) - 4) < 1e-5)

    # 3) CLCR：完全一致时为 0，差异越大越大
    t = torch.randn(2, 32, 32)
    same = [t, t.clone(), t.clone(), t.clone()]   # 同一个图层的副本
    check("CLCR(相同图层) = 0", float(cross_layer_consistency(same)) < 1e-6)
    diff = [torch.zeros(2, 32, 32), torch.ones(2, 32, 32) * 8,
            torch.zeros(2, 32, 32), torch.ones(2, 32, 32) * 8]
    l1 = cross_layer_consistency(diff)
    check("CLCR(不同图层) > 0", float(l1) > 1e-3, f"{float(l1):.4f}")

    # 4) 单层输入不报错
    check("CLCR 单层返回 0", float(cross_layer_consistency([torch.randn(1, 8, 8)])) == 0.0)

    print()
    if failures:
        print("FAILED:", failures)
        return 1
    print("全部不变量通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
