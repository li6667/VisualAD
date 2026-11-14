"""ATS-ViT: 效率基准（可训练参数量 / 单图推理延迟 / FPS）。

用法: python scripts/bench_efficiency.py [--device cuda:0] [--iters 30]
输出: ../results_ats_vit/bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = Path("D:/ls/Anomaly Detection/results_ats_vit")

from scripts.generate_paper_figures import Pipeline  # noqa: E402


def count_trainable(pipe) -> float:
    total = 0
    total += pipe.model.visual.anomaly_token.numel()
    total += pipe.model.visual.normal_token.numel()
    total += pipe.model.visual.ln_post.weight.numel() + pipe.model.visual.ln_post.bias.numel()
    for t in pipe.layer_transforms.values():
        total += sum(p.numel() for p in t.parameters() if p.requires_grad)
    if pipe.cross_attn is not None:
        total += sum(p.numel() for p in pipe.cross_attn.parameters() if p.requires_grad)
    return total / 1e6


def bench(pipe, iters=30, warmup=5):
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.default_rng(0).random((518, 518, 3)) * 255).astype("uint8"))
    for _ in range(warmup):
        pipe.forward(img)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        pipe.forward(img)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    ckpts = {
        "VisualAD (baseline)": str(REPO / "weight/train_on_visa/CLIP.pth"),
        "ATS-ViT (Ours)": str(OUT / "ckpt/finetune_ats.pth"),
    }
    res = {}
    for name, ck in ckpts.items():
        if not Path(ck).exists():
            print(f"skip {name}: {ck} 不存在")
            continue
        pipe = Pipeline(ck, args.device)
        params = count_trainable(pipe)
        lat = bench(pipe, args.iters)
        res[name] = {
            "trainable_params_m": round(params, 3),
            "latency_ms": round(lat * 1000, 2),
            "fps": round(1.0 / lat, 2),
            "ckpt": ck,
        }
        print(f"{name:22s} trainable={params:6.2f}M  latency={lat*1000:7.2f} ms  fps={1/lat:5.2f}")
        del pipe
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写出:", OUT / "bench.json")


if __name__ == "__main__":
    main()
