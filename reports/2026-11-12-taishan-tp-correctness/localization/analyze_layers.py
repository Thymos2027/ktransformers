"""Compare isolated TP1/TP2 MoE layer traces without imposing a pass threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(root: Path, phase: str, layer: int, rank: int, call: int, kind: str):
    path = root / phase / f"layer{layer}-rank{rank}-call{call:03d}-{kind}.pt"
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Invalid trace: {path}")
    return tensor.float()


def metrics(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Trace shape mismatch: {a.shape} vs {b.shape}")
    d = b - a
    return {
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs": float(d.abs().max()),
        "rmse": float(d.square().mean().sqrt()),
        "reference_rms": float(a.square().mean().sqrt()),
        "normalized_rmse": float(d.square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--call", type=int, required=True)
    args = parser.parse_args()
    root = args.run / "remote" / "layer-trace"
    rows = []
    for layer in range(43):
        tp1 = load(root, "tp1_eager", layer, 0, args.call, "post")
        tp2_r0_pre = load(root, "tp2_eager", layer, 0, args.call, "pre")
        tp2_r1_pre = load(root, "tp2_eager", layer, 1, args.call, "pre")
        tp2_r0_post = load(root, "tp2_eager", layer, 0, args.call, "post")
        tp2_r1_post = load(root, "tp2_eager", layer, 1, args.call, "post")
        rows.append({
            "layer": layer,
            "tp1_vs_tp2_post": metrics(tp1, tp2_r0_post),
            "tp2_rank_post": metrics(tp2_r0_post, tp2_r1_post),
            "tp2_pre_sum_vs_post": metrics(tp2_r0_pre + tp2_r1_pre, tp2_r0_post),
            "tp1_vs_tp2_sum": metrics(tp1, tp2_r0_pre + tp2_r1_pre),
            "tp2_rank0_pre_vs_post": metrics(tp2_r0_pre, tp2_r0_post),
            "tp2_rank1_pre_vs_post": metrics(tp2_r1_pre, tp2_r1_post),
            "tp2_rank1_pre_rms": float(tp2_r1_pre.square().mean().sqrt()),
        })
    print(json.dumps({"call": args.call, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
