"""Inspect layer-0 component captures for matched TP1/TP2 calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


KINDS = ("moe_input", "kt_pre", "kt_post", "v2_pre_reduce", "v2_after")


def load(root: Path, phase: str, rank: int, call: int, kind: str):
    path = root / phase / f"layer0-rank{rank}-call{call:03d}-{kind}.pt"
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid component capture: {path}")
    return value


def metrics(a, b):
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    a, b = a.float(), b.float()
    diff = b - a
    rms = a.square().mean().sqrt()
    return {
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs": float(diff.abs().max()),
        "rmse": float(diff.square().mean().sqrt()),
        "nrmse": float(diff.square().mean().sqrt() / rms.clamp_min(1e-12)),
    }


def tensor_info(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "rms": float(value.float().square().mean().sqrt()),
            "nonzero": int(torch.count_nonzero(value))}


def analyze(root: Path, call: int):
    t1 = {kind: load(root, "tp1_eager", 0, call, kind) for kind in KINDS}
    t2 = {rank: {kind: load(root, "tp2_eager", rank, call, kind)
                 for kind in KINDS} for rank in (0, 1)}
    return {
        "call": call,
        "moe_input_r0": metrics(t1["moe_input"]["hidden"], t2[0]["moe_input"]["hidden"]),
        "moe_input_r1": metrics(t1["moe_input"]["hidden"], t2[1]["moe_input"]["hidden"]),
        "route_ids": {"tp1": t1["moe_input"]["topk_ids"].tolist(),
                      "tp2_r0": t2[0]["moe_input"]["topk_ids"].tolist(),
                      "tp2_r1": t2[1]["moe_input"]["topk_ids"].tolist()},
        "route_weights_r0": metrics(t1["moe_input"]["topk_weights"],
                                    t2[0]["moe_input"]["topk_weights"]),
        "route_weights_r1": metrics(t1["moe_input"]["topk_weights"],
                                    t2[1]["moe_input"]["topk_weights"]),
        "cpu": {"tp1": tensor_info(t1["kt_post"]["cpu"]),
                "tp2_r0": tensor_info(t2[0]["kt_post"]["cpu"]),
                "tp2_r1": tensor_info(t2[1]["kt_post"]["cpu"])},
        "npu": {"tp1": tensor_info(t1["kt_post"]["npu"]),
                "tp2_r0": tensor_info(t2[0]["kt_post"]["npu"]),
                "tp2_r1": tensor_info(t2[1]["kt_post"]["npu"])},
        "npu_sum": metrics(t1["kt_post"]["npu"],
                           t2[0]["kt_post"]["npu"] + t2[1]["kt_post"]["npu"]),
        "cpu_r0": metrics(t1["kt_post"]["cpu"], t2[0]["kt_post"]["cpu"]),
        "kt_sum": metrics(t1["kt_post"]["combined"],
                          t2[0]["kt_post"]["combined"] + t2[1]["kt_post"]["combined"]),
        "v2_reduce_needed": {"tp1": t1["v2_pre_reduce"]["reduce_needed"],
                             "tp2_r0": t2[0]["v2_pre_reduce"]["reduce_needed"],
                             "tp2_r1": t2[1]["v2_pre_reduce"]["reduce_needed"]},
        "v2_pre_sum": metrics(t1["v2_pre_reduce"]["hidden"],
                              t2[0]["v2_pre_reduce"]["hidden"] +
                              t2[1]["v2_pre_reduce"]["hidden"]),
        "v2_post_ranks": metrics(t2[0]["v2_after"]["hidden"],
                                 t2[1]["v2_after"]["hidden"]),
        "v2_post_vs_pre_sum": metrics(
            t2[0]["v2_pre_reduce"]["hidden"] + t2[1]["v2_pre_reduce"]["hidden"],
            t2[0]["v2_after"]["hidden"]),
        "v2_tp1_vs_tp2": metrics(t1["v2_after"]["hidden"],
                                  t2[0]["v2_after"]["hidden"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calls", type=int, nargs="+", required=True)
    args = parser.parse_args()
    root = args.run / "remote" / "components"
    print(json.dumps([analyze(root, call) for call in args.calls], indent=2))


if __name__ == "__main__":
    main()
