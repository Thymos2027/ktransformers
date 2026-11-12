"""Summarize full-vocabulary pilot logits without declaring an acceptance pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PHASES = ("tp1_eager", "tp2_eager", "tp1_graph", "tp2_graph")
PAIRS = (("tp1_eager", "tp2_eager"), ("tp1_graph", "tp2_graph"),
         ("tp1_eager", "tp1_graph"), ("tp2_eager", "tp2_graph"))


def load(root: Path, phase: str, case: str, repeat: int, step: int, rank: int = 0):
    path = root / "probe" / f"{phase}-{case}-r{repeat}-rank{rank}-step{step:03d}.pt"
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"Unexpected logits tensor: {path}")
    return value.float().numpy().astype(np.float64), str(value.dtype)


def sha(value) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def metrics(a, b):
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"Vocab shape mismatch: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"finite": False, "pass": False}
    delta = b - a
    az = a - np.max(a)
    bz = b - np.max(b)
    pa = np.exp(az) / np.sum(np.exp(az))
    pb = np.exp(bz) / np.sum(np.exp(bz))
    order = np.argsort(a)[-2:]
    gap = float(a[order[-1]] - a[order[-2]])
    return {
        "finite": True,
        "max_abs": float(np.max(np.abs(delta))),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "normalized_rmse": float(np.sqrt(np.mean(delta * delta)) /
                                 max(float(np.sqrt(np.mean(a * a))), 1e-12)),
        "total_variation": float(0.5 * np.sum(np.abs(pa - pb))),
        "reference_top1": int(np.argmax(a)),
        "candidate_top1": int(np.argmax(b)),
        "reference_top1_top2_gap": gap,
        "top1_changed": bool(np.argmax(a) != np.argmax(b)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        for step in range(len(case["forced_ids"]) + 1):
            for first, second in PAIRS:
                for repeat in range(2):
                    a, dtype_a = load(root, first, case["name"], repeat, step)
                    b, dtype_b = load(root, second, case["name"], repeat, step)
                    rows.append({"case": case["name"], "step": step,
                                 "first": first, "second": second, "repeat": repeat,
                                 "first_dtype": dtype_a, "second_dtype": dtype_b,
                                 "first_sha256_fp64": sha(a),
                                 "second_sha256_fp64": sha(b),
                                 **metrics(a, b)})
    output = {"scope": "diagnostic pilot only; no numeric threshold or task quality verdict",
              "cases": len(cases), "comparisons": len(rows), "rows": rows}
    target = root / "comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"{target}: {len(rows)} comparisons")


if __name__ == "__main__":
    main()
