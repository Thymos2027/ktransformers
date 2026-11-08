"""Measure W8A8 expert host-to-NPU copy on the idle second Ascend device.

This is a copy microbenchmark, not an end-to-end cache-switch benchmark.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # registers torch.npu
from safetensors import safe_open


MODEL = Path("/workspace/models/DeepSeek-V4-Flash-W8A8")
OUT = Path("/workspace/expert-cache-study-20261107/transfer.json")
KEYS = [f"layers.0.ffn.experts.0.w{i}.{suffix}" for i in (1, 2, 3)
        for suffix in ("weight", "weight_scale")]
DEVICE = "npu:1"


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def measure(sources, destinations, rounds=30):
    samples = []
    for idx in range(rounds + 5):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        for src, dst in zip(sources, destinations):
            dst.copy_(src, non_blocking=True)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if idx >= 5:
            samples.append(elapsed_ms)
    return {"rounds": rounds, "median_ms": statistics.median(samples),
            "p95_ms": percentile(samples, 0.95), "min_ms": min(samples),
            "max_ms": max(samples)}


def main():
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    missing = [key for key in KEYS if key not in weight_map]
    if missing:
        raise KeyError(f"Required W8A8 expert tensors missing: {missing}")
    by_shard = {}
    for key in KEYS:
        by_shard.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for shard, keys in by_shard.items():
        with safe_open(MODEL / shard, framework="pt", device="cpu") as source:
            for key in keys:
                tensors[key] = source.get_tensor(key).contiguous()
    weights = [tensors[key] for key in KEYS]
    total_bytes = sum(t.numel() * t.element_size() for t in weights)
    torch.npu.set_device(1)
    destination = [torch.empty(t.shape, dtype=t.dtype, device=DEVICE) for t in weights]
    output = {"device": DEVICE, "keys": KEYS, "total_bytes": total_bytes,
              "total_mib": total_bytes / 2**20,
              "tensor_shapes": {key: list(tensors[key].shape) for key in KEYS},
              "pageable_to_preallocated_npu": measure(weights, destination)}
    try:
        pinned = [t.pin_memory() for t in weights]
        output["pinned_to_preallocated_npu"] = measure(pinned, destination)
    except Exception as error:
        output["pinned_error"] = f"{type(error).__name__}: {error}"
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
