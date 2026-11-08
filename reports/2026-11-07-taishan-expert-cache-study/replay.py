"""Replay SGLang per-pass expert counts without changing model execution.

Input is a trusted local .pt dump from SGLang's expert distribution recorder.
Each pass is evaluated against the cache state *before* that pass; swaps only
become visible to the next pass. This deliberately avoids crediting a policy
with instantaneous, free expert loads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_counts(path: Path) -> list[list[list[int]]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    records = data["records"]
    mapping = data["last_physical_to_logical_map"].long().cpu()
    if mapping.ndim != 2:
        raise ValueError(f"Unexpected expert map shape: {tuple(mapping.shape)}")
    layers, physical_experts = mapping.shape
    logical_experts = int(mapping.max().item()) + 1
    result = []
    for record in records:
        physical = record["global_physical_count"].long().cpu()
        if tuple(physical.shape) != (layers, physical_experts):
            raise ValueError(f"Unexpected count shape: {tuple(physical.shape)}")
        logical = torch.zeros((layers, logical_experts), dtype=torch.long)
        logical.scatter_add_(1, mapping, physical)
        if int(logical.sum()) > 0:
            result.append(logical.tolist())
    if not result:
        raise ValueError("The recorder dump has no nonempty passes")
    return result


def evaluate(counts: list[list[list[int]]], slots: int, train_fraction: float, swaps_per_layer: int):
    n_passes, n_layers, n_experts = len(counts), len(counts[0]), len(counts[0][0])
    if not 0 < slots < n_experts:
        raise ValueError(f"slots must be between 1 and {n_experts - 1}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split = max(1, min(n_passes - 1, round(n_passes * train_fraction)))
    if split == n_passes:
        raise ValueError("At least two nonempty passes are required")

    frequency = [[0] * n_experts for _ in range(n_layers)]
    for step in counts[:split]:
        for layer, row in enumerate(step):
            for expert, hits in enumerate(row):
                frequency[layer][expert] += hits
    baseline = [set(range(slots)) for _ in range(n_layers)]
    hottest = [
        set(sorted(range(n_experts), key=lambda e: (-frequency[layer][e], e))[:slots])
        for layer in range(n_layers)
    ]

    def score_fixed(cache):
        total = hits = decode_total = decode_hits = decode_steps = 0
        for step in counts[split:]:
            step_total = sum(sum(row) for row in step)
            step_hits = sum(sum(row[e] for e in cache[layer]) for layer, row in enumerate(step))
            total += step_total
            hits += step_hits
            if step_total <= n_layers * 16:  # single-token decode; top-k is at most 16
                decode_steps += 1
                decode_total += step_total
                decode_hits += step_hits
        return dict(hits=hits, accesses=total, hit_rate=hits / total,
                    decode_hits=decode_hits, decode_accesses=decode_total,
                    decode_steps=decode_steps, decode_swaps=0,
                    decode_hit_rate=decode_hits / decode_total if decode_total else None,
                    swaps=0)

    def score_dynamic(kind):
        cache = [set(row) for row in baseline]
        last_seen = [[-1] * n_experts for _ in range(n_layers)]
        heat = [[0.0] * n_experts for _ in range(n_layers)]
        total = hits = decode_total = decode_hits = decode_steps = swaps = decode_swaps = 0
        for offset, step in enumerate(counts[split:]):
            step_total = sum(sum(row) for row in step)
            step_hits = sum(sum(row[e] for e in cache[layer]) for layer, row in enumerate(step))
            total += step_total
            hits += step_hits
            is_decode = step_total <= n_layers * 16
            if is_decode:
                decode_steps += 1
                decode_total += step_total
                decode_hits += step_hits
            for layer, row in enumerate(step):
                for expert, count in enumerate(row):
                    if count:
                        last_seen[layer][expert] = offset
                    heat[layer][expert] = 0.9 * heat[layer][expert] + count
                if kind == "lru":
                    candidates = sorted((e for e in range(n_experts) if row[e] and e not in cache[layer]),
                                        key=lambda e: (-row[e], e))
                    victims = sorted(cache[layer], key=lambda e: (last_seen[layer][e], e))
                else:
                    candidates = sorted((e for e in range(n_experts) if e not in cache[layer]),
                                        key=lambda e: (-heat[layer][e], e))
                    victims = sorted(cache[layer], key=lambda e: (heat[layer][e], e))
                for incoming, outgoing in zip(candidates[:swaps_per_layer], victims[:swaps_per_layer]):
                    if kind == "ewma" and heat[layer][incoming] <= 1.2 * heat[layer][outgoing]:
                        break
                    cache[layer].remove(outgoing)
                    cache[layer].add(incoming)
                    swaps += 1
                    if is_decode:
                        decode_swaps += 1
        return dict(hits=hits, accesses=total, hit_rate=hits / total,
                    decode_hits=decode_hits, decode_accesses=decode_total,
                    decode_steps=decode_steps, decode_swaps=decode_swaps,
                    decode_hit_rate=decode_hits / decode_total if decode_total else None,
                    swaps=swaps)

    return dict(passes=n_passes, train_passes=split, test_passes=n_passes - split,
                layers=n_layers, experts_per_layer=n_experts, slots_per_layer=slots,
                swaps_per_layer_per_pass=swaps_per_layer,
                policies={"current_first_n": score_fixed(baseline),
                          "static_train_frequency": score_fixed(hottest),
                          "dynamic_lru": score_dynamic("lru"),
                          "dynamic_ewma": score_dynamic("ewma")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--swaps-per-layer", type=int, default=2)
    args = parser.parse_args()
    result = evaluate(load_counts(args.trace), args.slots, args.train_fraction,
                      args.swaps_per_layer)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
