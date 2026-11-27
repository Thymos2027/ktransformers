"""Read saved requests and summarize within-request timing; never contact a server."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO = Path(__file__).resolve().parents[2]
SOURCES = (
    ("prefill96", "reports/2026-11-25-taishan-prefill-streaming/runs/"
     "prefill-20261125-224217-first/remote/requests.jsonl", 48, 36),
    ("tutorial32-short", "reports/2026-11-26-taishan-tutorial-benchmark/runs/"
     "tutorial-20261126-111251-swa-audit-fix/remote/requests.jsonl", 16, 12),
    ("tutorial32-long", "reports/2026-11-26-taishan-tutorial-benchmark/runs/"
     "tutorial-20261126-144145-capacity-parser-fix/remote/requests.jsonl", 8, 6),
)


def window_ms(events, start, end):
    return 1000 * (events[end - 1]["seconds"] - events[start - 1]["seconds"]) / (end - start)


def summarize():
    sources, groups = [], []
    total_formal_events = 0
    for label, relative, expected_rows, expected_formal in SOURCES:
        raw = (REPO / relative).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        formal = [row for row in rows if row["kind"] == "formal"]
        if (len(rows), len(formal)) != (expected_rows, expected_formal):
            raise ValueError(f"Unexpected request matrix: {relative}")
        by_group = defaultdict(list)
        seen = set()
        for row in formal:
            identity = (row["phase"], row.get("case", row.get("target")), row["iteration"])
            if identity in seen:
                raise ValueError(f"Duplicate formal request: {relative}: {identity}")
            seen.add(identity)
            count = row["usage"]["completion_tokens"]
            events = row["token_events"]
            if len(row["output_ids"]) != count or len(events) != count:
                raise ValueError(f"Missing IDs/events: {relative}: {identity}")
            if [event["completion_tokens"] for event in events] != list(range(1, count + 1)):
                raise ValueError(f"Aggregated or missing token events: {relative}: {identity}")
            if not row["done"] or row["finish_reason"] != "length":
                raise ValueError(f"Incomplete fixed-length response: {relative}: {identity}")
            if any(b["seconds"] <= a["seconds"] for a, b in zip(events, events[1:])):
                raise ValueError(f"Non-increasing token timestamps: {relative}: {identity}")
            if count not in (64, 1000):
                raise ValueError(f"Unexpected output budget: {relative}: {identity}")
            tpot = window_ms(events, 1, count)
            if abs(tpot - row["tpot_ms"]) > 1e-7:
                raise ValueError(f"Saved TPOT differs from raw events: {relative}: {identity}")
            if abs(events[0]["seconds"] - row["ttft_seconds"]) > 1e-7:
                raise ValueError(f"Saved TTFT differs from raw events: {relative}: {identity}")
            total_formal_events += count
            by_group[(row["phase"], row["usage"]["prompt_tokens"])].append(row)
        for (phase, prompt_tokens), selected in by_group.items():
            if len(selected) != 3 or len({row["usage"]["completion_tokens"] for row in selected}) != 1:
                raise ValueError(f"Unbalanced formal group: {relative}: {phase}/{prompt_tokens}")
            count = selected[0]["usage"]["completion_tokens"]
            group = dict(source=label, phase=phase, prompt_tokens=prompt_tokens,
                         output_tokens=count, formal_requests=len(selected),
                         content_anomalies=sum(row["content_anomaly"] for row in selected),
                         distinct_outputs=len({tuple(row["output_ids"]) for row in selected}))
            for key in ("ttft_seconds", "tpot_ms", "decode_tokens_per_second", "total_seconds"):
                group[key] = dict(median=median(row[key] for row in selected),
                                  minimum=min(row[key] for row in selected),
                                  maximum=max(row[key] for row in selected))
            windows = [(1, 64)] if count == 64 else [(1, 64), (65, 256), (257, 512), (513, 1000)]
            group["window_tpot_ms"] = {
                f"{start}-{end}": median(window_ms(row["token_events"], start, end) for row in selected)
                for start, end in windows
            }
            if count == 1000:
                group["tail_to_first64_tpot_ratio"] = median(
                    window_ms(row["token_events"], 513, 1000) / window_ms(row["token_events"], 1, 64)
                    for row in selected
                )
            group["ttft_fraction_of_total"] = median(
                row["ttft_seconds"] / row["total_seconds"] for row in selected
            )
            group["zero_ttft_speedup_at_fixed_decode"] = median(
                row["total_seconds"] / (row["total_seconds"] - row["ttft_seconds"])
                for row in selected
            )
            groups.append(group)
        sources.append(dict(label=label, path=relative, sha256=hashlib.sha256(raw).hexdigest(),
                            requests=len(rows), formal_requests=len(formal)))
    return dict(scope="Offline saved-request checks and descriptive timing only; no causal attribution",
                sources=sources, formal_requests=sum(source["formal_requests"] for source in sources),
                formal_token_events=total_formal_events, groups=groups,
                limitation="Different sources/configurations/prompts/output budgets are not paired A/B data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("formal_requests", "formal_token_events")}, ensure_ascii=False))
    for group in result["groups"]:
        print(json.dumps(dict(source=group["source"], phase=group["phase"],
                              input=group["prompt_tokens"], output=group["output_tokens"],
                              ttft=round(group["ttft_seconds"]["median"], 3),
                              tpot_ms=round(group["tpot_ms"]["median"], 3),
                              window_tpot_ms={key: round(value, 3) for key, value in group["window_tpot_ms"].items()},
                              tail_ratio=group.get("tail_to_first64_tpot_ratio"),
                              ttft_fraction=round(group["ttft_fraction_of_total"], 4),
                              zero_ttft_bound=round(group["zero_ttft_speedup_at_fixed_decode"], 3)), ensure_ascii=False))


if __name__ == "__main__":
    main()
