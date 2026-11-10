"""Bounded client-side baseline for the existing taishan service."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import requests


BASE = "http://192.168.137.11:18080"
MODEL_LIMIT = 8192
TARGETS = (128, 2048, 6144)
SEED = (
    "NUMA placement changes which memory controller serves CPU expert weights. "
    "Measure prefill and decode separately, and keep the model and request fixed. "
)


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_json(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def timed_request(session: requests.Session, path: str, payload: dict) -> dict:
    start = time.monotonic()
    result = session.post(BASE + path, json=payload, timeout=(10, 30))
    result.raise_for_status()
    body = result.json()
    return {"body": body, "seconds": time.monotonic() - start}


def token_count(session: requests.Session, prompt: str) -> int:
    body = timed_request(session, "/tokenize", {
        "messages": [{"role": "user", "content": prompt}]
    })["body"]
    return int(body["count"])


def make_prompt(session: requests.Session, target: int) -> tuple[str, int]:
    low, high = 1, max(2, target)
    while token_count(session, SEED * high) < target:
        high *= 2
    while high - low > 1:
        mid = (low + high) // 2
        if token_count(session, SEED * mid) < target:
            low = mid
        else:
            high = mid
    prompt = SEED * high + "\nSummarize the measurement principle in numbered points."
    count = token_count(session, prompt)
    if count >= MODEL_LIMIT - 256:
        raise ValueError(f"prompt {count} plus output budget exceeds {MODEL_LIMIT}")
    return prompt, count


def stream_chat(session: requests.Session, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started_at = now()
    start = time.monotonic()
    first = last = None
    parts: list[str] = []
    usage = None
    finish_reason = None
    done = False
    with session.post(BASE + "/v1/chat/completions", json=payload,
                      stream=True, timeout=(15, 900)) as response:
        response.raise_for_status()
        for raw in response.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    instant = time.monotonic()
                    if first is None:
                        first = instant
                    last = instant
                    parts.append(content)
    end = time.monotonic()
    completion = (usage or {}).get("completion_tokens")
    decode = None
    if first is not None and last is not None and last > first and completion and completion > 1:
        decode = (completion - 1) / (last - first)
    return {
        "started_at": started_at, "prompt": prompt, "max_tokens": max_tokens,
        "content": "".join(parts), "done_marker": done,
        "finish_reason": finish_reason, "usage": usage,
        "ttft_seconds": None if first is None else first - start,
        "last_content_seconds": None if last is None else last - start,
        "total_seconds": end - start, "decode_tokens_per_second": decode,
    }


def resources(output: Path, stage: str) -> None:
    command = (
        "date -Is; free -b; numastat -m; npu-smi info; "
        "sudo -n docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "
        "dsv4-kt-20261104"
    )
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "taishan", command],
            capture_output=True, text=True, timeout=45, check=False,
        )
        item = {"at": now(), "stage": stage, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except Exception as error:
        item = {"at": now(), "stage": stage, "error": repr(error)}
    append_json(output / "resources.jsonl", item)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"phase": "starting", "started_at": now(), "output": str(output)}
    atomic_json(output / "status.json", status)
    session = requests.Session()
    session.trust_env = False
    try:
        health = session.get(BASE + "/health", timeout=15)
        health.raise_for_status()
        models = session.get(BASE + "/v1/models", timeout=15)
        models.raise_for_status()
        model_info = models.json()["data"][0]
        model = model_info["id"]
        if int(model_info["max_model_len"]) != MODEL_LIMIT:
            raise RuntimeError("deployed context length changed; inspect config first")
        atomic_json(output / "model.json", model_info)
        resources(output, "before")
        status.update(phase="correctness", model=model, updated_at=now())
        atomic_json(output / "status.json", status)
        checks = (
            ("capital", "法国的首都是哪里？只回答城市名称。", "巴黎"),
            ("arithmetic", "计算 17 + 25。只输出数字答案。", "42"),
            ("sse", "Reply with only the word READY.", "READY"),
        )
        for name, prompt, expected in checks:
            item = stream_chat(session, model, prompt, 64)
            item.update(kind="correctness", name=name,
                        passed=item["done_marker"] and expected in item["content"])
            append_json(output / "events.jsonl", item)
            print(f"correctness {name}: {item['passed']}", flush=True)
            if not item["passed"]:
                raise RuntimeError(f"correctness gate failed: {name}")
        cases = []
        for target in TARGETS:
            prompt, count = make_prompt(session, target)
            cases.append({"target_tokens": target, "prompt_tokens": count, "prompt": prompt})
        atomic_json(output / "requests.json", cases)
        all_results = []
        for case in cases:
            target = case["target_tokens"]
            resources(output, f"before_{target}")
            for index in range(args.warmup + args.repetitions):
                kind = "warmup" if index < args.warmup else "formal"
                status.update(phase="running", target_tokens=target, kind=kind,
                              iteration=index, updated_at=now())
                atomic_json(output / "status.json", status)
                item = stream_chat(session, model, case["prompt"], 256)
                item.update(kind=kind, target_tokens=target, iteration=index,
                            requested_prompt_tokens=case["prompt_tokens"])
                append_json(output / "events.jsonl", item)
                if kind == "formal":
                    all_results.append(item)
                print(f"{target} {kind} {index}: {item['total_seconds']:.2f}s "
                      f"{(item['usage'] or {}).get('completion_tokens')} tokens", flush=True)
                if not item["done_marker"] or not item["usage"]:
                    raise RuntimeError("missing SSE end marker or usage")
            resources(output, f"after_{target}")
        resources(output, "after")
        summary = {"model": model, "finished_at": now(), "repetitions": args.repetitions,
                   "warmup": args.warmup, "cases": []}
        for target in TARGETS:
            group = [x for x in all_results if x["target_tokens"] == target]
            metrics = {}
            for field in ("ttft_seconds", "total_seconds", "decode_tokens_per_second"):
                values = [x[field] for x in group if x[field] is not None]
                metrics[field] = ({"median": statistics.median(values),
                                   "p95": percentile(values, 0.95), "values": values}
                                  if values else None)
            summary["cases"].append({"target_tokens": target, "metrics": metrics,
                                     "finish_reasons": [x["finish_reason"] for x in group],
                                     "prompt_tokens": [x["usage"]["prompt_tokens"] for x in group],
                                     "completion_tokens": [x["usage"]["completion_tokens"] for x in group]})
        atomic_json(output / "summary.json", summary)
        status.update(phase="complete", completed_at=now(), formal_requests=len(all_results))
        atomic_json(output / "status.json", status)
        return 0
    except Exception as error:
        status.update(phase="failed", updated_at=now(), error=repr(error))
        atomic_json(output / "status.json", status)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
