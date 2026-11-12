"""Streaming client and frozen workload checks for the paired TP experiment."""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request


def check_graph(log, tp):
    if not re.search(r"\btp_size=" + str(tp) + r"\b", log):
        raise RuntimeError("Resolved TP size missing or mismatched")
    if tp == 1:
        # SGLang omits the TP rank label for the single-rank logger.
        if "Capture target decode NPU graph end" not in log:
            raise RuntimeError("Single-rank decode graph success marker missing")
    else:
        for rank in range(tp):
            if not re.search(r"TP" + str(rank) + r"\] Capture target decode NPU graph end", log):
                raise RuntimeError(f"No decode graph for TP{rank}")


def stream(core, model, messages, max_tokens, ignore_eos=False):
    payload = dict(model=model, messages=messages, temperature=0, seed=20261111,
                   max_tokens=max_tokens, ignore_eos=ignore_eos, stream=True,
                   stream_options=dict(include_usage=True))
    request = urllib.request.Request("http://127.0.0.1:18082/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    first = last = None
    content, usage, finish, done = [], None, None, False
    started_at = core.now()
    start = time.monotonic()
    chunks = []
    with core.HTTP.open(request, timeout=600) as response:
        for line in response:
            core.safe()
            elapsed = time.monotonic() - start
            if elapsed > 600:
                raise TimeoutError("Request exceeded 600 second wall time")
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                done = True
                break
            value = json.loads(raw)
            usage = value.get("usage") or usage
            for choice in value.get("choices", []):
                finish = choice.get("finish_reason") or finish
                text = (choice.get("delta") or {}).get("content") or ""
                if text:
                    if first is None:
                        first = elapsed
                    last = elapsed
                    content.append(text)
                    chunks.append(dict(seconds=elapsed, text=text))
    total = time.monotonic() - start
    output = "".join(content)
    count = (usage or {}).get("completion_tokens")
    speed = None
    if first is not None and last is not None and last > first and count and count > 1:
        speed = (count - 1) / (last - first)
    return dict(started_at=started_at, payload=payload, content=output, usage=usage,
                finish_reason=finish, done=done, ttft_seconds=first, total_seconds=total,
                last_content_seconds=last, decode_tokens_per_second=speed, chunks=chunks,
                content_sha256=hashlib.sha256(output.encode()).hexdigest())


def check_transport(value):
    if not value["done"] or not value["usage"] or not value["content"] or not value["finish_reason"]:
        raise RuntimeError("Missing SSE end, usage, output, or finish reason")


def gate(core, model, phase):
    cases = [
        ("capital", "法国的首都是哪里？只回答城市名称。", "巴黎"),
        ("arithmetic", "计算 17 + 25。只输出数字答案。", "42"),
        ("multiply", "Reply with only the decimal result of 12 * 13.", "156"),
        ("sse", "Reply with only the word READY.", "READY"),
        ("sorting", "Return only a JSON array sorting [9,1,5,2] in ascending order.", [1, 2, 5, 9]),
    ]
    results = []
    for name, prompt, expected in cases:
        value = stream(core, model, [dict(role="user", content=prompt)], 64)
        check_transport(value)
        actual = value["content"].strip()
        if isinstance(expected, list):
            try:
                passed = json.loads(actual) == expected
            except ValueError:
                passed = False
        else:
            passed = actual == expected
        value.update(name=name, passed=passed and value["finish_reason"] == "stop")
        results.append(value)
        core.atomic(core.ROOT / f"{phase}-gate.json", results)
        if not value["passed"]:
            raise RuntimeError(f"{phase} quality gate failed: {name}: {actual!r}")


def measure(core, phase, tp, cases):
    model = core.request("http://127.0.0.1:18082", "/v1/models")["data"][0]["id"]
    gate(core, model, phase)
    for case in cases:
        tokenization = core.request("http://127.0.0.1:18082", "/tokenize",
                                    dict(messages=case["messages"]))
        if tokenization["count"] != case["prompt_tokens"] or tokenization["tokens"] != case["token_ids"]:
            raise RuntimeError("Prompt tokenization changed across environments")
        for index in range(4):
            kind = "warmup" if index == 0 else "formal"
            core.status("measuring", phase_id=phase, tp=tp, case=case["name"], kind=kind, iteration=index)
            value = stream(core, model, case["messages"], 512, ignore_eos=True)
            value.update(phase=phase, tp=tp, case=case["name"], kind=kind, iteration=index)
            with (core.ROOT / "requests.jsonl").open("a") as output:
                output.write(json.dumps(value, ensure_ascii=False) + "\n")
                output.flush()
            check_transport(value)
            if value["usage"]["completion_tokens"] != 512 or value["finish_reason"] != "length":
                raise RuntimeError("Performance request did not generate the full 512 token budget")
            if value["usage"]["prompt_tokens"] != case["prompt_tokens"]:
                raise RuntimeError("Actual prompt token count differs from frozen workload")
            if value["decode_tokens_per_second"] is None:
                raise RuntimeError("Generation interval unavailable")
