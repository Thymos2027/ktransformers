from __future__ import annotations

import hashlib

import json

import re

import time

import urllib.request

def stream(core, model, messages, max_tokens, ignore_eos=False, routed_dp_rank=None):
    payload = dict(model=model, messages=messages, temperature=0, seed=20261111,
                   max_tokens=max_tokens, ignore_eos=ignore_eos, stream=True,
                   stream_options=dict(include_usage=True))
    if routed_dp_rank is not None:
        payload["routed_dp_rank"] = routed_dp_rank
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
