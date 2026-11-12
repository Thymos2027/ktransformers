"""Exercise frozen histories on real prefill and decode paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


URL = "http://127.0.0.1:18082/generate"


def generate(core, rid, case, processor, capture):
    payload = {
        "rid": rid,
        "input_ids": case["prompt_ids"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": len(case["forced_ids"]) + 1,
            "ignore_eos": True,
            "custom_params": {
                "kt_forced_ids": case["forced_ids"],
                "kt_capture": capture,
            },
        },
        "custom_logit_processor": processor,
    }
    request = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with core.HTTP.open(request, timeout=300) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"{rid}: HTTP {exc.code}: {body}") from exc
    if not isinstance(value, dict) or "text" not in value:
        raise RuntimeError(f"Malformed generation response for {rid}: {str(value)[:300]}")
    return {"rid": rid, "text": value["text"], "meta_info": value.get("meta_info"),
            "capture": capture, "case": case["name"]}


def check_capture(root: Path, rid: str, prompt_len: int, frozen_ids: list[int],
                  tp: int, graph: bool):
    steps = len(frozen_ids) + 1
    for rank in range(tp):
        partial_path = root / f"rank{rank}-partial.jsonl"
        if partial_path.exists():
            partial = [json.loads(line) for line in partial_path.read_text(
                encoding="utf-8").splitlines() if line]
            partial = [item for item in partial if item["rid"] == rid]
            if any(not 0 < item["seq_len"] < prompt_len or
                   item["mode"] not in ("EXTEND", "MIXED") for item in partial):
                raise RuntimeError(f"{rid} rank{rank}: invalid partial prefill: {partial}")
        records = []
        for line in (root / f"rank{rank}.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["rid"] == rid:
                records.append(item)
        if len(records) != steps:
            raise RuntimeError(f"{rid} rank{rank}: expected {steps} captures, got {len(records)}")
        for step, item in enumerate(records):
            if item["step"] != step:
                raise RuntimeError(f"{rid} rank{rank}: capture order mismatch")
            if item["prompt_len"] != prompt_len or item["seq_len"] != prompt_len + step:
                raise RuntimeError(f"{rid} rank{rank}: token position/KV length mismatch: {item}")
            if item["call_for_position"] != 0:
                raise RuntimeError(f"{rid} rank{rank}: repeated forward at position {step}")
            expected_history_hash = hashlib.sha256(
                json.dumps(frozen_ids[:step], separators=(",", ":")).encode()
            ).hexdigest()
            if item["model_history_sha256"] != expected_history_hash:
                raise RuntimeError(f"{rid} rank{rank}: frozen model history mismatch")
            if item["forced_token"] != (frozen_ids[step] if step < len(frozen_ids) else None):
                raise RuntimeError(f"{rid} rank{rank}: forced token mismatch")
            if step > 0 and item["decode_input_id"] != frozen_ids[step - 1]:
                raise RuntimeError(f"{rid} rank{rank}: decode input mismatch")
            if item["mode"] != ("EXTEND" if step == 0 else "DECODE"):
                raise RuntimeError(f"{rid} rank{rank}: wrong forward path: {item}")
            if item["graph_replay"] is not (graph and step > 0):
                raise RuntimeError(f"{rid} rank{rank}: graph replay mismatch: {item}")
            if not (root / f"{rid}-rank{rank}-step{step:03d}.pt").is_file():
                raise RuntimeError(f"{rid} rank{rank}: missing logits step {step}")
        extras_path = root / f"rank{rank}-extra.jsonl"
        if extras_path.exists():
            extras = [json.loads(line) for line in extras_path.read_text(
                encoding="utf-8").splitlines() if line]
            extras = [item for item in extras if item["rid"] == rid]
            if len(extras) > 1 or any(item["step"] != steps for item in extras):
                raise RuntimeError(f"{rid} rank{rank}: unexpected extra forwards: {extras}")


def final_hash(root: Path, rid: str, rank: int = 0) -> str:
    rows = [json.loads(line) for line in (root / f"rank{rank}.jsonl").read_text(
        encoding="utf-8").splitlines()]
    matching = [row for row in rows if row["rid"] == rid]
    if not matching:
        raise RuntimeError(f"No logits digest for {rid} rank{rank}")
    return matching[-1]["logits_fp32_sha256"]


def run_phase(core, phase: str, tp: int, graph: bool, cases: list[dict]):
    processor = (core.ROOT / "probe-processor.txt").read_text(encoding="utf-8").strip()
    if not processor.startswith("{"):
        raise RuntimeError("Probe processor serialization unavailable")
    root = core.ROOT / "probe"
    results = []
    for case in cases:
        captured = []
        for repeat in range(2):
            core.safe()
            rid = f"{phase}-{case['name']}-r{repeat}"
            core.status("probing", phase_id=phase, case=case["name"], repeat=repeat)
            result = generate(core, rid, case, processor, capture=True)
            check_capture(root, rid, len(case["prompt_ids"]), case["forced_ids"], tp, graph)
            results.append(result)
            captured.append(result)
            core.atomic(core.ROOT / f"{phase}-responses.json", results)
        if captured[0]["text"] != captured[1]["text"]:
            raise RuntimeError(f"{phase} {case['name']}: repeated output changed")
        if case == cases[0]:
            no_capture_rid = f"{phase}-{case['name']}-no-capture"
            uninstrumented = generate(core, no_capture_rid,
                                      case, processor, capture=False)
            if captured[0]["text"] != uninstrumented["text"]:
                raise RuntimeError(f"{phase}: capture changed final sampled output")
            for rank in range(tp):
                if final_hash(root, captured[0]["rid"], rank) != final_hash(
                    root, no_capture_rid, rank
                ):
                    raise RuntimeError(f"{phase} rank{rank}: capture changed final logits")
            results.append(uninstrumented)
            core.atomic(core.ROOT / f"{phase}-responses.json", results)
    return {"phase": phase, "tp": tp, "graph": graph, "requests": len(results),
            "captured_requests": len(cases) * 2,
            "positions_per_request": len(cases[0]["forced_ids"]) + 1}
