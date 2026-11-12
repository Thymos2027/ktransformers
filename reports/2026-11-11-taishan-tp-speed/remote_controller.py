"""TP1/TP2 ABBA sequence, repeated fixed-token requests, and baseline restore."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
import time

import tp_runtime as core
import workload


PHASES = (("tp1a", 1), ("tp2a", 2), ("tp2b", 2), ("tp1b", 1))
IMAGE = "sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e"
core.IMAGE = IMAGE
core.NAMES = {name: f"dsv4-tp2-{core.RUN_ID}-{name}" for name, _ in PHASES}
core.DEADLINE = time.monotonic() + 3 * 3600


def summarize():
    rows = [json.loads(x) for x in (core.ROOT / "requests.jsonl").read_text().splitlines()]
    formal = [x for x in rows if x["kind"] == "formal"]
    summary = {"cases": [], "formal_requests": len(formal), "scope": "same-workload, fixed-512-token ABBA"}
    for name in sorted({x["case"] for x in formal}):
        group = [x for x in formal if x["case"] == name]
        item = {"case": name, "tp": {}, "paired": []}
        for tp in (1, 2):
            values = [x for x in group if x["tp"] == tp]
            item["tp"][str(tp)] = {"requests": len(values), "metrics": {
                key: {"median": statistics.median(x[key] for x in values),
                      "min": min(x[key] for x in values), "max": max(x[key] for x in values)}
                for key in ("ttft_seconds", "decode_tokens_per_second", "total_seconds")}}
        for first, second in (("tp1a", "tp2a"), ("tp1b", "tp2b")):
            for index in (1, 2, 3):
                a = next(x for x in group if x["phase"] == first and x["iteration"] == index)
                b = next(x for x in group if x["phase"] == second and x["iteration"] == index)
                item["paired"].append({"tp1_phase": first, "tp2_phase": second, "iteration": index,
                    "decode_speed_ratio_tp2_over_tp1": b["decode_tokens_per_second"] / a["decode_tokens_per_second"],
                    "ttft_time_ratio_tp1_over_tp2": a["ttft_seconds"] / b["ttft_seconds"],
                    "total_time_ratio_tp1_over_tp2": a["total_seconds"] / b["total_seconds"],
                    "output_identical": a["content_sha256"] == b["content_sha256"]})
        summary["cases"].append(item)
    if len(formal) != 24:
        raise RuntimeError("Incomplete formal matrix; expected 24 requests")
    core.atomic(core.ROOT / "summary.json", summary)


def snapshot(phase):
    value = core.run(["bash", "-c", "date -Is; free -b; numastat -m; npu-smi info"], timeout=30)
    core.atomic(core.ROOT / f"{phase}-resources.json", {"at": core.now(), "raw": value})


def main():
    guard = None
    stopped = False
    restored = False
    was_running = False
    error = None
    core.LOGS.mkdir(exist_ok=True)
    core.status("preflight")
    try:
        was_running = core.docker("inspect", "-f", "{{.State.Running}}", core.ORIGINAL) == "true"
        if not was_running:
            raise RuntimeError("Baseline must be running before benchmark")
        current = set(core.docker("ps", "--format", "{{.Names}}").splitlines())
        if current != {core.ORIGINAL}:
            raise RuntimeError(f"Other running containers: {current}")
        if core.docker("image", "inspect", "-f", "{{.Id}}", IMAGE) != IMAGE:
            raise RuntimeError("Frozen environment image absent")
        existing = set(core.docker("ps", "-a", "--format", "{{.Names}}").splitlines())
        if existing.intersection(core.NAMES.values()):
            raise RuntimeError("Benchmark container names already exist")
        core.read_cards()
        if time.time() - (core.ROOT / "heartbeat").stat().st_mtime > 45:
            raise RuntimeError("Fan-monitor heartbeat stale")
        guard = threading.Thread(target=core.watchdog, daemon=True)
        guard.start()
        cases = json.loads((core.ROOT / "requests.json").read_text())
        if len(cases) != 2 or any(x["prompt_tokens"] + 512 > 1536 for x in cases):
            raise RuntimeError("Frozen workload exceeds tested SWA budget")
        core.prepare()
        core.atomic(core.ROOT / "benchmark-config.json", {
            "at": core.now(), "phases": PHASES, "image": IMAGE,
            "max_total_tokens": 16384, "context_length": 8192, "experts": 32,
            "cpu_threads": 96, "threadpool_count": 4, "random_seed": 20261111,
            "output_tokens": 512, "ignore_eos": True, "warmup_per_case": 1,
            "formal_repeats_per_case_per_start": 3,
            "requests_sha256": hashlib.sha256((core.ROOT / "requests.json").read_bytes()).hexdigest(),
        })
        core.status("stopping_original")
        stopped = True
        core.stop_original()
        for phase, tp in PHASES:
            core.safe()
            core.status("loading", phase_id=phase, tp=tp)
            core.create_container(phase)
            try:
                core.wait_health("http://127.0.0.1:18082", core.NAMES[phase], 1800)
                core.capture_logs(phase)
                log = (core.LOGS / f"{phase}.log").read_text(errors="replace")
                workload.check_graph(log, tp)
                snapshot(phase + "-before")
                workload.measure(core, phase, tp, cases)
                snapshot(phase + "-after")
            finally:
                core.docker("stop", "-t", "30", core.NAMES[phase], timeout=65, check=False)
                core.settled_stopped(core.NAMES[phase])
                core.capture_logs(phase)
        summarize()
        core.status("measurements_complete")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        core.atomic(core.ROOT / "failure.json", {"at": core.now(), "error": error})
        print(error, flush=True)
    finally:
        core.stop_all()
        if stopped and not core.ALARM.is_set() and not (core.ROOT / "abort.json").exists():
            try:
                # Reserve time for restoration even if the measurement deadline was reached.
                core.DEADLINE = max(core.DEADLINE, time.monotonic() + 1800)
                core.read_cards()
                core.status("restoring_original", benchmark_error=error)
                core.settled_stopped(core.ORIGINAL)
                core.docker("start", core.ORIGINAL, timeout=60)
                core.wait_health("http://192.168.137.11:18080", core.ORIGINAL, 1800)
                core.smoke("http://192.168.137.11:18080", "restored-original")
                restored = True
            except Exception as exc:
                error = (error or "") + f"; restoration failed: {exc}"
                core.docker("stop", "-t", "10", core.ORIGINAL, timeout=35, check=False)
        elif not stopped:
            restored = was_running and not core.ALARM.is_set()
        core.DONE.set()
        if guard:
            guard.join(timeout=60)
        if core.ALARM.is_set() or (core.ROOT / "abort.json").exists():
            core.stop_all(include_original=True)
            core.status("aborted_thermal_or_telemetry", error=error, original_restored=False)
        else:
            core.status("failed" if error else "complete", error=error,
                        original_restored=restored, results_require_review=True)
    return 1 if error or core.ALARM.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
