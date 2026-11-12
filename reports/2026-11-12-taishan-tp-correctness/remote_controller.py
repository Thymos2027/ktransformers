"""Guarded four-path numerical probe with original-service restoration."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time

import tp_runtime as core
import probe_client


PHASES = (("tp1_eager", 1, False), ("tp2_eager", 2, False),
          ("tp1_graph", 1, True), ("tp2_graph", 2, True))
IMAGE = "sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e"
core.IMAGE = IMAGE
core.NAMES = {phase: f"dsv4-tp2-{core.RUN_ID}-{phase}" for phase, _, _ in PHASES}
core.DEADLINE = time.monotonic() + 4 * 3600


def main():
    guard = None
    stopped = False
    restored = False
    was_running = False
    error = None
    scope = None
    core.LOGS.mkdir(exist_ok=True)
    core.status("preflight")
    try:
        scope = json.loads((core.ROOT / "scope.json").read_text(encoding="utf-8"))
        if scope["mode"] == "long_prefill_check":
            active_phases = PHASES[:1]
        elif scope["mode"] == "full_pilot":
            active_phases = PHASES
        else:
            raise RuntimeError(f"Unknown probe scope: {scope}")
        core.NAMES = {phase: f"dsv4-tp2-{core.RUN_ID}-{phase}"
                      for phase, _, _ in active_phases}
        was_running = core.docker("inspect", "-f", "{{.State.Running}}", core.ORIGINAL) == "true"
        if not was_running:
            raise RuntimeError("Original service must be running")
        current = set(core.docker("ps", "--format", "{{.Names}}").splitlines())
        if current != {core.ORIGINAL}:
            raise RuntimeError(f"Unexpected running containers: {current}")
        if core.docker("image", "inspect", "-f", "{{.Id}}", IMAGE) != IMAGE:
            raise RuntimeError("Frozen environment image absent")
        existing = set(core.docker("ps", "-a", "--format", "{{.Names}}").splitlines())
        if existing.intersection(core.NAMES.values()):
            raise RuntimeError("Probe container names already exist")
        cases_file = core.ROOT / "cases.json"
        cases = json.loads(cases_file.read_text(encoding="utf-8"))
        if len(cases) != 8 or any(len(c["forced_ids"]) != 8 for c in cases):
            raise RuntimeError("Frozen eight-case pilot missing or changed")
        # Exercise chunked prefill first, before the shorter diagnostic prompts.
        cases = sorted(cases, key=lambda c: (-len(c["prompt_ids"]), c["name"]))
        if scope["mode"] == "long_prefill_check":
            cases = [c for c in cases if c["name"] == "prior_tp2"]
            if len(cases) != 1:
                raise RuntimeError("Long prefill case missing")
        core.atomic(core.ROOT / "probe-config.json", {
            "cases_sha256": hashlib.sha256(cases_file.read_bytes()).hexdigest(),
            "case_order": [c["name"] for c in cases],
            "phases": active_phases, "image": IMAGE, "scope": scope["mode"],
            "numeric_acceptance": "not set; requires review after pilot",
        })
        core.read_cards()
        if time.time() - (core.ROOT / "heartbeat").stat().st_mtime > 45:
            raise RuntimeError("Fan-monitor heartbeat stale")
        guard = threading.Thread(target=core.watchdog, daemon=True)
        guard.start()
        core.prepare()
        core.status("stopping_original")
        stopped = True
        core.stop_original()
        summaries = []
        for phase, tp, graph in active_phases:
            core.safe()
            core.status("loading", phase_id=phase, tp=tp, graph=graph)
            core.create_container(phase)
            try:
                core.wait_health("http://127.0.0.1:18082", core.NAMES[phase], 1800)
                core.capture_logs(phase)
                log = (core.LOGS / f"{phase}.log").read_text(errors="replace")
                if graph:
                    if tp == 1 and "Capture target decode NPU graph end" not in log:
                        raise RuntimeError("TP1 decode graph capture marker missing")
                    for rank in range(tp):
                        if tp > 1 and not re.search(
                            r"TP" + str(rank) + r"\] Capture target decode NPU graph end", log
                        ):
                            raise RuntimeError(f"TP2 rank {rank} decode graph capture missing")
                summaries.append(probe_client.run_phase(core, phase, tp, graph, cases))
                core.atomic(core.ROOT / "pilot-summary.json", summaries)
            finally:
                core.docker("stop", "-t", "30", core.NAMES[phase], timeout=65, check=False)
                core.settled_stopped(core.NAMES[phase])
                core.capture_logs(phase)
        core.status("pilot_collected", scope=scope["mode"],
                    verdict="structural checks only; numerical acceptance pending")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        core.atomic(core.ROOT / "failure.json", {"at": core.now(), "error": error})
        print(error, flush=True)
    finally:
        core.stop_all()
        if stopped and not core.ALARM.is_set() and not (core.ROOT / "abort.json").exists():
            try:
                core.DEADLINE = max(core.DEADLINE, time.monotonic() + 1800)
                core.read_cards()
                core.status("restoring_original", probe_error=error)
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
                        original_restored=restored,
                        scope=scope["mode"] if scope else "unknown",
                        verdict="correctness and task quality not yet accepted")
    return 1 if error or core.ALARM.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
