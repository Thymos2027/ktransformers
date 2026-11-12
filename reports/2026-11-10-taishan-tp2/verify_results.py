"""Verify archived TP2 functional gates and write a small reviewable summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run):
    root = run / "remote"
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    state = read("status.json")
    require(state["phase"] == "complete" and state["original_restored"], "Workflow not complete/restored")
    result = {"run_id": run.name, "remote_completed_at": state["at"],
              "scope": "TP2/EP1 short functional validation only",
              "source": read("source.json"), "candidate": read("candidate.json"),
              "environment": read("environment.json"), "hccl": [], "modes": {}}
    for rank in (0, 1):
        hccl = read(f"hccl-rank{rank}.json")
        require(hccl["rank"] == rank and hccl["world_size"] == 2 and hccl["passed"], "HCCL rank failed")
        require([x["count"] for x in hccl["checks"]] == [1, 1024, 262144], "HCCL cases changed")
        require(all(x["passed"] for x in hccl["checks"]), "HCCL sum mismatch")
        result["hccl"].append(hccl)
    for mode in ("eager", "graph", "restored-original"):
        smoke = read(f"{mode}-smoke.json")
        require([x["name"] for x in smoke] == ["capital", "arithmetic", "sse"], "Missing smoke cases")
        for item, expected in zip(smoke, ("巴黎", "42", "READY")):
            require(item["passed"] and item["content"] == expected and item["done"]
                    and item["finish_reason"] == "stop", f"{mode} {item['name']} failed")
            require(item["usage"]["completion_tokens"] > 0, "Missing completion usage")
        summary = {"smoke": smoke}
        if mode != "restored-original":
            log = (root / "logs" / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
            require("max_total_tokens=16384" in log, "Token cap not in resolved arguments")
            require(log.count("DSV4 pool sizes: full=16384") == 2, "Both rank pools must be capped")
            summary["layers_per_rank"] = {}
            for rank in (0, 1):
                matches = re.findall(
                    r"\[KT TP2 EXPERIMENT\] layer=(\d+) rank=" + str(rank)
                    + r" moe_tp=2 local_intermediate=1024 full_intermediate=2048 "
                    + r"resident_experts=32 cpu_owner=" + ("True" if rank == 0 else "False"), log)
                require(set(map(int, matches)) == set(range(43)), f"{mode}: rank {rank} layer diagnostics missing")
                summary["layers_per_rank"][str(rank)] = len(matches)
                capture = re.search(r"TP" + str(rank) + r"\] Capture target decode NPU graph end", log)
                require(bool(capture) == (mode == "graph"), f"{mode}: capture state wrong for rank {rank}")
            # Retain shutdown errors, but distinguish them from serving errors.
            serving, separator, shutdown = log.partition("SIGTERM received.")
            require(bool(separator), "Expected controlled SIGTERM shutdown")
            require(not re.search(r"Traceback|OutOfMemoryError|Scheduler hit an exception", serving),
                    f"{mode}: exception during serving")
            summary["shutdown_traceback_present"] = "Traceback" in shutdown
            snapshot = read(f"{mode}-npu.json")
            hbm = list(map(int, re.findall(r"(\d+)\s*/\s*65536", snapshot["raw"])))
            require(len(hbm) == 2, "Missing two-card HBM snapshot")
            require(len(re.findall(r"\|\s*[25]\s+0\s*\|\s*\d+\s*\|\s*sglangschedul", snapshot["raw"])) == 2,
                    "Missing scheduler process on a physical NPU")
            summary["hbm_mib"] = dict(zip(("2", "5"), hbm))
        result["modes"][mode] = summary
    rows = [json.loads(x) for x in (run / "thermal.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [x for x in rows if "cards" in x]
    failures = [x for x in rows if "error" in x]
    require(samples, "No local thermal samples")
    result["monitoring"] = {
        "valid_samples": len(samples), "command_error_events": len(failures),
        "max_consecutive_misses": max((x["misses"] for x in failures), default=0),
        "maximum_temperature_c": {key: max(x["cards"][key]["temperature_c"] for x in samples)
                                  for key in ("2", "5")},
        "minimum_fan_rpm": min(v for x in samples for v in x["fans"].values()),
    }
    require(result["monitoring"]["max_consecutive_misses"] < 2, "Telemetry stop threshold crossed")
    require(not (root / "alarm.json").exists() and not (root / "abort.json").exists(), "Alarm/abort artifact present")
    sources = [x for x in root.glob("*.json") if x.is_file()] + list((root / "logs").glob("*.log"))
    sources += [root / "candidate.diff", root / "launch.sh", root / "controller.stdout.log", run / "thermal.jsonl"]
    result["evidence_sha256"] = {x.relative_to(run).as_posix(): hashlib.sha256(x.read_bytes()).hexdigest()
                                 for x in sorted(sources)}
    result["passed"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = verify(args.run)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TP2 functional gates verified; summary:", args.output)
