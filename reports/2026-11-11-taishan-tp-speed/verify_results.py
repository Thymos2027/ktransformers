"""Independently validate the frozen ABBA evidence and recompute speed ratios."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from workload import check_graph


PHASES = (("tp1a", 1), ("tp2a", 2), ("tp2b", 2), ("tp1b", 1))
FIELDS = ("ttft_seconds", "decode_tokens_per_second", "total_seconds")


def require(value, message):
    if not value:
        raise ValueError(message)


def stats(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)


def verify(run):
    root = run / "remote"
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    state = read("status.json")
    require(state["phase"] == "complete" and state["original_restored"] and not state["error"],
            "Workflow did not complete and restore baseline")
    config = read("benchmark-config.json")
    require(config["max_total_tokens"] == 16384 and config["experts"] == 32
            and config["cpu_threads"] == 96 and config["output_tokens"] == 512
            and config["ignore_eos"] is True and config["phases"] == [list(x) for x in PHASES],
            "Benchmark configuration differs from frozen protocol")
    cases = read("requests.json")
    require(hashlib.sha256((root / "requests.json").read_bytes()).hexdigest() == config["requests_sha256"],
            "Frozen request digest mismatch")
    by_case = {x["name"]: x for x in cases}
    require([x["prompt_tokens"] for x in cases] == [144, 792], "Unexpected input lengths")
    rows = [json.loads(x) for x in (root / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    formal = [x for x in rows if x["kind"] == "formal"]
    require(len(rows) == 32 and len(formal) == 24, "Incomplete request matrix")
    identity = [(x["phase"], x["case"], x["iteration"]) for x in rows]
    require(len(set(identity)) == 32, "Duplicate request identity")
    expected = {(phase, case["name"], iteration) for phase, _ in PHASES for case in cases
                for iteration in range(4)}
    require(set(identity) == expected, "Missing phase/case/iteration")
    for x in rows:
        case = by_case[x["case"]]
        require(x["payload"]["messages"] == case["messages"] and x["payload"]["ignore_eos"]
                and x["payload"]["seed"] == 20261111 and x["payload"]["temperature"] == 0,
                "Request payload changed")
        require(x["tp"] == dict(PHASES)[x["phase"]] and x["done"]
                and x["finish_reason"] == "length" and x["usage"]["completion_tokens"] == 512
                and x["usage"]["prompt_tokens"] == case["prompt_tokens"], "Request gate failed")
        require(hashlib.sha256(x["content"].encode()).hexdigest() == x["content_sha256"], "Output digest mismatch")
        require("".join(c["text"] for c in x["chunks"]) == x["content"], "Chunk content mismatch")
        times = [c["seconds"] for c in x["chunks"]]
        require(times == sorted(times) and times[0] == x["ttft_seconds"]
                and times[-1] == x["last_content_seconds"] and times[-1] <= x["total_seconds"],
                "Streaming timestamps invalid")
        speed = 511 / (times[-1] - times[0])
        require(math.isclose(speed, x["decode_tokens_per_second"], rel_tol=1e-12), "Decode calculation mismatch")

    result = dict(passed=True, run_id=run.name, completed_at=state["at"], configuration=config,
                  source=read("source.json"), candidate=read("candidate.json"),
                  formal_requests=24, warmup_requests=8, phases={}, cases=[])
    for phase, tp in PHASES:
        gate = read(phase + "-gate.json")
        require(len(gate) == 5 and all(x["passed"] and x["done"] and x["finish_reason"] == "stop" for x in gate),
                f"{phase}: task gate failed")
        require([x["content"].strip() for x in gate[:4]] == ["巴黎", "42", "156", "READY"]
                and json.loads(gate[4]["content"]) == [1, 2, 5, 9], "Task answers incorrect")
        log = (root / "logs" / (phase + ".log")).read_text(encoding="utf-8", errors="replace")
        check_graph(log, tp)
        for marker in ("max_total_tokens=16384", "kt_num_gpu_experts=32", "kt_cpuinfer=96",
                       "random_seed=20261111", "max_running_requests=1"):
            require(marker in log, f"{phase}: resolved config missing {marker}")
        serving, separator, shutdown = log.partition("SIGTERM received.")
        require(separator and not re.search(r"Traceback|OutOfMemoryError|Scheduler hit an exception", serving),
                f"{phase}: unexpected error before controlled stop")
        phase_data = dict(tp=tp, tasks_passed=5, shutdown_traceback_present="Traceback" in shutdown,
                          by_case={}, hbm_mib={})
        for case in cases:
            group = [x for x in formal if x["phase"] == phase and x["case"] == case["name"]]
            phase_data["by_case"][case["name"]] = {field: stats([x[field] for x in group]) for field in FIELDS}
        for stage in ("before", "after"):
            raw = read(f"{phase}-{stage}-resources.json")["raw"]
            hbm = list(map(int, re.findall(r"(\d+)\s*/\s*65536", raw)))
            require(len(hbm) == 2, "HBM snapshot incomplete")
            phase_data["hbm_mib"][stage] = dict(zip(("2", "5"), hbm))
        result["phases"][phase] = phase_data

    automatic = read("summary.json")
    for case in cases:
        name = case["name"]
        group = [x for x in formal if x["case"] == name]
        item = dict(case=name, prompt_tokens=case["prompt_tokens"], completion_tokens=512,
                    by_tp={}, pairs=[])
        for tp in (1, 2):
            subset = [x for x in group if x["tp"] == tp]
            item["by_tp"][str(tp)] = {field: stats([x[field] for x in subset]) for field in FIELDS}
            for field in FIELDS:
                actual = next(c for c in automatic["cases"] if c["case"] == name)["tp"][str(tp)]["metrics"][field]["median"]
                require(actual == item["by_tp"][str(tp)][field]["median"], "Automatic summary mismatch")
        for p1, p2 in (("tp1a", "tp2a"), ("tp1b", "tp2b")):
            for iteration in (1, 2, 3):
                a = next(x for x in group if x["phase"] == p1 and x["iteration"] == iteration)
                b = next(x for x in group if x["phase"] == p2 and x["iteration"] == iteration)
                item["pairs"].append(dict(tp1_phase=p1, tp2_phase=p2, iteration=iteration,
                    decode_ratio=b["decode_tokens_per_second"] / a["decode_tokens_per_second"],
                    total_time_ratio=a["total_seconds"] / b["total_seconds"],
                    output_identical=a["content_sha256"] == b["content_sha256"]))
        t1, t2 = item["by_tp"]["1"], item["by_tp"]["2"]
        item["decode_gain_percent_ratio_of_medians"] = 100 * (t2["decode_tokens_per_second"]["median"] / t1["decode_tokens_per_second"]["median"] - 1)
        item["total_time_reduction_percent_ratio_of_medians"] = 100 * (1 - t2["total_seconds"]["median"] / t1["total_seconds"]["median"])
        item["output_hashes_by_tp"] = {str(tp): sorted({x["content_sha256"] for x in group if x["tp"] == tp}) for tp in (1, 2)}
        result["cases"].append(item)
    for name in ("restored-original-smoke.json",):
        smoke = read(name)
        require(len(smoke) == 3 and all(x["passed"] and x["done"] for x in smoke), "Restoration smoke failed")
    thermal = [json.loads(x) for x in (run / "thermal.jsonl").read_text(encoding="utf-8").splitlines()]
    valid = [x for x in thermal if "cards" in x]
    errors = [x for x in thermal if "error" in x]
    result["monitoring"] = dict(valid_samples=len(valid), command_errors=len(errors),
        max_consecutive_misses=max((x.get("misses", 0) for x in errors), default=0),
        max_temperature_c={k: max(x["cards"][k]["temperature_c"] for x in valid) for k in ("2", "5")},
        min_fan_rpm=min(v for x in valid for v in x["fans"].values()))
    require(result["monitoring"]["max_consecutive_misses"] < 2
            and max(result["monitoring"]["max_temperature_c"].values()) < 70
            and result["monitoring"]["min_fan_rpm"] >= 4500, "Thermal guard threshold crossed")
    result["formal_rows"] = [{k: x[k] for k in ("phase", "tp", "case", "iteration", *FIELDS, "content_sha256")}
                             for x in formal]
    evidence = list(root.glob("*.json")) + list((root / "logs").glob("*.log"))
    evidence += [root / "requests.jsonl", root / "launch.sh", root / "workload.py",
                 root / "remote_controller.py", root / "tp_runtime.py", run / "thermal.jsonl"]
    result["evidence_sha256"] = {p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(evidence)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(case["case"], "decode gain %:", round(case["decode_gain_percent_ratio_of_medians"], 3),
              "total time reduction %:", round(case["total_time_reduction_percent_ratio_of_medians"], 3))
