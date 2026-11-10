"""Emit one read-only JSON sample from the taishan host (run with sudo)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess


def fields(path: Path, wanted: set[str]) -> dict[str, int]:
    result = {}
    for line in path.read_text().splitlines():
        pieces = line.split()
        if pieces and pieces[0].rstrip(":") in wanted:
            try:
                result[pieces[0].rstrip(":")] = int(pieces[1])
            except (IndexError, ValueError):
                pass
    return result


def process_sample(pid: int) -> dict:
    root = Path(f"/proc/{pid}")
    info = fields(root / "status", {"VmRSS", "VmSwap", "Threads", "voluntary_ctxt_switches",
                                     "nonvoluntary_ctxt_switches"})
    info["io"] = fields(root / "io", {"read_bytes", "write_bytes", "rchar", "wchar"})
    stat = (root / "stat").read_text().split(") ", 1)[1].split()
    info.update(minflt=int(stat[7]), majflt=int(stat[9]),
                utime_ticks=int(stat[11]), stime_ticks=int(stat[12]))
    sched = (root / "schedstat").read_text().split()
    info["schedstat"] = {"runtime_ns": int(sched[0]), "wait_ns": int(sched[1]),
                         "timeslices": int(sched[2])}
    info["pid"] = pid
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numa", action="store_true")
    args = parser.parse_args()
    processes = subprocess.check_output(["ps", "-eo", "pid=,comm="], text=True)
    pids = [int(line.split()[0]) for line in processes.splitlines()
            if len(line.split()) >= 2 and line.split()[1].startswith("sglang::schedul")]
    output = {
        "at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "vmstat": fields(Path("/proc/vmstat"), {"pgmajfault", "pswpin", "pswpout",
                                                "pgscan_direct", "pgscan_kswapd",
                                                "numa_miss", "numa_foreign"}),
        "meminfo": fields(Path("/proc/meminfo"), {"MemAvailable", "SwapFree", "MemFree"}),
        "pressure": {name: Path(f"/proc/pressure/{name}").read_text().strip()
                     for name in ("cpu", "io", "memory")},
        "processes": [],
    }
    for pid in pids:
        try:
            item = process_sample(pid)
            if args.numa:
                result = subprocess.run(["numastat", "-p", str(pid)], capture_output=True,
                                        text=True, timeout=15)
                item["numastat"] = {"exit_code": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr}
            output["processes"].append(item)
        except (FileNotFoundError, ProcessLookupError):
            pass
    if args.numa:
        for name, command in {
            "free": ["free", "-b"], "npu": ["npu-smi", "info"],
            "docker_stats": ["docker", "stats", "--no-stream", "--format",
                             "{{.Name}} {{.CPUPerc}} {{.MemUsage}}", "dsv4-kt-20261104"],
        }.items():
            result = subprocess.run(command, capture_output=True, text=True, timeout=25)
            output[name] = {"exit_code": result.returncode, "stdout": result.stdout,
                            "stderr": result.stderr}
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
