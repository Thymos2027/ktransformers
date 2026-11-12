"""Detached Windows fan monitor and launcher for the remote TP2 trial."""
from __future__ import annotations

import datetime as dt
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time


HERE = Path(__file__).resolve().parent
RUN = None
REMOTE = None
NAMES = None
ORIGINAL = "dsv4-kt-20261104"
BUNDLE_FILES = ("prepare_candidate.py", "hccl_probe.py", "launch.sh", "remote_controller.py")
BUNDLE_SOURCES = {}
MAX_SECONDS = 2 * 3600
CONTAINER_PHASES = ("hccl", "eager", "graph")


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def status(phase, **fields):
    value = dict(phase=phase, at=now(), pid=os.getpid(), remote=REMOTE, **fields)
    atomic(RUN / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def ssh(alias, command, timeout=20):
    value = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            alias, command], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
    if value.returncode:
        raise RuntimeError(f"{alias} exit {value.returncode}: {(value.stderr or value.stdout)[-800:]}")
    return value.stdout


def sample():
    raw = ssh("taishan", "npu-smi info")
    cards = {}
    for m in re.finditer(r"\|\s*(\d+)\s+910B4-1\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s+(\d+)\s+", raw):
        key, health, power, temp = m.groups()
        cards[key] = dict(health=health, temperature_c=int(temp), power_w=float(power))
    raw = ssh("taishan-ibmc", "ipmcget -t sensor -d list")
    fans = {}
    for m in re.finditer(r"(?m)^\s*0x[0-9a-f]+\s*\|\s*(FAN[1-4] Speed)\s*\|\s*([\d.]+)", raw):
        fans[m.group(1)] = float(m.group(2))
    if set(cards) != {"2", "5"} or len(fans) != 4:
        raise RuntimeError(f"Missing telemetry: cards={cards}, fans={fans}")
    return dict(at=now(), cards=cards, fans=fans)


def abnormal(value):
    if any(v["health"] != "OK" or v["temperature_c"] >= 70 for v in value["cards"].values()):
        return f"NPU health/temperature limit: {value['cards']}"
    if min(value["fans"].values()) < 4500:
        return f"Fan below 4500 RPM: {value['fans']}"
    return None


def emergency_stop(reason):
    atomic(RUN / "alarm.json", dict(at=now(), reason=reason))
    # The remote watchdog treats this marker as a persistent no-restore request.
    try:
        ssh("taishan", f"touch {REMOTE}/abort.json")
    finally:
        for name in NAMES + [ORIGINAL]:
            try:
                ssh("taishan", f"sudo -n docker stop -t 3 {name}", timeout=35)
            except Exception as exc:
                print(f"emergency stop {name}: {exc}", flush=True)


def collect():
    archive = RUN / "remote-results.tar.gz"
    with archive.open("wb") as output:
        result = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "taishan",
            f"tar -C {REMOTE} --exclude=./sglang --exclude=./sglang-source.tar "
            "--exclude=./kernel_meta -czf - .",
        ], stdout=output, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise RuntimeError("Result archive retrieval failed: " + result.stderr.decode(errors="replace"))
    destination = (RUN / "remote").resolve()
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            target = (destination / member.name).resolve()
            target.relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as output:
                output.write(source.read())


def main():
    global RUN, REMOTE, NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20261111-first")
    parser.add_argument("--resume-stopped-baseline", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.run_id):
        parser.error("run-id must contain lowercase letters, digits or hyphens")
    RUN = HERE / "runs" / args.run_id
    REMOTE = "/data/deploy/dsv4-tp2-" + args.run_id
    NAMES = [f"dsv4-tp2-{args.run_id}-{mode}" for mode in CONTAINER_PHASES]
    RUN.mkdir(parents=True, exist_ok=False)
    launched = False
    misses = 0
    count = 0
    status("preflight")
    try:
        if "Current mode: auto" not in ssh("taishan-ibmc", "ipmcget -d faninfo"):
            raise RuntimeError("iBMC fan mode changed; review before trial")
        first = sample()
        reason = abnormal(first)
        if reason:
            raise RuntimeError(reason)
        if ssh("taishan", f"test -e {REMOTE} && echo exists || echo absent").strip() != "absent":
            raise RuntimeError("Remote trial directory already exists")
        ssh("taishan", f"mkdir {REMOTE}")
        for name in BUNDLE_FILES:
            # Enforce LF independently of Windows working-tree conversion.
            content = Path(BUNDLE_SOURCES.get(name, HERE / name)).read_bytes().replace(b"\r\n", b"\n")
            staged = RUN / name
            staged.write_bytes(content)
            result = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(staged),
                                     "taishan:" + REMOTE + "/" + name],
                                    capture_output=True, timeout=40)
            if result.returncode:
                raise RuntimeError("Upload failed: " + result.stderr.decode(errors="replace"))
        ssh("taishan", f"touch {REMOTE}/heartbeat")
        status("launching_remote")
        launched = True
        resume_flag = " --resume-stopped-baseline" if args.resume_stopped_baseline else ""
        remote_pid = ssh("taishan", f"nohup python3 -u {REMOTE}/remote_controller.py{resume_flag} > {REMOTE}/controller.stdout.log 2> {REMOTE}/controller.stderr.log < /dev/null & echo $!").strip()
        atomic(RUN / "launch.json", dict(at=now(), remote_pid=remote_pid, remote=REMOTE))
        deadline = time.monotonic() + MAX_SECONDS + 180
        while time.monotonic() < deadline:
            try:
                reading = sample()
                count += 1
                with (RUN / "thermal.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(reading) + "\n")
                reason = abnormal(reading)
                if reason:
                    emergency_stop(reason)
                    status("aborted_thermal_or_telemetry", reason=reason, samples=count)
                    return 1
                ssh("taishan", f"touch {REMOTE}/heartbeat")
                raw = ssh("taishan", f"cat {REMOTE}/status.json")
                remote_status = json.loads(raw)
                atomic(RUN / "remote-status.json", remote_status)
                misses = 0
                status("running", samples=count, remote_status=remote_status, last=reading)
                if remote_status["phase"] in ("complete", "failed", "aborted_thermal_or_telemetry"):
                    retrieval_error = None
                    try:
                        collect()
                    except Exception as exc:
                        retrieval_error = str(exc)
                    status(remote_status["phase"], samples=count, remote_status=remote_status,
                           results_downloaded=retrieval_error is None,
                           retrieval_error=retrieval_error)
                    return 0 if remote_status["phase"] == "complete" else 1
            except Exception as exc:
                misses += 1
                with (RUN / "thermal.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(dict(at=now(), error=str(exc), misses=misses)) + "\n")
                if misses >= 2:
                    raise RuntimeError(f"Two telemetry/controller reads failed: {exc}") from exc
            time.sleep(5)
        raise TimeoutError("Remote controller exceeded its configured time limit plus grace period")
    except Exception as exc:
        if launched:
            try:
                emergency_stop(str(exc))
            except Exception as stop_error:
                print(f"Emergency stop could not reach host: {stop_error}", flush=True)
        status("failed", error=str(exc), samples=count)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
