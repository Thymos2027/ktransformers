"""Remote detached TP2 trial, device watchdog, and original-service restoration."""
from __future__ import annotations

import datetime as dt
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parent
ORIGINAL = "dsv4-kt-20261104"
SOURCE = Path("/data/deploy/dsv4-20261104/sources/sglang")
COMMIT = "ff46d660d16cbf11ef393c80cc6b3ab7f6259780"
RUN_ID = ROOT.name.removeprefix("dsv4-tp2-")
IMAGE = "dsv4-tp2-trial:" + RUN_ID
NAMES = {mode: f"dsv4-tp2-{RUN_ID}-{mode}" for mode in ("hccl", "eager", "graph")}
LOGS = ROOT / "logs"
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ALARM = threading.Event()
DONE = threading.Event()
DEADLINE = time.monotonic() + 2 * 3600


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def status(phase, **fields):
    value = dict(phase=phase, at=now(), pid=os.getpid(), **fields)
    atomic(ROOT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(args, timeout=60, check=True):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    with (ROOT / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(at=now(), args=args, exit=result.returncode,
                                     stdout=result.stdout, stderr=result.stderr)) + "\n")
    if check and result.returncode:
        raise RuntimeError(f"{args[:4]} exited {result.returncode}: {result.stderr[-1200:]}")
    return result.stdout.strip()


def docker(*args, **kwargs):
    return run(["sudo", "-n", "docker", *args], **kwargs)


def safe():
    if ALARM.is_set() or (ROOT / "abort.json").exists():
        raise RuntimeError("Thermal/telemetry stop; original service will remain stopped")
    if time.monotonic() > DEADLINE:
        raise TimeoutError("TP2 workflow exceeded two hours")


def stop_all(include_original=False):
    names = list(NAMES.values()) + ([ORIGINAL] if include_original else [])
    for name in names:
        try:
            docker("stop", "-t", "3", name, timeout=25, check=False)
        except Exception as exc:
            print(f"stop {name}: {exc}", flush=True)


def settled_stopped(container, seconds=90):
    """Wait through delayed containerd exit events before any restart/snapshot."""
    end = time.monotonic() + seconds
    stable = 0
    while time.monotonic() < end:
        safe()
        state = json.loads(docker("inspect", "-f", "{{json .State}}", container))
        stopped = (not state["Running"] and state["Pid"] == 0
                   and state["Status"] == "exited" and not state["Dead"])
        stable = stable + 1 if stopped else 0
        if stable >= 3:
            atomic(ROOT / (container + "-stopped.json"), dict(at=now(), state=state))
            return
        time.sleep(3)
    raise TimeoutError(f"{container} did not reach a stable exited state")


def stop_original():
    # A nonzero docker stop can precede an asynchronous successful task deletion.
    docker("stop", "-t", "90", ORIGINAL, timeout=125, check=False)
    settled_stopped(ORIGINAL)


def read_cards():
    raw = run(["npu-smi", "info"], timeout=12)
    cards = {}
    for m in re.finditer(r"\|\s*(\d+)\s+910B4-1\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s+(\d+)\s+", raw):
        key, health, power, temp = m.groups()
        cards[key] = dict(health=health, power_w=float(power), temperature_c=int(temp))
    if set(cards) != {"2", "5"}:
        raise RuntimeError(f"Missing device: {cards}")
    if any(c["health"] != "OK" or c["temperature_c"] >= 70 for c in cards.values()):
        raise RuntimeError(f"NPU guard triggered: {cards}")
    return cards


def watchdog():
    misses = 0
    while not DONE.is_set():
        try:
            cards = read_cards()
            if (ROOT / "abort.json").exists():
                raise RuntimeError("Local fan/telemetry guard requested stop")
            heartbeat = ROOT / "heartbeat"
            if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 90:
                raise RuntimeError("Local fan-monitor heartbeat stale for 90 seconds")
            misses = 0
            with (ROOT / "thermal.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(at=now(), cards=cards)) + "\n")
        except Exception as exc:
            misses += 1
            with (ROOT / "thermal.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(at=now(), error=str(exc), misses=misses)) + "\n")
            # Explicit abnormal readings/abort stop immediately; command failures allow one retry.
            if misses >= 2 or "guard triggered" in str(exc) or "requested stop" in str(exc):
                ALARM.set()
                atomic(ROOT / "alarm.json", dict(at=now(), reason=str(exc)))
                stop_all(include_original=True)
                return
        DONE.wait(3)


def request(base, path, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else None


def wait_health(base, container, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        safe()
        if docker("inspect", "-f", "{{.State.Running}}", container) != "true":
            raise RuntimeError(f"{container} stopped during startup; see logs")
        try:
            request(base, "/health", timeout=3)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"{container} did not become healthy in {seconds} seconds")


def capture_logs(mode):
    with (LOGS / f"{mode}.log").open("wb") as stream:
        subprocess.run(["sudo", "-n", "docker", "logs", NAMES[mode]],
                       stdout=stream, stderr=subprocess.STDOUT, timeout=45)


def create_container(mode):
    safe()
    args = ["run", "-d", "--name", NAMES[mode], "--restart", "no",
            "--network", "host", "--ipc", "host", "--privileged",
            "--security-opt", "seccomp=unconfined", "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864", "--volumes-from", ORIGINAL + ":ro",
            "-v", str(ROOT) + ":/experiment:rw", "-w", "/experiment"]
    for device in ("davinci2", "davinci5", "davinci_manager", "devmm_svm", "hisi_hdc"):
        args += ["--device", "/dev/" + device]
    args += ["--entrypoint", "/bin/bash", IMAGE, "/experiment/launch.sh", mode]
    result = docker(*args, timeout=60)
    safe()
    return result


def smoke(base, mode):
    model = request(base, "/v1/models")["data"][0]["id"]
    results = []
    for name, prompt, expected in (
        ("capital", "法国的首都是哪里？只回答城市名称。", "巴黎"),
        ("arithmetic", "计算 17 + 25。只输出数字答案。", "42"),
        ("sse", "Reply with only the word READY.", "READY"),
    ):
        safe()
        payload = dict(model=model, messages=[dict(role="user", content=prompt)],
                       temperature=0, max_tokens=64, stream=True,
                       stream_options=dict(include_usage=True))
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic()
        parts, usage, finish, done, first = [], None, None, False, None
        with HTTP.open(req, timeout=300) as response:
            for line in response:
                safe()
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                item = json.loads(raw)
                usage = item.get("usage") or usage
                for choice in item.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    content = (choice.get("delta") or {}).get("content") or ""
                    if content:
                        if first is None:
                            first = time.monotonic() - start
                        parts.append(content)
        content = "".join(parts)
        passed = expected in content and done and bool(usage) and finish == "stop"
        results.append(dict(name=name, content=content, passed=passed, usage=usage,
                            finish_reason=finish, done=done, ttft_seconds=first,
                            seconds=time.monotonic() - start))
        atomic(ROOT / f"{mode}-smoke.json", results)
        if not passed:
            raise RuntimeError(f"{mode} smoke failed: {name}: {content!r}")
    return results


def prepare():
    status("preparing_source")
    prefix = ["git", "-c", f"safe.directory={SOURCE}", "-C", str(SOURCE)]
    if run(prefix + ["rev-parse", "HEAD"]) != COMMIT:
        raise RuntimeError("Baseline SGLang commit changed")
    if run(prefix + ["diff", "--name-only"], timeout=120):
        raise RuntimeError("Baseline SGLang has tracked edits")
    target = ROOT / "sglang"
    target.mkdir(exist_ok=False)
    archive = ROOT / "sglang-source.tar"
    run(prefix + ["archive", "--format=tar", "--output=" + str(archive), COMMIT], timeout=300)
    run(["tar", "-xf", str(archive), "-C", str(target)], timeout=300)
    run([sys.executable, str(ROOT / "prepare_candidate.py"), str(target)])
    run(["bash", "-n", str(ROOT / "launch.sh")])
    atomic(ROOT / "source.json", dict(base_commit=COMMIT, baseline_source=str(SOURCE),
                                      isolated_source=str(target)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-stopped-baseline", action="store_true")
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    original_was_running = False
    original_stop_attempted = False
    failure = None
    restored = False
    guard = None
    status("preflight")
    try:
        original_was_running = docker("inspect", "-f", "{{.State.Running}}", ORIGINAL) == "true"
        if not original_was_running and not args.resume_stopped_baseline:
            raise RuntimeError("Original baseline is not running; do not change it")
        if not original_was_running:
            settled_stopped(ORIGINAL)
            original_stop_attempted = True
        existing = set(docker("ps", "-a", "--format", "{{.Names}}").splitlines())
        if existing.intersection(NAMES.values()):
            raise RuntimeError("Trial containers already exist; preserve prior run")
        read_cards()
        if time.time() - (ROOT / "heartbeat").stat().st_mtime > 45:
            raise RuntimeError("Fan-monitor heartbeat not fresh")
        guard = threading.Thread(target=watchdog, daemon=True)
        guard.start()
        prepare()
        safe()
        status("stopping_original")
        original_stop_attempted = True
        if original_was_running:
            stop_original()
        else:
            settled_stopped(ORIGINAL)
        safe()
        status("snapshotting_environment")
        image_id = docker("commit", "--pause=false", ORIGINAL, IMAGE, timeout=900)
        atomic(ROOT / "environment.json", dict(snapshot_image=IMAGE, image_id=image_id,
                                               original_container=ORIGINAL, at=now()))
        status("hccl_probe")
        create_container("hccl")
        try:
            code = docker("wait", NAMES["hccl"], timeout=300)
            if code != "0":
                raise RuntimeError(f"HCCL probe exited {code}")
            for rank in (0, 1):
                value = json.loads((ROOT / f"hccl-rank{rank}.json").read_text())
                if value.get("passed") is not True:
                    raise RuntimeError(f"HCCL rank {rank} failed")
        finally:
            docker("stop", "-t", "3", NAMES["hccl"], check=False)
            capture_logs("hccl")
        for mode in ("eager", "graph"):
            safe()
            status("loading_" + mode)
            create_container(mode)
            try:
                wait_health("http://127.0.0.1:18082", NAMES[mode], 1800)
                status("validating_" + mode)
                capture_logs(mode)
                log = (LOGS / f"{mode}.log").read_text(errors="replace")
                for rank in (0, 1):
                    if not re.search(r"\[KT TP2 EXPERIMENT\].*rank=" + str(rank), log):
                        raise RuntimeError(f"Missing KT TP rank {rank} diagnostics")
                if mode == "graph" and "Capture target decode NPU graph end" not in log:
                    raise RuntimeError("Graph capture success marker absent")
                atomic(ROOT / f"{mode}-npu.json", dict(at=now(), raw=run(["npu-smi", "info"])))
                smoke("http://127.0.0.1:18082", mode)
            finally:
                docker("stop", "-t", "30", NAMES[mode], timeout=60, check=False)
                capture_logs(mode)
        status("trial_passed_smoke_only")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        atomic(ROOT / "failure.json", dict(at=now(), error=failure))
        print(failure, flush=True)
    finally:
        stop_all()
        if original_stop_attempted and not ALARM.is_set() and not (ROOT / "abort.json").exists():
            try:
                safe()
                read_cards()
                status("restoring_original", trial_error=failure)
                settled_stopped(ORIGINAL)
                docker("start", ORIGINAL, timeout=60)
                wait_health("http://192.168.137.11:18080", ORIGINAL, 1800)
                smoke("http://192.168.137.11:18080", "restored-original")
                read_cards()
                restored = True
            except Exception as exc:
                failure = (failure or "") + f"; original restoration: {exc}"
                try:
                    docker("stop", "-t", "10", ORIGINAL, timeout=35, check=False)
                except Exception as stop_error:
                    failure += f"; restoration cleanup: {stop_error}"
        elif not original_stop_attempted:
            restored = original_was_running and not ALARM.is_set()
        DONE.set()
        if guard is not None:
            guard.join(timeout=90)
        if ALARM.is_set() or (ROOT / "abort.json").exists():
            stop_all(include_original=True)
            status("aborted_thermal_or_telemetry", error=failure, original_restored=False)
        elif failure:
            status("failed", error=failure, original_restored=restored)
        else:
            status("complete", eager_smoke=True, graph_smoke=True, original_restored=restored,
                   scope="short functional trial only; performance and numeric equivalence unverified")
    return 1 if failure or ALARM.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
