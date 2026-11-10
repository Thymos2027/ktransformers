"""Resume the model experiment with an independent thermal stop loop."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "guarded-thread-budget"
RUN.mkdir(parents=True, exist_ok=False)
STATUS = RUN / "status.json"
SAMPLES = RUN / "thermal.jsonl"
CONTAINER = "dsv4-kt-20261104"
CONTROLLER = HERE.parent / "2026-11-08-taishan-dual-instance-thread-budget" / "retry_controller.py"
EXPECTED = {"2", "5"}
TRIP_TEMP_C = 70  # Conservative test stop point, not a device specification.
TRIP_FAN_RPM = 4500
MAX_SECONDS = 3 * 3600


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def status(phase: str, **fields: object) -> None:
    path = STATUS.with_suffix(".json.tmp")
    path.write_text(json.dumps({"phase": phase, "at": now(), **fields}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(path, STATUS)


def ssh(alias: str, command: str, timeout: int = 15) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, command],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{alias} exit {result.returncode}: {(result.stderr or result.stdout).strip()[:300]}")
    return result.stdout


def read_sample() -> dict[str, object]:
    raw = ssh("taishan", "npu-smi info", timeout=15)
    cards = {}
    for match in re.finditer(
        r"\|\s*(\d+)\s+910B4-1\s*\|\s*(OK|Warning|Alarm|Critical|UNKNOWN)\s*\|\s*([\d.]+)\s+(\d+)\s+",
        raw,
    ):
        card_id, health, power, temperature = match.groups()
        cards[card_id] = {"health": health, "power_w": float(power), "temperature_c": int(temperature)}
    raw = ssh("taishan-ibmc", "ipmcget -t sensor -d list", timeout=15)
    fans = {}
    for line in raw.splitlines():
        match = re.match(r"\s*0x[0-9a-f]+\s*\|\s*(FAN[1-4] Speed)\s*\|\s*([\d.]+)", line)
        if match:
            fans[match.group(1)] = float(match.group(2))
    if set(cards) != EXPECTED or len(fans) != 4:
        raise RuntimeError(f"Incomplete telemetry: cards={sorted(cards)}, fans={sorted(fans)}")
    return {"at": now(), "cards": cards, "fans": fans}


def abnormal(sample: dict[str, object]) -> str | None:
    cards = sample["cards"]
    fans = sample["fans"]
    if any(card["health"] != "OK" for card in cards.values()):
        return f"NPU health abnormal: {cards}"
    if any(card["temperature_c"] >= TRIP_TEMP_C for card in cards.values()):
        return f"NPU reached {TRIP_TEMP_C} C test limit: {cards}"
    if min(fans.values()) < TRIP_FAN_RPM:
        return f"Fan below {TRIP_FAN_RPM} RPM: {fans}"
    return None


def stop_model(reason: str) -> None:
    status("stopping_model", reason=reason)
    with (RUN / "emergency-stop.log").open("a", encoding="utf-8") as output:
        for command in (
            "sudo -n docker stop -t 2 " + CONTAINER,
            "sudo -n docker inspect -f '{{.State.Status}}' " + CONTAINER,
        ):
            try:
                result = ssh("taishan", command, timeout=35)
                output.write(f"{now()} {command}: {result.strip()}\n")
            except Exception as exc:
                output.write(f"{now()} {command}: {exc}\n")
            output.flush()


def main() -> int:
    controller: subprocess.Popen[bytes] | None = None
    stop_event = threading.Event()
    alarm: list[str] = []
    misses = 0
    count = 0
    model_started = False
    watcher: threading.Thread | None = None
    deadline = time.monotonic() + MAX_SECONDS

    def watch() -> None:
        nonlocal misses, count
        with SAMPLES.open("a", encoding="utf-8") as output:
            while not stop_event.is_set():
                try:
                    result = read_sample()
                    misses = 0
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    count += 1
                    reason = abnormal(result)
                    status("running", samples=count, last=result)
                    if reason:
                        alarm.append(reason)
                        stop_model(reason)
                        return
                except Exception as exc:
                    misses += 1
                    output.write(json.dumps({"at": now(), "error": str(exc), "consecutive_misses": misses}) + "\n")
                    output.flush()
                    if misses >= 2:
                        reason = f"NPU or fan telemetry unavailable twice: {exc}"
                        alarm.append(reason)
                        stop_model(reason)
                        return
                stop_event.wait(3)

    status("preflight")
    try:
        if "Current mode: auto" not in ssh("taishan-ibmc", "ipmcget -d faninfo"):
            raise RuntimeError("iBMC fan mode is not auto")
        first = read_sample()
        reason = abnormal(first)
        if reason:
            raise RuntimeError(f"Preflight failed: {reason}")
        if ssh("taishan", "sudo -n docker ps --format '{{.Names}}'").strip():
            raise RuntimeError("Other Docker containers are already running")
        if ssh("taishan", "sudo -n docker inspect -f '{{.State.Status}} {{.HostConfig.RestartPolicy.Name}}' " + CONTAINER).strip() != "exited no":
            raise RuntimeError("Original container is not stopped with restart policy no")
        if (CONTROLLER.parent / "runs" / "retry-thermal-guard").exists():
            raise RuntimeError("Retry output directory already exists")
        if (CONTROLLER.parent / "runs" / "second-thermal-guard").exists():
            raise RuntimeError("Experiment output directory already exists")
        if ssh("taishan", "test -e /data/deploy/dsv4-20261104/dual-instance-thread-budget-20261110 && echo exists || echo absent").strip() != "absent":
            raise RuntimeError("Remote experiment directory already exists")
        watcher = threading.Thread(target=watch, name="thermal-watch", daemon=True)
        watcher.start()
        for _ in range(15):
            if count >= 1 or alarm:
                break
            time.sleep(1)
        if count < 1 or alarm:
            raise RuntimeError(f"Thermal watcher did not start: {alarm}")
        status("starting_original", samples=count)
        model_started = True
        ssh("taishan", "sudo -n docker start " + CONTAINER, timeout=40)
        if alarm:
            raise RuntimeError(alarm[0])
        with (RUN / "controller.stdout.log").open("wb") as stdout, (RUN / "controller.stderr.log").open("wb") as stderr:
            controller = subprocess.Popen([os.sys.executable, "-u", str(CONTROLLER)], stdout=stdout, stderr=stderr)
            while controller.poll() is None:
                if alarm:
                    controller.terminate()
                    raise RuntimeError(alarm[0])
                if time.monotonic() >= deadline:
                    stop_model("Guarded experiment exceeded 3 hours")
                    controller.terminate()
                    raise RuntimeError("Experiment timed out")
                time.sleep(2)
            if controller.returncode != 0:
                raise RuntimeError(f"Retry controller exited {controller.returncode}")
        if alarm:
            raise RuntimeError(alarm[0])
        stop_event.set()
        watcher.join(timeout=25)
        if watcher.is_alive():
            raise RuntimeError("Thermal watcher did not exit")
        status("complete", samples=count)
        return 0
    except Exception as exc:
        if model_started:
            stop_model(f"Guarded controller failed: {exc}")
        status("failed", error=str(exc), samples=count, alarm=alarm)
        return 1
    finally:
        stop_event.set()
        if controller and controller.poll() is None:
            controller.terminate()
        if watcher and watcher.is_alive():
            watcher.join(timeout=25)


if __name__ == "__main__":
    raise SystemExit(main())
