"""Record a short, read-only post-thermal power-on idle check."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


RUN = Path(__file__).resolve().parent / "runs" / "post-power-on-idle"
RUN.mkdir(parents=True, exist_ok=True)
STATUS = RUN / "status.json"
SAMPLES = RUN / "samples.jsonl"
INTERVAL_SECONDS = 10
DURATION_SECONDS = 300


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(**fields: object) -> None:
    payload = {"at": now(), **fields}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def read_ssh(alias: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", alias, command],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{alias}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def sample() -> dict[str, object]:
    npu_output = read_ssh("taishan", "npu-smi info")
    sensor_output = read_ssh("taishan-ibmc", "ipmcget -t sensor -d list")

    cards: dict[str, dict[str, object]] = {}
    for match in re.finditer(
        r"\|\s*(\d+)\s+910B4-1\s*\|\s*(OK|Warning|Alarm|Critical|UNKNOWN)\s*\|\s*([\d.]+)\s+(\d+)\s+",
        npu_output,
    ):
        card_id, health, power, temperature = match.groups()
        cards[card_id] = {
            "health": health,
            "power_w": float(power),
            "temperature_c": int(temperature),
        }

    sensors: dict[str, float] = {}
    for line in sensor_output.splitlines():
        match = re.match(r"\s*0x[0-9a-f]+\s*\|\s*(Inlet Temp|Outlet Temp|FAN[1-4] Speed)\s*\|\s*([\d.]+)", line)
        if match:
            sensors[match.group(1)] = float(match.group(2))

    if set(cards) != {"2", "5"}:
        raise RuntimeError(f"Expected NPU 2 and 5, found {sorted(cards)}: {npu_output[-500:]}")
    if len(sensors) != 6:
        raise RuntimeError(f"Expected inlet, outlet and four fan speeds, found {sorted(sensors)}")
    if any(card["health"] != "OK" for card in cards.values()):
        raise RuntimeError(f"NPU health changed: {cards}")
    if any(card["temperature_c"] >= 80 for card in cards.values()):
        raise RuntimeError(f"Idle NPU temperature reached conservative check limit: {cards}")
    return {"at": now(), "cards": cards, "sensors": sensors}


def main() -> int:
    deadline = time.monotonic() + DURATION_SECONDS
    count = 0
    write_status(phase="running", samples=count, duration_seconds=DURATION_SECONDS)
    try:
        with SAMPLES.open("a", encoding="utf-8") as output:
            while True:
                result = sample()
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
                write_status(phase="running", samples=count, last=result)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(INTERVAL_SECONDS, remaining))
        write_status(phase="complete", samples=count, last=result)
        return 0
    except Exception as exc:
        write_status(phase="failed", samples=count, error=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
