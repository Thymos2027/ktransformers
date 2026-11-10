"""Cold power cycle taishan and restore the original service, with durable status."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "first"
CONTAINER = "dsv4-kt-20261104"
MODEL = "/workspace/models/DeepSeek-V4-Flash-W8A8"


def stamp() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def status(phase: str, **extra: object) -> None:
    payload = {"phase": phase, "at": stamp(), **extra}
    temporary = RUN / "status.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, RUN / "status.json")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def command(alias: str, remote: str, timeout: int = 30,
            answer: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", alias, remote],
        input=answer, capture_output=True, timeout=timeout,
    )
    output = result.stdout.decode(errors="replace")
    error = result.stderr.decode(errors="replace")
    with (RUN / "commands.log").open("a", encoding="utf-8") as log:
        log.write(f"\n[{stamp()}] {alias}: {remote}\nexit={result.returncode}\n{output}{error}\n")
    return result


def ibmc_power() -> str:
    result = command("taishan-ibmc", "ipmcget -d powerstate", timeout=20)
    if result.returncode:
        raise RuntimeError("iBMC power query failed")
    output = result.stdout.decode(errors="replace")
    for state in ("On", "Off"):
        if f"Power state   : {state}" in output:
            return state
    raise RuntimeError(f"unrecognized iBMC power state: {output[-200:]}")


def set_power(value: int) -> None:
    result = command("taishan-ibmc", f"ipmcset -d powerstate -v {value}",
                     timeout=75, answer=b"Y\n")
    if result.returncode:
        raise RuntimeError(f"iBMC power command {value} failed: "
                           f"{result.stdout.decode(errors='replace')[-300:]}")


def wait_for_power(expected: str, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if ibmc_power() == expected:
                return
        except Exception as exc:
            print(f"iBMC status probe: {exc!r}", flush=True)
        time.sleep(10)
    raise TimeoutError(f"iBMC did not report power {expected} within {seconds}s")


def wait_for_host(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = command("taishan", "date -Is", timeout=20)
            if result.returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(15)
    raise TimeoutError("taishan SSH did not return after power on")


def npu_count() -> int:
    result = command("taishan", "lspci -Dnn | grep -c '19e5:d802'", timeout=30)
    try:
        return int(result.stdout.decode().strip())
    except ValueError:
        raise RuntimeError("could not parse NPU PCIe count") from None


def wait_for_npu(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if npu_count() == 2:
            result = command("taishan", "npu-smi info", timeout=35)
            output = result.stdout.decode(errors="replace")
            if result.returncode == 0 and "910B" in output:
                (RUN / "npu-smi.txt").write_text(output, encoding="utf-8")
                return
        time.sleep(20)
    raise TimeoutError(f"two 910B devices not ready; PCIe count={npu_count()}")


def wait_for_health(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    session = requests.Session()
    session.trust_env = False
    try:
        while time.monotonic() < deadline:
            result = command("taishan", f"sudo -n docker inspect -f '{{{{.State.Status}}}}' {CONTAINER}",
                             timeout=25)
            if result.returncode or result.stdout.decode().strip() != "running":
                raise RuntimeError("original container exited before health check")
            try:
                response = session.get("http://192.168.137.11:18080/health", timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(20)
    finally:
        session.close()
    raise TimeoutError("original service did not become healthy within 3 hours")


def smoke_request() -> None:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            "http://192.168.137.11:18080/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
                  "temperature": 0, "max_tokens": 32},
            timeout=(10, 300),
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not content or not payload.get("usage"):
            raise RuntimeError("smoke request missing content or usage")
        (RUN / "smoke.json").write_text(
            json.dumps({"at": stamp(), "content": content, "usage": payload["usage"]},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        session.close()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=False)
    status("preflight")
    power_off_attempted = False
    try:
        if ibmc_power() != "On":
            raise RuntimeError("taishan not powered on at preflight")
        if npu_count() != 0:
            raise RuntimeError("PCIe baseline changed; refusing cold cycle")
        result = command("taishan", "sudo -n docker ps --format '{{.Names}}'", timeout=30)
        if result.returncode or result.stdout.strip():
            raise RuntimeError("Docker has running containers; refusing cold cycle")
        status("powering_off")
        power_off_attempted = True
        set_power(0)  # Normal power off, deactivate FRU.
        wait_for_power("Off", 240)
        status("powered_off")
        time.sleep(30)
        status("powering_on")
        set_power(1)
        wait_for_power("On", 300)
        status("waiting_for_host")
        wait_for_host(1800)
        status("waiting_for_npu")
        wait_for_npu(900)
        status("npu_ready")
        result = command("taishan", "sudo -n systemctl start docker.service && "
                         f"sudo -n docker start {CONTAINER}", timeout=120)
        if result.returncode:
            raise RuntimeError("Docker or original container failed to start")
        status("waiting_for_service")
        wait_for_health(10800)
        status("service_healthy")
        smoke_request()
        status("complete", smoke="passed", container=CONTAINER)
        return 0
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if power_off_attempted:
            try:
                if ibmc_power() == "Off":
                    status("power_on_recovery")
                    set_power(1)
                    wait_for_power("On", 300)
                    status("failed_host_power_restored")
            except Exception as exc:
                print(f"ALERT: failed to confirm power on: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
