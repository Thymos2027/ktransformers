"""Wait for the restored original service, verify it, then run the 32-thread A-E experiment."""

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
STATE_DIR = HERE / "runs" / "retry-controller"
BASE = "http://192.168.137.11:18080"
MODEL = "/workspace/models/DeepSeek-V4-Flash-W8A8"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def status(phase: str, **extra: object) -> None:
    value = {"phase": phase, "at": now(), **extra}
    temp = STATE_DIR / "status.json.tmp"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, STATE_DIR / "status.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def ssh(command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                           "taishan", command], capture_output=True, text=True, timeout=timeout)


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=False)
    status("waiting_for_original_service")
    session = requests.Session()
    session.trust_env = False
    try:
        deadline = time.monotonic() + 3 * 3600
        while time.monotonic() < deadline:
            try:
                result = ssh("sudo -n docker inspect -f '{{.State.Status}}' dsv4-kt-20261104")
            except subprocess.TimeoutExpired:
                time.sleep(20)
                continue
            if result.returncode or result.stdout.strip() != "running":
                raise RuntimeError(f"original container stopped: {result.stdout} {result.stderr}")
            try:
                if session.get(BASE + "/health", timeout=5).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(20)
        else:
            raise TimeoutError("original service was not healthy within 3 hours")

        status("original_healthy_smoke_running")
        response = session.post(BASE + "/v1/chat/completions",
                                json={"model": MODEL,
                                      "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
                                      "temperature": 0, "max_tokens": 32},
                                timeout=(15, 300))
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not content or not payload.get("usage"):
            raise RuntimeError("original smoke request missing content or usage")
        (STATE_DIR / "original_smoke.json").write_text(
            json.dumps({"at": now(), "content": content, "usage": payload["usage"]},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        check = ssh("npu-smi info", timeout=45)
        if check.returncode or "No running processes found in NPU 5" not in check.stdout:
            raise RuntimeError("NPU 5 is not idle after original service recovery")
        (STATE_DIR / "npu-smi-before.txt").write_text(check.stdout, encoding="utf-8")
        if (HERE / "runs" / "second").exists():
            raise RuntimeError("runs/second already exists")
        status("experiment_running", output="runs/second")
        with (STATE_DIR / "experiment.stdout.log").open("wb") as stdout, \
             (STATE_DIR / "experiment.stderr.log").open("wb") as stderr:
            result = subprocess.run([sys.executable, "-u", str(HERE / "diagnose.py"),
                                     "--output", str(HERE / "runs" / "second")],
                                    stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError(f"diagnose.py failed with exit {result.returncode}")
        experiment_status = json.loads((HERE / "runs" / "second" / "status.json").read_text(encoding="utf-8"))
        if experiment_status.get("phase") != "complete":
            raise RuntimeError(f"experiment ended without verified cleanup: {experiment_status}")
        status("complete", output="runs/second")
        return 0
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
