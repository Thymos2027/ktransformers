"""Wait for the first controller's verified restoration, then run the fixed benchmark."""
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
RUN = HERE / "runs" / "retry-graph-check-controller"
PREVIOUS = HERE / "runs" / "speed-20261111-first" / "status.json"
NEXT_ID = "speed-20261111-graph-check"


def status(phase, **fields):
    value = dict(phase=phase, at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                 pid=os.getpid(), next_run=NEXT_ID, **fields)
    temp = RUN / "status.json.tmp"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, RUN / "status.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    status("waiting_for_previous_restoration")
    session = requests.Session()
    session.trust_env = False
    try:
        deadline = time.monotonic() + 2400
        while time.monotonic() < deadline:
            previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
            if previous["phase"] in ("failed", "complete", "aborted_thermal_or_telemetry"):
                remote = previous.get("remote_status", {})
                if previous["phase"] != "failed" or not remote.get("original_restored"):
                    raise RuntimeError(f"Previous run not safely restored: {previous}")
                if "No decode graph for tp1a rank 0" not in remote.get("error", ""):
                    raise RuntimeError("Unexpected previous failure; do not auto-retry")
                response = session.get("http://192.168.137.11:18080/health", timeout=10)
                response.raise_for_status()
                break
            time.sleep(10)
        else:
            raise TimeoutError("Previous controller did not finish restoration in 40 minutes")
        if (HERE / "runs" / NEXT_ID).exists():
            raise RuntimeError("New run already exists; preserve its evidence")
        status("launching_fixed_benchmark")
        with (RUN / "benchmark.stdout.log").open("wb") as stdout, (RUN / "benchmark.stderr.log").open("wb") as stderr:
            result = subprocess.run([sys.executable, "-u", str(HERE / "run_guarded.py"),
                                     "--run-id", NEXT_ID], cwd=HERE, stdout=stdout, stderr=stderr)
        status("complete" if result.returncode == 0 else "failed", benchmark_exit=result.returncode,
               benchmark_status=str(HERE / "runs" / NEXT_ID / "status.json"))
        return result.returncode
    except Exception as exc:
        status("failed", error=str(exc))
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
