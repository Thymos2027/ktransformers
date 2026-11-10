"""Power off taishan through iBMC after the NPU thermal alarm."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent / "runs" / "thermal-shutdown"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def status(phase: str, **extra: object) -> None:
    value = {"phase": phase, "at": now(), **extra}
    temp = ROOT / "status.json.tmp"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, ROOT / "status.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def ibmc(command: str, answer: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                             "taishan-ibmc", command], input=answer,
                            capture_output=True, timeout=45)
    with (ROOT / "commands.log").open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] {command} exit={result.returncode}\n")
        log.write(result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace") + "\n")
    return result


def power() -> str:
    result = ibmc("ipmcget -d powerstate")
    if result.returncode:
        raise RuntimeError("iBMC power state query failed")
    output = result.stdout.decode(errors="replace")
    if "Power state   : On" in output:
        return "On"
    if "Power state   : Off" in output:
        return "Off"
    raise RuntimeError("unrecognized iBMC power state")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=False)
    try:
        current = power()
        if current == "Off":
            status("complete", power="Off", already_off=True)
            return 0
        status("powering_off", power="On")
        result = ibmc("ipmcset -d powerstate -v 0", answer=b"Y\n")
        if result.returncode or b"normal power off successfully" not in result.stdout:
            raise RuntimeError("iBMC did not accept normal power off")
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if power() == "Off":
                status("complete", power="Off")
                return 0
            time.sleep(10)
        raise TimeoutError("taishan did not reach Off within 6 minutes")
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
