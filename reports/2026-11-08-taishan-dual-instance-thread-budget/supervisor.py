"""Keep the diagnostic second server bounded and stop it on request."""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


ROOT = Path("/workspace/dual-instance-thread-budget-20261110")
STOP = False


def stamp() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def status(phase: str, **details: object) -> None:
    payload = {"phase": phase, "at": stamp(), "supervisor_pid": os.getpid(), **details}
    temporary = ROOT / "status.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, ROOT / "status.json")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def stop_signal(_number: int, _frame: object) -> None:
    global STOP
    STOP = True


def child_setup() -> None:
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)


def healthy() -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:18081/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    for sig, limit in ((signal.SIGINT, 90), (signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        if server.poll() is not None:
            break
        os.killpg(server.pid, sig)
        try:
            server.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    status("starting", npu_device=1, port=18081)
    deadline = time.monotonic() + 3600
    server = None
    phase = "failed"
    error = None
    try:
        with (ROOT / "logs" / "server.log").open("ab") as log:
            server = subprocess.Popen(["bash", os.environ["DSV4_TOOLS"] + "/serve.sh",
                                       "--foreground"], env=os.environ.copy(), stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True,
                                      preexec_fn=child_setup)
            status("loading", server_pid=server.pid)
            loading_deadline = time.monotonic() + 2400
            while not STOP and time.monotonic() < loading_deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"second server exited: {server.returncode}")
                if healthy():
                    break
                time.sleep(10)
            else:
                if not STOP:
                    raise TimeoutError("second server did not become healthy in 40 minutes")
            if STOP:
                phase = "stopped_before_ready"
                return 0
            status("ready", server_pid=server.pid, health=True)
            while not STOP and time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"second server exited unexpectedly: {server.returncode}")
                if (ROOT / "stop.request").exists():
                    break
                time.sleep(5)
            phase = "stopped" if time.monotonic() < deadline else "timeout_stopped"
            return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server is not None:
            stop_server(server)
        status(phase, server_pid=server.pid if server else None,
               server_stopped=server is None or server.poll() is not None, error=error)


if __name__ == "__main__":
    raise SystemExit(main())
