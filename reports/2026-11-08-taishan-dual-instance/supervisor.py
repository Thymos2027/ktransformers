"""Bounded second NPU server lifetime inside the existing taishan container."""

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


ROOT = Path("/workspace/dual-instance-20261108")
MAX_SECONDS = 3600
STOP = False


def stamp() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def write_status(phase: str, **details: object) -> None:
    payload = {"phase": phase, "at": stamp(), "supervisor_pid": os.getpid(), **details}
    temp = ROOT / "status.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, ROOT / "status.json")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def on_stop(_signal: int, _frame: object) -> None:
    global STOP
    STOP = True


def child_setup() -> None:
    # Linux sends SIGTERM to the server if this supervisor disappears unexpectedly.
    libc = ctypes.CDLL(None)
    libc.prctl(1, signal.SIGTERM)


def health() -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:18081/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGINT)
    try:
        server.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    write_status("starting", npu_device=1, port=18081)
    deadline = time.monotonic() + MAX_SECONDS
    launcher = os.environ["DSV4_TOOLS"] + "/serve.sh"
    server = None
    final_phase = "failed"
    error = None
    try:
        with (ROOT / "logs" / "server.log").open("ab") as log:
            server = subprocess.Popen(["bash", launcher, "--foreground"],
                                      env=os.environ.copy(), stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True,
                                      preexec_fn=child_setup)
            write_status("loading", server_pid=server.pid)
            startup_deadline = time.monotonic() + 2400
            while not STOP and time.monotonic() < startup_deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {server.returncode}")
                if health():
                    break
                time.sleep(10)
            else:
                if not STOP:
                    raise TimeoutError("second server did not become healthy in 40 minutes")
            if STOP:
                final_phase = "stopped_before_ready"
                return 0
            write_status("ready", server_pid=server.pid, health=True)
            while not STOP and time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited unexpectedly: {server.returncode}")
                if (ROOT / "stop.request").exists():
                    break
                time.sleep(5)
            final_phase = "stopped" if time.monotonic() < deadline else "timeout_stopped"
            return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server is not None:
            stop_server(server)
        write_status(final_phase, server_pid=server.pid if server else None,
                     server_stopped=server is None or server.poll() is not None,
                     error=error)


if __name__ == "__main__":
    raise SystemExit(main())
