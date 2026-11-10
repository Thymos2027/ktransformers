"""Stage the one-versus-two-instance diagnostic with host-side telemetry."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import requests


HERE = Path(__file__).resolve().parent
REMOTE = "/data/deploy/dsv4-20261104/dual-instance-diagnosis-20261108"
CONTAINER = "dsv4-kt-20261104"
ORIGINAL = "http://127.0.0.1:18180"
SECOND = "http://127.0.0.1:18181"
MODEL = "/workspace/models/DeepSeek-V4-Flash-W8A8"
REFERENCE = HERE.parent / "2026-11-08-taishan-single-card-baseline" / "runs" / "first"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_json(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def ssh(command: str, *, content: bytes | None = None, timeout: int = 60,
        check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "taishan", command],
        input=content, capture_output=True, timeout=timeout,
    )
    if check and result.returncode:
        raise RuntimeError(f"ssh exit {result.returncode}: "
                           f"{result.stderr.decode(errors='replace')[:500]}")
    return result


def status(output: Path, phase: str, **details: object) -> None:
    value = {"phase": phase, "at": now(), **details}
    atomic_json(output / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def remote_status() -> dict | None:
    result = ssh(f"cat {REMOTE}/status.json", check=False, timeout=20)
    return json.loads(result.stdout) if result.returncode == 0 else None


def health(base: str) -> bool:
    session = requests.Session()
    session.trust_env = False
    try:
        return session.get(base + "/health", timeout=5).status_code == 200
    except Exception:
        return False
    finally:
        session.close()


def host_sample(numa: bool = False) -> dict:
    command = f"sudo -n python3 {REMOTE}/sample_host.py"
    if numa:
        command += " --numa"
    result = ssh(command, timeout=60 if numa else 20)
    return json.loads(result.stdout)


class Sampler:
    def __init__(self, output: Path, stage: str) -> None:
        self.output = output
        self.stage = stage
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"sample-{stage}", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                value = host_sample()
                value["stage"] = self.stage
            except Exception as exc:
                value = {"stage": self.stage, "at": now(), "error": repr(exc)}
            append_json(self.output / "samples.jsonl", value)
            self.stop_event.wait(5)

    def __enter__(self) -> "Sampler":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=25)


def request(base: str, prompt: str, barrier: threading.Barrier | None = None) -> dict:
    session = requests.Session()
    session.trust_env = False
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 64, "stream": True,
               "stream_options": {"include_usage": True}}
    if barrier is not None:
        barrier.wait(timeout=20)
    started_at = now()
    begin = time.monotonic()
    first = last = None
    parts: list[str] = []
    usage = None
    finish_reason = None
    done = False
    try:
        with session.post(base + "/v1/chat/completions", json=payload,
                          stream=True, timeout=(15, 900)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                if line[6:] == b"[DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    text = (choice.get("delta") or {}).get("content") or ""
                    if text:
                        instant = time.monotonic()
                        first = first or instant
                        last = instant
                        parts.append(text)
        end = time.monotonic()
        body = "".join(parts)
        completion = (usage or {}).get("completion_tokens")
        decode = ((completion - 1) / (last - first)
                  if completion and completion > 1 and first and last and last > first else None)
        return {"base": base, "started_at": started_at, "done_marker": done,
                "finish_reason": finish_reason, "usage": usage, "content": body,
                "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "ttft_seconds": None if first is None else first - begin,
                "total_seconds": end - begin, "decode_tokens_per_second": decode}
    finally:
        session.close()


def run_once(bases: tuple[str, ...], prompt: str) -> tuple[list[dict], float]:
    if len(bases) == 1:
        begin = time.monotonic()
        result = request(bases[0], prompt)
        return [result], time.monotonic() - begin
    barrier = threading.Barrier(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(request, base, prompt, barrier) for base in bases]
        barrier.wait(timeout=20)
        begin = time.monotonic()
        results = [task.result(timeout=900) for task in tasks]
        return results, time.monotonic() - begin


def stage(output: Path, name: str, bases: tuple[str, ...], prompt: str,
          reference_hash: str | None) -> tuple[list[dict], str]:
    status(output, name + "_running")
    before = host_sample(numa=True)
    append_json(output / "snapshots.jsonl", {"stage": name, "edge": "before", "sample": before})
    results = []
    with Sampler(output, name):
        for index in range(2):
            responses, wall = run_once(bases, prompt)
            if reference_hash is None:
                reference_hash = responses[0]["content_sha256"]
            comparable = all(x["done_marker"] and x["usage"] and x["content_sha256"] == reference_hash
                             for x in responses)
            record = {"stage": name, "iteration": index, "kind": "warmup" if index == 0 else "formal",
                      "at": now(), "wall_seconds": wall, "comparable": comparable,
                      "requests": responses}
            append_json(output / "requests.jsonl", record)
            results.append(record)
            print(f"{name} {index}: {wall:.2f}s, "
                  f"decode={[round(x['decode_tokens_per_second'], 3) if x['decode_tokens_per_second'] else None for x in responses]}, "
                  f"comparable={comparable}", flush=True)
            if not all(x["done_marker"] and x["usage"] for x in responses):
                raise RuntimeError(f"{name} request missing SSE end or usage")
    after = host_sample(numa=True)
    append_json(output / "snapshots.jsonl", {"stage": name, "edge": "after", "sample": after})
    return results, reference_hash


def stop_second() -> dict:
    ssh(f"touch {REMOTE}/stop.request")
    for _ in range(36):
        state = remote_status()
        if state and state.get("server_stopped"):
            return state
        time.sleep(5)
    state = remote_status()
    if state and state.get("supervisor_pid"):
        ssh(f"sudo -n docker exec {CONTAINER} kill -TERM {int(state['supervisor_pid'])}", check=False)
    return remote_status() or {}


def summary(records: list[dict]) -> dict:
    result = {}
    for item in records:
        if item["kind"] != "formal":
            continue
        result[item["stage"]] = {
            "wall_seconds": item["wall_seconds"], "comparable": item["comparable"],
            "requests": [{"base": x["base"], "ttft_seconds": x["ttft_seconds"],
                          "total_seconds": x["total_seconds"],
                          "decode_tokens_per_second": x["decode_tokens_per_second"],
                          "completion_tokens": (x["usage"] or {}).get("completion_tokens"),
                          "finish_reason": x["finish_reason"]} for x in item["requests"]],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tunnel = None
    remote_started = False
    second_stopped = False
    records: list[dict] = []
    status(output, "preflight", started_at=now())
    try:
        prompt = json.loads((REFERENCE / "requests.json").read_text(encoding="utf-8"))[0]["prompt"]
        atomic_json(output / "request.json", {"prompt": prompt, "max_tokens": 64,
                                               "temperature": 0})
        check = ssh("sudo -n docker ps --filter name=^dsv4-kt-20261104$ --format '{{.Names}}'; "
                    "ss -lnt | grep ':18081' || true; npu-smi info | tail -12")
        atomic_json(output / "preflight.json", {"at": now(), "stdout": check.stdout.decode()})
        if "dsv4-kt-20261104" not in check.stdout.decode():
            raise RuntimeError("original container not running")
        if ":18081" in check.stdout.decode():
            raise RuntimeError("port 18081 occupied")
        if "No running processes found in NPU 5" not in check.stdout.decode():
            raise RuntimeError("NPU 5 not idle")
        if ssh(f"test -e {REMOTE}", check=False).returncode == 0:
            raise RuntimeError("remote experiment directory already exists")
        ssh(f"mkdir -p {REMOTE}/logs")
        ssh(f"cat > {REMOTE}/sample_host.py", content=(HERE / "sample_host.py").read_bytes())
        container_ip = ssh("sudo -n docker inspect -f "
                           "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "
                           "dsv4-kt-20261104").stdout.decode().strip()
        if not container_ip.startswith("172."):
            raise RuntimeError(f"unexpected container IP: {container_ip}")
        with (output / "tunnel.log").open("wb") as log:
            tunnel = subprocess.Popen([
                "ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-L", f"127.0.0.1:18180:{container_ip}:18080",
                "-L", f"127.0.0.1:18181:{container_ip}:18081", "taishan",
            ], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for _ in range(20):
                if tunnel.poll() is not None:
                    raise RuntimeError("SSH tunnel exited")
                if health(ORIGINAL):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("original service unavailable through tunnel")
            baseline, reference_hash = stage(output, "A_original_only", (ORIGINAL,), prompt, None)
            records.extend(baseline)
            status(output, "loading_second")
            for filename in ("run.sh", "supervisor.py"):
                ssh(f"cat > {REMOTE}/{filename}", content=(HERE / filename).read_bytes())
            ssh(f"sudo -n docker exec -d {CONTAINER} /bin/bash "
                "/workspace/dual-instance-diagnosis-20261108/run.sh")
            remote_started = True
            deadline = time.monotonic() + 2500
            while time.monotonic() < deadline:
                state = remote_status()
                if state and state.get("phase") == "ready" and health(SECOND):
                    break
                if state and state.get("phase") in ("failed", "stopped_before_ready"):
                    raise RuntimeError(f"second server failed: {state}")
                time.sleep(10)
            else:
                raise TimeoutError("second server not healthy by startup deadline")
            graph = ssh(f"grep -F 'Capture target decode NPU graph end' "
                        f"{REMOTE}/logs/server.log | tail -1", check=False)
            if graph.returncode or not graph.stdout.strip():
                raise RuntimeError("second server graph capture completion absent")
            ready_snapshot = host_sample(numa=True)
            append_json(output / "snapshots.jsonl", {"stage": "second_ready", "edge": "after",
                                                        "sample": ready_snapshot})
            if ready_snapshot["meminfo"]["MemAvailable"] < 64 * 1024 * 1024:
                raise RuntimeError("less than 64 GiB MemAvailable after second load")
            atomic_json(output / "second_ready.json", {"status": state,
                         "graph_log": graph.stdout.decode(errors="replace")})
            for name, bases in (
                ("B_original_second_idle", (ORIGINAL,)),
                ("C_second_original_idle", (SECOND,)),
                ("D_both_active", (ORIGINAL, SECOND)),
            ):
                observed, _ = stage(output, name, bases, prompt, reference_hash)
                records.extend(observed)
            status(output, "stopping_second")
            second_state = stop_second()
            second_stopped = bool(second_state.get("server_stopped"))
            atomic_json(output / "second_stopped.json", second_state)
            if not second_stopped:
                raise RuntimeError("second server cleanup not confirmed")
            observed, _ = stage(output, "E_original_after_stop", (ORIGINAL,), prompt,
                                reference_hash)
            records.extend(observed)
            atomic_json(output / "summary.json", summary(records))
            status(output, "benchmarks_complete", formal_stages=5)
            return 0
    except Exception as exc:
        status(output, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup: dict[str, object] = {"at": now(), "remote_started": remote_started}
        if remote_started:
            try:
                state = remote_status()
                if not state or not state.get("server_stopped"):
                    state = stop_second()
                cleanup["second_server_status"] = state
                cleanup["second_stopped"] = bool(state and state.get("server_stopped"))
            except Exception as exc:
                cleanup["stop_error"] = repr(exc)
        cleanup["original_healthy"] = health("http://192.168.137.11:18080")
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        atomic_json(output / "cleanup.json", cleanup)
        current = json.loads((output / "status.json").read_text(encoding="utf-8"))
        if current["phase"] == "benchmarks_complete":
            status(output, "complete" if cleanup["original_healthy"] and
                   cleanup.get("second_stopped") else "cleanup_incomplete", cleanup=cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
