"""Compare two concurrent requests on one 910B versus two 910B instances."""

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
REMOTE = "/data/deploy/dsv4-20261104/dual-instance-20261108"
CONTAINER = "dsv4-kt-20261104"
CONTAINER_IP = "172.17.0.2"
BASE_A = "http://127.0.0.1:18180"
BASE_B = "http://127.0.0.1:18181"
MODEL = "/workspace/models/DeepSeek-V4-Flash-W8A8"
REFERENCE = HERE.parent / "2026-11-08-taishan-single-card-baseline" / "runs" / "first"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_json(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def ssh(command: str, *, input_bytes: bytes | None = None,
        timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "taishan", command],
        input=input_bytes, capture_output=True, timeout=timeout,
    )
    if check and result.returncode:
        raise RuntimeError(f"ssh command failed ({result.returncode}): "
                           f"{result.stderr.decode(errors='replace')[:500]}")
    return result


def status(output: Path, phase: str, **details: object) -> None:
    payload = {"phase": phase, "at": now(), **details}
    atomic_json(output / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def resource(output: Path, stage: str) -> None:
    command = ("date -Is; free -b; npu-smi info; "
               "sudo -n docker stats --no-stream --format "
               "'{{.Name}} {{.CPUPerc}} {{.MemUsage}}' dsv4-kt-20261104")
    try:
        result = ssh(command, timeout=45, check=False)
        payload = {"at": now(), "stage": stage, "exit_code": result.returncode,
                   "stdout": result.stdout.decode(errors="replace"),
                   "stderr": result.stderr.decode(errors="replace")}
    except Exception as exc:
        payload = {"at": now(), "stage": stage, "error": repr(exc)}
    append_json(output / "resources.jsonl", payload)


def server_status() -> dict | None:
    result = ssh(f"cat {REMOTE}/status.json", check=False, timeout=20)
    if result.returncode:
        return None
    return json.loads(result.stdout)


def health(base: str) -> bool:
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.get(base + "/health", timeout=5)
        session.close()
        return response.status_code == 200
    except Exception:
        return False


def stream_request(base: str, prompt: str, barrier: threading.Barrier) -> dict:
    session = requests.Session()
    session.trust_env = False
    barrier.wait(timeout=20)
    start = time.monotonic()
    first = last = None
    parts: list[str] = []
    usage = None
    finish_reason = None
    done = False
    payload = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 256, "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        with session.post(base + "/v1/chat/completions", json=payload,
                          stream=True, timeout=(15, 900)) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                if raw[6:] == b"[DONE]":
                    done = True
                    break
                event = json.loads(raw[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    content = (choice.get("delta") or {}).get("content") or ""
                    if content:
                        stamp = time.monotonic()
                        if first is None:
                            first = stamp
                        last = stamp
                        parts.append(content)
        end = time.monotonic()
        output = "".join(parts)
        return {
            "base": base, "at": now(), "ttft_seconds": None if first is None else first - start,
            "total_seconds": end - start, "last_content_seconds": None if last is None else last - start,
            "done_marker": done, "usage": usage, "finish_reason": finish_reason,
            "content_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "content": output,
        }
    finally:
        session.close()


def pair(output: Path, stage: str, iteration: int, bases: tuple[str, str],
         prompt: str, reference_hash: str) -> dict:
    barrier = threading.Barrier(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(stream_request, base, prompt, barrier) for base in bases]
        barrier.wait(timeout=20)
        begin = time.monotonic()
        results = [task.result(timeout=900) for task in tasks]
        wall = time.monotonic() - begin
    valid = all(x["done_marker"] and x["usage"] and x["finish_reason"] == "stop"
                and x["content_sha256"] == reference_hash for x in results)
    tokens = sum((x["usage"] or {}).get("completion_tokens", 0) for x in results)
    record = {"stage": stage, "iteration": iteration, "at": now(), "bases": bases,
              "pair_wall_seconds": wall, "completion_tokens": tokens,
              "aggregate_tokens_per_second": tokens / wall, "comparable": valid,
              "requests": results}
    append_json(output / "pairs.jsonl", record)
    print(f"{stage} pair {iteration}: {wall:.2f}s, {tokens} tokens, "
          f"{tokens/wall:.2f} tok/s, comparable={valid}", flush=True)
    if not all(x["done_marker"] and x["usage"] for x in results):
        raise RuntimeError(f"{stage} pair {iteration} missing SSE end or usage")
    return record


def summarize(records: list[dict]) -> dict:
    summary = {}
    for stage in ("single", "dual"):
        group = [x for x in records if x["stage"] == stage and x["iteration"] > 0]
        rates = [x["aggregate_tokens_per_second"] for x in group]
        summary[stage] = {
            "formal_pairs": len(group), "comparable_pairs": sum(x["comparable"] for x in group),
            "aggregate_tokens_per_second": {"median": statistics.median(rates), "values": rates},
            "pair_wall_seconds": {"median": statistics.median(x["pair_wall_seconds"] for x in group),
                                  "values": [x["pair_wall_seconds"] for x in group]},
            "request_total_seconds": [x["total_seconds"] for pair in group for x in pair["requests"]],
            "request_ttft_seconds": [x["ttft_seconds"] for pair in group for x in pair["requests"]],
        }
    if summary["single"]["comparable_pairs"] == 3 and summary["dual"]["comparable_pairs"] == 3:
        summary["dual_to_single_rate_ratio"] = (
            summary["dual"]["aggregate_tokens_per_second"]["median"] /
            summary["single"]["aggregate_tokens_per_second"]["median"]
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tunnel = None
    remote_started = False
    records: list[dict] = []
    status(output, "preflight", started_at=now())
    try:
        case = json.loads((REFERENCE / "requests.json").read_text(encoding="utf-8"))[0]
        prompt = case["prompt"]
        reference_events = [json.loads(line) for line in
                            (REFERENCE / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        reference_text = next(x["content"] for x in reference_events
                              if x.get("kind") == "formal" and x.get("target_tokens") == 128)
        reference_hash = hashlib.sha256(reference_text.encode()).hexdigest()
        atomic_json(output / "request.json", {"prompt": prompt,
                    "prompt_tokens": case["prompt_tokens"], "reference_sha256": reference_hash})
        check = ssh("sudo -n docker ps --filter name=^dsv4-kt-20261104$ --format '{{.Names}}'; "
                    "ss -lnt | grep ':18081' || true; npu-smi info | tail -12")
        atomic_json(output / "preflight.json", {"at": now(), "stdout": check.stdout.decode()})
        if "dsv4-kt-20261104" not in check.stdout.decode():
            raise RuntimeError("original container is not running")
        if ":18081" in check.stdout.decode():
            raise RuntimeError("port 18081 is already listening")
        if "No running processes found in NPU 5" not in check.stdout.decode():
            raise RuntimeError("NPU 5 is not idle")
        if ssh(f"test -e {REMOTE}", check=False).returncode == 0:
            raise RuntimeError("remote experiment directory already exists")
        with (output / "tunnel.log").open("wb") as log:
            tunnel = subprocess.Popen([
                "ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-L", f"127.0.0.1:18180:{CONTAINER_IP}:18080",
                "-L", f"127.0.0.1:18181:{CONTAINER_IP}:18081", "taishan",
            ], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for _ in range(20):
                if tunnel.poll() is not None:
                    raise RuntimeError(f"SSH tunnel exited: {tunnel.returncode}")
                if health(BASE_A):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("original service not healthy through tunnel")
            resource(output, "before_single")
            status(output, "single_running")
            for index in range(4):
                records.append(pair(output, "single", index, (BASE_A, BASE_A),
                                    prompt, reference_hash))
            resource(output, "after_single")
            status(output, "deploying_second")
            ssh(f"mkdir -p {REMOTE}/logs")
            for filename in ("run.sh", "supervisor.py"):
                ssh(f"cat > {REMOTE}/{filename}", input_bytes=(HERE / filename).read_bytes())
            ssh(f"sudo -n docker exec -d {CONTAINER} /bin/bash "
                "/workspace/dual-instance-20261108/run.sh")
            remote_started = True
            status(output, "loading_second", remote_dir=REMOTE)
            deadline = time.monotonic() + 2500
            while time.monotonic() < deadline:
                state = server_status()
                if state and state.get("phase") == "ready" and health(BASE_B):
                    break
                if state and state.get("phase") in ("failed", "stopped_before_ready"):
                    raise RuntimeError(f"second server failed: {state}")
                time.sleep(10)
            else:
                raise TimeoutError("second server not healthy by startup deadline")
            graph = ssh(f"grep -F 'Capture target decode NPU graph end' "
                        f"{REMOTE}/logs/server.log | tail -1", check=False)
            if graph.returncode:
                raise RuntimeError("second server is healthy but graph capture completion is absent")
            atomic_json(output / "second_ready.json", {"at": now(), "status": state,
                         "graph_log": graph.stdout.decode(errors="replace")})
            resource(output, "before_dual")
            status(output, "dual_running")
            for index in range(4):
                records.append(pair(output, "dual", index, (BASE_A, BASE_B),
                                    prompt, reference_hash))
            resource(output, "after_dual")
            atomic_json(output / "summary.json", summarize(records))
            status(output, "benchmarks_complete", formal_pairs=6)
            return 0
    except Exception as exc:
        status(output, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup: dict[str, object] = {"at": now(), "remote_started": remote_started}
        if remote_started:
            try:
                ssh(f"touch {REMOTE}/stop.request")
                for _ in range(24):
                    state = server_status()
                    if state and state.get("server_stopped"):
                        cleanup["second_server_status"] = state
                        break
                    time.sleep(5)
                else:
                    state = server_status()
                    if state and state.get("supervisor_pid"):
                        ssh(f"sudo -n docker exec {CONTAINER} kill -TERM "
                            f"{int(state['supervisor_pid'])}", check=False)
                    cleanup["second_server_status"] = server_status()
            except Exception as exc:
                cleanup["stop_error"] = repr(exc)
        try:
            cleanup["original_healthy"] = health("http://192.168.137.11:18080")
        except Exception as exc:
            cleanup["health_error"] = repr(exc)
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        atomic_json(output / "cleanup.json", cleanup)
        if (output / "status.json").exists():
            current = json.loads((output / "status.json").read_text(encoding="utf-8"))
            if current.get("phase") == "benchmarks_complete":
                status(output, "complete" if cleanup.get("original_healthy") and
                       (not remote_started or cleanup.get("second_server_status", {}).get("server_stopped"))
                       else "cleanup_incomplete", cleanup=cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
