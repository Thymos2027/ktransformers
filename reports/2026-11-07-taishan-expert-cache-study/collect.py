"""Run an isolated trace server on taishan's second 910B and replay its routes.

Launch inside the existing container after sourcing /workspace/dsv4.env. The
production server on port 18080 and physical NPU 2 is never stopped or changed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path("/workspace/expert-cache-study-20261107")
PORT = 18081
URL = f"http://127.0.0.1:{PORT}"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPTS = [
    "请用三点解释数据库索引如何影响查询延迟。",
    "Write a Python function that merges two sorted lists and explain its complexity.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h? Explain.",
    "Summarize the difference between optimistic and pessimistic locking in four sentences.",
    "请分析一台双路服务器上 NUMA 亲和性可能影响内存带宽的原因。",
    "Create a small SQL query that groups orders by month and counts them.",
    "Explain why a cache can improve average latency but worsen a tail latency spike.",
    "用一个生活中的例子解释混合专家模型的路由器如何工作。",
    "Design a checklist for verifying a large file download with SHA-256 and resumable transfers.",
    "If 7 workers process 21 tasks evenly, how many tasks does each worker receive?",
    "What are the key tradeoffs between quantizing model weights to 4 bits and 8 bits?",
    "请给出一个 Rust Result 错误处理的简短示例并解释每一步。",
    "Describe how a streaming HTTP client detects the end of a server-sent event response.",
    "Write a JavaScript function that counts the frequencies of words in a sentence.",
    "为什么测试推理系统时要区分首 token 延迟和生成吞吐？",
    "Explain the difference between a logical expert ID and a physical GPU slot.",
    "A cache holds 32 of 256 experts. If 20 of 80 accesses hit, what is the hit rate?",
    "请比较按访问频率、最近访问时间和预测路由来替换缓存的优缺点。",
    "Describe a safe rollout plan for changing a model serving configuration.",
    "Write a Python snippet that parses newline-delimited JSON and skips invalid lines.",
]


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status(phase: str, **details):
    payload = dict(phase=phase, time=stamp(), pid=os.getpid(), **details)
    atomic_json(ROOT / "status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def request(path: str, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw and raw[:1] in (b"{", b"[") else raw.decode()


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "traces").mkdir(exist_ok=True)
    status("starting", requests_planned=len(PROMPTS), npu_logical_device=1, port=PORT)
    env = dict(os.environ)
    env.update(DSV4_NPU_DEVICE_ID="1", DSV4_PORT=str(PORT),
               DSV4_HOST="127.0.0.1", DSV4_LOG_DIR=str(ROOT / "logs"),
               DSV4_EXTRA_FLAGS="--expert-distribution-recorder-mode per_pass",
               SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=str(ROOT / "traces"),
               SGLANG_SET_CPU_AFFINITY="0")
    launcher = env["DSV4_TOOLS"] + "/serve.sh"
    proc = None
    try:
        with (ROOT / "logs" / "server.log").open("ab") as log:
            proc = subprocess.Popen(["bash", launcher, "--foreground"], env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            status("loading", server_pid=proc.pid)
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Trace server exited during startup: {proc.returncode}")
                try:
                    request("/health", timeout=5)
                    break
                except Exception:
                    time.sleep(10)
            else:
                raise TimeoutError("Trace server did not become healthy within 40 minutes")
            status("recording", server_pid=proc.pid, completed=0)
            request("/start_expert_distribution_record", body={})
            model = request("/v1/models")["data"][0]["id"]
            completions = []
            for index, prompt in enumerate(PROMPTS):
                answer = request("/v1/chat/completions", body={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 96,
                }, timeout=900)
                usage = answer.get("usage") or {}
                completions.append(dict(index=index,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=answer["choices"][0].get("finish_reason")))
                atomic_json(ROOT / "requests.json", completions)
                status("recording", server_pid=proc.pid, completed=len(completions))
            request("/stop_expert_distribution_record", body={})
            request("/dump_expert_distribution_record", body={}, timeout=300)
            traces = sorted((ROOT / "traces").glob("expert_distribution_recorder_*.pt"),
                            key=lambda p: p.stat().st_mtime)
            if not traces:
                raise RuntimeError("Recorder returned successfully but no trace file appeared")
            status("replaying", trace=str(traces[-1]), completed=len(completions))
            with (ROOT / "logs" / "replay.log").open("wb") as replay_log:
                subprocess.run(["python3", str(ROOT / "replay.py"), str(traces[-1]),
                                "--output", str(ROOT / "replay.json")],
                               stdout=replay_log, stderr=subprocess.STDOUT, check=True)
            status("collected_and_replayed", trace=str(traces[-1]),
                   replay=str(ROOT / "replay.json"), completed=len(completions))
    except Exception as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if proc is not None:
            stop(proc)
            if (ROOT / "status.json").exists():
                current = json.loads((ROOT / "status.json").read_text())
                if current["phase"] == "collected_and_replayed":
                    status("complete", trace=current["trace"], replay=current["replay"],
                           completed=current["completed"], server_stopped=True)


if __name__ == "__main__":
    main()
