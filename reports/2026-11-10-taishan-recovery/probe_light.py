"""Run one bounded NPU compute probe with local, independent telemetry."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from monitor_idle import read_ssh, sample


RUN = Path(__file__).resolve().parent / "runs" / "post-power-on-light-npu2-retry"
RUN.mkdir(parents=True, exist_ok=True)
STATUS = RUN / "status.json"
SAMPLES = RUN / "samples.jsonl"
CONTAINER = "kt-thermal-probe-npu2-retry-20261110"
IMAGE = "quay.io/ascend/sglang@sha256:5ed0acf504181fe092f9badbd8f77611b985ec65df0f93b66ab4b9616231f70b"
WORK_SECONDS = 30
MAX_RUN_SECONDS = 75
MAX_TEMP_C = 70  # Conservative experiment stop point, not a hardware limit.
MIN_FAN_RPM = 3000

WORKLOAD = f"""import time
import torch
import torch_npu

assert torch.npu.device_count() == 1, 'Expected exactly one visible NPU'
torch.npu.set_device(0)
x = torch.randn((1024, 1024), dtype=torch.float16, device='npu:0')
y = torch.randn((1024, 1024), dtype=torch.float16, device='npu:0')
torch.npu.synchronize()
print('ready', flush=True)
start = time.monotonic()
iterations = 0
while time.monotonic() - start < {WORK_SECONDS}:
    output = x.mm(y)
    torch.npu.synchronize()
    iterations += 1
    time.sleep(0.25)
print(f'iterations={{iterations}} checksum={{float(output[0, 0].cpu())}}', flush=True)
"""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(**fields: object) -> None:
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"at": now(), **fields}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def stop_container() -> None:
    try:
        read_ssh("taishan", f"sudo -n docker stop -t 2 {CONTAINER}")
    except Exception as exc:
        with (RUN / "cleanup.log").open("a", encoding="utf-8") as output:
            output.write(f"{now()} docker stop failed: {exc}\n")


def guarded_sample() -> dict[str, object]:
    result = sample()
    cards = result["cards"]
    sensors = result["sensors"]
    if any(card["temperature_c"] >= MAX_TEMP_C for card in cards.values()):
        raise RuntimeError(f"Temperature reached {MAX_TEMP_C} C experiment stop point: {cards}")
    if any(sensors[f"FAN{i} Speed"] < MIN_FAN_RPM for i in range(1, 5)):
        raise RuntimeError(f"Fan fell below {MIN_FAN_RPM} RPM: {sensors}")
    return result


def main() -> int:
    process: subprocess.Popen[str] | None = None
    count = 0
    last: dict[str, object] | None = None
    write_status(phase="preflight", samples=0, container=CONTAINER)
    try:
        mode = read_ssh("taishan-ibmc", "ipmcget -d faninfo")
        if "Current mode: auto" not in mode:
            raise RuntimeError(f"Unexpected fan mode: {mode.strip()}")
        baseline = guarded_sample()
        if any(card["temperature_c"] >= 60 for card in baseline["cards"].values()):
            raise RuntimeError(f"Baseline temperature not cool enough: {baseline['cards']}")
        if read_ssh("taishan", "sudo -n docker ps --format '{{.Names}}'").strip():
            raise RuntimeError("Another container is running; refusing the isolated probe")

        command = (
            "sudo -n docker run --rm -i --name " + CONTAINER
            + " --network none --ipc=host --stop-timeout 2"
            + " --device=/dev/davinci2 --device=/dev/davinci_manager"
            + " --device=/dev/devmm_svm --device=/dev/hisi_hdc"
            + " --volume=/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
            + " --volume=/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro"
            + " --volume=/usr/local/dcmi:/usr/local/dcmi:ro"
            + " --volume=/usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro"
            + " --security-opt seccomp=unconfined"
            + " -e ASCEND_RT_VISIBLE_DEVICES=0"
            + " --entrypoint python " + IMAGE + " -u -"
        )
        with (RUN / "workload.stdout.log").open("w", encoding="utf-8") as stdout, (RUN / "workload.stderr.log").open("w", encoding="utf-8") as stderr, SAMPLES.open("w", encoding="utf-8") as samples:
            process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "taishan", command],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            assert process.stdin is not None
            process.stdin.write(WORKLOAD)
            process.stdin.close()
            deadline = time.monotonic() + MAX_RUN_SECONDS
            write_status(phase="running", samples=0, container=CONTAINER, baseline=baseline)
            while True:
                last = guarded_sample()
                samples.write(json.dumps(last, ensure_ascii=False) + "\n")
                samples.flush()
                count += 1
                write_status(phase="running", samples=count, container=CONTAINER, last=last)
                if process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Workload exceeded {MAX_RUN_SECONDS} seconds")
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError(f"Workload exited {process.returncode}; see workload.stderr.log")
        write_status(phase="complete", samples=count, container=CONTAINER, last=last)
        return 0
    except Exception as exc:
        write_status(phase="stopping", samples=count, container=CONTAINER, error=str(exc), last=last)
        if process is not None:
            stop_container()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
        write_status(phase="failed", samples=count, container=CONTAINER, error=str(exc), last=last)
        return 1
    finally:
        if process is not None and process.poll() is None:
            stop_container()
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
