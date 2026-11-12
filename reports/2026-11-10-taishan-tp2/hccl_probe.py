"""Two local NPU ranks: validate small HCCL sums before loading weights."""
import datetime
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401


def worker(rank):
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method="tcp://127.0.0.1:29577", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=120))
    records = []
    try:
        for count in (1, 1024, 262144):
            value = torch.full((count,), rank + 1, dtype=torch.float32, device=f"npu:{rank}")
            torch.npu.synchronize()
            started = time.monotonic()
            dist.all_reduce(value)
            torch.npu.synchronize()
            elapsed = time.monotonic() - started
            if not torch.equal(value.cpu(), torch.full((count,), 3.0)):
                raise RuntimeError(f"HCCL sum mismatch on rank {rank}")
            records.append({"count": count, "seconds": elapsed, "passed": True})
        Path(f"/experiment/hccl-rank{rank}.json").write_text(json.dumps({
            "rank": rank, "world_size": 2, "checks": records, "passed": True,
        }, indent=2) + "\n")
        print(json.dumps({"rank": rank, "passed": True, "checks": records}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=2, join=True)
