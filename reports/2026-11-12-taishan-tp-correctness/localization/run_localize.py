"""Start the guarded two-path MoE layer localization in the background."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent.parent / "2026-11-10-taishan-tp2"


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("tp_guard", SUPPORT / "run_guarded.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.HERE = HERE
    guard.MAX_SECONDS = 3 * 3600
    guard.CONTAINER_PHASES = ("tp1_eager", "tp2_eager")
    guard.BUNDLE_FILES = (
        "prepare_candidate.py", "gate_candidate.py",
        "probe_processor_fragment.py.txt", "launch.sh",
        "remote_controller.py", "tp_runtime.py", "probe_client.py",
        "cases.json", "scope.json",
    )
    guard.BUNDLE_SOURCES = {"tp_runtime.py": SUPPORT / "remote_controller.py"}
    raise SystemExit(guard.main())
