"""Reuse the verified two-sided thermal guard with a frozen benchmark bundle."""
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "2026-11-10-taishan-tp2"


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("tp_guard", SUPPORT / "run_guarded.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.HERE = HERE
    guard.MAX_SECONDS = 3 * 3600 + 1800
    guard.CONTAINER_PHASES = ("tp1a", "tp2a", "tp2b", "tp1b")
    guard.BUNDLE_FILES = ("prepare_candidate.py", "launch.sh", "remote_controller.py",
                          "tp_runtime.py", "workload.py", "requests.json")
    guard.BUNDLE_SOURCES = {
        "prepare_candidate.py": SUPPORT / "prepare_candidate.py",
        "tp_runtime.py": SUPPORT / "remote_controller.py",
    }
    raise SystemExit(guard.main())
