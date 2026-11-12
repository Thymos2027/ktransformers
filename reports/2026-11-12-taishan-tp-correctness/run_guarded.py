"""Launch a detached local thermal guard and remote correctness pilot."""
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "2026-11-10-taishan-tp2"


if __name__ == "__main__":
    narrow = "--narrow" in sys.argv
    if narrow:
        sys.argv.remove("--narrow")
    spec = importlib.util.spec_from_file_location("tp_guard", SUPPORT / "run_guarded.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.HERE = HERE
    guard.MAX_SECONDS = 4 * 3600 + 1800
    guard.CONTAINER_PHASES = (("tp1_eager",) if narrow else
                              ("tp1_eager", "tp2_eager", "tp1_graph", "tp2_graph"))
    guard.BUNDLE_FILES = ("prepare_candidate.py", "gate_candidate.py",
                          "probe_processor_fragment.py.txt", "launch.sh",
                          "remote_controller.py", "tp_runtime.py", "probe_client.py",
                          "cases.json", "scope.json")
    guard.BUNDLE_SOURCES = {
        "gate_candidate.py": SUPPORT / "prepare_candidate.py",
        "tp_runtime.py": SUPPORT / "remote_controller.py",
        "scope.json": HERE / ("scope-narrow.json" if narrow else "scope-full.json"),
    }
    raise SystemExit(guard.main())
