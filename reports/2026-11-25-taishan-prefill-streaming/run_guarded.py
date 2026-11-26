"""Reuse the sustained-temperature observer and prior result retrieval."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERFORMANCE = HERE.parent/'2026-11-25-taishan-performance'
PREVIOUS = HERE.parent/'2026-11-24-taishan-tp2-expert64'
SUPPORT = HERE.parent/'2026-11-10-taishan-tp2'
ATTENTION = HERE.parent/'2026-11-24-taishan-attn-tp1-ep2'
spec = importlib.util.spec_from_file_location('performance_launcher',PERFORMANCE/'run_guarded.py')
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)
guard = performance.guard
guard.HERE = HERE
guard.PHASES = ('environment','baseline','stream','bundle')
guard.BUNDLE = ('prepare_candidate.py','launch.sh','remote_controller.py','prefill_gate.py',
    'kernel_probe.py','workload.py','test_prefill.py','performance_support.py','resume_performance.py',
    'startup_gate.py','graph_gate.py','tp_runtime.py','base_runtime.py','thermal_policy.py',
    'test_thermal_policy.py','transport_policy.py','environment_probe.py','requests.json',
    'frozen-workloads.json','experiment-plan.md','PUBLIC_BENCHMARKS.md')
guard.SOURCES = {name:PREVIOUS/name for name in ('tp_runtime.py','thermal_policy.py',
    'test_thermal_policy.py','transport_policy.py','environment_probe.py')}
guard.SOURCES.update({name:PERFORMANCE/name for name in ('resume_performance.py','startup_gate.py')})
guard.SOURCES.update({'performance_support.py':PERFORMANCE/'remote_controller.py',
    'base_runtime.py':SUPPORT/'remote_controller.py','graph_gate.py':ATTENTION/'graph_gate.py',
    'requests.json':HERE.parent/'2026-11-11-taishan-tp-speed/requests.json',
    'frozen-workloads.json':PERFORMANCE/'runs/performance-20261125-113036-profiler-resume/remote/frozen-workloads.json'})


if __name__ == '__main__':
    raise SystemExit(guard.main())
