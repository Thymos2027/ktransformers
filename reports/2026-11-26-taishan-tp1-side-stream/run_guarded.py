"""Use the existing detached cooling observer with this scoped TP1 bundle."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
PREVIOUS = REPORTS / '2026-11-24-taishan-tp2-expert64'
PREFILL = REPORTS / '2026-11-25-taishan-prefill-streaming'
SUPPORT = REPORTS / '2026-11-10-taishan-tp2'
sys.path.insert(1, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('tp1_side_guard', PREVIOUS / 'run_guarded.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.HERE = HERE
guard.PHASES = ('environment', 'h0', 'h1', 's0', 's1')
guard.BUNDLE = ('remote_controller.py', 'runtime_support.py', 'launch.sh', 'prepare_candidate.py',
                'prefill_gate.py', 'test_experiment.py', 'kernel_probe.py', 'environment_probe.py',
                'workload.py', 'base_runtime.py', 'tp_runtime.py', 'thermal_policy.py',
                'test_thermal_policy.py', 'transport_policy.py', 'requests.json',
                'frozen-workloads.json', 'weight-reference.json', 'experiment-plan.md')
guard.SOURCES = {name: PREVIOUS/name for name in ('environment_probe.py', 'tp_runtime.py',
                  'thermal_policy.py', 'test_thermal_policy.py', 'transport_policy.py')}
guard.SOURCES.update({name: PREFILL/name for name in ('kernel_probe.py', 'workload.py')})
guard.SOURCES['base_runtime.py'] = SUPPORT/'remote_controller.py'
guard.SOURCES['requests.json'] = REPORTS/'2026-11-11-taishan-tp-speed/requests.json'
guard.SOURCES['frozen-workloads.json'] = (
    REPORTS/'2026-11-25-taishan-performance/runs/performance-20261125-113036-profiler-resume'
    /'remote/frozen-workloads.json')
guard.SOURCES['weight-reference.json'] = (
    PREFILL/'runs/prefill-20261125-224217-first/remote/weight-preflight.json')

if __name__ == '__main__':
    raise SystemExit(guard.main())
