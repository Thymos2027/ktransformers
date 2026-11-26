"""Reuse the detached cooling observer; upload a scoped single-device bundle."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '2026-11-24-taishan-tp2-expert64'
PREFILL = HERE.parent / '2026-11-25-taishan-prefill-streaming'
SUPPORT = HERE.parent / '2026-11-10-taishan-tp2'
sys.path.insert(1, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('tutorial_guard', PREVIOUS/'run_guarded.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.HERE = HERE
guard.PHASES = ('environment', 'table12', 'table19', 'table39')
guard.BUNDLE = ('remote_controller.py', 'launch.sh', 'benchmark.py', 'test_benchmark.py',
    'tutorial_gate.py', 'startup-reference.txt', 'capacity-reference.txt',
    'prepare_candidate.py', 'kernel_probe.py', 'environment_probe.py', 'workload.py',
    'prefill_gate.py', 'base_runtime.py', 'tp_runtime.py', 'thermal_policy.py',
    'test_thermal_policy.py', 'transport_policy.py', 'sources.json', 'tutorial.md',
    'decode_throughput_test.upstream.sh', 'experiment-plan.md')
guard.SOURCES = {n: PREVIOUS/n for n in ('environment_probe.py', 'tp_runtime.py',
    'thermal_policy.py', 'test_thermal_policy.py', 'transport_policy.py')}
guard.SOURCES.update({n: PREFILL/n for n in ('prepare_candidate.py', 'kernel_probe.py',
    'workload.py', 'prefill_gate.py')})
guard.SOURCES['base_runtime.py'] = SUPPORT/'remote_controller.py'

if __name__ == '__main__':
    if '--hardware-long' in sys.argv:
        sys.argv.remove('--hardware-long')
        guard.PHASES=('environment','hardwarelong')
        guard.BUNDLE=(*guard.BUNDLE,'runtime-profile.json','hardware-plan.md')
        guard.SOURCES['runtime-profile.json']=HERE/'hardware-profile.json'
    raise SystemExit(guard.main())
