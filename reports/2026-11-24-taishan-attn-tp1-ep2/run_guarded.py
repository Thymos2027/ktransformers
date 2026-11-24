"""Launch attention TP1/DP2 and MoE EP2 with the existing cooling observer."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-24-taishan-tp2-expert64'
SUPPORT=HERE.parent/'2026-11-10-taishan-tp2'
SPEED=HERE.parent/'2026-11-11-taishan-tp-speed'
sys.path.insert(1,str(PREVIOUS))
spec=importlib.util.spec_from_file_location('guard_launcher',PREVIOUS/'run_guarded.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.HERE=HERE
guard.PHASES=('environment','hccl','ep2eager','ep2graph')
guard.BUNDLE=('run_guarded.py','guard_launcher.py','prepare_candidate.py','ep_prepare.py',
    'remote_controller.py','tp2_controller.py','tp_runtime.py','base_runtime.py',
    'thermal_policy.py','transport_policy.py','resume_checkpoint.py','workload.py',
    'hccl_probe.py','requests.json','environment_probe.py','ep_layout.py','device_probe.py',
    'graph_gate.py','request_budget_probe.py','combine_probe.py','test_combine.py',
    'test_ep2.py','test_attention.py','test_thermal_policy.py','launch.sh','experiment-plan.md')
guard.SOURCES={name:PREVIOUS/name for name in ('tp_runtime.py','thermal_policy.py','transport_policy.py',
    'resume_checkpoint.py','environment_probe.py','test_thermal_policy.py')}
guard.SOURCES.update({'guard_launcher.py':PREVIOUS/'run_guarded.py',
    'tp2_controller.py':PREVIOUS/'remote_controller.py','base_runtime.py':SUPPORT/'remote_controller.py',
    'workload.py':SPEED/'workload.py','requests.json':SPEED/'requests.json'})
if __name__=='__main__':
    if '--diagnose-format' in sys.argv:
        sys.argv.remove('--diagnose-format')
        guard.BUNDLE+=('attention_controller.py','diagnostic_controller.py','diagnostic_workload.py','test_diagnostic.py')
        guard.SOURCES.update({'remote_controller.py':HERE/'diagnostic_controller.py',
            'attention_controller.py':HERE/'remote_controller.py','launch.sh':HERE/'launch_diagnostic.sh'})
    raise SystemExit(guard.main())
