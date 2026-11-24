"""Launch isolated EP2 validation using the reviewed cooling/SSH observer."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-24-taishan-tp2-expert64'
SUPPORT=HERE.parent/'2026-11-10-taishan-tp2'
SPEED=HERE.parent/'2026-11-11-taishan-tp-speed'
sys.path.insert(1,str(PREVIOUS))

if (HERE/'guard_launcher.py').exists():
    import guard_launcher as guard
else:
    spec=importlib.util.spec_from_file_location('guard_launcher',PREVIOUS/'run_guarded.py')
    guard=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
guard.HERE=HERE
guard.PHASES=('environment','hccl','ep2eager','ep2graph')
guard.BUNDLE=('run_guarded.py','guard_launcher.py','prepare_candidate.py','remote_controller.py',
    'tp2_controller.py','tp_runtime.py','base_runtime.py','thermal_policy.py','transport_policy.py',
    'resume_checkpoint.py','workload.py','hccl_probe.py','requests.json','environment_probe.py',
    'ep_layout.py','device_probe.py','graph_gate.py','ep_checkpoint.py','test_graph_gate.py',
    'test_ep2.py','test_thermal_policy.py','launch.sh','experiment-plan.md')
guard.SOURCES={name:PREVIOUS/name for name in ('tp_runtime.py','thermal_policy.py','transport_policy.py',
    'resume_checkpoint.py','environment_probe.py','test_thermal_policy.py')}
guard.SOURCES.update({'guard_launcher.py':PREVIOUS/'run_guarded.py',
    'tp2_controller.py':PREVIOUS/'remote_controller.py','base_runtime.py':SUPPORT/'remote_controller.py',
    'workload.py':SPEED/'workload.py','hccl_probe.py':SUPPORT/'hccl_probe.py','requests.json':SPEED/'requests.json'})
if __name__=='__main__':raise SystemExit(guard.main())
