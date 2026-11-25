"""Use the existing detached observer with a bounded performance bundle."""
import importlib.util
from pathlib import Path
import sys
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-24-taishan-tp2-expert64'
ATTENTION=HERE.parent/'2026-11-24-taishan-attn-tp1-ep2'
SUPPORT=HERE.parent/'2026-11-10-taishan-tp2'
SPEED=HERE.parent/'2026-11-11-taishan-tp-speed'
sys.path.insert(1,str(PREVIOUS))
spec=importlib.util.spec_from_file_location('guard_launcher',PREVIOUS/'run_guarded.py')
guard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.HERE=HERE
guard.PHASES=('environment','hccl','dpaeager','dpagraph','epgraph','epprefill','dpaprefill')
guard.BUNDLE=('prepare_candidate.py','attn_prepare.py','ep_prepare.py','ep_layout.py',
    'launch.sh','remote_controller.py','resume_performance.py','startup_gate.py','workload.py','test_performance.py','test_resume.py',
    'tp_runtime.py','base_runtime.py','thermal_policy.py','test_thermal_policy.py','transport_policy.py',
    'environment_probe.py','device_probe.py','request_budget_probe.py','hccl_probe.py','combine_probe.py',
    'graph_gate.py','requests.json','experiment-plan.md')
guard.SOURCES={name:PREVIOUS/name for name in ('tp_runtime.py','thermal_policy.py',
    'test_thermal_policy.py','transport_policy.py','environment_probe.py')}
guard.SOURCES.update({name:ATTENTION/name for name in ('ep_prepare.py','ep_layout.py','device_probe.py',
    'request_budget_probe.py','hccl_probe.py','combine_probe.py','graph_gate.py')})
guard.SOURCES.update({'attn_prepare.py':ATTENTION/'prepare_candidate.py',
    'base_runtime.py':SUPPORT/'remote_controller.py','requests.json':SPEED/'requests.json'})


def collect():
    # Raw NPU traces can be large. Keep them durably on taishan and download
    # the trace inventory, measurements and logs without duplicating the traces.
    helpers=guard.helpers
    archive=helpers.RUN/'remote-results.tar.gz'
    with archive.open('wb') as output:
        result=subprocess.run(['ssh','-o','BatchMode=yes','taishan',
            f'tar -C {helpers.REMOTE} --exclude=./sglang --exclude=./sglang-source.tar '
            '--exclude=./kernel_meta --exclude=./profiles -czf - .'],
            stdout=output,stderr=subprocess.PIPE,timeout=120)
    if result.returncode:
        raise RuntimeError('Result retrieval failed: '+result.stderr.decode(errors='replace'))
    destination=(helpers.RUN/'remote').resolve()
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            target=(destination/member.name).resolve()
            target.relative_to(destination)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('wb') as output:
                output.write(source.read())


guard.helpers.collect=collect
if __name__=='__main__':
    raise SystemExit(guard.main())
