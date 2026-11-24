"""Validate and reuse only the completed EP2 eager phase."""
import hashlib
import json
from pathlib import Path


def read(root, name):
    return json.loads((root/name).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source, target, startup_check):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError('Eager checkpoint must come from a different run')
    old, new = read(source,'benchmark-config.json'), read(target,'benchmark-config.json')
    keys=('phases','image','max_total_tokens','cpu_threads','threadpool_count','random_seed',
          'output_tokens','ignore_eos','warmups_per_case','formal_repeats_per_case','requests_sha256','thermal_policy')
    for key in keys:
        if old[key] != new[key]:raise ValueError('EP2 checkpoint configuration changed: '+key)
    for name in ('candidate.json','environment-preflight.json'):
        if read(source,name) != read(target,name):raise ValueError('EP2 implementation identity changed: '+name)
    for key in ('metadata_sha256','model_shards','cpu_expert_files'):
        if read(source,'weight-preflight.json')[key] != read(target,'weight-preflight.json')[key]:
            raise ValueError('EP2 weights changed: '+key)
    if read(source,'source.json')['base_commit'] != read(target,'source.json')['base_commit']:
        raise ValueError('EP2 base commit changed')
    for name in ('launch.sh','workload.py','requests.json'):
        if digest(source/name)!=digest(target/name):raise ValueError('EP2 workload changed: '+name)
    cleanup=read(source,'cleanup.json')
    if not cleanup['passed'] or cleanup['original_running']!='false' or cleanup['port_18082_open']:
        raise ValueError('Previous EP2 trial did not clean up')
    if read(source,'status.json')['phase'] not in ('failed','aborted_thermal_or_telemetry'):
        raise ValueError('Previous EP2 trial is not a terminated interrupted run')
    log=(source/'logs/ep2eager.log').read_text(encoding='utf-8',errors='replace').partition('SIGTERM received.')[0]
    if startup_check(log,2,64,False)!=read(source,'ep2eager-startup-audit.json'):
        raise ValueError('Eager residency audit differs')
    smoke=read(source,'ep2eager-smoke.json')
    if len(smoke)!=3:raise ValueError('Incomplete eager smoke')
    for row,answer in zip(smoke,('巴黎','42','READY')):
        if not(row['passed'] and row['done'] and row['finish_reason']=='stop' and row['content'].strip()==answer):
            raise ValueError('Eager short task did not pass')
    copied=['logs/ep2eager.log','ep2eager-startup-audit.json','ep2eager-smoke.json',
            'ep2eager-before-resources.json','ep2eager-after-resources.json']
    checked=['benchmark-config.json','candidate.json','environment-preflight.json','weight-preflight.json',
             'source.json','launch.sh','workload.py','requests.json','cleanup.json','status.json']
    for name in copied+checked:
        path=(source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file():raise ValueError('Invalid checkpoint artifact: '+name)
    return dict(source_root=str(source),reused_phases=['ep2eager'],copied_files=copied,
                source_sha256={name:digest(source/name) for name in copied+checked},
                scope='Reuse completed EP2 eager only; graph tasks and performance run afresh')
