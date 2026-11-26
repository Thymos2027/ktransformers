"""Detached tutorial-table experiment; only its own NPU2 containers are stopped."""
import hashlib
import json
from pathlib import Path
import re
import socket
import threading
import time
import traceback

import tp_runtime as core
import benchmark
import prefill_gate
import workload
import tutorial_gate

IMAGE='sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e'
core.IMAGE=IMAGE
core.NAMES={p:f'dsv4-tp2-{core.RUN_ID}-{p}' for p in ('environment',*benchmark.PHASES)}
core.DEADLINE=time.monotonic()+3*3600
PROFILE_PATH=core.ROOT/'runtime-profile.json'
PROFILE=json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else None
if PROFILE:
    benchmark.PHASES={k:tuple(v) for k,v in PROFILE['phases'].items()}
    core.NAMES={p:f'dsv4-tp2-{core.RUN_ID}-{p}' for p in ('environment',*benchmark.PHASES)}
original_stop_all=core.stop_all
def scoped_stop(include_original=False):
    original_stop_all(include_original=False)
core.stop_all=scoped_stop


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1',18082))==0


def snapshot(name):
    core.atomic(core.ROOT/(name+'-resources.json'),dict(at=core.now(),
        raw=core.run(['bash','-c','npu-smi info; free -b; numastat -m; ps -eo pid,pcpu,pmem,args --sort=-pcpu | head -12']),
        running_containers=core.docker('ps','--format','{{.Names}}'),
        loadavg=Path('/proc/loadavg').read_text(),meminfo=Path('/proc/meminfo').read_text()))


def preflight():
    raw=core.run(['npu-smi','info'])
    if 'No running processes found in NPU 2' not in raw or port_open():
        raise RuntimeError('NPU2 must be idle and port18082 free')
    existing=core.docker('ps','--format','{{.Names}}').splitlines()
    original=core.docker('inspect','-f','{{.State.Running}}',core.ORIGINAL)
    core.atomic(core.ROOT/'initial-state.json',dict(original_running=original,
        running_containers=existing,npu_smi=raw))
    if original!='false' or set(core.NAMES.values()).intersection(
            core.docker('ps','-a','--format','{{.Names}}').splitlines()):
        raise RuntimeError('Original service or experiment names changed')
    if core.docker('image','inspect','-f','{{.Id}}',IMAGE)!=IMAGE:
        raise RuntimeError('Frozen image missing')
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<220*1024**2:
        raise RuntimeError('Insufficient available host RAM')
    model=Path('/data/llm/DeepSeek-V4-Flash-W8A8')
    metadata=[model/'config.json',model/'model.safetensors.index.json']
    shards=[model/n for n in set(json.loads(metadata[1].read_text())['weight_map'].values())]
    gguf=list(Path('/data/deploy/dsv4-20261104/gguf').glob('*.gguf'))
    if not shards or len(gguf)!=43 or any(not p.is_file() or p.stat().st_size<8 for p in shards+gguf):
        raise RuntimeError('Model files missing')
    core.atomic(core.ROOT/'weight-preflight.json',dict(
        metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata},
        file_sizes={str(p):p.stat().st_size for p in shards+gguf},reuse_prior_integrity_audit=True))
    core.read_cards()
    if time.time()-(core.ROOT/'heartbeat').stat().st_mtime>45:
        raise RuntimeError('Cooling observer heartbeat stale')
    snapshot('initial')


def create(phase):
    core.safe()
    args=['run','-d','--name',core.NAMES[phase],'--restart','no','--network','host',
          '--ipc','host','--privileged','--security-opt','seccomp=unconfined',
          '--ulimit','memlock=-1','--ulimit','stack=67108864','--cpuset-cpus','0-39',
          '--volumes-from',core.ORIGINAL+':ro','-v',str(core.ROOT)+':/experiment:rw','-w','/experiment']
    for device in ('davinci2','davinci_manager','devmm_svm','hisi_hdc'):
        args+=['--device','/dev/'+device]
    args+=['--entrypoint','/bin/bash',IMAGE,'/experiment/launch.sh',phase]
    return core.docker(*args)


def read_log(phase):
    core.capture_logs(phase)
    return (core.LOGS/(phase+'.log')).read_text(errors='replace')


def startup(log,phase):
    return tutorial_gate.startup(log,phase,PROFILE['settings'] if PROFILE else None)


def cleanup():
    core.stop_all()
    states={}
    for phase,name in core.NAMES.items():
        raw=core.docker('inspect','-f','{{json .State}}',name,check=False)
        if raw:
            states[name]=json.loads(raw)
            core.capture_logs(phase)
    raw=core.run(['npu-smi','info'])
    original=core.docker('inspect','-f','{{.State.Running}}',core.ORIGINAL)
    passed=(all(not s['Running'] and s['Pid']==0 for s in states.values())
            and 'No running processes found in NPU 2' in raw and not port_open() and original=='false')
    core.atomic(core.ROOT/'cleanup.json',dict(passed=passed,states=states,npu_smi=raw,
        original_running=original,port_open=port_open(),other_workloads_never_stopped=True,
        final_running_containers=core.docker('ps','--format','{{.Names}}').splitlines()))
    return passed


def main():
    core.LOGS.mkdir(exist_ok=True)
    started,guard,error,clean=False,None,None,False
    core.status('preflight')
    try:
        preflight()
        started=True
        guard=threading.Thread(target=core.watchdog,daemon=True)
        guard.start()
        core.prepare()
        core.atomic(core.ROOT/'bundle-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in core.ROOT.iterdir() if p.suffix in ('.py','.sh','.md') or p.name in ('sources.json','runtime-profile.json')})
        core.atomic(core.ROOT/'benchmark-config.json',dict(image=IMAGE,sglang_commit=core.COMMIT,
            physical_npu=2,container_logical_npu=0,cpu_threads=32,threadpools=1,cpuset='0-39',
            context_length=65536,chunked_prefill_size=32768,
            max_total_tokens=PROFILE['settings']['max_total_tokens'] if PROFILE else 'auto',
            runtime_profile=PROFILE,
            output_tokens=1000,repeats=3,warmups=1,phases=benchmark.PHASES,
            software_policy='Reuse pinned working environment; do not claim software/hardware identity',
            shared_host_load=bool(json.loads((core.ROOT/'initial-state.json').read_text())['running_containers']),
            thermal_policy=core.THERMAL_POLICY))
        core.status('environment_probe')
        create('environment')
        code=core.docker('wait',core.NAMES['environment'],timeout=600)
        core.capture_logs('environment')
        if code!='0' or not json.loads((core.ROOT/'kernel-probe.json').read_text())['passed']:
            raise RuntimeError('Environment/AscendC probe failed')
        cases=None
        rows=[]
        for phase in benchmark.PHASES:
            core.safe()
            core.status('loading',phase_id=phase)
            create(phase)
            try:
                core.wait_health(workload.BASE,core.NAMES[phase],2400)
                before=read_log(phase)
                core.atomic(core.ROOT/(phase+'-startup-audit.json'),startup(before,phase))
                info=core.request(workload.BASE,'/server_info')
                core.atomic(core.ROOT/(phase+'-server-info.json'),info)
                if cases is None:
                    core.status('freezing_inputs',phase_id=phase)
                    cases=benchmark.build_cases(core)
                phase_cases=[c for c in cases if c['phase']==phase]
                core.atomic(core.ROOT/(phase+'-capacity-audit.json'),
                            tutorial_gate.capacity(before,phase_cases,server_info=info))
                snapshot(phase+'-before')
                for case in phase_cases:
                    for iteration in range(4):
                        core.safe()
                        core.status('measuring',phase_id=phase,target=case['target'],
                            prompt_tokens=case['prompt_tokens'],iteration=iteration,output_tokens=1000)
                        result=benchmark.generate(core,case,iteration)
                        after=read_log(phase)
                        if not after.startswith(before):
                            raise RuntimeError('Container log history changed')
                        result['streaming_audit']=prefill_gate.request_audit(
                            after[len(before):],'bundle',case['prompt_tokens'])
                        workload.append(core,'requests.jsonl',result)
                        rows.append(result)
                        before=after
                        snapshot(f'{phase}-{case["target"]}-{iteration}')
                if 'inline resident:' not in before or not re.search(r'Decode batch[^\n]+npu graph: True',before):
                    raise RuntimeError('Dynamic residency or actual graph replay missing')
                core.atomic(core.ROOT/(phase+'-complete.json'),dict(passed=True,at=core.now()))
            finally:
                core.docker('stop','-t','30',core.NAMES[phase],timeout=65,check=False)
                core.settled_stopped(core.NAMES[phase])
                core.capture_logs(phase)
        summary=benchmark.summarize(rows)
        if PROFILE:
            summary.update(runtime_profile=PROFILE,
                comparison='Hardware-adjusted long-input run; not identical to tutorial parameters or historical short-input rows')
        core.atomic(core.ROOT/'summary.json',summary)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        core.atomic(core.ROOT/'failure.json',dict(error=error,traceback=traceback.format_exc()))
    finally:
        if started:
            try:
                clean=cleanup()
                if not clean:
                    error=(error or '')+'; cleanup verification failed'
            except Exception as exc:
                error=(error or '')+'; cleanup: '+str(exc)
        core.DONE.set()
        if guard:
            guard.join(timeout=15)
        alarm=core.ALARM.is_set() or (core.ROOT/'abort.json').exists()
        core.status('aborted_thermal_or_telemetry' if alarm else ('failed' if error else 'complete'),
            error=error,cleanup_verified=clean,results_require_review=True)
    return int(bool(error or alarm))


if __name__=='__main__':
    raise SystemExit(main())
