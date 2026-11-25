"""Bounded performance run, short traces and stopped-state restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
import traceback

import tp_runtime as core
import startup_gate
import workload
import resume_performance

IMAGE = 'sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e'
# The first three phases retain the prior short-request pool/chunk geometry.
PHASES = (('dpaeager',True,False,False), ('dpagraph',True,True,False),
          ('epgraph',False,True,False), ('epprefill',False,True,True),
          ('dpaprefill',True,True,True))
core.IMAGE = IMAGE
core.NAMES = {p:f'dsv4-tp2-{core.RUN_ID}-{p}' for p in ('environment','hccl',*(x[0] for x in PHASES))}
core.DEADLINE = time.monotonic()+3*3600


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1',18082))==0


def snapshot(phase,stage):
    raw = core.run(['bash','-c','date -Is; free -b; numastat -m; npu-smi info'],timeout=30)
    processes = core.docker('top',core.NAMES[phase],'-eo','pid,ppid,rss,comm,args',timeout=20)
    core.atomic(core.ROOT/f'{phase}-{stage}-resources.json',dict(at=core.now(),raw=raw,
        processes=processes,meminfo=Path('/proc/meminfo').read_text()))


def preflight():
    initial = dict(at=core.now(),original_running=core.docker('inspect','-f','{{.State.Running}}',core.ORIGINAL),
                   running_containers=core.docker('ps','--format','{{.Names}}').splitlines())
    core.atomic(core.ROOT/'initial-state.json',initial)
    if initial['original_running']!='false' or initial['running_containers'] or port_open():
        raise RuntimeError('Host must be idle with original stopped and port free')
    if core.docker('image','inspect','-f','{{.Id}}',IMAGE)!=IMAGE:
        raise RuntimeError('Frozen environment image is absent')
    if set(core.docker('ps','-a','--format','{{.Names}}').splitlines()).intersection(core.NAMES.values()):
        raise RuntimeError('Experiment container names already exist')
    meminfo={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if meminfo['MemAvailable']<200*1024*1024:
        raise RuntimeError('Less than 200 GiB available RAM before loading CPU experts')
    core.read_cards()
    if time.time()-(core.ROOT/'heartbeat').stat().st_mtime>45:
        raise RuntimeError('Cooling observer heartbeat is stale')
    model = Path('/data/llm/DeepSeek-V4-Flash-W8A8')
    config,index = model/'config.json',model/'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    files = [model/name for name in shards]
    gguf = sorted(Path('/data/deploy/dsv4-20261104/gguf').glob('*.gguf'))
    if not files or len(gguf)!=43 or any(not p.is_file() or p.stat().st_size<8 for p in files+gguf):
        raise RuntimeError('Model weight files are incomplete')
    core.atomic(core.ROOT/'weight-preflight.json',dict(
        metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (config,index)},
        files={str(p):p.stat().st_size for p in files+gguf},
        verification='Reuse prior complete integrity audit; check current existence/size/metadata identity'))
    raw = (core.ROOT/'requests.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='da4c23cec376ecb3b5db41ac0550bd6f0c3bdfa167c0c721f7657f2812b02aa1':
        raise RuntimeError('Historical workload changed')


def probe(phase):
    core.status(phase+'_probe')
    core.create_container(phase)
    try:
        code = core.docker('wait',core.NAMES[phase],timeout=300)
        core.capture_logs(phase)
        if code!='0':
            raise RuntimeError(f'{phase} probe exited {code}')
        if phase=='hccl':
            for rank in (0,1):
                for prefix in ('hccl','combine'):
                    if json.loads((core.ROOT/f'{prefix}-rank{rank}.json').read_text()).get('passed') is not True:
                        raise RuntimeError(f'{prefix} rank {rank} did not pass')
    finally:
        core.docker('stop','-t','3',core.NAMES[phase],check=False)
        core.capture_logs(phase)


def defer_profile(phase):
    core.atomic(core.ROOT/f'{phase}-profile.json',dict(phase=phase,status='deferred',
        reason='Prior eight-step auto-stop during DP NPU graph replay hit device synchronization timeout 107020',
        policy='No profiling API calls in the performance workflow; isolate any future profiler experiment',
        performance_measurement_unchanged=True))


def cleanup():
    core.DEADLINE=max(core.DEADLINE,time.monotonic()+600)
    core.stop_all()
    states={}
    for phase,name in core.NAMES.items():
        raw = core.docker('inspect','-f','{{json .State}}',name,check=False)
        if not raw:
            continue
        core.capture_logs(phase)
        state=json.loads(raw)
        states[name]=state
    # Give completed task deletion a bounded settling window.
    end=time.monotonic()+45
    while True:
        raw=core.run(['npu-smi','info'],timeout=15)
        no_processes=all(f'No running processes found in NPU {rank}' in raw for rank in (2,5))
        if no_processes or time.monotonic()>=end:
            break
        time.sleep(3)
    original=core.docker('inspect','-f','{{.State.Running}}',core.ORIGINAL)
    passed=all(not x['Running'] and x['Pid']==0 for x in states.values()) and original=='false' and not port_open() and no_processes
    core.atomic(core.ROOT/'cleanup.json',dict(at=core.now(),passed=passed,states=states,
        original_running=original,port_open=port_open(),npu_smi=raw))
    return passed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args()
    core.LOGS.mkdir(exist_ok=True)
    guard,error,started,clean = None,None,False,False
    reused=set()
    core.status('preflight')
    try:
        preflight()
        started=True
        guard=threading.Thread(target=core.watchdog,daemon=True)
        guard.start()
        core.prepare()
        core.atomic(core.ROOT/'bundle-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in core.ROOT.iterdir() if p.suffix in ('.py','.sh','.md')})
        core.atomic(core.ROOT/'benchmark-config.json',dict(at=core.now(),phases=PHASES,image=IMAGE,
            seed=20261111,cpu_threads=96,threadpools=4,experts=64,
            interface='native_generate_token_stream',short_output=512,prefill_output=64,
            repeats=3,warmups=1,thermal_policy=core.THERMAL_POLICY,
            profile_policy='deferred_after_profiler_stop_107020; no profiling API in performance processes',
            quality_policy='Content observations recorded; JSON mismatch does not block performance',
            prefill_metric='Scheduler forward_entry to prefill_finished wall interval; includes host scheduling/forward completion',
            original_final_state='stopped'))
        probe('environment')
        probe('hccl')
        cases=None
        if args.resume_from:
            source=args.resume_from.resolve()
            if source.parent!=Path('/data/deploy') or not source.name.startswith('dsv4-tp2-'):
                raise ValueError('Checkpoint must be a prior isolated /data/deploy/dsv4-tp2-* run')
            core.status('validating_checkpoint',source=str(source))
            record=resume_performance.restore(source,core.ROOT,startup_gate.check)
            cases=record.pop('frozen')
            core.atomic(core.ROOT/'checkpoint-reuse.json',record)
            reused=set(record['reused_phases'])
        for phase,dpa,graph,wide in PHASES:
            core.safe()
            if phase in reused:
                core.status('reusing_completed_phase',phase_id=phase,source=str(args.resume_from))
                if graph:defer_profile(phase)
                continue
            core.status('loading',phase_id=phase,attention_dp=dpa,graph=graph,wide_prefill=wide)
            core.create_container(phase)
            try:
                core.wait_health(workload.BASE,core.NAMES[phase],2400)
                core.capture_logs(phase)
                log=(core.LOGS/f'{phase}.log').read_text(errors='replace')
                core.atomic(core.ROOT/f'{phase}-startup-audit.json',startup_gate.check(log,dpa,graph,wide))
                if cases is None:
                    core.status('freezing_inputs',phase_id=phase)
                    cases=workload.freeze(core)
                    core.atomic(core.ROOT/'frozen-workloads-sha256.json',dict(
                        sha256=hashlib.sha256((core.ROOT/'frozen-workloads.json').read_bytes()).hexdigest()))
                else:
                    for case in cases['prefill'] if wide else cases['short']:
                        actual=workload.tokenize(core,case['messages'])
                        if actual['tokens']!=case['token_ids']:
                            raise RuntimeError('Resumed model tokenizer differs from frozen input')
                core.atomic(core.ROOT/f'{phase}-server-info.json',core.request(workload.BASE,'/server_info'))
                snapshot(phase,'before')
                workload.functional(core,phase,dpa)
                workload.measure(core,phase,cases['prefill'] if wide else cases['short'],64 if wide else 512,dpa)
                snapshot(phase,'after')
                if graph and not wide:
                    defer_profile(phase)
            finally:
                core.docker('stop','-t','30',core.NAMES[phase],timeout=65,check=False)
                core.settled_stopped(core.NAMES[phase])
                core.capture_logs(phase)
        workload.summarize(core,PHASES)
        summary=json.loads((core.ROOT/'summary.json').read_text())
        summary.update(reused_phases=sorted(reused),checkpoint_source=str(args.resume_from) if reused else None,
                       profiling_completed=False,profiling_status='deferred_after_profiler_stop_107020')
        core.atomic(core.ROOT/'summary.json',summary)
        core.status('measurements_complete')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        core.atomic(core.ROOT/'failure.json',dict(at=core.now(),error=error,traceback=traceback.format_exc(),
            phase_status=json.loads((core.ROOT/'status.json').read_text())))
        print(error,flush=True)
    finally:
        if started:
            try:
                clean=cleanup()
                if not clean:error=(error or '')+'; cleanup verification failed'
            except Exception as exc:
                error=(error or '')+f'; cleanup error: {exc}'
        profiles=core.ROOT/'profiles'
        core.atomic(core.ROOT/'profile-inventory.json',dict(at=core.now(),
            raw_profiles_retained_on_remote=True,
            files={str(p.relative_to(core.ROOT)):p.stat().st_size for p in profiles.rglob('*') if p.is_file()}
                  if profiles.exists() else {}))
        core.DONE.set()
        if guard:guard.join(timeout=25)
        alarm=core.ALARM.is_set() or (core.ROOT/'abort.json').exists()
        core.status('aborted_thermal_or_telemetry' if alarm else ('failed' if error else 'complete'),
                    error=error,cleanup_verified=clean,original_final_state='stopped',
                    results_require_review=True,correctness_accepted=False)
    return int(bool(error or alarm))


if __name__=='__main__':
    raise SystemExit(main())
