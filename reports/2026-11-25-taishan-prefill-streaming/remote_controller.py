"""Detached, bounded single-die prefill comparison with verified streaming."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback

import performance_support as support
import prefill_gate
import workload

core = support.core
PHASES = prefill_gate.PHASES
core.NAMES = {phase:f'dsv4-tp2-{core.RUN_ID}-{phase}' for phase in ('environment',*PHASES)}
core.DEADLINE = time.monotonic()+3*3600
FROZEN_SHA = '83584956389bd4d179c2ac78d8f6ff36fba3fb0adc736673c483e19d269de52f'


def cases_from_frozen(value):
    cases = [value['short'][0],*value['prefill']]
    if [c['prompt_tokens'] for c in cases] != [144,1024,4096,7168]:
        raise ValueError('Frozen prompt lengths changed')
    if any(len(c['token_ids']) != c['prompt_tokens'] for c in cases):
        raise ValueError('Frozen input ID length changed')
    return cases


def read_log(phase):
    core.capture_logs(phase)
    return (core.LOGS/f'{phase}.log').read_text(errors='replace')


def measure(phase, cases):
    before = read_log(phase)
    for case in cases:
        for iteration in range(4):
            core.safe()
            core.status('measuring',phase_id=phase,case=case['name'],iteration=iteration,output_tokens=64)
            result = workload.generate(core,case['token_ids'],64)
            result.update(phase=phase,case=case['name'],iteration=iteration,
                          kind='warmup' if iteration == 0 else 'formal',dp_rank=None)
            after = read_log(phase)
            if not after.startswith(before):
                raise RuntimeError('Container log history changed during measurement')
            try:
                result['streaming_audit'] = prefill_gate.request_audit(after[len(before):],phase,case['prompt_tokens'])
            except Exception as exc:
                result['streaming_audit'] = dict(passed=False,error=str(exc))
                workload.append(core,'requests.jsonl',result)
                raise
            workload.append(core,'requests.jsonl',result)
            if not result['content'].strip() or not any(c.isalnum() for c in result['content']):
                raise RuntimeError('Empty or unusable model output; raw output IDs preserved')
            before = after


def probe():
    phase = 'environment'
    core.status('environment_probe')
    core.create_container(phase)
    try:
        code = core.docker('wait',core.NAMES[phase],timeout=600)
        core.capture_logs(phase)
        if code != '0' or not json.loads((core.ROOT/'kernel-probe.json').read_text())['passed']:
            raise RuntimeError('Environment/AscendC build probe failed; see environment.log')
    finally:
        core.docker('stop','-t','3',core.NAMES[phase],check=False)
        core.capture_logs(phase)


def main():
    core.LOGS.mkdir(exist_ok=True)
    guard,error,started,clean = None,None,False,False
    core.status('preflight')
    try:
        support.preflight()
        started = True
        guard = threading.Thread(target=core.watchdog,daemon=True)
        guard.start()
        frozen_path = core.ROOT/'frozen-workloads.json'
        if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != FROZEN_SHA:
            raise ValueError('Previous experiment frozen workload identity changed')
        cases = cases_from_frozen(json.loads(frozen_path.read_text()))
        core.prepare()
        core.atomic(core.ROOT/'frozen-workloads-sha256.json',dict(sha256=FROZEN_SHA,
            source_run='performance-20261125-113036-profiler-resume',selected_cases=[c['name'] for c in cases]))
        core.atomic(core.ROOT/'bundle-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in core.ROOT.iterdir() if p.suffix in ('.py','.sh','.md')})
        core.atomic(core.ROOT/'benchmark-config.json',dict(at=core.now(),phases=PHASES,image=core.IMAGE,
            sglang_commit=core.COMMIT,tp=1,ep=1,dp=1,physical_npu=2,logical_npu=0,
            experts=32,cpu_threads=96,threadpools=4,output_tokens=64,repeats=3,warmups=1,
            task_queue_enable=1,shared_experts_stream=0,graph='decode_only',
            original_final_state='stopped',thermal_policy=core.THERMAL_POLICY,
            prefill_metric='scheduler forward_entry to prefill_finished wall interval',
            profile_policy='No profiling API calls; prior stop caused 107020',
            content_policy='Preserve raw output IDs; flag short/repetitive content separately from timing'))
        probe()
        for phase in PHASES:
            core.safe()
            core.status('loading',phase_id=phase)
            loading_started = time.monotonic()
            core.create_container(phase)
            try:
                core.wait_health(workload.BASE,core.NAMES[phase],2400)
                core.atomic(core.ROOT/f'{phase}-loading.json',dict(seconds=time.monotonic()-loading_started))
                log = read_log(phase)
                core.atomic(core.ROOT/f'{phase}-startup-audit.json',prefill_gate.startup(log,phase))
                for case in cases:
                    if workload.tokenize(core,case['messages'])['tokens'] != case['token_ids']:
                        raise RuntimeError('Tokenizer/chat encoding differs from frozen inputs')
                core.atomic(core.ROOT/f'{phase}-server-info.json',core.request(workload.BASE,'/server_info'))
                support.snapshot(phase,'before')
                workload.functional(core,phase,False)
                measure(phase,cases)
                support.snapshot(phase,'after')
                final_log = read_log(phase)
                if phase == 'bundle' and 'inline resident:' not in final_log:
                    raise RuntimeError('Dynamic resident success was not observed')
                rows = [json.loads(line) for line in (core.ROOT/'requests.jsonl').read_text().splitlines()]
                core.atomic(core.ROOT/f'{phase}-complete.json',dict(at=core.now(),
                    requests=sum(r['phase']==phase for r in rows),passed=True))
            finally:
                core.docker('stop','-t','30',core.NAMES[phase],timeout=65,check=False)
                core.settled_stopped(core.NAMES[phase])
                core.capture_logs(phase)
        rows = [json.loads(line) for line in (core.ROOT/'requests.jsonl').read_text().splitlines()]
        core.atomic(core.ROOT/'summary.json',prefill_gate.summarize(rows,cases))
        core.status('measurements_complete')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        core.atomic(core.ROOT/'failure.json',dict(at=core.now(),error=error,traceback=traceback.format_exc(),
            phase_status=json.loads((core.ROOT/'status.json').read_text())))
        print(error,flush=True)
    finally:
        if started:
            try:
                clean = support.cleanup()
                if not clean:
                    error = (error or '')+'; cleanup verification failed'
            except Exception as exc:
                error = (error or '')+f'; cleanup error: {exc}'
        core.DONE.set()
        if guard:
            guard.join(timeout=25)
        alarm = core.ALARM.is_set() or (core.ROOT/'abort.json').exists()
        core.status('aborted_thermal_or_telemetry' if alarm else ('failed' if error else 'complete'),
                    error=error,cleanup_verified=clean,original_final_state='stopped',
                    results_require_review=True,correctness_accepted=False)
    return int(bool(error or alarm))


if __name__ == '__main__':
    raise SystemExit(main())
