"""Detached K64 deployment, contemporary K32 controls, and cleanup."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time

import tp_runtime as core
import workload
import resume_checkpoint


PHASES = (('tp2k64eager', 2, 64, False), ('tp1k32graph', 1, 32, True),
          ('tp2k32graph', 2, 32, True), ('tp2k64graph', 2, 64, True))
IMAGE = 'sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e'
REQUEST_SHA256 = 'da4c23cec376ecb3b5db41ac0550bd6f0c3bdfa167c0c721f7657f2812b02aa1'
core.IMAGE = IMAGE
core.NAMES = {phase: f'dsv4-tp2-{core.RUN_ID}-{phase}'
              for phase in ('environment', 'hccl', *(p[0] for p in PHASES))}
core.DEADLINE = time.monotonic() + 3 * 3600


def check_startup(log, tp, experts, graph):
    if not re.search(r'\btp_size=' + str(tp) + r'\b', log):
        raise RuntimeError('Resolved TP differs from experiment')
    for marker in (f'kt_num_gpu_experts={experts}', 'max_total_tokens=16384',
                   'kt_cpuinfer=96', 'max_running_requests=1', 'random_seed=20261111'):
        if marker not in log:
            raise RuntimeError(f'Missing resolved startup option: {marker}')
    pattern = (r'\[KT RESIDENCY AUDIT\] layer=(\d+) rank=(\d+) resident=(\d+) '
               r'cpu_experts=(\d+) cpu_owner=(True|False) weight_shapes=([^\n]+)')
    audits = re.findall(pattern, log)
    expected = {(layer, rank) for layer in range(43) for rank in range(tp)}
    seen = set()
    for layer, rank, resident, cpu, owner, shapes in audits:
        pair = (int(layer), int(rank))
        if pair in seen or pair not in expected:
            raise RuntimeError(f'Unexpected or repeated residency audit {pair}')
        seen.add(pair)
        if (int(resident) != experts or int(cpu) != 256 - experts
                or (owner == 'True') != (int(rank) == 0)):
            raise RuntimeError(f'Residency/CPU ownership mismatch: {pair}')
    if seen != expected:
        raise RuntimeError(f'Incomplete residency audit: {len(seen)}/{len(expected)}')
    if tp == 2:
        partition = re.findall(r'\[KT TP2 EXPERIMENT\] layer=(\d+) rank=(\d+) '
                              r'moe_tp=(\d+) local_intermediate=(\d+) '
                              r'full_intermediate=(\d+) resident_experts=(\d+) '
                              r'cpu_owner=(True|False)', log)
        if len(partition) != 86 or {(int(x[0]), int(x[1])) for x in partition} != expected:
            raise RuntimeError('Incomplete TP2 intermediate-dimension audit')
        if any(x[2:6] != ('2', '1024', '2048', str(experts)) for x in partition):
            raise RuntimeError('TP2 expert tensor partition differs from expected dimensions')
    if graph:
        workload.check_graph(log, tp)
    elif 'disable_cuda_graph=True' not in log:
        raise RuntimeError('Eager mode was not resolved')
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception', log):
        raise RuntimeError('Startup reported an exception')
    return dict(layers=43, ranks=tp, resident_experts=experts,
                cpu_experts=256-experts, audit_records=len(audits), graph=graph)


def snapshot(phase, stage):
    # Host counters and the complete process tree remain raw evidence.
    name = core.NAMES[phase]
    raw = core.run(['bash', '-c', 'date -Is; free -b; numastat -m; npu-smi info'], timeout=30)
    top = core.docker('top', name, '-eo', 'pid,ppid,rss,comm,args', timeout=20)
    mem = Path('/proc/meminfo').read_text()
    core.atomic(core.ROOT / f'{phase}-{stage}-resources.json',
                dict(at=core.now(), raw=raw, processes=top, meminfo=mem))


def summarize():
    rows = [json.loads(x) for x in (core.ROOT / 'requests.jsonl').read_text().splitlines()]
    formal = [x for x in rows if x['kind'] == 'formal']
    if len(rows) != 24 or len(formal) != 18:
        raise RuntimeError('Incomplete benchmark matrix: expected 24 requests / 18 formal')
    expected = {(phase, case, i) for phase, _, _, graph in PHASES if graph
                for case in ('input_128', 'input_768') for i in range(4)}
    if len({(x['phase'], x['case'], x['iteration']) for x in rows}) != 24:
        raise RuntimeError('Repeated request identities')
    if {(x['phase'], x['case'], x['iteration']) for x in rows} != expected:
        raise RuntimeError('Request identities differ from frozen matrix')
    for x in rows:
        workload.check_transport(x)
        if x['usage']['completion_tokens'] != 512 or x['finish_reason'] != 'length':
            raise RuntimeError('Unexpected output token budget or stop reason')
        if hashlib.sha256(x['content'].encode()).hexdigest() != x['content_sha256']:
            raise RuntimeError('Output digest mismatch')
        times = [chunk['seconds'] for chunk in x['chunks']]
        if (not times or times != sorted(times) or times[0] != x['ttft_seconds']
                or times[-1] != x['last_content_seconds']
                or not math.isclose(511/(times[-1]-times[0]), x['decode_tokens_per_second'], rel_tol=1e-12)):
            raise RuntimeError('Streaming time/counter evidence mismatch')
    reuse_path = core.ROOT / 'checkpoint-reuse.json'
    output = dict(formal_requests=18, warmup_requests=6, cases=[],
                  checkpoint_reuse=json.loads(reuse_path.read_text()) if reuse_path.exists() else None,
                  interpretation='Exploratory same-workload performance; numerical and independent task quality remain unaccepted')
    for case in ('input_128', 'input_768'):
        item = dict(case=case, configurations={}, comparisons={})
        for phase, tp, experts, graph in PHASES:
            if not graph:
                continue
            group = [x for x in formal if x['phase'] == phase and x['case'] == case]
            item['configurations'][phase] = dict(tp=tp, experts=experts, requests=len(group), metrics={
                key: dict(median=statistics.median(x[key] for x in group),
                          min=min(x[key] for x in group), max=max(x[key] for x in group))
                for key in ('ttft_seconds', 'decode_tokens_per_second', 'total_seconds')},
                output_sha256=sorted({x['content_sha256'] for x in group}))
        target = item['configurations']['tp2k64graph']['metrics']
        for control in ('tp1k32graph', 'tp2k32graph'):
            base = item['configurations'][control]['metrics']
            item['comparisons']['tp2k64graph_vs_' + control] = dict(
                decode_gain_percent=100*(target['decode_tokens_per_second']['median']/base['decode_tokens_per_second']['median']-1),
                total_time_reduction_percent=100*(1-target['total_seconds']['median']/base['total_seconds']['median']),
                ttft_reduction_percent=100*(1-target['ttft_seconds']['median']/base['ttft_seconds']['median']))
        output['cases'].append(item)
    core.atomic(core.ROOT / 'summary.json', output)


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', 18082)) == 0


def weight_preflight():
    model = Path('/data/llm/DeepSeek-V4-Flash-W8A8')
    index = model / 'model.safetensors.index.json'
    config = model / 'config.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    files = [model / name for name in shards]
    if not files or any(not p.is_file() or p.stat().st_size < 8 for p in files):
        raise RuntimeError('Missing W8A8 model shards')
    gguf = sorted(Path('/data/deploy/dsv4-20261104/gguf').glob('*.gguf'))
    if len(gguf) != 43:
        raise RuntimeError('Expected 43 CPU expert GGUF files')
    for p in gguf:
        with p.open('rb') as stream:
            if stream.read(4) != b'GGUF' or p.stat().st_size < 1024:
                raise RuntimeError(f'Invalid CPU expert file: {p}')
    core.atomic(core.ROOT / 'weight-preflight.json', dict(at=core.now(),
        scope='Existence, size, GGUF magic and metadata identity; reuses prior full integrity audit',
        model_path=str(model), metadata_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in (config, index)},
        model_shards={p.name: p.stat().st_size for p in files},
        cpu_expert_files={p.name: p.stat().st_size for p in gguf}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    guard = None
    error = None
    cleanup_ok = False
    preflight_passed = False
    reused = set()
    core.LOGS.mkdir(exist_ok=True)
    core.status('preflight')
    try:
        initial = dict(at=core.now(), original_running=core.docker(
            'inspect', '-f', '{{.State.Running}}', core.ORIGINAL) == 'true',
            running_containers=core.docker('ps', '--format', '{{.Names}}').splitlines())
        core.atomic(core.ROOT / 'initial-state.json', initial)
        if initial['original_running'] or initial['running_containers']:
            raise RuntimeError('Expected idle host with original stopped; refuse to change another service')
        if port_open():
            raise RuntimeError('Experiment port 18082 already in use')
        if core.docker('image', 'inspect', '-f', '{{.Id}}', IMAGE) != IMAGE:
            raise RuntimeError('Frozen environment image absent')
        existing = set(core.docker('ps', '-a', '--format', '{{.Names}}').splitlines())
        if existing.intersection(core.NAMES.values()):
            raise RuntimeError('Trial container name collision')
        core.read_cards()
        if time.time() - (core.ROOT / 'heartbeat').stat().st_mtime > 45:
            raise RuntimeError('Fan monitor heartbeat stale')
        if hashlib.sha256((core.ROOT / 'requests.json').read_bytes()).hexdigest() != REQUEST_SHA256:
            raise RuntimeError('Historical frozen request identity mismatch')
        cases = json.loads((core.ROOT / 'requests.json').read_text())
        if [x['prompt_tokens'] for x in cases] != [144, 792]:
            raise RuntimeError('Unexpected input token budgets')
        for case in cases:
            if case['prompt_tokens'] + 512 > 1536:
                raise RuntimeError('Request exceeds validated short-workload budget')
        weight_preflight()
        preflight_passed = True
        guard = threading.Thread(target=core.watchdog, daemon=True)
        guard.start()
        core.prepare()
        core.atomic(core.ROOT / 'benchmark-config.json', dict(
            at=core.now(), phases=PHASES, image=IMAGE, max_total_tokens=16384,
            cpu_threads=96, threadpool_count=4, random_seed=20261111, output_tokens=512,
            ignore_eos=True, warmups_per_case=1, formal_repeats_per_case=3,
            requests_sha256=REQUEST_SHA256, original_final_state='stopped',
            thermal_policy=core.THERMAL_POLICY))
        core.atomic(core.ROOT / 'bundle-sha256.json', {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in core.ROOT.iterdir() if p.suffix in ('.py', '.sh', '.md')})
        core.status('environment_probe')
        core.create_container('environment')
        try:
            code = core.docker('wait', core.NAMES['environment'], timeout=300)
            core.capture_logs('environment')
            log = (core.LOGS / 'environment.log').read_text(errors='replace')
            if code != '0':
                raise RuntimeError(f'KT environment probe exited {code}; see logs/environment.log')
            records = [json.loads(line.split('KT environment: ', 1)[1])
                       for line in log.splitlines() if line.startswith('KT environment: ')]
            if len(records) != 1 or records[0].get('passed') is not True:
                raise RuntimeError('KT environment probe produced no successful identity record')
            core.atomic(core.ROOT / 'environment-preflight.json', records[0])
        finally:
            core.docker('stop', '-t', '3', core.NAMES['environment'], check=False)
            core.capture_logs('environment')
        core.status('hccl_probe')
        core.create_container('hccl')
        try:
            code = core.docker('wait', core.NAMES['hccl'], timeout=300)
            if code != '0':
                raise RuntimeError(f'HCCL probe exited {code}')
            for rank in (0, 1):
                if json.loads((core.ROOT / f'hccl-rank{rank}.json').read_text()).get('passed') is not True:
                    raise RuntimeError(f'HCCL rank {rank} failed')
        finally:
            core.docker('stop', '-t', '3', core.NAMES['hccl'], check=False)
            core.capture_logs('hccl')
        if args.resume_from:
            source = args.resume_from.resolve()
            if source.parent != Path('/data/deploy') or not source.name.startswith('dsv4-tp2-'):
                raise ValueError('Checkpoint must be a prior isolated trial under /data/deploy')
            core.status('validating_checkpoint', source=str(source))
            record = resume_checkpoint.validate(source, core.ROOT, check_startup)
            resume_checkpoint.copy_verified(record, core.ROOT)
            core.atomic(core.ROOT / 'checkpoint-reuse.json', record)
            reused = set(record['reused_phases'])
            config = json.loads((core.ROOT / 'benchmark-config.json').read_text())
            config.update(checkpoint_source=str(source), reused_phases=sorted(reused))
            core.atomic(core.ROOT / 'benchmark-config.json', config)
        for phase, tp, experts, graph in PHASES:
            core.safe()
            if phase in reused:
                core.status('reusing_completed_phase', phase_id=phase, source=str(args.resume_from))
                continue
            core.status('loading', phase_id=phase, tp=tp, resident_experts=experts, graph=graph)
            core.create_container(phase)
            try:
                core.wait_health('http://127.0.0.1:18082', core.NAMES[phase], 2400)
                core.capture_logs(phase)
                log = (core.LOGS / f'{phase}.log').read_text(errors='replace')
                audit = check_startup(log, tp, experts, graph)
                core.atomic(core.ROOT / f'{phase}-startup-audit.json', audit)
                snapshot(phase, 'before')
                if graph:
                    workload.measure(core, phase, tp, cases)
                else:
                    core.status('smoke', phase_id=phase)
                    core.smoke('http://127.0.0.1:18082', phase)
                snapshot(phase, 'after')
            finally:
                core.docker('stop', '-t', '30', core.NAMES[phase], timeout=65, check=False)
                core.settled_stopped(core.NAMES[phase])
                core.capture_logs(phase)
        summarize()
        core.status('measurements_complete')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        core.atomic(core.ROOT / 'failure.json', dict(at=core.now(), error=error))
        print(error, flush=True)
    finally:
        # Retain initial stopped state; never start the old baseline automatically.
        if preflight_passed:
            core.DEADLINE = max(core.DEADLINE, time.monotonic() + 600)
            core.stop_all()
            for phase, name in core.NAMES.items():
                if phase in reused:
                    continue
                try:
                    core.capture_logs(phase)
                except Exception as exc:
                    error = (error or '') + f'; could not capture {phase} log: {exc}'
            try:
                states = {name: core.docker('inspect', '-f', '{{.State.Running}}', name, check=False)
                          for name in core.NAMES.values()}
                original = core.docker('inspect', '-f', '{{.State.Running}}', core.ORIGINAL)
                raw = core.run(['npu-smi', 'info'], timeout=15)
                cleanup_ok = all(x != 'true' for x in states.values()) and original == 'false' and not port_open()
                cleanup_ok = cleanup_ok and 'No running processes found in NPU 2' in raw and 'No running processes found in NPU 5' in raw
                core.atomic(core.ROOT / 'cleanup.json', dict(at=core.now(), containers=states,
                            original_running=original, port_18082_open=port_open(), npu_smi=raw, passed=cleanup_ok))
                if not cleanup_ok:
                    error = (error or '') + '; cleanup verification failed'
            except Exception as exc:
                error = (error or '') + f'; cleanup verification error: {exc}'
        core.DONE.set()
        if guard:
            guard.join(timeout=25)
        alarm = core.ALARM.is_set() or (core.ROOT / 'abort.json').exists()
        core.status('aborted_thermal_or_telemetry' if alarm else ('failed' if error else 'complete'),
                    error=error, cleanup_verified=cleanup_ok, original_final_state='stopped',
                    results_require_review=True, correctness_accepted=False)
    return 1 if error or alarm else 0


if __name__ == '__main__':
    raise SystemExit(main())
