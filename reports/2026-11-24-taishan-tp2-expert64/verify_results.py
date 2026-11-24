"""Independently recompute the completed benchmark and verify its source lineage."""
import argparse
import ast
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

PHASES = {'tp1k32graph': (1, 32), 'tp2k32graph': (2, 32), 'tp2k64graph': (2, 64)}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run):
    root = run / 'remote'
    state, cleanup = read(root / 'status.json'), read(root / 'cleanup.json')
    require(state['phase'] == 'complete' and not state['error'] and state['cleanup_verified'], 'Workflow incomplete')
    require(cleanup['passed'] and cleanup['original_running'] == 'false'
            and not cleanup['port_18082_open'], 'Cleanup failed')
    checks = read(run / 'remote-hash-check.json')
    for name, expected in checks['current'].items():
        require(sha(root / name) == expected, f'Remote/local digest mismatch: {name}')
    frozen = read(root / 'bundle-sha256.json')
    for name, expected in frozen.items():
        require(sha(root / name) == expected, f'Frozen runtime changed: {name}')
    reuse = read(root / 'checkpoint-reuse.json')
    predecessor = run.parent / 'expert64-20261124-cooling90' / 'remote'
    require(reuse['source_sha256'] == checks['predecessor'], 'Predecessor manifest differs')
    for name, expected in reuse['source_sha256'].items():
        require(sha(predecessor / name) == expected, f'Predecessor digest mismatch: {name}')
    for name in reuse['copied_files']:
        if name != 'requests.jsonl':
            require(sha(root / name) == reuse['source_sha256'][name], f'Reused artifact changed: {name}')
    require((root / 'requests.jsonl').read_bytes().startswith((predecessor / 'requests.jsonl').read_bytes()),
            'Reused baseline is not preserved byte for byte')
    config, automatic = read(root / 'benchmark-config.json'), read(root / 'summary.json')
    require(config['cpu_threads'] == 96 and config['threadpool_count'] == 4
            and config['random_seed'] == 20261111 and config['output_tokens'] == 512
            and config['thermal_policy']['temperature_limit_c'] == 90
            and config['thermal_policy']['temperature_hold_seconds'] == 10, 'Unexpected settings')
    require(sha(root / 'requests.json') == config['requests_sha256'], 'Prompt identity changed')
    cases = {x['name']: x for x in read(root / 'requests.json')}
    require(sorted(x['prompt_tokens'] for x in cases.values()) == [144, 792], 'Unexpected input token counts')
    require(all(len(x['token_ids']) == x['prompt_tokens'] for x in cases.values()), 'Frozen token sequence incomplete')
    rows = [json.loads(x) for x in (root / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {(phase, case, i) for phase in PHASES for case in cases for i in range(4)}
    require(len(rows) == 24 and {(x['phase'], x['case'], x['iteration']) for x in rows} == expected,
            'Incomplete or duplicate request matrix')
    computed = []
    for row in rows:
        case = cases[row['case']]
        payload = row['payload']
        require(row['tp'] == PHASES[row['phase']][0]
                and row['kind'] == ('warmup' if row['iteration'] == 0 else 'formal'), 'Request grouping differs')
        require(payload['messages'] == case['messages'] and payload['max_tokens'] == 512
                and payload['temperature'] == 0 and payload['seed'] == 20261111
                and payload['ignore_eos'] and row['done'] and row['finish_reason'] == 'length', 'Request gate differs')
        require(row['usage']['prompt_tokens'] == case['prompt_tokens']
                and row['usage']['completion_tokens'] == 512, 'Token count differs')
        text = ''.join(x['text'] for x in row['chunks'])
        require(text == row['content'] and hashlib.sha256(text.encode()).hexdigest() == row['content_sha256'],
                'Output/chunk digest differs')
        times = [x['seconds'] for x in row['chunks']]
        require(times == sorted(times) and times[-1] > times[0] > 0 and times[-1] <= row['total_seconds'],
                'Invalid streaming timestamps')
        rate = (row['usage']['completion_tokens'] - 1) / (times[-1] - times[0])
        require(times[0] == row['ttft_seconds'] and times[-1] == row['last_content_seconds']
                and math.isclose(rate, row['decode_tokens_per_second'], rel_tol=1e-12), 'Client metric differs')
        computed.append(dict(phase=row['phase'], case=row['case'], iteration=row['iteration'], kind=row['kind'],
                             started_at=row['started_at'], prompt_tokens=case['prompt_tokens'],
                             ttft_seconds=times[0], decode_tokens_per_second=rate,
                             total_seconds=row['total_seconds'], output_sha256=row['content_sha256']))
    result = dict(passed=True, readiness='Share with caveats', reviewed_at=dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))).isoformat(), functional_and_measurement_checks_passed=True,
        numerical_correctness_accepted=False, formal_requests=18, warmup_requests=6,
        source_hashes_verified=len(checks['current']), predecessor_hashes_verified=len(checks['predecessor']),
        checkpoint_reuse=reuse, cases=[], phases={}, formal_rows=[x for x in computed if x['kind'] == 'formal'])
    for case in cases:
        item = dict(case=case, prompt_tokens=cases[case]['prompt_tokens'], output_tokens=512,
                    configurations={}, comparisons={})
        auto = next(x for x in automatic['cases'] if x['case'] == case)
        for phase in PHASES:
            group = [x for x in computed if x['phase'] == phase and x['case'] == case and x['kind'] == 'formal']
            metrics = {key: dict(median=statistics.median(x[key] for x in group),
                                min=min(x[key] for x in group), max=max(x[key] for x in group),
                                values=[x[key] for x in group])
                       for key in ('ttft_seconds', 'decode_tokens_per_second', 'total_seconds')}
            for key, stats in metrics.items():
                for field in ('median', 'min', 'max'):
                    require(math.isclose(stats[field], auto['configurations'][phase]['metrics'][key][field], rel_tol=1e-12),
                            f'Automatic summary differs: {case}/{phase}/{key}/{field}')
            item['configurations'][phase] = dict(metrics=metrics, output_sha256=sorted({x['output_sha256'] for x in group}))
        for baseline in ('tp1k32graph', 'tp2k32graph'):
            target, base = item['configurations']['tp2k64graph']['metrics'], item['configurations'][baseline]['metrics']
            comparison = dict(decode_gain_percent=100*(target['decode_tokens_per_second']['median']/base['decode_tokens_per_second']['median']-1),
                              ttft_reduction_percent=100*(1-target['ttft_seconds']['median']/base['ttft_seconds']['median']),
                              total_time_reduction_percent=100*(1-target['total_seconds']['median']/base['total_seconds']['median']))
            name = 'tp2k64graph_vs_' + baseline
            for key, value in comparison.items():
                require(math.isclose(value, auto['comparisons'][name][key], abs_tol=1e-10), 'Automatic ratio differs')
            item['comparisons'][name] = comparison
        result['cases'].append(item)
    for phase, (tp, experts) in PHASES.items():
        log = (root / 'logs' / f'{phase}.log').read_text(encoding='utf-8', errors='replace')
        serving, separator, shutdown = log.partition('SIGTERM received.')
        require(separator and not re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception', serving),
                f'{phase} failed before controlled stop')
        audit = re.findall(r'\[KT RESIDENCY AUDIT\] layer=(\d+) rank=(\d+) resident=(\d+) '
                           r'cpu_experts=(\d+) cpu_owner=(True|False) weight_shapes=([^\n]+)', serving)
        require(len(audit) == 43*tp and {(int(x[0]), int(x[1])) for x in audit} ==
                {(layer, rank) for layer in range(43) for rank in range(tp)}, 'Missing layer/rank audits')
        for layer, rank, resident, cpu, owner, shapes in audit:
            shape = ast.literal_eval(shapes)
            require(int(resident) == experts and int(cpu) == 256-experts
                    and (owner == 'True') == (int(rank) == 0)
                    and shape['w13_weight'][0] == shape['w2_weight'][0] == experts, 'Actual residency differs')
        if tp == 2:
            for rank in (0, 1):
                require(f'TP{rank}] Capture target decode NPU graph end' in serving, f'{phase} graph capture missing')
        else:
            require('Capture target decode NPU graph end' in serving, 'TP1 graph missing')
        require('npu graph: True' in serving, f'{phase} no actual graph replay log')
        gate = read(root / f'{phase}-gate.json')
        require(len(gate) == 5, 'Short task gate incomplete')
        for task, answer in zip(gate, ('巴黎', '42', '156', 'READY', [1, 2, 5, 9])):
            actual = task['content'].strip()
            require(task['passed'] and task['done'] and task['finish_reason'] == 'stop'
                    and (json.loads(actual) if isinstance(answer, list) else actual) == answer, 'Task answer/transport failed')
        resources = {}
        for stage in ('before', 'after'):
            snapshot = read(root / f'{phase}-{stage}-resources.json')
            hbm = list(map(int, re.findall(r'(\d+)\s*/\s*65536', snapshot['raw'])))
            require(len(hbm) == 2, 'HBM snapshot incomplete')
            memory = dict((k, int(v)) for k, v in re.findall(r'^(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+) kB', snapshot['meminfo'], re.M))
            resources[stage] = dict(at=snapshot['at'], hbm_gib={k:v/1024 for k,v in zip(('2','5'),hbm)},
                                    system_total_minus_available_gib=(memory['MemTotal']-memory['MemAvailable'])/1024**2,
                                    swap_used_gib=(memory['SwapTotal']-memory['SwapFree'])/1024**2)
        result['phases'][phase] = dict(tp=tp, resident_experts=experts, cpu_experts=256-experts,
            short_tasks_passed=5, audited_layer_rank_pairs=len(audit), resources=resources,
            shutdown_traceback_present='Traceback' in shutdown)
    telemetry = [json.loads(x) for x in (root / 'thermal.jsonl').read_text(encoding='utf-8').splitlines()]
    valid = [x for x in telemetry if 'cards' in x]
    require(valid and all(c['health'] == 'OK' for x in valid for c in x['cards'].values()), 'Remote device health failed')
    for x in valid:
        require(not any(c['temperature_c'] >= 90 and c.get('over_limit_seconds', 0) >= 10 for c in x['cards'].values()), 'Temperature stop condition reached')
    service = [json.loads(x)['service'] for x in (run / 'cooling-service-samples.jsonl').read_text(encoding='utf-8').splitlines()]
    require(service and all(x['status'] == 'running' and x['sample_valid'] and not x['controls_workloads'] for x in service),
            'External cooling observer invalid')
    result['monitoring'] = dict(remote_valid_samples=len(valid), remote_errors=len(telemetry)-len(valid),
        max_temperature_c={k:max(x['cards'][k]['temperature_c'] for x in valid) for k in ('2','5')},
        cooling_service_pids=sorted({x['pid'] for x in service}),
        min_fan_rpm=min(v for x in service for v in x['sample']['fans_rpm'].values()),
        ssh_transport_errors=len((run / 'transport.jsonl').read_text(encoding='utf-8').splitlines()) if (run / 'transport.jsonl').exists() else 0)
    for name, value in checks['containers'].items():
        state = value['state']
        require(not state['Running'] and state['Pid'] == 0 and not state['OOMKilled'], f'Container not clean: {name}')
    result['container_states'] = {name:{k:value['state'][k] for k in ('Status','Pid','ExitCode','OOMKilled')}
                                  for name,value in checks['containers'].items()}
    result['limitations'] = ['Only three formal repeats per configuration and input length; one model start per configuration',
        'TP1 baseline reused from interrupted predecessor; sequential time-separated measurements',
        'Generated output hashes differ across configurations, so decode histories and routing are not held equal',
        'K32->K64 changes expert placement and compute precision; no full-logit or independent task-quality acceptance',
        'Short sequential requests only; no concurrent, long-context or sustained-production validation',
        'Model containers stopped with exit 137 and OOMKilled=false; graceful shutdown remains unresolved']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    output = args.run.resolve() / 'reviewed-results.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','readiness','formal_requests','warmup_requests','source_hashes_verified',
                                           'predecessor_hashes_verified','monitoring')}, ensure_ascii=False, indent=2))
    for case in result['cases']:
        print(case['prompt_tokens'], json.dumps(case['comparisons']))
    print('Reviewed evidence:', output)
