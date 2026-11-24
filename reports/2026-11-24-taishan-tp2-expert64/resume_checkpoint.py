"""Validate completed phases before copying them into an explicitly resumed run."""
import hashlib
import json
import math
from pathlib import Path
import shutil


REUSED_PHASES = ('tp2k64eager', 'tp1k32graph')
CONFIG_KEYS = ('phases', 'image', 'max_total_tokens', 'cpu_threads', 'threadpool_count',
               'random_seed', 'output_tokens', 'ignore_eos', 'warmups_per_case',
               'formal_repeats_per_case', 'requests_sha256', 'thermal_policy')


def require(value, message):
    if not value:
        raise ValueError('Checkpoint rejected: ' + message)


def read(root, name):
    return json.loads((root / name).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(source, target, startup_check):
    source, target = Path(source).resolve(), Path(target).resolve()
    require(source != target, 'source and target must differ')
    before, current = read(source, 'benchmark-config.json'), read(target, 'benchmark-config.json')
    for key in CONFIG_KEYS:
        require(before[key] == current[key], f'configuration changed: {key}')
    for key in ('base_sha256', 'candidate_sha256'):
        require(read(source, 'candidate.json')[key] == read(target, 'candidate.json')[key],
                f'forward implementation changed: {key}')
    require(read(source, 'source.json')['base_commit'] == read(target, 'source.json')['base_commit'],
            'SGLang base commit changed')
    for key in ('metadata_sha256', 'model_shards', 'cpu_expert_files'):
        require(read(source, 'weight-preflight.json')[key] == read(target, 'weight-preflight.json')[key],
                f'weight metadata changed: {key}')
    require(read(source, 'environment-preflight.json') == read(target, 'environment-preflight.json'),
            'native backend identity changed')
    for name in ('requests.json', 'launch.sh', 'workload.py'):
        require(digest(source / name) == digest(target / name), f'input/launch code changed: {name}')
    require(digest(source / 'requests.json') == before['requests_sha256'], 'frozen request digest differs')
    cleanup = read(source, 'cleanup.json')
    require(cleanup['passed'] and cleanup['original_running'] == 'false'
            and not cleanup['port_18082_open'], 'predecessor did not clean up')
    require(read(source, 'status.json')['phase'] in ('failed', 'aborted_thermal_or_telemetry'),
            'predecessor is not a stopped interrupted run')
    copied = ['requests.jsonl']
    checked = ['benchmark-config.json', 'candidate.json', 'source.json', 'weight-preflight.json',
               'environment-preflight.json', 'requests.json', 'launch.sh', 'workload.py',
               'cleanup.json', 'status.json']
    for phase, tp, experts, graph in (('tp2k64eager', 2, 64, False), ('tp1k32graph', 1, 32, True)):
        logfile = f'logs/{phase}.log'
        log = (source / logfile).read_text(encoding='utf-8', errors='replace')
        serving = log.partition('SIGTERM received.')[0]
        audit = startup_check(serving, tp, experts, graph)
        require(audit == read(source, f'{phase}-startup-audit.json'), f'{phase} audit differs')
        copied += [logfile, f'{phase}-startup-audit.json',
                   f'{phase}-before-resources.json', f'{phase}-after-resources.json']
        if not graph:
            smoke = read(source, f'{phase}-smoke.json')
            require(len(smoke) == 3, 'incomplete eager smoke')
            for result, expected in zip(smoke, ('巴黎', '42', 'READY')):
                require(result['passed'] and result['done'] and result['finish_reason'] == 'stop'
                        and result['content'].strip() == expected, 'eager smoke failed')
            copied.append(f'{phase}-smoke.json')
        else:
            gate = read(source, f'{phase}-gate.json')
            require(len(gate) == 5, 'incomplete baseline task gate')
            for result, expected in zip(gate, ('巴黎', '42', '156', 'READY', [1, 2, 5, 9])):
                actual = result['content'].strip()
                if isinstance(expected, list):
                    actual = json.loads(actual)
                require(result['passed'] and result['done'] and result['finish_reason'] == 'stop'
                        and actual == expected, 'baseline task answer/transport failed')
            copied.append(f'{phase}-gate.json')
    cases = {x['name']: x for x in read(source, 'requests.json')}
    rows = [json.loads(line) for line in (source / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(rows) == 8, 'expected exactly 8 completed baseline requests')
    expected_ids = {(name, i) for name in cases for i in range(4)}
    require({(x['case'], x['iteration']) for x in rows} == expected_ids, 'missing/repeated baseline request')
    for row in rows:
        case = cases[row['case']]
        require(row['phase'] == 'tp1k32graph' and row['tp'] == 1
                and row['kind'] == ('warmup' if row['iteration'] == 0 else 'formal'), 'baseline phase identity changed')
        require(row['done'] and row['finish_reason'] == 'length'
                and row['usage']['completion_tokens'] == 512
                and row['usage']['prompt_tokens'] == case['prompt_tokens'], 'incomplete request transport/token count')
        payload = row['payload']
        require(payload['messages'] == case['messages'] and payload['max_tokens'] == 512
                and payload['ignore_eos'] and payload['temperature'] == 0
                and payload['seed'] == 20261111, 'request payload changed')
        require(hashlib.sha256(row['content'].encode()).hexdigest() == row['content_sha256'], 'output hash differs')
        require(''.join(x['text'] for x in row['chunks']) == row['content'], 'SSE chunks differ from output')
        times = [x['seconds'] for x in row['chunks']]
        require(times == sorted(times) and len(times) > 1 and times[-1] > times[0]
                and times[0] == row['ttft_seconds'] and times[-1] == row['last_content_seconds']
                and times[-1] <= row['total_seconds'], 'SSE timing evidence invalid')
        require(math.isclose(511/(times[-1]-times[0]), row['decode_tokens_per_second'], rel_tol=1e-12),
                'decode metric differs from raw timestamps')
    all_names = sorted(set(copied + checked))
    for name in all_names:
        path = (source / name).resolve()
        require(path.is_relative_to(source) and path.is_file(), f'missing or external artifact: {name}')
    return dict(source_root=str(source), reused_phases=list(REUSED_PHASES), baseline_requests=8,
                baseline_formal_requests=6, copied_files=copied,
                source_sha256={name: digest(source / name) for name in all_names},
                comparison_limit='TP1 control is reused across a transport interruption; exploratory sequential comparison')


def copy_verified(record, target):
    source, target = Path(record['source_root']), Path(target)
    for name, expected in record['source_sha256'].items():
        require(digest(source / name) == expected, f'source changed after validation: {name}')
    for name in record['copied_files']:
        destination = target / name
        require(not destination.exists(), f'target artifact already exists: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        require(digest(destination) == record['source_sha256'][name], f'copied artifact hash mismatch: {name}')
