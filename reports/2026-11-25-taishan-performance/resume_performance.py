"""Reuse only complete performance evidence collected before profiler failure."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

REUSED_PHASES = ('dpaeager','dpagraph')
IDENTITY_FILES = ('launch.sh','workload.py','prepare_candidate.py','attn_prepare.py',
    'ep_prepare.py','ep_layout.py','startup_gate.py','tp_runtime.py','base_runtime.py',
    'environment_probe.py','device_probe.py','request_budget_probe.py','hccl_probe.py',
    'combine_probe.py','graph_gate.py','requests.json','thermal_policy.py','transport_policy.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_rows(rows, frozen, phases=REUSED_PHASES):
    by_case={case['name']:case for case in frozen['short']}
    expected={(phase,case,i) for phase in phases for case in by_case for i in range(4)}
    if len(rows)!=len(expected) or {(r['phase'],r['case'],r['iteration']) for r in rows}!=expected:
        raise ValueError('Checkpoint performance matrix is missing or duplicated')
    for row in rows:
        case=by_case[row['case']]
        payload=row['payload']
        if payload['input_ids']!=case['token_ids'] or payload['sampling_params']!=dict(
                temperature=0,sampling_seed=20261111,max_new_tokens=512,ignore_eos=True):
            raise ValueError('Checkpoint input or sampling parameters changed')
        if row['kind']!=('warmup' if row['iteration']==0 else 'formal'):
            raise ValueError('Checkpoint warmup/formal identity changed')
        if row['dp_rank']!=row['iteration']%2 or payload['routed_dp_rank']!=row['dp_rank']:
            raise ValueError('Checkpoint DP routing changed')
        if row['usage']!=dict(prompt_tokens=case['prompt_tokens'],completion_tokens=512):
            raise ValueError('Checkpoint output token budget is incomplete')
        if not row['done'] or row['finish_reason']!='length':
            raise ValueError('Checkpoint native stream did not complete')
        if hashlib.sha256(row['content'].encode()).hexdigest()!=row['content_sha256']:
            raise ValueError('Checkpoint output digest mismatch')
        events=row['token_events']
        counts=[e['completion_tokens'] for e in events]
        times=[e['seconds'] for e in events]
        if counts[0]!=1 or counts[-1]!=512 or any(a>=b for a,b in zip(counts,counts[1:])):
            raise ValueError('Checkpoint token event counts are invalid')
        if times!=sorted(times) or times[-1]<=times[0] or times[0]!=row['ttft_seconds'] or times[-1]>row['total_seconds']:
            raise ValueError('Checkpoint stream event timestamps are invalid')
        tpot=(times[-1]-times[0])/511
        if not math.isclose(row['tpot_ms'],1000*tpot,rel_tol=1e-12) or not math.isclose(
                row['decode_tokens_per_second'],1/tpot,rel_tol=1e-12):
            raise ValueError('Checkpoint TPOT/decode calculation mismatch')
        meta=row['meta_info']
        prefill=meta['prefill_finished_time']-meta['forward_entry_time']
        if prefill<=0 or not math.isclose(prefill,row['server_prefill_seconds'],rel_tol=1e-12) or not math.isclose(
                case['prompt_tokens']/prefill,row['server_prefill_tokens_per_second'],rel_tol=1e-12):
            raise ValueError('Checkpoint scheduler prefill calculation mismatch')
    return len(rows)


def environment(root):
    lines=(root/'logs/environment.log').read_text(encoding='utf-8').splitlines()
    records=[json.loads(line.split('KT environment: ',1)[1]) for line in lines if line.startswith('KT environment: ')]
    if len(records)!=1 or not records[0]['passed']:
        raise ValueError('Checkpoint native environment identity is absent')
    return {key:records[0][key] for key in ('torch','torch_npu','extension_sha256')}


def restore(source, target, startup_check):
    source,target=Path(source).resolve(),Path(target).resolve()
    if source==target or not source.is_dir():
        raise ValueError('Checkpoint source must be an existing, distinct run')
    cleanup=json_read(source/'cleanup.json')
    if not cleanup['passed'] or cleanup['original_running']!='false':
        raise ValueError('Checkpoint run did not verify cleanup')
    for name in IDENTITY_FILES:
        if sha(source/name)!=sha(target/name):
            raise ValueError('Checkpoint runtime identity mismatch: '+name)
    for name in ('candidate.json','weight-preflight.json'):
        if json_read(source/name)!=json_read(target/name):
            raise ValueError('Checkpoint source/weight identity mismatch: '+name)
    if environment(source)!=environment(target):
        raise ValueError('Checkpoint native extension or torch identity mismatch')
    config=json_read(source/'benchmark-config.json')
    new_config=json_read(target/'benchmark-config.json')
    for key in ('image','phases','seed','cpu_threads','threadpools','experts','interface',
                'short_output','prefill_output','repeats','warmups','prefill_metric','thermal_policy'):
        if config[key]!=new_config[key]:
            raise ValueError('Checkpoint measurement configuration changed: '+key)
    frozen=json_read(source/'frozen-workloads.json')
    if sha(source/'frozen-workloads.json')!=json_read(source/'frozen-workloads-sha256.json')['sha256']:
        raise ValueError('Checkpoint frozen workload digest mismatch')
    for case,historical in zip(frozen['short'],json_read(source/'requests.json')):
        if case!=historical:
            raise ValueError('Checkpoint historical input identity changed')
    rows=[json.loads(x) for x in (source/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    verify_rows(rows,frozen)
    frozen_bundle=json_read(source/'bundle-sha256.json')
    for name,digest in frozen_bundle.items():
        if sha(source/name)!=digest:
            raise ValueError('Source bundle changed after launch: '+name)
    copy_files=['requests.jsonl','functional-observations.jsonl','frozen-workloads.json','frozen-workloads-sha256.json']
    cutoffs={}
    for phase in REUSED_PHASES:
        log=(source/'logs'/f'{phase}.log').read_text(encoding='utf-8')
        marker='SIGTERM received' if phase=='dpaeager' else 'Profiling starts.'
        if marker not in log:
            raise ValueError('Expected post-measurement boundary missing: '+phase)
        prefix=log.split(marker,1)[0]
        startup_check(prefix,True,phase=='dpagraph',False)
        cutoffs[phase]=dict(boundary=marker,prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                           full_log_contains_later_failure=phase=='dpagraph')
        copy_files += [f'logs/{phase}.log',f'{phase}-startup-audit.json',f'{phase}-server-info.json',
                       f'{phase}-before-resources.json',f'{phase}-after-resources.json']
    observed=[json.loads(x) for x in (source/'functional-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(observed)!=8 or any(x['phase'] not in REUSED_PHASES or not x['done'] for x in observed):
        raise ValueError('Checkpoint functional observations incomplete')
    for row in observed:
        if hashlib.sha256(row['content'].encode()).hexdigest()!=row['content_sha256']:
            raise ValueError('Checkpoint functional output digest mismatch')
    digests={name:sha(source/name) for name in copy_files}
    for name in copy_files:
        destination=target/name
        if destination.exists():
            raise ValueError('Refuse to overwrite current run evidence: '+name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,destination)
        if sha(destination)!=digests[name]:
            raise ValueError('Checkpoint copy digest mismatch: '+name)
    return dict(source=str(source),reused_phases=list(REUSED_PHASES),requests=16,formal_requests=12,
                evidence_sha256=digests,accepted_log_boundaries=cutoffs,
                limitations='Prior phases precede a profiler-triggered failure; sequential cross-run comparison, no interleaved A/B',
                frozen=frozen)
