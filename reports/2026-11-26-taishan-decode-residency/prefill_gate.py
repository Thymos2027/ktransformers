"""Same token protocol as R1; verify actual prefill and decode promotion paths."""
import json
import re
import statistics

import prior_gate

PHASES=('a0','b0','b1','a1')
CANDIDATE={'b0','b1'}
OUTPUT_TOKENS=1000
validate_request=prior_gate.validate_request
capacity=prior_gate.capacity
FALLBACK=prior_gate.FALLBACK


def startup(log,phase):
    configs=re.findall(r'\[RESIDENCY CONFIG\] (\{[^\n]+\})',log)
    if len(configs)!=1: raise RuntimeError('Missing unique resolved residency environment')
    config=json.loads(configs[0])
    expected=dict(KT_PREFILL_STREAM='1',KT_DYNAMIC_RESIDENT='1',KT_SIDE_STREAM='1',
                  KT_MXFP4_DEPOOL='1',KT_MXFP4_GGUF_DEDUP='1',KT_STREAM_WARMUP='0',
                  KT_DECODE_RESIDENCY=str(int(phase in CANDIDATE)),KT_RESIDENCY_INTERVAL='32',
                  KT_RESIDENCY_BUDGET='8',KT_RESIDENCY_CACHE_MIB='1024',TASK_QUEUE_ENABLE='1')
    if any(config.get(k)!=v for k,v in expected.items()): raise RuntimeError('Resolved A/B configuration differs')
    for marker in ('tp_size=1','ep_size=1','dp_size=1','kt_num_gpu_experts=32','kt_cpuinfer=96',
                   'kt_threadpool_count=4','max_total_tokens=16384','max_running_requests=1',
                   'chunked_prefill_size=8192','swa_full_tokens_ratio=0.5',
                   'disable_cuda_graph=False','Capture target decode NPU graph end'):
        if marker not in log: raise RuntimeError('Missing startup marker: '+marker)
    side=sorted(map(int,re.findall(prior_gate.SIDE_AUDIT,log)))
    if side!=list(range(43)): raise RuntimeError('Captured side-stream forks differ')
    records=events(log)
    registered=sorted(r['layer'] for r in records if r['event']=='registered')
    if registered!=(list(range(43)) if phase in CANDIDATE else []):
        raise RuntimeError('Residency registration differs from A/B phase')
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception|'+FALLBACK,log):
        raise RuntimeError('Runtime failure or fallback')
    return dict(passed=True,config=config,registered_layers=registered)


def events(log):
    return [json.loads(s) for s in re.findall(r'\[KT RESIDENCY\] (\{[^\n]+\})',log)]


def request_audit(delta,phase,prompt_tokens):
    if re.search(FALLBACK,delta): raise RuntimeError('Streaming/resident update fallback')
    layers=[(int(a),int(b)) for a,b in re.findall(prior_gate.AUDIT,delta)]
    if layers!=[(i,prompt_tokens) for i in range(43)]: raise RuntimeError('Missing actual streaming layers')
    if 'inline resident:' not in delta: raise RuntimeError('Missing per-request prefill resident reset')
    updates=[r for r in events(delta) if r['event']=='update']
    if phase in CANDIDATE:
        if not updates:
            raise RuntimeError('Candidate did not execute periodic decode updates')
        for row in updates:
            if (row['completed_decodes']%32 or not 0<=row['promotions']<=8
                    or row['route_choices']!=row['completed_decodes']*43*6
                    or row['cache_bytes']>1024**3):
                raise RuntimeError('Residency budget or graph observation contract violated')
    elif updates:
        raise RuntimeError('Disabled baseline performed promotions')
    return dict(passed=True,streamed_layers=43,updates=updates,
                candidate_triggered=any(r['promotions'] for r in updates),
                promotions=updates[-1]['promotions_total'] if updates else 0,
                observation_coverage='Periodic boundaries; final partial window is not included in route counters')


def summarize(rows,cases):
    expected={(p,c['name'],i) for p in PHASES for c in cases for i in range(4)}
    if len(rows)!=len(expected) or {(r['phase'],r['case'],r['iteration']) for r in rows}!=expected:
        raise ValueError('Incomplete ABBA matrix')
    groups=[]
    for p in PHASES:
        for c in cases:
            selected=[r for r in rows if r['phase']==p and r['case']==c['name'] and r['kind']=='formal']
            for row in selected:
                validate_request(row,c['prompt_tokens'])
                if not row['streaming_audit']['passed']: raise ValueError('Missing execution evidence')
            groups.append(dict(phase=p,case=c['name'],metrics={k:statistics.median(r[k] for r in selected)
                          for k in ('ttft_seconds','tpot_ms','decode_tokens_per_second','total_seconds')},
                          promotions=[r['streaming_audit']['promotions'] for r in selected]))
    return dict(groups=groups,formal_requests=24,warmup_requests=8,independent_loads_per_arm=2,
                load_order=list(PHASES),quality_accepted=False,results_require_review=True,
                interpretation='Net feature A/B; does not isolate policy, host cache and observation overhead separately')
