"""Reject silent hybrid fallback and summarize a complete controlled matrix."""
import json
import math
import re
import statistics

PHASES = ('baseline','stream','bundle')
METRICS = ('ttft_seconds','server_prefill_seconds','server_prefill_tokens_per_second',
           'tpot_ms','decode_tokens_per_second','total_seconds')
FALLBACK = r'streaming failed|hybrid fallback|missing dims .*hybrid|inline resident L\d+ failed|dynamic residency failed'
AUDIT = r'\[KT PREFILL AUDIT\] layer=(\d+) tokens=(\d+) handled=True'


def startup(log, phase):
    records = re.findall(r'\[PREFILL CONFIG\] (\{[^\n]+\})',log)
    if len(records) != 1:
        raise RuntimeError('Missing or duplicate resolved prefill environment')
    config = json.loads(records[0])
    stream, bundle = int(phase != 'baseline'), int(phase == 'bundle')
    for key, expected in dict(PREFILL_PHASE=phase, ASCEND_RT_VISIBLE_DEVICES='0',
            TASK_QUEUE_ENABLE='1', KT_PREFILL_STREAM=str(stream), KT_DYNAMIC_RESIDENT=str(bundle),
            KT_SIDE_STREAM=str(bundle), KT_SHARED_EXPERTS_STREAM='0',
            KT_MXFP4_DEPOOL=str(stream), KT_MXFP4_GGUF_DEDUP=str(stream),
            KT_PREFILL_STREAM_THRESHOLD='512', KT_STREAM_WARMUP='1').items():
        if config.get(key) != expected:
            raise RuntimeError('Prefill environment mismatch: '+key)
    for marker in ('tp_size=1','ep_size=1','dp_size=1','kt_num_gpu_experts=32',
            'kt_cpuinfer=96','kt_threadpool_count=4','max_total_tokens=16384',
            'max_running_requests=1','chunked_prefill_size=8192','swa_full_tokens_ratio=0.5',
            'enable_metrics=True','stream_interval=1','disable_cuda_graph=False',
            'Capture target decode NPU graph end'):
        if marker not in log:
            raise RuntimeError('Missing resolved option/capture: '+marker)
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception|'+FALLBACK,log):
        raise RuntimeError('Startup exception or streaming fallback')
    if stream and '[KT_STREAM][depool] reserved ND streaming slot' not in log:
        raise RuntimeError('Streaming HBM slot was not reserved')
    return dict(passed=True, config=config, graph=True, tp=1,ep=1,experts=32)


def request_audit(delta, phase, prompt_tokens):
    if re.search(FALLBACK,delta):
        raise RuntimeError('Streaming/dynamic resident failed; hybrid timing must not be labeled optimized')
    records = [(int(layer),int(tokens)) for layer,tokens in re.findall(AUDIT,delta)]
    expected = [(layer,prompt_tokens) for layer in range(43)] if phase != 'baseline' and prompt_tokens >= 512 else []
    if records != expected:
        raise RuntimeError(f'Unexpected streaming execution: expected {len(expected)} layers, got {len(records)}')
    return dict(passed=True, streamed_layers=len(records), threshold=512,
                stream_expected=bool(expected), records=records)


def summarize(rows, cases):
    expected = {(p,c['name'],i) for p in PHASES for c in cases for i in range(4)}
    if len(rows) != len(expected) or {(r['phase'],r['case'],r['iteration']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicated prefill matrix')
    lengths = {c['name']:c['prompt_tokens'] for c in cases}
    for row in rows:
        if row['kind'] != ('warmup' if row['iteration'] == 0 else 'formal'):
            raise ValueError('Warmup/formal identity changed')
        if row['usage'] != dict(prompt_tokens=lengths[row['case']],completion_tokens=64):
            raise ValueError('Token budget changed')
        if len(row['output_ids']) != 64 or not row['streaming_audit']['passed']:
            raise ValueError('Missing output IDs or streaming audit')
        if any(not math.isfinite(row[key]) or row[key] <= 0 for key in METRICS):
            raise ValueError('Missing/invalid timing')
    result = dict(formal_requests=36,warmup_requests=12,groups=[],comparisons=[],
                  results_require_review=True,correctness_accepted=False,
                  cold_first_long_request_is_warmup=True,independent_loads_per_phase=1)
    for phase in PHASES:
        for case in cases:
            group = [r for r in rows if r['phase']==phase and r['case']==case['name'] and r['kind']=='formal']
            qualified = [r for r in group if not r['content_anomaly']]
            result['groups'].append(dict(phase=phase,case=case['name'],prompt_tokens=case['prompt_tokens'],
                requests=len(group),normal_content_requests=len(qualified),metrics={
                    key:dict(median=statistics.median(r[key] for r in group),
                             min=min(r[key] for r in group),max=max(r[key] for r in group)) for key in METRICS},
                normal_content_decode_tokens_per_second=(statistics.median(r['decode_tokens_per_second'] for r in qualified)
                                                         if qualified else None)))
    by_key = {(g['phase'],g['case']):g for g in result['groups']}
    for case in cases:
        baseline = by_key['baseline',case['name']]['metrics']
        for phase in PHASES[1:]:
            actual = by_key[phase,case['name']]['metrics']
            result['comparisons'].append(dict(phase=phase,case=case['name'],
                ttft_speedup=baseline['ttft_seconds']['median']/actual['ttft_seconds']['median'],
                prefill_speedup=baseline['server_prefill_seconds']['median']/actual['server_prefill_seconds']['median']))
    return result
