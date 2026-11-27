"""Validate the four-arm TP1 comparison, actual paths and 1000-token evidence."""
import json
import math
import re
import statistics

PHASES = ('h0', 'h1', 's0', 's1')
STREAM = {'s0', 's1'}
SIDE = {'h1', 's1'}
OUTPUT_TOKENS = 1000
METRICS = ('ttft_seconds', 'server_prefill_seconds', 'server_prefill_tokens_per_second',
           'tpot_ms', 'decode_tokens_per_second', 'total_seconds')
FALLBACK = r'streaming failed|hybrid fallback|missing dims .*hybrid|inline resident L\d+ failed|dynamic residency failed'
AUDIT = r'\[KT PREFILL AUDIT\] layer=(\d+) tokens=(\d+) handled=True'
SIDE_AUDIT = r'\[KT SIDE AUDIT\] layer=(\d+) forked=True'


def startup(log, phase):
    if phase not in PHASES:
        raise ValueError('Unknown experiment phase')
    records = re.findall(r'\[PREFILL CONFIG\] (\{[^\n]+\})', log)
    if len(records) != 1:
        raise RuntimeError('Missing or ambiguous resolved environment')
    config = json.loads(records[0])
    stream, side = int(phase in STREAM), int(phase in SIDE)
    expected = dict(PREFILL_PHASE=phase, ASCEND_RT_VISIBLE_DEVICES='0', TASK_QUEUE_ENABLE='1',
        SGLANG_SET_CPU_AFFINITY='0', KT_PREFILL_STREAM=str(stream), KT_DYNAMIC_RESIDENT='0',
        KT_SIDE_STREAM=str(side), KT_SHARED_EXPERTS_STREAM='0', KT_MXFP4_DEPOOL=str(stream),
        KT_MXFP4_GGUF_DEDUP=str(stream), KT_PREFILL_STREAM_THRESHOLD='512', KT_STREAM_WARMUP='1',
        KT_HOT_TAIL_TOKENS='0')
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Resolved feature switches do not match phase')
    for marker in ('tp_size=1', 'ep_size=1', 'dp_size=1', 'kt_num_gpu_experts=32',
                   'kt_cpuinfer=96', 'kt_threadpool_count=4', 'max_total_tokens=16384',
                   'max_running_requests=1', 'chunked_prefill_size=8192',
                   'swa_full_tokens_ratio=0.5', 'enable_metrics=True', 'stream_interval=1',
                   'disable_cuda_graph=False', 'Capture target decode NPU graph end'):
        if marker not in log:
            raise RuntimeError('Missing startup evidence: '+marker)
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception|'+FALLBACK, log):
        raise RuntimeError('Startup/runtime failure')
    if stream and '[KT_STREAM][depool] reserved ND streaming slot' not in log:
        raise RuntimeError('Streaming HBM slot not reserved')
    side_layers = sorted(map(int, re.findall(SIDE_AUDIT, log)))
    if side_layers != (list(range(43)) if side else []):
        raise RuntimeError('Actual captured side-stream forks differ from configured phase')
    return dict(passed=True, config=config, side_layers=side_layers, graph=True, tp=1, ep=1)


def capacity(log, info):
    # The first line may be an unbounded budget; the last is the finalized allocation.
    pools = re.findall(r'DSV4 pool sizes: full=(\d+), swa=(\d+)', log)
    if not pools:
        raise RuntimeError('Actual full/SWA allocation missing')
    full, swa = map(int, pools[-1])
    required = 4096+1024+512+128
    if (full != 16384 or swa != 8192 or info.get('max_total_num_tokens') != full
            or info.get('context_length') != 8192 or min(full, swa) < required):
        raise RuntimeError('Finalized KV pools do not meet frozen request budget')
    return dict(passed=True, full=full, swa=swa, required_positions=required,
                budget_lines=len(pools), selection='Last finalized allocation plus server_info')


def request_audit(delta, phase, prompt_tokens):
    if re.search(FALLBACK, delta):
        raise RuntimeError('Streaming failed or fell back')
    records = [(int(layer), int(tokens)) for layer, tokens in re.findall(AUDIT, delta)]
    expected = [(layer, prompt_tokens) for layer in range(43)] if phase in STREAM and prompt_tokens >= 512 else []
    if records != expected:
        raise RuntimeError('Unexpected successful streaming layers')
    if 'inline resident:' in delta or 'dynamic resident applied:' in delta:
        raise RuntimeError('Dynamic residency must remain disabled')
    return dict(passed=True, records=records, streamed_layers=len(records), threshold=512)


def validate_request(row, prompt_tokens):
    events = row['token_events']
    if (row['usage'] != dict(prompt_tokens=prompt_tokens, completion_tokens=OUTPUT_TOKENS)
            or len(row['output_ids']) != OUTPUT_TOKENS
            or any(type(token) is not int for token in row['output_ids']) or not row['done']
            or row['finish_reason'] != 'length' or row['content_anomaly']):
        raise RuntimeError('Incomplete or anomalous fixed-length model output')
    counts = [event['completion_tokens'] for event in events]
    times = [event['seconds'] for event in events]
    if len(events) < 2 or any(type(count) is not int for count in counts):
        raise RuntimeError('Missing or invalid cumulative token notifications')
    if counts[0] != 1:
        raise RuntimeError('First token arrival was not individually observed; exact TTFT unavailable')
    if counts[-1] != OUTPUT_TOKENS:
        raise RuntimeError('Final token notification is incomplete')
    if any(not 1 <= count <= OUTPUT_TOKENS for count in counts) or any(
            b <= a for a, b in zip(counts, counts[1:])):
        raise RuntimeError('Cumulative token counts must strictly increase within the output budget')
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in times) or any(
            b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError('Invalid or non-increasing notification timestamps')
    if not math.isclose(times[0], row['ttft_seconds'], rel_tol=1e-12):
        raise RuntimeError('TTFT differs from first token notification')
    actual = 1000*(times[-1]-times[0])/(OUTPUT_TOKENS-1)
    if not math.isclose(actual, row['tpot_ms'], rel_tol=1e-12):
        raise RuntimeError('TPOT differs from raw events')
    # SGLang may emit the latest cumulative result after draining queued outputs.
    # Interior notification gaps do not change a TPOT measured from observed
    # endpoints. Never interpret notification array offsets as token numbers.
    by_count = dict(zip(counts, times))
    windows = ((1,64), (65,256), (257,512), (513,1000))
    missing = sorted({count for pair in windows for count in pair if count not in by_count})
    jumps = [dict(previous_count=a, current_count=b, unobserved_counts=list(range(a+1,b)))
             for a, b in zip(counts, counts[1:]) if b-a > 1]
    if row.get('event_token_jumps', len(jumps)) != len(jumps):
        raise RuntimeError('Saved notification jump count differs from raw events')
    return dict(window_tpot_ms={
        f'{a}-{b}': (1000*(by_count[b]-by_count[a])/(b-a)
                    if a in by_count and b in by_count else None) for a,b in windows},
        timing_audit=dict(passed=True, notification_count=len(events), output_tokens=OUTPUT_TOKENS,
            cumulative_count_jumps=jumps, unobserved_intermediate_counts=OUTPUT_TOKENS-len(events),
            per_token_arrivals_complete=not jumps, missing_window_boundaries=missing,
            exact_whole_decode_endpoints=True, interpolated_timestamps=False))


def summarize(rows, cases):
    expected = {(phase, case['name'], i) for phase in PHASES for case in cases for i in range(4)}
    if len(rows) != len(expected) or {(r['phase'], r['case'], r['iteration']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate experiment matrix')
    lengths = {c['name']: c['prompt_tokens'] for c in cases}
    for row in rows:
        if row['kind'] != ('warmup' if row['iteration'] == 0 else 'formal'):
            raise ValueError('Warmup/formal identity changed')
        checked = validate_request(row, lengths[row['case']])
        if row['window_tpot_ms'] != checked['window_tpot_ms']:
            raise ValueError('Saved window timing differs from token-number endpoints')
        if not row['streaming_audit']['passed'] or any(not math.isfinite(row[k]) or row[k] <= 0 for k in METRICS):
            raise ValueError('Missing execution/timing evidence')
    groups = []
    for phase in PHASES:
        for case in cases:
            selected = [r for r in rows if r['phase'] == phase and r['case'] == case['name'] and r['kind'] == 'formal']
            valid_windows = {key: [r['window_tpot_ms'][key] for r in selected
                                  if r['window_tpot_ms'][key] is not None]
                             for key in selected[0]['window_tpot_ms']}
            audits = [validate_request(r, case['prompt_tokens'])['timing_audit'] for r in selected]
            groups.append(dict(phase=phase, case=case['name'], prompt_tokens=case['prompt_tokens'],
                metrics={key: dict(median=statistics.median(r[key] for r in selected),
                         minimum=min(r[key] for r in selected), maximum=max(r[key] for r in selected)) for key in METRICS},
                window_tpot_ms={key: statistics.median(values) if values else None
                                for key, values in valid_windows.items()},
                window_sample_counts={key: len(values) for key, values in valid_windows.items()},
                requests_with_count_jumps=sum(not audit['per_token_arrivals_complete'] for audit in audits),
                unobserved_intermediate_counts=sum(audit['unobserved_intermediate_counts'] for audit in audits)))
    return dict(formal_requests=24, warmup_requests=8, groups=groups, results_require_review=True,
                independent_loads_per_phase=1, shared_host=True, correctness_accepted=False)
