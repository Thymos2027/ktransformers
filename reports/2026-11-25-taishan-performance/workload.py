"""Native token streaming with scheduler timing and separate content observations."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import urllib.request

BASE = 'http://127.0.0.1:18082'


def consume(lines, started, clock=time.monotonic, safe=lambda: None):
    events, final, done = [], None, False
    previous = 0
    for line in lines:
        safe()
        elapsed = clock()-started
        if elapsed > 600:
            raise TimeoutError('Generation exceeded 600-second wall budget')
        if not line.startswith(b'data: '):
            continue
        raw = line[6:].strip()
        if raw == b'[DONE]':
            done = True
            break
        value = json.loads(raw)
        if 'error' in value:
            raise RuntimeError(f'Native generation error: {value["error"]}')
        meta = value.get('meta_info', {})
        count = meta.get('completion_tokens', 0)
        if count < previous:
            raise RuntimeError('Stream completion token count decreased')
        if count > previous:
            events.append(dict(seconds=elapsed, completion_tokens=count))
            previous = count
        final = value
    if not done or final is None or not events:
        raise RuntimeError('Incomplete native SSE response')
    meta = final.get('meta_info', {})
    count = meta.get('completion_tokens', 0)
    finish = (meta.get('finish_reason') or {}).get('type')
    if finish not in ('length', 'stop') or count < 1:
        raise RuntimeError(f'Invalid completion metadata: {meta}')
    prefill = meta.get('prefill_finished_time', 0)-meta.get('forward_entry_time', 0)
    if not math.isfinite(prefill) or prefill <= 0:
        raise RuntimeError('Scheduler prefill timestamps missing or invalid; metadata='+json.dumps(meta))
    first, last = events[0], events[-1]
    tpot = None
    if count > 1 and first['completion_tokens'] == 1 and last['seconds'] > first['seconds']:
        tpot = (last['seconds']-first['seconds'])/(count-1)
    text = final.get('text', '')
    return dict(content=text, content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        usage=dict(prompt_tokens=meta.get('prompt_tokens'), completion_tokens=count),
        done=done, finish_reason=finish, token_events=events, meta_info=meta,
        ttft_seconds=first['seconds'], total_seconds=clock()-started,
        tpot_ms=None if tpot is None else tpot*1000,
        decode_tokens_per_second=None if tpot is None else 1/tpot,
        server_prefill_seconds=prefill,
        server_prefill_tokens_per_second=meta['prompt_tokens']/prefill,
        event_token_jumps=sum(e['completion_tokens']-(events[i-1]['completion_tokens'] if i else 0)!=1
                              for i,e in enumerate(events)))


def generate(core, ids, count, rank=None, ignore_eos=True):
    payload = dict(input_ids=ids, stream=True, sampling_params=dict(
        temperature=0, sampling_seed=20261111, max_new_tokens=count, ignore_eos=ignore_eos))
    if rank is not None:
        payload['routed_dp_rank'] = rank
    req = urllib.request.Request(BASE+'/generate', data=json.dumps(payload).encode(),
                                 headers={'Content-Type':'application/json'})
    started_at, started = core.now(), time.monotonic()
    with core.HTTP.open(req, timeout=600) as response:
        result = consume(response, started, safe=core.safe)
    result.update(started_at=started_at, payload=payload, interface='native_generate_token_stream')
    if result['usage']['prompt_tokens'] != len(ids):
        raise RuntimeError('Model prompt token count differs from frozen input')
    if ignore_eos and (result['usage']['completion_tokens'] != count or result['finish_reason'] != 'length'):
        raise RuntimeError('Fixed-length performance request stopped early')
    if count > 1 and result['tpot_ms'] is None:
        raise RuntimeError('Cannot calculate token interval; first stream event was aggregated')
    return result


def append(core, name, value):
    with (core.ROOT/name).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')
        stream.flush()


def tokenize(core, messages):
    return core.request(BASE, '/tokenize', dict(messages=messages), timeout=30)


def freeze(core):
    historical = json.loads((core.ROOT/'requests.json').read_text())
    for case in historical:
        actual = tokenize(core, case['messages'])
        if actual['tokens'] != case['token_ids'] or actual['count'] != case['prompt_tokens']:
            raise RuntimeError('Historical tokenizer/chat encoding changed')
    filler = ('CPU expert weights live in host memory. NPU attention uses local KV state. '
              'Benchmark prefill and decode independently with fixed resource budgets. ')*1400
    long_cases = []
    for target in (1024,4096,7168):
        low, high, best = 0, len(filler), None
        while low <= high:
            size = (low+high)//2
            messages = [dict(role='user', content=filler[:size]+'\nWrite a detailed engineering guide on CPU and NPU inference.')]
            value = tokenize(core, messages)
            if value['count'] <= target:
                best = dict(name=f'prefill_{target}', target_tokens=target, prompt_tokens=value['count'],
                            messages=messages, token_ids=value['tokens'])
                low = size+1
            else:
                high = size-1
        if best is None or best['prompt_tokens'] < target-16:
            raise RuntimeError('Could not construct a near-target token budget')
        long_cases.append(best)
    result = dict(short=historical, prefill=long_cases, seed=20261111,
                  frozen_at=core.now(), method='tokenize before measurements; native input_ids reused unchanged')
    core.atomic(core.ROOT/'frozen-workloads.json', result)
    return result


def functional(core, phase, dpa):
    ranks = (0,1) if dpa else (None,)
    for rank in ranks:
        for name,prompt,expected in (
            ('arithmetic','Reply with only the decimal result of 12 * 13.','156'),
            ('sorting','Return only a JSON array sorting [9,1,5,2] in ascending order.',[1,2,5,9])):
            ids = tokenize(core,[dict(role='user',content=prompt)])['tokens']
            result = generate(core, ids, 64, rank=rank, ignore_eos=False)
            text = result['content'].strip()
            try:
                correct = json.loads(text)==expected if isinstance(expected,list) else text==expected
            except ValueError:
                correct = False
            result.update(phase=phase, case=name, dp_rank=rank, answer_matches=correct,
                          observation_only=True)
            append(core,'functional-observations.jsonl',result)
            if not text.strip() or not any(c.isalnum() for c in text):
                raise RuntimeError('Empty or structurally unusable output')


def measure(core, phase, cases, output_tokens, dpa):
    for case in cases:
        for iteration in range(4):
            rank = iteration%2 if dpa else None
            core.status('measuring', phase_id=phase, case=case['name'], iteration=iteration,
                        output_tokens=output_tokens, dp_rank=rank)
            result = generate(core,case['token_ids'],output_tokens,rank=rank)
            result.update(phase=phase, case=case['name'], iteration=iteration,
                          kind='warmup' if iteration==0 else 'formal', dp_rank=rank)
            append(core,'requests.jsonl',result)


def summarize(core, phases):
    rows = [json.loads(x) for x in (core.ROOT/'requests.jsonl').read_text().splitlines()]
    expected = {(phase,case,i) for phase,_,_,wide in phases
                for case in (('prefill_1024','prefill_4096','prefill_7168') if wide else ('input_128','input_768'))
                for i in range(4)}
    if len(rows)!=len(expected) or {(r['phase'],r['case'],r['iteration']) for r in rows}!=expected:
        raise RuntimeError('Incomplete or duplicate performance matrix')
    metrics = ('ttft_seconds','tpot_ms','decode_tokens_per_second','server_prefill_seconds',
               'server_prefill_tokens_per_second','total_seconds')
    output = dict(formal_requests=sum(r['kind']=='formal' for r in rows),
                  warmup_requests=sum(r['kind']=='warmup' for r in rows), groups=[],
                  quality_observations_are_not_performance_gates=True, results_require_review=True)
    for phase,case in sorted({(r['phase'],r['case']) for r in rows}):
        group = [r for r in rows if r['phase']==phase and r['case']==case and r['kind']=='formal']
        output['groups'].append(dict(phase=phase,case=case,requests=len(group),metrics={
            key:dict(median=statistics.median(r[key] for r in group),
                     min=min(r[key] for r in group),max=max(r[key] for r in group)) for key in metrics}))
    core.atomic(core.ROOT/'summary.json',output)
