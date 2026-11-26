"""Use the upstream prompts and report both text-growth and token-event timing."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import urllib.request

import prefill_gate
import workload

PHASES = {'table12': (130,1000,4000,8000), 'table19': (16000,), 'table39': (32000,)}


def tokenize_prompt(core, text):
    # /generate accepts text, whereas this fork's /tokenize schema requires prompt.
    return core.request(workload.BASE,'/tokenize',dict(prompt=text),timeout=30)


def prompt_builder(text):
    blocks = re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", text, re.S)
    matches = [b for b in blocks if 'NAT_SHORT' in b and 'chars_needed' in b]
    if len(matches) != 1:
        raise ValueError('Upstream prompt builder changed')
    return matches[0]


def build_cases(core):
    sources = json.loads((core.ROOT/'sources.json').read_text())
    for record in sources:
        if hashlib.sha256((core.ROOT/record['file']).read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError('Public source hash mismatch')
    code = prompt_builder((core.ROOT/'decode_throughput_test.upstream.sh').read_text())
    directory = core.ROOT/'prompts'
    directory.mkdir()
    builder = directory/'upstream_prompt_builder.py'
    builder.write_text(code+'\n')
    cases = []
    for phase, targets in PHASES.items():
        for target in targets:
            prompt = directory/f'prompt-{target}.txt'
            request = directory/f'request-{target}.json'
            core.run([sys.executable,str(builder),str(target),str(prompt),'4.6','1000',
                      str(directory/f'probe-{target}.json'),str(request)])
            payload = json.loads(request.read_text())
            if payload['text'] != prompt.read_text() or payload['sampling_params'] != dict(
                    max_new_tokens=1000,temperature=0,ignore_eos=True):
                raise RuntimeError('Upstream request settings differ')
            actual = tokenize_prompt(core,payload['text'])
            count, ids = actual['count'], actual['tokens']
            if count != len(ids) or not 0 < count <= 32768 or count+1000 > 65536:
                raise RuntimeError('Prompt exceeds tutorial single-chunk/context budget')
            cases.append(dict(phase=phase,target=target,prompt_tokens=count,token_ids=ids,
                              payload=payload,prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest()))
    core.atomic(core.ROOT/'frozen-cases.json',cases)
    return cases


def validate_result(result, count):
    if result['usage'] != dict(prompt_tokens=count,completion_tokens=1000):
        raise RuntimeError('Input/output token budget changed')
    if result['finish_reason'] != 'length' or not result['done']:
        raise RuntimeError('Generation did not complete')
    if len(result['output_ids']) != 1000:
        raise RuntimeError('Missing output token IDs')


def generate(core, case, iteration):
    payload = dict(case['payload'],stream=iteration!=0)
    started = time.monotonic()
    if iteration == 0:
        raw = core.request(workload.BASE,'/generate',payload,timeout=600)
        meta = raw.get('meta_info',{})
        core.atomic(core.ROOT/f"warmup-{case['target']}.json",raw)
        if (meta.get('completion_tokens') != 1000 or meta.get('prompt_tokens') != case['prompt_tokens']
                or (meta.get('finish_reason') or {}).get('type') != 'length'):
            raise RuntimeError('Non-stream warmup failed or stopped early')
        return dict(phase=case['phase'],target=case['target'],iteration=0,kind='warmup',
                    total_seconds=time.monotonic()-started,payload=payload,meta_info=meta,
                    content=raw.get('text'),output_ids=raw.get('output_ids'))
    request = urllib.request.Request(workload.BASE+'/generate',data=json.dumps(payload).encode(),
                                    headers={'Content-Type':'application/json'})
    growth = dict(length=0,first=None,last=None)
    def observed(response):
        for line in response:
            if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
                event = json.loads(line[6:])
                text = event.get('text')
                if isinstance(text,str) and len(text)>growth['length']:
                    elapsed = time.monotonic()-started
                    if growth['first'] is None:
                        growth['first'] = elapsed
                    growth['last'],growth['length'] = elapsed,len(text)
            yield line
    with core.HTTP.open(request,timeout=600) as response:
        result = workload.consume(observed(response),started,safe=core.safe)
    validate_result(result,case['prompt_tokens'])
    if growth['first'] is None or growth['last'] <= growth['first']:
        raise RuntimeError('Cannot reproduce upstream text-growth timing')
    result.update(phase=case['phase'],target=case['target'],iteration=iteration,kind='formal',
        payload=payload,public_prefill_seconds=growth['first'],
        public_decode_tokens_per_second=999/(growth['last']-growth['first']),
        public_tpot_ms=1000*(growth['last']-growth['first'])/999,
        last_text_growth_seconds=growth['last'])
    return result


def summarize(rows):
    expected={(phase,target,i) for phase,targets in PHASES.items() for target in targets for i in range(4)}
    if len(rows)!=len(expected) or {(r['phase'],r['target'],r['iteration']) for r in rows}!=expected:
        raise ValueError('Incomplete or duplicated matrix')
    groups=[]
    for phase,targets in PHASES.items():
        for target in targets:
            group=[r for r in rows if r['phase']==phase and r['target']==target and r['kind']=='formal']
            for row in group:
                validate_result(row,row['usage']['prompt_tokens'])
            keys=('public_prefill_seconds','public_decode_tokens_per_second','public_tpot_ms',
                  'ttft_seconds','decode_tokens_per_second','tpot_ms','server_prefill_seconds','total_seconds')
            groups.append(dict(phase=phase,target=target,prompt_tokens=group[0]['usage']['prompt_tokens'],
                content_anomalies=sum(r['content_anomaly'] for r in group),
                metrics={k:dict(mean=statistics.mean(r[k] for r in group),
                    median=statistics.median(r[k] for r in group),min=min(r[k] for r in group),
                    max=max(r[k] for r in group),values=[r[k] for r in group]) for k in keys}))
    case_count=sum(len(targets) for targets in PHASES.values())
    return dict(groups=groups,formal_requests=case_count*3,warmup_requests=case_count,results_require_review=True,
                comparison='Matched public knobs/workload; software and shared-host differences recorded separately')
