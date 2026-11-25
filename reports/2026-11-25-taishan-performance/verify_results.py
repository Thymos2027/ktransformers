"""Independently recompute all performance groups and verify source evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess

PHASES=(('dpaeager',True,False,False),('dpagraph',True,True,False),
        ('epgraph',False,True,False),('epprefill',False,True,True),('dpaprefill',True,True,True))
METRICS=('ttft_seconds','tpot_ms','decode_tokens_per_second','server_prefill_seconds',
         'server_prefill_tokens_per_second','total_seconds')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(median=statistics.median(values),min=min(values),max=max(values),values=values)


def thermal(path):
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    valid=[x for x in rows if 'cards' in x]
    errors=[x for x in rows if 'error' in x]
    require(valid,'No thermal evidence')
    require(all(c['health']=='OK' for x in valid for c in x['cards'].values()),'Abnormal device health')
    result=dict(valid_samples=len(valid),errors=len(errors),
        max_temperature_c={key:max(x['cards'][key]['temperature_c'] for x in valid) for key in ('2','5')})
    require(max(result['max_temperature_c'].values())<90,'Temperature limit reached')
    if 'fans' in valid[0]:
        result.update(min_fan_rpm=min(v for x in valid for v in x['fans'].values()),
                      cooling_pids=sorted({x['cooling_pid'] for x in valid}))
    return result


def verify(run):
    root=run/'remote'
    state=read(root/'status.json')
    require(state['phase']=='complete' and state['error'] is None and state['cleanup_verified'],
            'Controller did not complete and clean up')
    require(read(root/'cleanup.json')['passed'],'Cleanup evidence failed')
    config=read(root/'benchmark-config.json')
    require(config['phases']==[list(p) for p in PHASES] and config['seed']==20261111
            and config['cpu_threads']==96 and config['experts']==64,'Unexpected experiment configuration')
    frozen=read(root/'frozen-workloads.json')
    require(sha(root/'frozen-workloads.json')==read(root/'frozen-workloads-sha256.json')['sha256'],
            'Frozen workload digest mismatch')
    short={x['name']:x for x in frozen['short']}
    long={x['name']:x for x in frozen['prefill']}
    require([x['prompt_tokens'] for x in frozen['short']]==[144,792],'Historical inputs changed')
    require([x['prompt_tokens'] for x in frozen['prefill']]==[1024,4096,7168],'Long inputs changed')
    for x in (*short.values(),*long.values()):
        require(len(x['token_ids'])==x['prompt_tokens'],'Frozen input ID length mismatch')
    rows=[json.loads(x) for x in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(phase,case,i) for phase,dpa,graph,wide in PHASES
              for case in (long if wide else short) for i in range(4)}
    require(len(rows)==48 and len(expected)==48
            and {(x['phase'],x['case'],x['iteration']) for x in rows}==expected,'Missing or repeated request identity')
    for x in rows:
        phase,dpa,graph,wide=next(p for p in PHASES if p[0]==x['phase'])
        case=(long if wide else short)[x['case']]
        output=64 if wide else 512
        require(x['kind']==('warmup' if x['iteration']==0 else 'formal'),'Warmup classification mismatch')
        require(x['payload']['input_ids']==case['token_ids'],'Input token IDs changed')
        require(x['payload']['sampling_params']==dict(temperature=0,sampling_seed=20261111,
                max_new_tokens=output,ignore_eos=True),'Sampling configuration changed')
        rank=x['iteration']%2 if dpa else None
        require(x['dp_rank']==rank and x['payload'].get('routed_dp_rank')==rank,'DP request routing changed')
        require(x['done'] and x['finish_reason']=='length'
                and x['usage']==dict(prompt_tokens=case['prompt_tokens'],completion_tokens=output),
                'Token budget or stream completion failed')
        require(hashlib.sha256(x['content'].encode()).hexdigest()==x['content_sha256'],'Output digest mismatch')
        events=x['token_events']
        counts=[e['completion_tokens'] for e in events]
        times=[e['seconds'] for e in events]
        require(counts==list(range(1,output+1)) and x['event_token_jumps']==0,
                'Stream aggregated or omitted token events')
        require(times==sorted(times) and times[0]==x['ttft_seconds']
                and times[-1]<x['total_seconds'] and times[-1]>times[0],'Invalid stream timing')
        tpot=(times[-1]-times[0])/(output-1)
        require(math.isclose(1000*tpot,x['tpot_ms'],rel_tol=1e-12)
                and math.isclose(1/tpot,x['decode_tokens_per_second'],rel_tol=1e-12),'Decode recomputation differs')
        meta=x['meta_info']
        prefill=meta['prefill_finished_time']-meta['forward_entry_time']
        require(prefill>0 and meta['prompt_tokens']==case['prompt_tokens'] and meta['completion_tokens']==output,
                'Scheduler token/timing metadata mismatch')
        require(math.isclose(prefill,x['server_prefill_seconds'],rel_tol=1e-12)
                and math.isclose(case['prompt_tokens']/prefill,x['server_prefill_tokens_per_second'],rel_tol=1e-12),
                'Prefill recomputation differs')
    require(Counter(x['kind'] for x in rows)==dict(formal=36,warmup=12),'Unexpected request counts')
    for name,digest in read(root/'bundle-sha256.json').items():
        require(sha(root/name)==digest,'Frozen runtime changed: '+name)
    reuse=read(root/'checkpoint-reuse.json')
    require(reuse['reused_phases']==['dpaeager','dpagraph'] and reuse['requests']==16,'Unexpected checkpoint scope')
    for name,digest in reuse['evidence_sha256'].items():
        if name in ('requests.jsonl','functional-observations.jsonl'):
            # Continued measurements append to these files. The copied prefix,
            # rather than the finished file, must retain the source digest.
            lines=(root/name).read_bytes().splitlines(keepends=True)
            count=16 if name=='requests.jsonl' else 8
            actual=hashlib.sha256(b''.join(lines[:count])).hexdigest()
        else:
            actual=sha(root/name)
        require(actual==digest,'Copied checkpoint changed: '+name)
    spec=importlib.util.spec_from_file_location('frozen_graph_gate',root/'graph_gate.py')
    graph_gate=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_gate)
    scope=dict(re=re,check_decode_graph=graph_gate.check_decode_graph)
    # Execute the frozen gate after verifying its identity, not a later edit.
    source=(root/'startup_gate.py').read_text(encoding='utf-8')
    source=source.replace('from graph_gate import check_decode_graph','')
    exec(compile(source,'frozen startup gate','exec'),scope)
    groups=[]
    phases={}
    for phase,dpa,graph,wide in PHASES:
        log=(root/'logs'/f'{phase}.log').read_text(encoding='utf-8')
        if phase in reuse['reused_phases']:
            boundary=reuse['accepted_log_boundaries'][phase]
            prefix=log.split(boundary['boundary'],1)[0]
            require(hashlib.sha256(prefix.encode()).hexdigest()==boundary['prefix_sha256'],'Checkpoint log boundary changed')
        else:
            require('SIGTERM received' in log,'No controlled stop marker: '+phase)
            prefix=log.split('SIGTERM received',1)[0]
        audit=scope['check'](prefix,dpa,graph,wide)
        require('error code is' not in prefix,'Device error before accepted measurement boundary')
        pools=re.findall(r'max_num_reqs=1 swa_size=(\d+)',prefix)
        require(pools==(['8192']*2 if wide else ['1536']*2),'SWA pool geometry mismatch')
        graph_replays=len(re.findall(r'Decode batch[^\n]+npu graph: True',prefix))
        require(graph_replays>0 if graph else graph_replays==0,'Graph execution evidence mismatch')
        info=read(root/f'{phase}-server-info.json')
        require(info['context_length']==8192 and info['max_total_num_tokens']==16384
                and info['chunked_prefill_size']==(8192 if wide else 512),'Resolved server capacity differs')
        resource=read(root/f'{phase}-after-resources.json')
        hbm=list(map(int,re.findall(r'(\d+)\s*/\s*65536',resource['raw'])))
        require(len(hbm)==2,'HBM snapshot incomplete')
        mem={k:int(v) for k,v in re.findall(r'^(MemTotal|MemAvailable):\s+(\d+)',resource['meminfo'],re.M)}
        phases[phase]=dict(audit=audit,graph_decode_log_count=graph_replays,
            after_hbm_mib=dict(zip(('2','5'),hbm)),
            system_memtotal_minus_available_gib=(mem['MemTotal']-mem['MemAvailable'])/1024**2,
            prior_profiler_exception_retained=phase=='dpagraph')
        for case in (long if wide else short):
            formal=[x for x in rows if x['phase']==phase and x['case']==case and x['kind']=='formal']
            item=dict(phase=phase,case=case,prompt_tokens=formal[0]['usage']['prompt_tokens'],
                completion_tokens=formal[0]['usage']['completion_tokens'],formal_requests=3,
                metrics={k:stats([x[k] for x in formal]) for k in METRICS},rank_groups={})
            for rank in sorted({x['dp_rank'] for x in formal},key=lambda v:-1 if v is None else v):
                subset=[x for x in formal if x['dp_rank']==rank]
                item['rank_groups'][str(rank)]=dict(n=len(subset),
                    metrics={k:stats([x[k] for x in subset]) for k in METRICS},
                    output_sha256=sorted({x['content_sha256'] for x in subset}))
            automatic=next(g for g in read(root/'summary.json')['groups'] if g['phase']==phase and g['case']==case)
            require(all(automatic['metrics'][k]['median']==item['metrics'][k]['median'] for k in METRICS),
                    'Automatic medians differ from raw events')
            groups.append(item)
    observations=[json.loads(x) for x in (root/'functional-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(observations)==16,'Functional observation matrix incomplete')
    for x in observations:
        text=x['content'].strip()
        if x['case']=='arithmetic':correct=text=='156'
        else:
            try:correct=json.loads(text)==[1,2,5,9]
            except ValueError:correct=False
        require(correct==x['answer_matches'] and x['done'],'Functional observation scoring mismatch')
    comparisons={}
    def group(phase,case):return next(g for g in groups if (g['phase'],g['case'])==(phase,case))
    for case in short:
        a,b,c=(group(p,case)['metrics'] for p in ('dpagraph','epgraph','dpaeager'))
        comparisons[case]=dict(
            dpa_graph_vs_ep_graph_decode_change_pct=100*(a['decode_tokens_per_second']['median']/b['decode_tokens_per_second']['median']-1),
            dpa_graph_vs_ep_graph_tpot_change_pct=100*(a['tpot_ms']['median']/b['tpot_ms']['median']-1),
            dpa_graph_vs_ep_graph_ttft_change_pct=100*(a['ttft_seconds']['median']/b['ttft_seconds']['median']-1),
            dpa_graph_over_eager_decode_ratio=a['decode_tokens_per_second']['median']/c['decode_tokens_per_second']['median'])
    for case in long:
        a,b=(group(p,case)['metrics'] for p in ('dpaprefill','epprefill'))
        comparisons[case]=dict(dpa_vs_ep_prefill_throughput_change_pct=100*(a['server_prefill_tokens_per_second']['median']/b['server_prefill_tokens_per_second']['median']-1),
            dpa_vs_ep_decode_change_pct=100*(a['decode_tokens_per_second']['median']/b['decode_tokens_per_second']['median']-1))
    return dict(verified=True,run_id=run.name,formal_requests=36,warmup_requests=12,
        interface='native_generate_token_stream',groups=groups,phases=phases,comparisons=comparisons,
        functional_observations=dict(requests=16,answer_matches=sum(x['answer_matches'] for x in observations)),
        monitoring=dict(remote=thermal(root/'thermal.jsonl'),local=thermal(run/'thermal.jsonl')),
        checkpoint_reuse=reuse,profiling_completed=False,
        limitations=['Prior DP phases reused across a profiler interruption; all configurations measured sequentially with one model load each',
            'Three formal repeats per case, DP0 n=1 versus DP1 n=2; no representative P95 or general throughput claim',
            'DP graph 792-token case has rank-associated output and speed differences; preserve all observations',
            'Scheduler prefill wall timing is not pure NPU kernel time',
            'Wide prefill cases change SWA/chunk geometry and generate 64 tokens; short cases generate 512',
            'HBM/RAM after-request snapshots do not establish peak allocation or maximum model capacity',
            'Profiler deferred; no completed causal performance breakdown; numerical/task quality unaccepted'])


def check_remote(run):
    root=run/'remote'
    remote='/data/deploy/dsv4-tp2-'+run.name
    files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name!='heartbeat' and 'kernel_meta' not in p.parts]
    code='''from pathlib import Path
import hashlib,json,subprocess
root=Path(REMOTE)
files=NAMES
digests={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
candidate=json.loads((root/'candidate.json').read_text())
actual={name:hashlib.sha256((root/'sglang'/name).read_bytes()).hexdigest() for name in candidate['files']}
names=['dsv4-kt-20261104']+['dsv4-tp2-'+RUN+'-'+p for p in ['environment','hccl','epgraph','epprefill','dpaprefill']]
r=subprocess.run(['sudo','-n','docker','inspect',*names],capture_output=True,text=True,check=True)
containers={x['Name'].lstrip('/'):dict(state=x['State'],restart=x['HostConfig']['RestartPolicy']) for x in json.loads(r.stdout)}
print(json.dumps(dict(file_sha256=digests,candidate_actual_sha256=actual,containers=containers)))
'''
    code=code.replace('REMOTE',repr(remote)).replace('NAMES',repr([p.relative_to(root).as_posix() for p in files])).replace('RUN',repr(run.name))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','taishan',
                           'python3 -c '+shlex.quote(code)],capture_output=True,text=True,encoding='utf-8',timeout=45,check=True)
    value=json.loads(result.stdout)
    for p in files:
        name=p.relative_to(root).as_posix()
        require(sha(p)==value['file_sha256'][name],'Remote evidence hash mismatch: '+name)
    for name,identities in read(root/'candidate.json')['files'].items():
        require(value['candidate_actual_sha256'][name]==identities['candidate_sha256'],'Deployed source differs: '+name)
    for name,item in value['containers'].items():
        state=item['state']
        require(not state['Running'] and state['Pid']==0 and not state['OOMKilled'],'Container not cleanly idle: '+name)
        require(item['restart']['Name']=='no','Unexpected restart policy: '+name)
    value.update(verified=True,files_matched=len(files))
    (run/'remote-hash-check.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(files_matched=len(files),deployed_source_files_matched=len(value['candidate_actual_sha256']),
                containers_stopped=len(value['containers']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--remote-check',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    result=verify(run)
    if args.remote_check:result['remote_verification']=check_remote(run)
    result['reviewed_at']=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    (run/'reviewed-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=result['verified'],formal_requests=result['formal_requests'],
        warmup_requests=result['warmup_requests'],remote=result.get('remote_verification'),comparisons=result['comparisons']),indent=2))


if __name__=='__main__':main()
