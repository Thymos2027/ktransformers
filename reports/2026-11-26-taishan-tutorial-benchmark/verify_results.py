"""Offline review of raw token events, deployed identities and completed phases."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
SHORT='tutorial-20261126-111251-swa-audit-fix'
LONG='tutorial-20261126-144145-capacity-parser-fix'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    snapshot=read(HERE/'runs'/LONG/'remote-review-snapshot.json')
    require(not snapshot['running_containers'] and not snapshot['port_open']
            and snapshot['original_running']=='false','Remote host cleanup changed')
    require(all(f'No running processes found in NPU {n}' in snapshot['npu_smi'] for n in (2,5)),
            'NPU process remains')
    all_groups=[]
    source_files=evidence_files=events_total=0
    run_evidence={}
    public={118:(1.5,18.32),801:(15.6,20.12),3944:(16.0,19.58),
            7823:(16.5,19.86),15568:(17.5,20.48),31540:(20.4,19.84)}
    for name,phase,targets in ((SHORT,'table12',(130,1000,4000,8000)),
                                (LONG,'hardwarelong',(16000,32000))):
        root=HERE/'runs'/name/'remote'
        state=read(root/'status.json')
        require(state['cleanup_verified'] and read(root/'cleanup.json')['passed'],'Run cleanup failed')
        require(state['phase']==('complete' if name==LONG else 'failed'),'Run identity changed')
        require(read(root/(phase+'-complete.json'))['passed'],'Measured phase not completed')
        if name==SHORT:
            require('TimeoutError' in read(root/'failure.json')['error'],'Historical failure changed')
        for filename,digest in snapshot[name]['files'].items():
            require(sha(root/filename)==digest,'Local/remote evidence differs: '+name+'/'+filename)
            evidence_files+=1
        candidate=read(root/'candidate.json')
        identities=dict(candidate['before'],**candidate['after'])
        require(snapshot[name]['source']==identities,'Deployed source differs')
        source_files+=len(identities)
        for filename,digest in read(root/'bundle-sha256.json').items():
            require(sha(root/filename)==digest,'Deployed bundle changed: '+filename)
        for src in read(root/'sources.json'):
            require(sha(root/src['file'])==src['sha256'],'Frozen public source changed')
        frozen={c['target']:c for c in read(root/'frozen-cases.json') if c['target'] in targets}
        require(set(frozen)==set(targets),'Missing frozen cases')
        requests=[json.loads(l) for l in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(requests)==len(targets)*4 and {(r['target'],r['iteration']) for r in requests}=={
            (t,i) for t in targets for i in range(4)},'Incomplete/duplicated request matrix')
        log=(root/'logs'/(phase+'.log')).read_text(encoding='utf-8')
        require('SIGTERM received' in log,'No controlled stop boundary')
        before_stop=log.split('SIGTERM received',1)[0]
        require(not re.search(r'Traceback|OutOfMemoryError|streaming failed|hybrid fallback|'
                r'inline resident L\d+ failed|Scheduler hit an exception',before_stop),
                'Runtime error before controlled stop')
        require(re.search(r'Decode batch[^\n]+npu graph: True',before_stop),'No actual graph replay')
        actual=[(int(l),int(t)) for l,t in re.findall(
            r'\[KT PREFILL AUDIT\] layer=(\d+) tokens=(\d+) handled=True',before_stop)]
        expected=[]
        for r in requests:
            c=frozen[r['target']]
            require(r['phase']==phase and r['payload']==dict(c['payload'],stream=r['iteration']!=0),
                    'Generation payload changed')
            wanted=[(l,c['prompt_tokens']) for l in range(43)] if c['prompt_tokens']>=512 else []
            expected+=wanted
            require(r['streaming_audit']['passed'] and r['streaming_audit']['records']==[list(x) for x in wanted],
                    'Per-request streaming audit differs')
            if r['iteration']==0:
                require(r['kind']=='warmup' and r['meta_info']['completion_tokens']==1000,'Warmup budget differs')
                require(r['meta_info']['prompt_tokens']==c['prompt_tokens'],'Warmup input count differs')
                continue
            require(r['kind']=='formal' and r['usage']==dict(prompt_tokens=c['prompt_tokens'],completion_tokens=1000)
                    and len(r['output_ids'])==1000 and r['done'] and r['finish_reason']=='length',
                    'Output budget/SSE incomplete')
            require(hashlib.sha256(r['content'].encode()).hexdigest()==r['content_sha256'],'Text SHA differs')
            require(r['content_anomaly']==(len(r['content'].strip())<16 or len(set(r['output_ids']))<=2),
                    'Content anomaly flag differs')
            e=r['token_events']; times=[x['seconds'] for x in e]
            require([x['completion_tokens'] for x in e]==list(range(1,1001))
                    and times==sorted(times) and times[0]<times[-1]<r['total_seconds'],'Token events incomplete')
            events_total+=len(e)
            meta=r['meta_info'];prefill=meta['prefill_finished_time']-meta['forward_entry_time']
            token_interval=(times[-1]-times[0])/999
            public_interval=(r['last_text_growth_seconds']-r['public_prefill_seconds'])/999
            for measured,computed in ((r['ttft_seconds'],times[0]),(r['tpot_ms'],1000*token_interval),
                    (r['decode_tokens_per_second'],1/token_interval),(r['server_prefill_seconds'],prefill),
                    (r['public_tpot_ms'],1000*public_interval),(r['public_decode_tokens_per_second'],1/public_interval)):
                require(computed>0 and math.isclose(measured,computed,rel_tol=1e-12),'Timing recomputation differs')
        require(actual==expected,'Actual streaming log differs from request audits')
        require(before_stop.count('inline resident:')==sum(frozen[r['target']]['prompt_tokens']>=512 for r in requests),
                'Dynamic resident execution count differs')
        info=read(root/(phase+'-server-info.json'))
        require(info['tp_size']==info['ep_size']==info['dp_size']==1 and info['kt_cpuinfer']==32
                and info['kt_threadpool_count']==1 and info['max_running_requests']==1
                and info['chunked_prefill_size']==32768 and info['context_length']==65536,'Topology differs')
        require(info['kt_num_gpu_experts']==(28 if name==LONG else 32),'Resident expert count differs')
        if name==LONG:
            require(info['max_total_num_tokens']==73728 and info['swa_full_tokens_ratio']==0.5
                    and info['mem_fraction_static']==0.775,'Hardware profile differs')
            require(read(root/(phase+'-capacity-audit.json'))['passed'],'Capacity audit missing')
        for target in targets:
            g=[r for r in requests if r['target']==target and r['kind']=='formal']
            metrics={k:dict(mean=statistics.mean(r[k] for r in g),median=statistics.median(r[k] for r in g),
                min=min(r[k] for r in g),max=max(r[k] for r in g),values=[r[k] for r in g]) for k in (
                'public_prefill_seconds','public_decode_tokens_per_second','ttft_seconds','tpot_ms',
                'decode_tokens_per_second','server_prefill_seconds','total_seconds')}
            if name==LONG:
                automatic=next(x for x in read(root/'summary.json')['groups'] if x['target']==target)
                require(all(metrics[k][s]==automatic['metrics'][k][s] for k in metrics
                            for s in ('mean','median','min','max','values')),'Automatic summary differs')
            pt=frozen[target]['prompt_tokens']; pf,dec=public[pt]
            all_groups.append(dict(run=name,target=target,prompt_tokens=pt,metrics=metrics,
                content_anomalies=sum(r['content_anomaly'] for r in g),
                repeated_output_id_sequences=len({json.dumps(r['output_ids']) for r in g}),
                public_reference=dict(prefill_seconds=pf,decode_tokens_per_second=dec),
                comparison=dict(prefill_time_ratio=metrics['public_prefill_seconds']['mean']/pf,
                                decode_rate_ratio=metrics['public_decode_tokens_per_second']['median']/dec)))
        thermal={}
        for label,path in (('local',root.parent/'thermal.jsonl'),('remote',root/'thermal.jsonl')):
            values=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()]
            require(values and all('cards' in v and all(c['health']=='OK' and c['temperature_c']<90
                    for c in v['cards'].values()) for v in values),'Invalid/abnormal thermal telemetry')
            thermal[label]=dict(samples=len(values),max_temperature_c={d:max(v['cards'][d]['temperature_c']
                                        for v in values) for d in ('2','5')})
        run_evidence[name]=dict(requests=len(requests),streamed_layers=len(actual),thermal=thermal)
    return dict(passed=True,reviewed_at_beijing=snapshot['checked_at_beijing'],formal_requests=18,warmups=6,
                verified_token_events=events_total,remote_evidence_files=evidence_files,
                source_files_verified=source_files,groups=all_groups,runs=run_evidence,
                hardware_long_run_complete=True,historical_short_phase_complete=True,
                identical_parameter_matrix=False,quality_accepted=False,cleanup_verified=True)


if __name__=='__main__':
    result=verify()
    (HERE/'runs'/LONG/'reviewed-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','runs')},ensure_ascii=False))
