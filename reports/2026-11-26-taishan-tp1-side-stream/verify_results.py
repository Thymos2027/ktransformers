"""Independently verify the downloaded R1 results against saved remote evidence."""
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
import statistics

PHASES = ('h0', 'h1', 's0', 's1')
WINDOWS = ((1,64), (65,256), (257,512), (513,1000))
METRICS = ('ttft_seconds', 'tpot_ms', 'decode_tokens_per_second', 'total_seconds',
           'server_prefill_seconds', 'server_prefill_tokens_per_second')
FROZEN_SHA = '83584956389bd4d179c2ac78d8f6ff36fba3fb0adc736673c483e19d269de52f'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12)


def verify(run):
    root = run/'remote'
    state = read(root/'status.json')
    require(state['phase']=='complete' and state['error'] is None and state['cleanup_verified'],
            'Controller did not complete and clean up')
    require(read(root/'cleanup.json')['passed'], 'Recorded cleanup failed')
    require(not (root/'failure.json').exists(), 'Failure artifact exists')
    config = read(root/'benchmark-config.json')
    require(config['phases']==list(PHASES) and config['tp']==config['ep']==config['dp']==1
            and config['physical_npu']==2 and config['experts']==32 and config['cpu_threads']==96
            and config['threadpools']==4 and config['output_tokens']==1000 and config['dynamic_resident']==0,
            'Frozen topology or output budget changed')
    require(sha(root/'frozen-workloads.json')==FROZEN_SHA, 'Input manifest changed')
    frozen = read(root/'frozen-workloads.json')
    cases = [frozen['short'][0],next(c for c in frozen['prefill'] if c['prompt_tokens']==4096)]
    require([c['prompt_tokens'] for c in cases]==[144,4096], 'Frozen input lengths changed')
    requests = rows(root/'requests.jsonl')
    expected = {(p,c['name'],i) for p in PHASES for c in cases for i in range(4)}
    require(len(requests)==32 and {(r['phase'],r['case'],r['iteration']) for r in requests}==expected,
            'Missing or duplicate requests')
    require(Counter(r['kind'] for r in requests)==dict(formal=24,warmup=8), 'Wrong warmup/formal count')
    by_case = {c['name']: c for c in cases}
    event_count = jumps = 0
    for row in requests:
        case = by_case[row['case']]
        require(row['kind']==('warmup' if row['iteration']==0 else 'formal'), 'Wrong iteration kind')
        require(row['payload']==dict(input_ids=case['token_ids'],stream=True,sampling_params=dict(
                temperature=0,sampling_seed=20261111,max_new_tokens=1000,ignore_eos=True)), 'Payload changed')
        require(row['usage']==dict(prompt_tokens=case['prompt_tokens'],completion_tokens=1000)
                and row['done'] and row['finish_reason']=='length', 'Incomplete response')
        ids = row['output_ids']
        require(len(ids)==1000 and all(type(x) is int for x in ids), 'Incomplete raw output IDs')
        require(hashlib.sha256(row['content'].encode()).hexdigest()==row['content_sha256'], 'Text hash differs')
        anomaly = len(row['content'].strip())<16 or len(set(ids))<=2
        require(anomaly==row['content_anomaly'] and not anomaly, 'Anomalous output')
        events = row['token_events']
        counts = [e['completion_tokens'] for e in events]
        times = [e['seconds'] for e in events]
        require(len(events)>=2 and all(type(c) is int for c in counts)
                and counts[0]==1 and counts[-1]==1000
                and all(a<b for a,b in zip(counts,counts[1:])), 'Invalid cumulative counts')
        require(all(math.isfinite(t) and t>0 for t in times)
                and all(a<b for a,b in zip(times,times[1:]))
                and times[-1]<row['total_seconds'], 'Invalid notification times')
        jump_count = sum(b-a>1 for a,b in zip(counts,counts[1:]))
        audit = row['timing_audit']
        require(audit['passed'] and audit['notification_count']==len(events)
                and audit['unobserved_intermediate_counts']==1000-len(events)
                and audit['per_token_arrivals_complete']==(jump_count==0)
                and not audit['interpolated_timestamps'] and row['event_token_jumps']==jump_count,
                'Notification quality audit differs')
        event_count += len(events)
        jumps += jump_count
        meta = row['meta_info']
        tpot = (times[-1]-times[0])/999
        prefill = meta['prefill_finished_time']-meta['forward_entry_time']
        require(prefill>0 and meta['prompt_tokens']==case['prompt_tokens']
                and meta['completion_tokens']==1000, 'Server metadata differs')
        for key,value in dict(ttft_seconds=times[0],tpot_ms=1000*tpot,
                decode_tokens_per_second=1/tpot,server_prefill_seconds=prefill,
                server_prefill_tokens_per_second=case['prompt_tokens']/prefill).items():
            require(close(row[key],value), 'Independent timing differs: '+key)
        by_token = dict(zip(counts,times))
        for a,b in WINDOWS:
            value = 1000*(by_token[b]-by_token[a])/(b-a) if a in by_token and b in by_token else None
            require(close(row['window_tpot_ms'][f'{a}-{b}'],value), 'Window timing differs')
        expected_stream = [[i,case['prompt_tokens']] for i in range(43)] if (
            row['phase'] in ('s0','s1') and case['prompt_tokens']>=512) else []
        require(row['streaming_audit']['passed'] and row['streaming_audit']['records']==expected_stream,
                'Per-request streaming path differs')
    for name,digest in read(root/'bundle-sha256.json').items():
        require(sha(root/name)==digest, 'Frozen bundle changed: '+name)
    environment=[json.loads(line.split('KT environment: ',1)[1]) for line in
                 (root/'logs/environment.log').read_text(encoding='utf-8').splitlines()
                 if line.startswith('KT environment: ')]
    require(len(environment)==1 and environment[0]['passed'] and environment[0]['extension_sha256']==
            '9b21e209ff3cfd63fe1068b18516b686962f69b7f4f8342a4bcef366931ef8c8',
            'Native extension identity differs')
    kernel=read(root/'kernel-probe.json')
    require(kernel['passed'] and sha(root/'mxfp4-kernel/libmxfp4fused.so')==kernel['library_sha256'],
            'AscendC library identity differs')
    remote = read(run/'remote-hash-check.json')
    for name,digest in remote['files'].items():
        require(sha(root/name)==digest, 'Downloaded/remote evidence differs: '+name)
    require(remote['source']==read(root/'candidate.json')['after'], 'Deployed source differs')
    require(not remote['port_open'] and remote['original_running']=='false'
            and 'No running processes found in NPU 2' in remote['npu_smi'], 'NPU2 cleanup differs')
    for value in remote['states'].values():
        s=value['state']
        require(not s['Running'] and s['Pid']==0 and not s['OOMKilled']
                and value['restart']['Name']=='no', 'Container not stopped cleanly')
    spec=importlib.util.spec_from_file_location('frozen_r1_gate',root/'prefill_gate.py')
    gate=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    automatic=read(root/'summary.json')
    groups=[]
    phase_evidence={}
    for phase in PHASES:
        log=(root/'logs'/f'{phase}.log').read_text(encoding='utf-8')
        require('SIGTERM received' in log, 'Missing controlled-stop boundary')
        prefix=log.split('SIGTERM received',1)[0]
        gate.startup(prefix,phase)
        require(not re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception|error code is|'+gate.FALLBACK,prefix),
                'Model/runtime failure before controlled stop')
        info=read(root/f'{phase}-server-info.json')
        gate.capacity(prefix,info)
        require(info['tp_size']==info['ep_size']==info['dp_size']==1 and info['kt_cpuinfer']==96
                and info['kt_threadpool_count']==4 and info['kt_num_gpu_experts']==32,
                'Resolved model topology differs')
        stream_records=[[int(a),int(b)] for a,b in re.findall(gate.AUDIT,prefix)]
        require(stream_records==[e for r in requests if r['phase']==phase for e in r['streaming_audit']['records']],
                'Full log/per-request streaming evidence differs')
        require('npu graph: True' in prefix, 'Actual graph replay missing')
        require('inline resident:' not in prefix and 'dynamic resident applied:' not in prefix,
                'Unexpected dynamic residency')
        before=read(root/f'{phase}-before-resources.json')
        after=read(root/f'{phase}-after-resources.json')
        phase_evidence[phase]=dict(streamed_layers=len(stream_records),
            side_audit_layers=len(re.findall(gate.SIDE_AUDIT,prefix)),
            graph_decode_logs=len(re.findall(r'Decode batch[^\n]+npu graph: True',prefix)),
            before_hbm_mib=list(map(int,re.findall(r'(\d+)\s*/\s*65536',before['raw']))),
            after_hbm_mib=list(map(int,re.findall(r'(\d+)\s*/\s*65536',after['raw']))),
            loading_seconds=read(root/f'{phase}-loading.json')['seconds'])
        for case in cases:
            selected=[r for r in requests if r['phase']==phase and r['case']==case['name'] and r['kind']=='formal']
            metrics={key:dict(median=statistics.median(r[key] for r in selected),
                minimum=min(r[key] for r in selected),maximum=max(r[key] for r in selected),
                values=[r[key] for r in selected]) for key in METRICS}
            saved=next(g for g in automatic['groups'] if g['phase']==phase and g['case']==case['name'])
            require(all(close(saved['metrics'][key][stat],metrics[key][stat]) for key in METRICS
                        for stat in ('median','minimum','maximum')), 'Automatic metrics differ')
            windows={}
            for a,b in WINDOWS:
                key=f'{a}-{b}'
                values=[r['window_tpot_ms'][key] for r in selected if r['window_tpot_ms'][key] is not None]
                windows[key]=statistics.median(values) if values else None
                require(close(saved['window_tpot_ms'][key],windows[key])
                        and saved['window_sample_counts'][key]==len(values), 'Automatic windows differ')
            shared=[]
            for r in selected:
                for stage in ('before','after'):
                    snap=read(root/f"{phase}-{r['case']}-{r['iteration']}-{stage}-resources.json")
                    other=[n for n in snap['running_containers'] if not n.startswith('dsv4-tp2-'+run.name+'-')]
                    if other: shared.append(dict(iteration=r['iteration'],stage=stage,containers=other))
            groups.append(dict(phase=phase,prompt_tokens=case['prompt_tokens'],metrics=metrics,
                window_tpot_ms=windows,other_workload_snapshots=shared,
                distinct_outputs=len({tuple(r['output_ids']) for r in selected}),
                output_ids_sha256=hashlib.sha256(json.dumps(selected[0]['output_ids'],separators=(',',':')).encode()).hexdigest()))
    observed=rows(root/'functional-observations.jsonl')
    require(len(observed)==8 and all(r['done'] and r['answer_matches'] and (
        r['content'].strip()=='156' if r['case']=='arithmetic' else json.loads(r['content'])==[1,2,5,9])
        for r in observed), 'Short functional observations differ')
    require(all(r['finish_reason']=='stop' and len(r['output_ids'])==r['usage']['completion_tokens']
                and hashlib.sha256(r['content'].encode()).hexdigest()==r['content_sha256'] for r in observed),
            'Functional observation IDs/end state/hash differs')
    thermal={}
    for label,path in (('local',run/'thermal.jsonl'),('remote',root/'thermal.jsonl')):
        values=rows(path)
        require(values and all('cards' in r and set(r['cards'])=={'2','5'} and all(
            c['health']=='OK' and c['temperature_c']<90 for c in r['cards'].values()) for r in values),
            'Abnormal or invalid temperature sample')
        thermal[label]=dict(samples=len(values),max_temperature_c={d:max(r['cards'][d]['temperature_c'] for r in values) for d in ('2','5')})
        if label=='local':
            thermal[label].update(min_fan_rpm=min(v for r in values for v in r['fans'].values()),
                                  cooling_pids=sorted({r['cooling_pid'] for r in values}))
    commands=rows(root/'commands.jsonl')
    cleanup=read(root/'cleanup.json')
    for phase in PHASES:
        s=cleanup['states']['dsv4-tp2-'+run.name+'-'+phase]
        start,end=dt.datetime.fromisoformat(s['StartedAt']),dt.datetime.fromisoformat(s['FinishedAt'])
        samples=[]
        for command in commands:
            if command['args']==['npu-smi','info'] and start<=dt.datetime.fromisoformat(command['at'])<=end:
                values=re.findall(r'(\d+)\s*/\s*65536',command['stdout'])
                if len(values)==2:
                    samples.append(tuple(map(int,values)))
        require(samples, 'No HBM device samples for '+phase)
        phase_evidence[phase].update(hbm_device_samples=len(samples),
            sampled_max_hbm_mib={device:max(s[i] for s in samples) for i,device in enumerate(('2','5'))})
    comparisons=[]
    index={(g['phase'],g['prompt_tokens']):g for g in groups}
    for old,new in (('h0','h1'),('s0','s1'),('h0','s1'),('h1','s1')):
        for count in (144,4096):
            a,b=index[old,count],index[new,count]
            metric=lambda g,k:g['metrics'][k]['median']
            comparisons.append(dict(baseline=old,candidate=new,prompt_tokens=count,
                ttft_reduction_pct=100*(1-metric(b,'ttft_seconds')/metric(a,'ttft_seconds')),
                decode_increase_pct=100*(metric(b,'decode_tokens_per_second')/metric(a,'decode_tokens_per_second')-1),
                tpot_reduction_pct=100*(1-metric(b,'tpot_ms')/metric(a,'tpot_ms')),
                total_reduction_pct=100*(1-metric(b,'total_seconds')/metric(a,'total_seconds')),
                end_to_end_speedup=metric(a,'total_seconds')/metric(b,'total_seconds'),
                matching_output_ids=a['output_ids_sha256']==b['output_ids_sha256']))
    return dict(passed=True,reviewed_at_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        formal_requests=24,warmup_requests=8,output_tokens=32000,notification_events=event_count,
        count_jumps=jumps,remote_evidence_files_verified=len(remote['files']),
        deployed_source_files_verified=len(remote['source']),functional_passed=8,
        groups=groups,phases=phase_evidence,comparisons=comparisons,thermal=thermal,
        cleanup_verified=True,quality_accepted=False,limitations='One independent load per phase; fixed order; shared host; two prompts')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=verify(args.run)
    (args.run/'reviewed-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','phases')},ensure_ascii=False))
