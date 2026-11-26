"""Recompute timing from raw events and check frozen evidence and remote snapshot."""
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

PHASES = ('baseline','stream','bundle')
METRICS = ('ttft_seconds','server_prefill_seconds','server_prefill_tokens_per_second',
           'tpot_ms','decode_tokens_per_second','total_seconds')
FROZEN_SHA = '83584956389bd4d179c2ac78d8f6ff36fba3fb0adc736673c483e19d269de52f'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run):
    root = run/'remote'
    state = read(root/'status.json')
    require(state['phase']=='complete' and state['error'] is None and state['cleanup_verified'],
            'Controller did not complete and clean up')
    require(read(root/'cleanup.json')['passed'],'Recorded cleanup failed')
    config = read(root/'benchmark-config.json')
    require(config['phases']==list(PHASES) and config['tp']==config['ep']==config['dp']==1
            and config['experts']==32 and config['cpu_threads']==96 and config['threadpools']==4,
            'Experiment topology changed')
    require(sha(root/'frozen-workloads.json')==FROZEN_SHA,'Frozen input identity changed')
    frozen = read(root/'frozen-workloads.json')
    cases = [frozen['short'][0],*frozen['prefill']]
    require([c['prompt_tokens'] for c in cases]==[144,1024,4096,7168],'Frozen lengths changed')
    by_name = {c['name']:c for c in cases}
    requests = rows(root/'requests.jsonl')
    expected = {(p,c['name'],i) for p in PHASES for c in cases for i in range(4)}
    require(len(requests)==48 and {(r['phase'],r['case'],r['iteration']) for r in requests}==expected,
            'Missing or duplicate requests')
    require(Counter(r['kind'] for r in requests)==dict(formal=36,warmup=12),'Wrong repetition count')
    for r in requests:
        case = by_name[r['case']]
        require(r['kind']==('warmup' if r['iteration']==0 else 'formal'),'Wrong warmup classification')
        require(r['payload']==dict(input_ids=case['token_ids'],stream=True,sampling_params=dict(
            temperature=0,sampling_seed=20261111,max_new_tokens=64,ignore_eos=True)),'Payload differs')
        require(r['done'] and r['finish_reason']=='length' and r['usage']==dict(
            prompt_tokens=case['prompt_tokens'],completion_tokens=64),'Output budget or SSE incomplete')
        require(len(r['output_ids'])==64 and all(type(i) is int for i in r['output_ids']),
                'Missing raw output token IDs')
        require(hashlib.sha256(r['content'].encode()).hexdigest()==r['content_sha256'],'Text digest differs')
        anomaly = len(r['content'].strip())<16 or len(set(r['output_ids']))<=2
        require(anomaly==r['content_anomaly'],'Content anomaly classification differs')
        events = r['token_events']
        times = [e['seconds'] for e in events]
        require([e['completion_tokens'] for e in events]==list(range(1,65)) and r['event_token_jumps']==0,
                'Aggregated or missing token events')
        require(times==sorted(times) and times[0]==r['ttft_seconds'] and times[0]<times[-1]<r['total_seconds'],
                'Invalid client timing')
        tpot = (times[-1]-times[0])/63
        meta = r['meta_info']
        prefill = meta['prefill_finished_time']-meta['forward_entry_time']
        require(meta['completion_tokens']==64 and meta['prompt_tokens']==case['prompt_tokens'],
                'Server token count differs')
        for actual,calculated in ((r['tpot_ms'],1000*tpot),(r['decode_tokens_per_second'],1/tpot),
                (r['server_prefill_seconds'],prefill),
                (r['server_prefill_tokens_per_second'],case['prompt_tokens']/prefill)):
            require(actual>0 and math.isclose(actual,calculated,rel_tol=1e-12),'Timing recomputation differs')
        expected_records = [[layer,case['prompt_tokens']] for layer in range(43)] if (
            r['phase']!='baseline' and case['prompt_tokens']>=512) else []
        audit = r['streaming_audit']
        require(audit['passed'] and audit['records']==expected_records
                and audit['streamed_layers']==len(expected_records),'Request streaming audit differs')
    for name,digest in read(root/'bundle-sha256.json').items():
        require(sha(root/name)==digest,'Frozen runtime changed: '+name)
    snapshot = read(run/'remote-hash-check.json')
    for name,digest in snapshot['files'].items():
        require(sha(root/name)==digest,'Local/remote evidence differs: '+name)
    candidate = read(root/'candidate.json')
    identities = dict(candidate['before'],**candidate['after'])
    require(snapshot['source']==identities,'Deployed source differs from candidate')
    require(not snapshot['port_open'] and snapshot['original_running']=='false'
            and not snapshot['running_containers'],'Host not idle at remote review')
    for name,record in snapshot['states'].items():
        s = record['state']
        require(not s['Running'] and s['Pid']==0 and not s['OOMKilled'] and record['restart']['Name']=='no',
                'Container cleanup failed: '+name)
    require(all(f'No running processes found in NPU {n}' in snapshot['npu_smi'] for n in (2,5)),
            'NPU process remains')
    spec = importlib.util.spec_from_file_location('frozen_prefill_gate',root/'prefill_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    groups,phase_evidence = [],{}
    automatic = read(root/'summary.json')
    for phase in PHASES:
        log = (root/'logs'/f'{phase}.log').read_text(encoding='utf-8')
        require('SIGTERM received' in log,'Missing controlled-stop boundary')
        prefix = log.split('SIGTERM received',1)[0]
        gate.startup(prefix,phase)
        require(not re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception|error code is|'+gate.FALLBACK,prefix),
                'Runtime failure before controlled stop')
        actual_records = [[int(l),int(n)] for l,n in re.findall(gate.AUDIT,prefix)]
        expected_records = [record for r in requests if r['phase']==phase for record in r['streaming_audit']['records']]
        require(actual_records==expected_records,'Actual log streaming execution differs from per-request evidence')
        require(log.count('inline resident:')==(12 if phase=='bundle' else 0),'Dynamic resident audit differs')
        info = read(root/f'{phase}-server-info.json')
        require(info['tp_size']==info['ep_size']==info['dp_size']==1 and info['max_total_num_tokens']==16384
                and info['chunked_prefill_size']==8192 and info['swa_full_tokens_ratio']==0.5,
                'Resolved server topology/capacity differs')
        replays = len(re.findall(r'Decode batch[^\n]+npu graph: True',prefix))
        require(replays>0,'No actual graph decode replay evidence')
        resource = read(root/f'{phase}-after-resources.json')
        hbm = list(map(int,re.findall(r'(\d+)\s*/\s*65536',resource['raw'])))
        mem = {k:int(v) for k,v in re.findall(r'^(MemTotal|MemAvailable):\s+(\d+)',resource['meminfo'],re.M)}
        phase_evidence[phase] = dict(streamed_layers=len(actual_records),graph_decode_log_count=replays,
            dynamic_resident_updates=log.count('inline resident:'),after_hbm_mib=dict(zip(('2','5'),hbm)),
            system_memtotal_minus_available_gib=(mem['MemTotal']-mem['MemAvailable'])/1024**2,
            loading_seconds=read(root/f'{phase}-loading.json')['seconds'])
        for case in cases:
            group = [r for r in requests if r['phase']==phase and r['case']==case['name'] and r['kind']=='formal']
            metrics = {k:dict(median=statistics.median(r[k] for r in group),min=min(r[k] for r in group),
                             max=max(r[k] for r in group),values=[r[k] for r in group]) for k in METRICS}
            a = next(g for g in automatic['groups'] if g['phase']==phase and g['case']==case['name'])
            require(all(a['metrics'][k][s]==metrics[k][s] for k in METRICS for s in ('median','min','max')),
                    'Automatic summary differs')
            groups.append(dict(phase=phase,case=case['name'],prompt_tokens=case['prompt_tokens'],metrics=metrics,
                output_text_sha256=sorted({r['content_sha256'] for r in group}),
                output_id_sha256=sorted({hashlib.sha256(json.dumps(r['output_ids']).encode()).hexdigest() for r in group}),
                content_anomalies=sum(r['content_anomaly'] for r in group)))
    observations = rows(root/'functional-observations.jsonl')
    require(len(observations)==6,'Functional observation count differs')
    require(all(r['done'] and r['answer_matches'] and len(r['output_ids'])==r['usage']['completion_tokens']
                and (r['content'].strip()=='156' if r['case']=='arithmetic' else json.loads(r['content'])==[1,2,5,9])
                for r in observations),'Functional observation failed')
    thermal = {}
    for label,path in (('local',run/'thermal.jsonl'),('remote',root/'thermal.jsonl')):
        readings = rows(path)
        require(readings and all('cards' in r and all(c['health']=='OK' and c['temperature_c']<90
                for c in r['cards'].values()) for r in readings),'Invalid/abnormal thermal sample')
        thermal[label] = dict(samples=len(readings),max_temperature_c={d:max(r['cards'][d]['temperature_c']
            for r in readings) for d in ('2','5')})
        if label=='local':
            thermal[label].update(min_fan_rpm=min(v for r in readings for v in r['fans'].values()),
                                  cooling_pids=sorted({r['cooling_pid'] for r in readings}))
    return dict(reviewed_at_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                passed=True,formal_requests=36,warmup_requests=12,token_events=3072,
                remote_evidence_files_verified=len(snapshot['files']),source_files_verified=len(identities),
                functional_passed=6,content_anomalies=sum(r['content_anomaly'] for r in requests),
                phases=phase_evidence,groups=groups,thermal=thermal,cleanup_verified=True,
                comparisons=automatic['comparisons'],quality_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    destination = args.run/'reviewed-results.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','phases','comparisons')},ensure_ascii=False))
