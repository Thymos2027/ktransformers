"""Diagnose short-output differences by mode and routed attention rank."""
import json

import attention_controller as attention
import diagnostic_workload as client

trial=attention.trial
core=trial.core
CASES=(
    ('sorting_json','Return only a JSON array sorting [9,1,5,2] in ascending order.',[1,2,5,9]),
    ('sorting_plain','Sort [9,1,5,2] in ascending order. Return only the four numbers separated by commas, with no spaces or other text.','1,2,5,9'),
    ('arithmetic','Reply with only the decimal result of 12 * 13.','156'),
)


def assess(content,expected):
    actual=content.strip()
    if isinstance(expected,list):
        try:return json.loads(actual)==expected
        except ValueError:return False
    return actual==expected


def diagnose(phase):
    model=core.request('http://127.0.0.1:18082','/v1/models')['data'][0]['id']
    for name,prompt,expected in CASES:
        for rank in (0,1):
            for repeat in range(2):
                core.status('diagnosing_short_output',phase_id=phase,case=name,routed_dp_rank=rank,repeat=repeat)
                value=client.stream(core,model,[dict(role='user',content=prompt)],64,routed_dp_rank=rank)
                client.check_transport(value)
                value.update(phase=phase,case=name,routed_dp_rank=rank,repeat=repeat,
                    passed=assess(value['content'],expected) and value['finish_reason']=='stop')
                with (core.ROOT/'diagnostic-requests.jsonl').open('a',encoding='utf-8') as output:
                    output.write(json.dumps(value,ensure_ascii=False)+'\n')
                # Content failures are observations here, never permission to
                # run performance requests or mark the original gate accepted.


def summarize():
    rows=[json.loads(x) for x in (core.ROOT/'diagnostic-requests.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(p,c,r,i) for p in ('ep2eager','ep2graph') for c,_,_ in CASES for r in (0,1) for i in range(2)}
    if len(rows)!=24 or {(x['phase'],x['case'],x['routed_dp_rank'],x['repeat']) for x in rows}!=expected:
        raise RuntimeError('Incomplete short-output diagnostic matrix')
    groups=[]
    for phase in ('ep2eager','ep2graph'):
        for case,_,_ in CASES:
            for rank in (0,1):
                group=[x for x in rows if (x['phase'],x['case'],x['routed_dp_rank'])==(phase,case,rank)]
                groups.append(dict(phase=phase,case=case,routed_dp_rank=rank,passed=[x['passed'] for x in group],
                    contents=[x['content'] for x in group],hashes=[x['content_sha256'] for x in group]))
    replay={phase:(core.LOGS/f'{phase}.log').read_text(errors='replace').count('npu graph: True')
        for phase in ('ep2eager','ep2graph')}
    core.atomic(core.ROOT/'diagnostic-summary.json',dict(diagnostic_complete=True,requests=24,
        correct_answers=sum(x['passed'] for x in rows),groups=groups,graph_decode_log_counts=replay,
        performance_requests=0,original_quality_gate_accepted=False,correctness_accepted=False))


original_atomic=core.atomic


def atomic(path,value):
    if path.name=='benchmark-config.json':
        value.update(trial_kind='short_output_diagnostic',output_tokens=64,ignore_eos=False,
            warmups_per_case=0,formal_repeats_per_case=2,performance_requests=0,
            diagnostic_cases=CASES,routed_dp_ranks=[0,1],original_quality_gate_accepted=False)
    original_atomic(path,value)


core.atomic=atomic
core.smoke=lambda url,phase:diagnose(phase)
trial.workload.measure=lambda core,phase,tp,cases:diagnose(phase)
trial.summarize=summarize
if __name__=='__main__':raise SystemExit(trial.main())
