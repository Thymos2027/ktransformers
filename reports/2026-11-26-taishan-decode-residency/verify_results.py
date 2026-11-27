"""Read-only verification of downloaded ABBA artifacts; does not contact taishan."""
import argparse
import hashlib
import json
from pathlib import Path

import prefill_gate as gate


def measurement_log(text):
    """Keep shutdown evidence, while checking errors before controlled SIGTERM."""
    active, separator, teardown=text.partition('SIGTERM received.')
    if separator and ('OutOfMemoryError' in teardown or 'Scheduler hit an exception' in teardown):
        raise ValueError('Unexpected failure during shutdown')
    return active, dict(controlled_sigterm=bool(separator),
                        traceback_after_sigterm='Traceback' in teardown,
                        cancelled_error_after_sigterm='CancelledError' in teardown,
                        teardown_sha256=hashlib.sha256(teardown.encode()).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(run):
    root=run/'remote'
    state=read(root/'status.json')
    if state['phase']!='complete' or not state['cleanup_verified'] or not read(root/'cleanup.json')['passed']:
        raise ValueError('Background measurements or cleanup did not complete')
    if not read(root/'residency-hardware-gate.json')['passed']:
        raise ValueError('Native hardware gate did not pass')
    for name,expected in read(root/'artifact-manifest.json').items():
        target=(root/name).resolve()
        target.relative_to(root.resolve())
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('Artifact identity mismatch: '+name)
    candidate=read(root/'candidate.json')
    if candidate['after']!=read(root/'deployed-source-sha256.json'):
        raise ValueError('Actual deployed sources differ from candidate')
    frozen=read(root/'frozen-workloads.json')
    cases=[next(c for c in frozen['prefill'] if c['prompt_tokens']==n) for n in (1024,4096)]
    rows=[json.loads(s) for s in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:
        case=next(c for c in cases if c['name']==row['case'])
        if row['payload']['input_ids']!=case['token_ids']:
            raise ValueError('Input IDs differ from frozen prompt')
        if hashlib.sha256(row['content'].encode()).hexdigest()!=row['content_sha256']:
            raise ValueError('Output text identity mismatch')
        gate.validate_request(row,case['prompt_tokens'])
        if row['streaming_audit']['updates'] and row['phase'] not in gate.CANDIDATE:
            raise ValueError('Baseline performed decode updates')
    shutdown={}
    for phase in gate.PHASES:
        active,shutdown[phase]=measurement_log((root/'logs'/f'{phase}.log').read_text(encoding='utf-8'))
        gate.startup(active,phase)
    summary=gate.summarize(rows,cases)
    if summary!=read(root/'summary.json'):
        raise ValueError('Summary differs from independent recomputation')
    comparisons=[]
    for case in cases:
        selected=[g for g in summary['groups'] if g['case']==case['name']]
        values={arm:[g['metrics'] for g in selected if (g['phase'] in gate.CANDIDATE)==(arm=='candidate')]
                for arm in ('baseline','candidate')}
        # Equal weight to each independent model start. Raw per-start results stay available.
        baseline={k:sum(v[k] for v in values['baseline'])/2 for k in values['baseline'][0]}
        candidate_metrics={k:sum(v[k] for v in values['candidate'])/2 for k in values['candidate'][0]}
        comparisons.append(dict(case=case['name'],baseline=baseline,candidate=candidate_metrics,
             decode_change_percent=100*(candidate_metrics['decode_tokens_per_second']/baseline['decode_tokens_per_second']-1),
             total_time_reduction_percent=100*(1-candidate_metrics['total_seconds']/baseline['total_seconds'])))
    return dict(passed=True,comparisons=comparisons,summary=summary,quality_accepted=False,shutdown=shutdown,
                limitation='Two frozen prompts; free generation may differ; shared host; not an independent task-quality score')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=verify(args.run)
    (args.run/'reviewed-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
