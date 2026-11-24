"""Read-only evidence checks and independent EP2 metric recomputation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess

from test_graph_gate import startup_function


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run, remote_check=False):
    root=run/'remote'
    state=read(root/'status.json')
    assert state['phase']=='complete' and state['error'] is None and state['cleanup_verified']
    cleanup=read(root/'cleanup.json')
    assert cleanup['passed'] and cleanup['original_running']=='false' and not cleanup['port_18082_open']
    for name,digest in read(root/'bundle-sha256.json').items():
        assert sha(root/name)==digest, name
    reuse=read(root/'checkpoint-reuse.json')
    predecessor=run.parent/Path(reuse['source_root']).name.removeprefix('dsv4-tp2-')/'remote'
    for name,digest in reuse['source_sha256'].items():
        assert sha(predecessor/name)==digest, name
    for name in reuse['copied_files']:
        assert sha(root/name)==reuse['source_sha256'][name], name
    assert reuse['reused_phases']==['ep2eager']
    check=startup_function()
    for phase,graph in [('ep2eager',False),('ep2graph',True)]:
        log=(root/f'logs/{phase}.log').read_text(encoding='utf-8').partition('SIGTERM received.')[0]
        assert check(log,2,64,graph)==read(root/f'{phase}-startup-audit.json')
        gate=read(root/(f'{phase}-gate.json' if graph else f'{phase}-smoke.json'))
        expected=['巴黎','42','156','READY',[1,2,5,9]] if graph else ['巴黎','42','READY']
        assert len(gate)==len(expected)
        for row,answer in zip(gate,expected):
            actual=row['content'].strip()
            assert (json.loads(actual) if isinstance(answer,list) else actual)==answer
            assert row['passed'] and row['done'] and row['finish_reason']=='stop'
    config=read(root/'benchmark-config.json')
    assert sha(root/'requests.json')==config['requests_sha256']
    cases={x['name']:x for x in read(root/'requests.json')}
    rows=[json.loads(x) for x in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==8 and {(x['case'],x['iteration']) for x in rows}=={(c,i) for c in cases for i in range(4)}
    for row in rows:
        case=cases[row['case']];p=row['payload']
        assert row['phase']=='ep2graph' and row['tp']==2
        assert row['kind']==('warmup' if row['iteration']==0 else 'formal')
        assert row['done'] and row['finish_reason']=='length'
        assert row['usage']['completion_tokens']==512 and row['usage']['prompt_tokens']==case['prompt_tokens']
        assert p['messages']==case['messages'] and p['max_tokens']==512 and p['ignore_eos']
        assert p['seed']==20261111 and p['temperature']==0
        assert sha_text(row['content'])==row['content_sha256']
        assert ''.join(c['text'] for c in row['chunks'])==row['content']
        times=[c['seconds'] for c in row['chunks']]
        assert times==sorted(times) and times[0]==row['ttft_seconds'] and times[-1]==row['last_content_seconds']
        assert times[-1]<=row['total_seconds'] and math.isclose(511/(times[-1]-times[0]),row['decode_tokens_per_second'],rel_tol=1e-12)
    baseline_root=run.parents[2]/'2026-11-24-taishan-tp2-expert64/runs/expert64-20261124-ssh-resume/remote'
    baseline=[json.loads(x) for x in (baseline_root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sha(root/'requests.json')==sha(baseline_root/'requests.json')
    output=dict(passed=True,run_id=run.name,formal_requests=6,warmups=2,correctness_accepted=False,cases=[])
    automatic={x['case']:x for x in read(root/'summary.json')['cases']}
    for name,case in cases.items():
        formal=[x for x in rows if x['case']==name and x['kind']=='formal']
        metrics={k:statistics.median(x[k] for x in formal) for k in ('ttft_seconds','decode_tokens_per_second','total_seconds')}
        assert all(metrics[k]==automatic[name]['metrics'][k]['median'] for k in metrics)
        comparisons={}
        for phase in ('tp1k32graph','tp2k32graph','tp2k64graph'):
            group=[x for x in baseline if x['case']==name and x['kind']=='formal' and x['phase']==phase]
            previous={k:statistics.median(x[k] for x in group) for k in metrics}
            comparisons[phase]=dict(metrics=previous,change_percent={k:100*(metrics[k]/previous[k]-1) for k in metrics})
        output['cases'].append(dict(input_tokens=case['prompt_tokens'],metrics=metrics,comparisons=comparisons,
            output_sha256=sorted({x['content_sha256'] for x in formal})))
    resource=read(root/'ep2graph-after-resources.json')
    output['hbm_mib']=[int(x) for x in re.findall(r'(\d+)\s*/\s*65536',resource['raw'])]
    assert len(output['hbm_mib'])==2
    mem={k:int(v) for k,v in re.findall(r'^(MemTotal|MemAvailable):\s*(\d+)',resource['meminfo'],re.M)}
    output['system_used_gib']=(mem['MemTotal']-mem['MemAvailable'])/1024**2
    samples=[json.loads(x) for x in (root/'thermal.jsonl').read_text().splitlines()]
    assert all('cards' in x and all(c['health']=='OK' for c in x['cards'].values()) for x in samples)
    output['thermal']=dict(samples=len(samples),max_c={card:max(x['cards'][card]['temperature_c'] for x in samples) for card in ('2','5')})
    output['graph_replay_log_count']=(root/'logs/ep2graph.log').read_text(encoding='utf-8').count('npu graph: True')
    assert output['graph_replay_log_count']>0
    if remote_check:
        remote=read(run/'status.json')['remote']
        code="import hashlib,json;from pathlib import Path;p=Path("+repr(remote)+");print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and (len(f.relative_to(p).parts)==1 or f.parent==p/'logs')}))"
        hashes=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','taishan','python3 -c '+shlex.quote(code)],text=True,timeout=30))
        checked={name:digest for name,digest in hashes.items() if (root/name).is_file() and name!='heartbeat'}
        assert all(sha(root/name)==digest for name,digest in checked.items())
        output['remote_files_matched']=len(checked)
        (run/'remote-hash-check.json').write_text(json.dumps(checked,indent=2)+'\n',encoding='utf-8')
    (run/'reviewed-results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return output


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--remote-check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.run.resolve(),args.remote_check),ensure_ascii=False,indent=2))
