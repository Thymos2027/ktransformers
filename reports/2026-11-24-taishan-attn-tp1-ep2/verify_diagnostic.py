"""Recompute the routed short-output matrix from raw SSE evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from test_attention import startup


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run,remote_check=False):
    root=run/'remote'
    state,cleanup=read(root/'status.json'),read(root/'cleanup.json')
    assert state['phase']=='complete' and state['error'] is None and state['cleanup_verified']
    assert cleanup['passed'] and cleanup['original_running']=='false' and not cleanup['port_18082_open']
    for name,digest in read(root/'bundle-sha256.json').items():
        assert sha(root/name)==digest,name
    previous=run.parent/'attn-tp1-ep2-20261124-211254-hccl-combine-fix/remote'
    assert read(root/'candidate.json')==read(previous/'candidate.json'),'Forward implementation changed'
    launch=(previous/'launch.sh').read_text(encoding='utf-8')
    assert (root/'launch.sh').read_text(encoding='utf-8')==launch.replace(
        '--max-running-requests 2 --host','--decode-log-interval 1 --max-running-requests 2 --host')
    tree=ast.parse((root/'diagnostic_controller.py').read_text(encoding='utf-8'))
    cases=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets)))
    specs={name:(prompt,answer) for name,prompt,answer in cases}
    rows=[json.loads(x) for x in (root/'diagnostic-requests.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={(phase,case,rank,repeat) for phase in ('ep2eager','ep2graph') for case in specs
        for rank in (0,1) for repeat in range(2)}
    assert len(rows)==24 and {(x['phase'],x['case'],x['routed_dp_rank'],x['repeat']) for x in rows}==expected
    assert not (root/'requests.jsonl').exists(),'Unexpected performance requests'
    for row in rows:
        prompt,answer=specs[row['case']]
        payload=row['payload']
        assert payload['messages']==[dict(role='user',content=prompt)]
        assert payload['routed_dp_rank']==row['routed_dp_rank']
        assert payload['max_tokens']==64 and payload['temperature']==0 and payload['seed']==20261111
        assert payload['ignore_eos'] is False and row['finish_reason']=='stop' and row['done']
        assert 0<row['usage']['completion_tokens']<64 and row['usage']['prompt_tokens']>0
        assert ''.join(x['text'] for x in row['chunks'])==row['content']
        assert hashlib.sha256(row['content'].encode()).hexdigest()==row['content_sha256']
        times=[x['seconds'] for x in row['chunks']]
        assert times==sorted(times) and times[0]==row['ttft_seconds'] and times[-1]<=row['total_seconds']
        actual=row['content'].strip()
        if isinstance(answer,list):
            try:passed=json.loads(actual)==answer
            except ValueError:passed=False
        else:passed=actual==answer
        assert passed==row['passed']
    summary=read(root/'diagnostic-summary.json')
    assert summary['requests']==24 and summary['correct_answers']==sum(x['passed'] for x in rows)
    assert summary['performance_requests']==0 and not summary['original_quality_gate_accepted']
    assert not summary['correctness_accepted']
    cases_out=[]
    for case in specs:
        group=[x for x in rows if x['case']==case]
        hashes={x['content_sha256'] for x in group}
        cases_out.append(dict(case=case,requests=len(group),passed=sum(x['passed'] for x in group),
            identical_across_modes_ranks_repeats=len(hashes)==1,contents=sorted({x['content'] for x in group})))
    resources={}
    for phase,graph in (('ep2eager',False),('ep2graph',True)):
        log=(root/f'logs/{phase}.log').read_text(encoding='utf-8')
        audit=startup()(log.partition('SIGTERM received.')[0],2,64,graph)
        assert audit==read(root/f'{phase}-startup-audit.json')
        assert log.count('npu graph: True')==summary['graph_decode_log_counts'][phase]
        snapshot=read(root/f'{phase}-after-resources.json')
        hbm=[int(x) for x in re.findall(r'(\d+)\s*/\s*65536',snapshot['raw'])]
        assert len(hbm)==2
        mem={k:int(v) for k,v in re.findall(r'^(MemTotal|MemAvailable):\s*(\d+)',snapshot['meminfo'],re.M)}
        resources[phase]=dict(hbm_mib=hbm,system_used_gib=(mem['MemTotal']-mem['MemAvailable'])/1024**2)
    assert summary['graph_decode_log_counts']['ep2eager']==0
    assert summary['graph_decode_log_counts']['ep2graph']>0
    for rank in (0,1):
        assert read(root/f'hccl-rank{rank}.json')['passed']
        combine=read(root/f'combine-rank{rank}.json')
        assert combine['passed'] and len(combine['checks'])==4
        assert all(x['passed'] and x['eager'] and x['graph_replays']==2 for x in combine['checks'])
    thermal=[json.loads(x) for x in (root/'thermal.jsonl').read_text().splitlines()]
    assert all('cards' in x and all(c['health']=='OK' for c in x['cards'].values()) for x in thermal)
    output=dict(verified=True,run_id=run.name,requests=24,correct_answers=sum(x['passed'] for x in rows),
        cases=cases_out,graph_decode_log_counts=summary['graph_decode_log_counts'],resources=resources,
        thermal=dict(samples=len(thermal),max_c={r:max(x['cards'][r]['temperature_c'] for x in thermal) for r in ('2','5')}),
        performance_requests=0,original_quality_gate_accepted=False,correctness_accepted=False)
    if remote_check:
        remote=read(run/'status.json')['remote']
        code="import hashlib,json;from pathlib import Path;p=Path("+repr(remote)+");fs=list(p.iterdir())+list((p/'logs').iterdir());print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in fs if f.is_file() and f.name not in ('heartbeat','sglang-source.tar')}))"
        hashes=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','taishan','python3 -c '+shlex.quote(code)],text=True,timeout=30))
        mismatches=[name for name,digest in hashes.items() if not (root/name).is_file() or sha(root/name)!=digest]
        assert not mismatches,mismatches
        output['remote_files_matched']=len(hashes)
        (run/'remote-hash-check.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    (run/'reviewed-diagnostic.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--remote-check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.run.resolve(),args.remote_check),ensure_ascii=False,indent=2))
