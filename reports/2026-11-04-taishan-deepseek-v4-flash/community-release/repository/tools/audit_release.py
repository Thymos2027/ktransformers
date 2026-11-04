# SPDX-License-Identifier: Apache-2.0
"""Read-only source/GGUF audit; stage hard links only after all checks pass."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

# The audit uses CPU tensors only; do not initialize an optional NPU backend.
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')

def stamp():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def state(work,phase,**kw):
    data={'phase':phase,'updated_at':stamp(),**kw}
    write_json(Path(work)/'status.json',data)
    print(json.dumps(data,ensure_ascii=False),flush=True)
def signature(p):
    s=p.stat();return s.st_size,s.st_mtime_ns

def audit_layer(args):
    layer,gguf_dir,source,tools,logs=args
    import numpy as np
    import gguf
    sys.path.insert(0,str(tools))
    from convert_mxfp4_gguf import _load_weight_map,_detect_experts_prefix,_open_shard,_as_u8
    p=Path(gguf_dir)/f'dsv4_layer{layer}_mxfp4.gguf'
    initial=signature(p)
    assert initial[0]==3422552640,(p,initial)
    sha=digest(p)
    reader=gguf.GGUFReader(str(p))
    tensors={t.name:t for t in reader.tensors}
    expected={f'blk.{layer}.ffn_{name}_exps.weight' for name in ('gate','up','down')}
    assert set(tensors)==expected,(layer,set(tensors))
    wm=_load_weight_map(Path(source));prefix=_detect_experts_prefix(wm,layer);cache={}
    checked=0
    for name,proj in (('gate','w1'),('up','w3'),('down','w2')):
        t=tensors[f'blk.{layer}.ffn_{name}_exps.weight']
        assert int(t.tensor_type)==39
        k,n,e=map(int,t.shape)
        assert e==256 and k%32==0
        blocks=np.asarray(t.data).reshape(e,n,k//32,17)
        for expert in range(e):
            wk=f'{prefix}.{expert}.{proj}.weight';sk=f'{prefix}.{expert}.{proj}.scale'
            h=_open_shard(Path(source),wm,cache,wk)
            native=_as_u8(h.get_tensor(wk)).reshape(n,k//32,16)
            scales=_as_u8(h.get_tensor(sk)).reshape(n,k//32)
            actual=blocks[expert]
            if not np.array_equal(actual[...,0],scales):raise ValueError(f'Scale bytes differ: layer {layer} {proj} expert {expert}')
            codes=np.empty((n,k//32,32),dtype=np.uint8)
            codes[...,0::2]=native&15;codes[...,1::2]=native>>4
            if not np.array_equal(actual[...,1:]&15,codes[...,:16]) or not np.array_equal(actual[...,1:]>>4,codes[...,16:]):
                raise ValueError(f'FP4 nibbles differ: layer {layer} {proj} expert {expert}')
            checked+=1
    log=Path(logs)/f'layer-{layer:02d}.log'
    with log.open('wb') as stream:
        run=subprocess.run([sys.executable,str(Path(tools)/'verify_mxfp4_gguf.py'),'layer','--gguf',str(p),'--model-dir',str(source),'--layer-idx',str(layer),'--n-experts-check','256'],stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
    output=log.read_text(errors='replace')
    if run.returncode or output.count('checked 256 experts')!=3 or 'FAIL' in output:raise ValueError(f'Upstream full-expert validation failed at layer {layer}; inspect {log.name}')
    assert signature(p)==initial,'GGUF changed during audit'
    result={'layer':layer,'file':p.name,'size':initial[0],'sha256':sha,'experts_per_projection':256,'projections':3,'expert_projections_checked':checked,'packed_nibbles_and_scale_bytes_exact':True,'upstream_dequantized_elementwise_exact':True,'passed':True}
    write_json(Path(logs)/f'layer-{layer:02d}.json',result)
    return result

def validate_sources(repo,source):
    manifest=json.loads((repo/'source-files.json').read_text())
    rehashed=0;cached=0;before={}
    for f in manifest['files']:
        name=f['Path'];p=source/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not p.resolve().is_relative_to(source.resolve()):raise ValueError('Unsafe source path')
        before[name]=signature(p)
        assert before[name][0]==f['Size'],p
        valid=False
        key=hashlib.sha256(name.encode()).hexdigest()+'.json'
        for record in source.glob('.transfer-*/verified/'+key):
            r=json.loads(record.read_text())
            if r.get('path')==name and r.get('size')==f['Size'] and r.get('sha256')==f['Sha256'] and r.get('mtime_ns')==before[name][1]:valid=True;break
        if valid:cached+=1
        else:
            assert digest(p)==f['Sha256'],p
            rehashed+=1
    return before,{'files':len(before),'unchanged_remote_sha256_records_reused':cached,'files_rehashed':rehashed}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repository',type=Path,required=True)
    ap.add_argument('--gguf-dir',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=2,choices=(1,2,3,4))
    args=ap.parse_args();repo=args.repository.resolve();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    logs=work/'layer-audits';logs.mkdir(exist_ok=True)
    lock=open(work/'audit.lock','a+')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=stamp();phase='source_verification'
    try:
        state(work,phase,started_at=started)
        before,source_report=validate_sources(repo,args.source)
        state(work,'auditing_all_experts',layers_done=0,layers_total=43,source_verification=source_report)
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            jobs=[pool.submit(audit_layer,(i,args.gguf_dir,args.source,repo/'tools',logs)) for i in range(43)]
            for future in concurrent.futures.as_completed(jobs):
                try:result=future.result()
                except Exception:
                    for job in jobs:job.cancel()
                    raise
                results.append(result)
                state(work,'auditing_all_experts',layers_done=len(results),layers_total=43,last_layer=result['layer'])
        for name,old in before.items():assert signature(args.source/name)==old,'Source changed during audit: '+name
        results.sort(key=lambda x:x['layer'])
        manifest=''.join(f"{r['sha256']}  {r['file']}\n" for r in results)
        (work/'SHA256SUMS.pending').write_text(manifest,encoding='utf-8')
        state(work,'independent_sha256_recheck',layers_done=43)
        with (work/'sha256-recheck.log').open('wb') as stream:
            check=subprocess.run(['sha256sum','-c',str(work/'SHA256SUMS.pending')],cwd=args.gguf_dir,stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
        assert check.returncode==0,'Independent SHA256 recheck failed'
        state(work,'runtime_probe')
        service=os.environ.get('AUDIT_SERVICE_URL')
        if not service:raise ValueError('Set AUDIT_SERVICE_URL for post-audit runtime verification')
        client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def request(path,payload=None):
            data=None if payload is None else json.dumps(payload).encode()
            req=urllib.request.Request(service+path,data=data,headers={'Content-Type':'application/json'})
            with client.open(req,timeout=300) as response:
                raw=response.read();return json.loads(raw) if raw else None
        request('/health');model=request('/v1/models')['data'][0]['id']
        response=request('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':'Calculate 17 + 25. Reply with only the number.'}],'temperature':0,'max_tokens':32})
        content=response['choices'][0]['message'].get('content','')
        assert '42' in content,content
        runtime={'passed':True,'tested_at':stamp(),'probe':'17 + 25','answer':content,'usage':response.get('usage')}
        write_json(repo/'validation'/'post-audit-runtime.json',runtime)
        state(work,'staging_candidate')
        for r in results:
            src=args.gguf_dir/r['file'];dest=repo/r['file']
            if dest.exists():assert dest.samefile(src),'Existing candidate file differs'
            else:os.link(src,dest)
        (repo/'SHA256SUMS').write_text(manifest,encoding='utf-8')
        report={'passed':True,'started_at':started,'completed_at':stamp(),'layers':43,'experts_per_layer':256,'projections_per_expert':3,'expert_projections_checked':sum(x['expert_projections_checked'] for x in results),'sha256_independently_rechecked':True,'source_verification':source_report,'source_files_unchanged_during_audit':True,'packed_nibbles_and_scale_bytes_exact':True,'upstream_dequantized_elementwise_exact':True,'runtime_probe':runtime,'files':results}
        write_json(repo/'validation'/'full-audit.json',report)
        card=repo/'README.md';text=card.read_text(encoding='utf-8');begin='<!-- AUDIT_STATUS_BEGIN -->';end='<!-- AUDIT_STATUS_END -->'
        section='\n**完整审计通过，候选包待发布审阅。** 全部 43 层 × 256 专家 × 3 投影（33,024 个专家投影）通过编码/缩放字节比较及上游逐元素 bit-exact 验证；43 个 GGUF 的 SHA-256 已独立重读复核。审计后实际聊天测试仍正常。详见 `validation/full-audit.json`。\n'
        text=text.split(begin)[0]+begin+section+end+text.split(end)[1]
        card.write_text(text,encoding='utf-8')
        metadata={p.relative_to(repo).as_posix():digest(p) for p in repo.rglob('*') if p.is_file() and p.suffix!='.gguf' and p.name!='metadata-sha256.json' and '__pycache__' not in p.parts}
        write_json(repo/'metadata-sha256.json',metadata)
        state(work,'ready_for_review',passed=True,layers=43,files=43,bytes=sum(x['size'] for x in results),published=False)
    except Exception as error:
        state(work,'failed',error=str(error),published=False)
        raise
if __name__=='__main__':main()
