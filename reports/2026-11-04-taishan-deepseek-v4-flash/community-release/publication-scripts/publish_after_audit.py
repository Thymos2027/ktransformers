"""Publish the user's audited CPU-expert bundle; never log credentials."""
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time

ROOT=Path('/publish')
CANDIDATE=Path('/candidate')
REPO=CANDIDATE/'repository'
REPO_ID='kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts'
URL='https://modelscope.cn/models/'+REPO_ID
LOCK=threading.RLock()
START=time.monotonic()
SECRET=''
def stamp():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
def clean(value):
    value=str(value)
    if SECRET:value=value.replace(SECRET,'[REDACTED]')
    return re.sub(r'https?://[^\s]+','[URL redacted]',value)[:800]
def write(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)
STATE={'repo_id':REPO_ID,'repo_url':URL,'published':False}
def status(phase=None,**extra):
    with LOCK:
        if phase:STATE['phase']=phase
        STATE.update(extra,updated_at=stamp())
        write(ROOT/'status.json',STATE)
        print(json.dumps(STATE,ensure_ascii=False),flush=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    global SECRET
    import fcntl
    lock=(ROOT/'publisher.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.umask(0o077)
    auth=ROOT/'.upload-auth'
    SECRET=auth.read_text().strip()
    auth.unlink()  # Credential exists only in this process from here onward.
    assert SECRET,'Empty credential'
    from modelscope_hub import HubApi
    logging.getLogger('modelscope_hub').setLevel(logging.ERROR)
    api=HubApi(token=SECRET,endpoint='https://modelscope.cn')
    status('authenticating')
    user=api.whoami()
    if user.username!='kmon03f':raise ValueError('Authenticated namespace does not match authorized account')
    status('waiting_for_full_audit',authenticated_username=user.username)
    deadline=time.monotonic()+48*3600
    while True:
        audit=json.loads((CANDIDATE/'status.json').read_text())
        phase=audit.get('phase')
        if phase=='failed':raise RuntimeError('Audit failed; no repository created or uploaded')
        if phase=='ready_for_review':break
        status('waiting_for_full_audit',audit_phase=phase,audit_layers_done=audit.get('layers_done',0),audit_layers_total=43)
        if time.monotonic()>deadline:raise TimeoutError('Audit wait exceeded 48 hours')
        time.sleep(60)
    full=json.loads((REPO/'validation/full-audit.json').read_text())
    assert full['passed'] and full['layers']==43 and full['expert_projections_checked']==33024
    assert full['sha256_independently_rechecked'] and full['packed_nibbles_and_scale_bytes_exact'] and full['upstream_dequantized_elementwise_exact']
    assert full['runtime_probe']['passed']
    expected_names={f'dsv4_layer{i}_mxfp4.gguf' for i in range(43)}
    weights={f['file']:f for f in full['files']}
    assert set(weights)==expected_names and sum(f['size'] for f in weights.values())==147169763520
    for name,item in weights.items():assert (REPO/name).is_file() and (REPO/name).stat().st_size==item['size']
    assert 'FULL_AUDIT_PENDING' not in (REPO/'README.md').read_text(encoding='utf-8')
    status('audit_passed_preparing_repository',audit_layers_done=43)
    marker=ROOT/'repository-created.json'
    if api.repo_exists(REPO_ID,'model'):
        if not marker.is_file() or json.loads(marker.read_text()).get('repo_id')!=REPO_ID:
            raise RuntimeError('Repository already exists without this workflow marker; preserved without changes')
    else:
        status('creating_repository')
        api.create_repo(REPO_ID,'model',visibility='public',license='mit',chinese_name='DeepSeek-V4-Flash KT MXFP4 CPU 专家',description='Community CPU-expert-only GGUF, 43 layers, paired W8A8 and pinned KT/SGLang required. Ascend-validated; not standalone.')
        write(marker,{'repo_id':REPO_ID,'created_at':stamp()})
    progress_keys=('event','batch_index','num_batches','batch_files','batch_bytes','committed_files','committed_bytes','uploaded_bytes','uploaded_files','total_files','total_bytes','skipped_files','elapsed')
    def progress(event):status('uploading',progress={k:event[k] for k in progress_keys if k in event})
    def upload(paths,message):
        last=None
        for attempt in range(1,7):
            try:
                status('uploading',upload_attempt=attempt)
                api.upload_folder(REPO_ID,'model',REPO,allow_patterns=paths,ignore_patterns=['**/__pycache__/**','**/*.pyc'],max_workers=2,use_cache=True,disable_tqdm=True,sync_remote_repo=False,tracker_path=ROOT/'upload-tracker.json',progress_callback=progress,commit_message=message)
                return
            except Exception as error:
                last=error
                status('upload_retry_wait',upload_attempt=attempt,last_error=clean(error))
                if attempt<6:time.sleep(min(60*2**(attempt-1),900))
        raise RuntimeError('Upload retries exhausted: '+clean(last))
    paths=sorted(p.relative_to(REPO).as_posix() for p in REPO.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    upload(paths,'Publish audited MXFP4 CPU experts, 43 layers, with provenance and validation')
    status('verifying_uploaded_weights')
    remote={f.path:f for f in api.list_repo_files(REPO_ID,'model',recursive=True) if not f.is_dir}
    for name,item in weights.items():
        f=remote.get(name)
        if f is None or f.size!=item['size'] or f.sha256!=item['sha256']:raise ValueError('Remote weight size/SHA256 mismatch: '+name)
    # Finalize release metadata only after all weight objects are committed and verified.
    provenance=json.loads((REPO/'provenance.json').read_text())
    provenance.update(published=True,modelscope_repository=REPO_ID,published_at=stamp())
    write(REPO/'provenance.json',provenance)
    card=REPO/'README.md'
    text=card.read_text(encoding='utf-8').replace('本候选包','本社区转换包').replace('候选包待发布审阅','已完成发布前验证')
    card.write_text(text,encoding='utf-8')
    metadata_paths=sorted(p for p in REPO.rglob('*') if p.is_file() and p.suffix!='.gguf' and p.name!='metadata-sha256.json' and '__pycache__' not in p.parts)
    write(REPO/'metadata-sha256.json',{p.relative_to(REPO).as_posix():sha(p) for p in metadata_paths})
    changed=['README.md','provenance.json','metadata-sha256.json']
    upload(changed,'Finalize release metadata after remote weight SHA256 verification')
    status('verifying_all_remote_files')
    remote={f.path:f for f in api.list_repo_files(REPO_ID,'model',recursive=True) if not f.is_dir}
    checked=[]
    for name in paths:
        local=REPO/name
        expected=weights[name]['sha256'] if name in weights else sha(local)
        f=remote.get(name)
        if f is None or f.size!=local.stat().st_size or f.sha256!=expected:raise ValueError('Remote file size/SHA256 mismatch: '+name)
        checked.append({'path':name,'size':f.size,'sha256':expected})
    result={'repo_id':REPO_ID,'repo_url':URL,'published':True,'remote_files_sha256_verified':True,'verified_files':checked,'weight_files':43,'weight_bytes':147169763520,'completed_at':stamp()}
    write(ROOT/'publication-result.json',result)
    status('published_and_verified',published=True,remote_files_sha256_verified=True,verified_files=len(checked),weight_files=43,weight_bytes=147169763520)
if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_needs_review',error_type=type(error).__name__,error=clean(error))
        raise SystemExit(1)
