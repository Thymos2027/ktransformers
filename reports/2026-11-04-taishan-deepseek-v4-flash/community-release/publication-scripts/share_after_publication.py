"""Open one documentation PR after ModelScope publication is fully verified."""
import base64
import datetime as dt
import difflib
import json
import os
from pathlib import Path
import subprocess
import time
import requests

ROOT=Path(__file__).resolve().parent
UPSTREAM='kvcache-ai/ktransformers'
FORK='Thymos2027/ktransformers'
OWNER='Thymos2027'
BRANCH='docs/dsv4-ascend-community-mxfp4-20261104'
DOC='doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md'
MODEL='kmon03f/DeepSeek-V4-Flash-KT-MXFP4-CPU-Experts'
URL='https://modelscope.cn/models/'+MODEL
PUBLICATION='/data/deploy/dsv4-modelscope-publish-20261104'
ENV=dict(os.environ,GH_PROMPT_DISABLED='1',GIT_TERMINAL_PROMPT='0')
ENV.pop('GH_DEBUG',None)
STATE={'upstream':UPSTREAM,'fork':FORK,'branch':BRANCH,'modelscope_repository':MODEL,'posted':False}
def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
def status(phase,**extra):
    STATE.update(phase=phase,updated_at=now(),**extra)
    temp=ROOT/'status.json.tmp';temp.write_text(json.dumps(STATE,indent=2),encoding='utf-8');temp.replace(ROOT/'status.json')
    with (ROOT/'share.log').open('a',encoding='utf-8') as f:f.write(json.dumps(STATE)+'\n')
def command(args,timeout=90):
    result=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW,env=ENV)
    if result.returncode:raise RuntimeError(result.stderr.strip()[:1200] or 'Command failed')
    return result.stdout
def gh_json(*args):return json.loads(command(['gh','api',*args]))
def remote(name):
    output=command(['ssh.exe','-n','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','taishan','sudo -n cat '+PUBLICATION+'/'+name],timeout=30)
    return json.loads(output)
def patch_document(raw):
    text=raw.decode('utf-8')
    if URL in text:return raw
    newline='\r\n' if '\r\n' in text else '\n'
    rows=text.splitlines(keepends=True)
    indices=[i for i,line in enumerate(rows) if line.startswith('| 43 per-layer MXFP4 GGUF |')]
    if len(indices)!=1:raise ValueError('Upstream artifact table changed; manual review required')
    paragraph=(
        'An optional **community-maintained, expert-only GGUF set** is available on '
        f'[ModelScope]({URL}) (43 per-layer files, about 137 GiB). '
        'The model card includes source revisions, SHA-256 manifests, full-expert '
        'bit-exact validation, pinned KT/SGLang versions and a Kunpeng 920 + '
        'Ascend 910B4-1 smoke test. These are CPU expert weights only; a compatible '
        'W8A8 checkpoint is still required. This is an unofficial third-party '
        'artifact; check the recorded versions and hashes before use.'
    )
    rows.insert(indices[0]+1,newline+paragraph+newline)
    return ''.join(rows).encode('utf-8')
def prepare():
    head=gh_json(f'repos/{UPSTREAM}/git/ref/heads/main')['object']['sha']
    document=gh_json(f'repos/{UPSTREAM}/contents/{DOC}','-f','ref='+head,'--method','GET')
    raw=base64.b64decode(document['content']);updated=patch_document(raw)
    (ROOT/'tutorial-before.md').write_bytes(raw);(ROOT/'tutorial-after.md').write_bytes(updated)
    diff=''.join(difflib.unified_diff(raw.decode().splitlines(keepends=True),updated.decode().splitlines(keepends=True),fromfile='a/'+DOC,tofile='b/'+DOC))
    (ROOT/'proposed-change.diff').write_text(diff,encoding='utf-8',newline='')
    return head,document['sha'],raw,updated
def write_request(name,value):
    p=ROOT/name;p.write_text(json.dumps(value),encoding='utf-8');return str(p)
def main():
    import msvcrt
    with (ROOT/'watcher.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        deadline=time.monotonic()+7*86400
        failures=0
        status('waiting_for_verified_modelscope_upload')
        while True:
            try:
                upload=remote('status.json');failures=0
            except Exception as error:
                failures+=1;status('waiting_for_server_connection',consecutive_errors=failures,error=str(error)[:500])
                if failures>=60:raise RuntimeError('Server inaccessible for 60 checks; no PR opened')
                time.sleep(60);continue
            if upload.get('phase')=='failed_needs_review':raise RuntimeError('ModelScope publication failed; no PR opened')
            if upload.get('phase')=='published_and_verified' and upload.get('published') and upload.get('remote_files_sha256_verified'):break
            status('waiting_for_verified_modelscope_upload',upload_phase=upload.get('phase'),audit_layers_done=upload.get('audit_layers_done'))
            if time.monotonic()>deadline:raise TimeoutError('Publication wait exceeded seven days')
            time.sleep(60)
        result=remote('publication-result.json')
        assert result['repo_id']==MODEL and result['published'] and result['remote_files_sha256_verified'] and result['weight_files']==43
        (ROOT/'modelscope-publication-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        status('checking_public_download_and_hashes')
        session=requests.Session();session.trust_env=False
        response=session.get(f'https://modelscope.cn/api/v1/models/{MODEL}/repo/files',params={'Revision':'master','Recursive':'true'},timeout=(10,60));response.raise_for_status()
        files={f['Path']:f for f in response.json()['Data']['Files'] if f.get('Type')=='blob'}
        for item in result['verified_files']:
            assert item['path'] in files
            actual=files[item['path']]
            assert actual['Size']==item['size'] and actual['Sha256']==item['sha256'],item['path']
        audit_response=session.get(f'https://modelscope.cn/api/v1/models/{MODEL}/repo',params={'Revision':'master','FilePath':'validation/full-audit.json'},timeout=(10,60));audit_response.raise_for_status()
        audit=audit_response.json()
        assert audit['passed'] and audit['expert_projections_checked']==33024 and audit['sha256_independently_rechecked']
        login=json.loads(command(['gh','api','user']))['login']
        if login!=OWNER:raise ValueError('GitHub account changed; no PR opened')
        existing=gh_json(f'repos/{UPSTREAM}/pulls','--method','GET','-f','state=all','-f','head='+OWNER+':'+BRANCH)
        if existing:
            status('pr_already_exists',posted=True,pr_url=existing[0]['html_url']);return
        head,blob,old,new=prepare()
        if new==old:status('already_documented',posted=False);return
        status('creating_documentation_branch')
        marker=ROOT/'branch-created.json'
        if marker.exists():
            previous=json.loads(marker.read_text());assert previous['branch']==BRANCH and previous['base_sha']==head,'Upstream changed since earlier attempt; inspect existing branch'
        else:
            request=write_request('branch-request.json',{'ref':'refs/heads/'+BRANCH,'sha':head})
            gh_json(f'repos/{FORK}/git/refs','--method','POST','--input',request)
            marker.write_text(json.dumps({'branch':BRANCH,'base_sha':head}),encoding='utf-8')
        committed=ROOT/'commit-created.json'
        if not committed.exists():
            payload={'message':'docs: link community MXFP4 CPU expert weights for Ascend','content':base64.b64encode(new).decode(),'sha':blob,'branch':BRANCH}
            request=write_request('file-request.json',payload)
            commit=gh_json(f'repos/{FORK}/contents/{DOC}','--method','PUT','--input',request)
            committed.write_text(json.dumps({'commit_sha':commit['commit']['sha']}),encoding='utf-8')
        status('opening_documentation_pr')
        output=command(['gh','pr','create','--repo',UPSTREAM,'--base','main','--head',OWNER+':'+BRANCH,'--title','docs: link community MXFP4 CPU expert weights for Ascend','--body-file',str(ROOT/'pr-body.md')],timeout=120)
        urls=[line.strip() for line in output.splitlines() if line.strip().startswith('https://github.com/')]
        if not urls:raise RuntimeError('PR create returned no URL; inspect GitHub before retrying')
        status('pr_opened',posted=True,pr_url=urls[-1])
if __name__=='__main__':
    import sys
    try:
        if '--prepare-only' in sys.argv:
            prepare();print('Prepared proposed-change.diff; no GitHub mutations performed.')
        else:main()
    except Exception as error:
        status('failed_needs_review',error=type(error).__name__+': '+str(error)[:1200]);raise SystemExit(1)
