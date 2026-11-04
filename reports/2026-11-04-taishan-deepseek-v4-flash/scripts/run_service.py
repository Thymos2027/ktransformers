import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT=Path('/workspace')
TOOLS=os.environ['DSV4_TOOLS']
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def status(phase,message=''):
    subprocess.run(['python3',str(ROOT/'status.py'),phase,message],check=True)
def request(path,payload=None,timeout=20):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request('http://127.0.0.1:18080'+path,data=data,headers={'Content-Type':'application/json'})
    with http.open(req,timeout=timeout) as response:
        raw=response.read()
        if path.startswith('/health'):return response.status
        return json.loads(raw) if raw else None
def stop(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGINT)
    try:proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=30)
profiles=[(32,False),(16,False),(16,True)]
for number,(resident,eager) in enumerate(profiles,1):
    env=dict(os.environ,DSV4_NUM_GPU_EXPERTS=str(resident))
    if eager:env['DSV4_EXTRA_FLAGS']='--disable-cuda-graph'
    log=ROOT/'logs'/f'serve-attempt-{number}.log'
    status('loading_model',f'attempt={number} resident_experts={resident} eager={eager}')
    with log.open('ab') as output:
        proc=subprocess.Popen(['bash',TOOLS+'/serve.sh','--foreground'],env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'server.pid').write_text(str(proc.pid))
        deadline=time.monotonic()+7200
        healthy=False
        while time.monotonic()<deadline and proc.poll() is None:
            try:
                request('/health',timeout=5)
                healthy=True;break
            except Exception:time.sleep(15)
        if not healthy:
            stop(proc)
            status('retrying_startup',f'attempt={number} failed; see {log.name}')
            continue
        try:
            status('validating_generation',f'attempt={number}')
            started=time.monotonic()
            generation=request('/generate',{'text':'The capital of France is','sampling_params':{'temperature':0,'max_new_tokens':32}},timeout=1800)
            text=generation.get('text','')
            if 'paris' not in text.lower():raise ValueError('Numerical probe did not answer Paris: '+repr(text))
            models=request('/v1/models')
            model=models['data'][0]['id']
            chat=request('/v1/chat/completions',{'model':model,'messages':[{'role':'user','content':'Reply with the single word READY.'}],'max_tokens':64,'temperature':0},timeout=1800)
            message=chat['choices'][0]['message']
            if not (message.get('content') or message.get('reasoning_content')):raise ValueError('Chat response empty')
            with (ROOT/'logs'/'acceptance.log').open('wb') as checklog:
                checks=subprocess.run(['timeout','1800','bash',TOOLS+'/verify.sh',str(log)],env=env,stdout=checklog,stderr=subprocess.STDOUT)
            if checks.returncode and not eager:raise ValueError('Official acceptance gates failed; see acceptance.log')
            result={'passed':True,'official_acceptance_exit':checks.returncode,'profile':{'resident_experts':resident,'eager':eager,'context_length':8192},'generation':generation,'chat':chat,'seconds_for_probes':time.monotonic()-started,'endpoint':'http://192.168.137.11:18080','server_pid':proc.pid,'completed_at':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()}
            (ROOT/'acceptance.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
            status('serving_verified_eager' if eager else 'serving_verified','Health, Paris probe and chat passed; see acceptance.json and logs/acceptance.log')
            code=proc.wait()
            status('failed_service_exited',f'exit={code}')
            raise SystemExit(1)
        except Exception as error:
            (ROOT/'logs'/f'validation-attempt-{number}.txt').write_text(str(error))
            stop(proc)
            status('retrying_validation',str(error))
status('failed_all_profiles','See logs/serve-attempt-*.log and validation-attempt-*.txt')
raise SystemExit(1)
