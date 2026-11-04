import datetime as dt
import json
from pathlib import Path
import time
import requests

root=Path(__file__).resolve().parent
session=requests.Session()
session.trust_env=False
base='http://192.168.137.11:18080'
report={'started_at':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),'endpoint':base+'/v1','tests':[]}
r=session.get(base+'/health',timeout=15);r.raise_for_status()
r=session.get(base+'/v1/models',timeout=15);r.raise_for_status()
model=r.json()['data'][0]['id']
report['model']=model
cases=[('chinese_capital','法国的首都是哪里？只回答城市名称。','巴黎'),('arithmetic','计算 17 + 25。只输出数字答案。','42')]
for name,prompt,expected in cases:
    start=time.monotonic()
    response=session.post(base+'/v1/chat/completions',json={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':128,'stream':False},timeout=(15,300))
    response.raise_for_status()
    data=response.json()
    content=data['choices'][0]['message'].get('content') or ''
    passed=expected in content
    item={'name':name,'passed':passed,'content':content,'usage':data.get('usage'),'seconds':time.monotonic()-start}
    report['tests'].append(item)
    print(json.dumps(item,ensure_ascii=False),flush=True)
    if not passed:raise ValueError(f'{name} failed: {content!r}')
start=time.monotonic()
parts=[]
done=False
first_token=None
with session.post(base+'/v1/chat/completions',json={'model':model,'messages':[{'role':'user','content':'Reply with only the word READY.'}],'temperature':0,'max_tokens':64,'stream':True},stream=True,timeout=(15,300)) as response:
    response.raise_for_status()
    response.encoding='utf-8'
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data: '):continue
        payload=line[6:]
        if payload=='[DONE]':done=True;break
        data=json.loads(payload)
        for choice in data.get('choices',[]):
            text=choice.get('delta',{}).get('content')
            if text:
                if first_token is None:first_token=time.monotonic()-start
                parts.append(text)
content=''.join(parts)
item={'name':'streaming','passed':done and 'READY' in content,'content':content,'done_marker':done,'first_content_seconds':first_token,'seconds':time.monotonic()-start}
report['tests'].append(item)
print(json.dumps(item,ensure_ascii=False),flush=True)
report['passed']=all(x['passed'] for x in report['tests'])
report['completed_at']=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
(root/'client-acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
if not report['passed']:raise SystemExit(1)
