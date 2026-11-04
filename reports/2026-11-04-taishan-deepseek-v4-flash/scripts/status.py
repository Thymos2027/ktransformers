import datetime as dt
import json
import os
from pathlib import Path
import sys

root=Path('/workspace')
path=root/'status.json'
data=json.loads(path.read_text()) if path.is_file() else {}
stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
data.setdefault('started_at',stamp)
data.update(phase=sys.argv[1],updated_at=stamp,message=' '.join(sys.argv[2:]))
tmp=path.with_suffix('.tmp')
tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False))
os.replace(tmp,path)
print(json.dumps(data,ensure_ascii=False),flush=True)
