import json
from pathlib import Path

root=Path('/workspace/sources/ktransformers')
changed=[]
for p in root.rglob('*'):
    if not p.is_file() or p.relative_to(root).parts[0]=='third_party':continue
    if p.suffix not in ('.sh','.patch','.cmake') and p.name!='CMakeLists.txt':continue
    data=p.read_bytes()
    if b'\r\n' in data:
        p.write_bytes(data.replace(b'\r\n',b'\n'))
        changed.append(str(p.relative_to(root)))
out=Path('/workspace/artifacts/source-line-ending-normalization.json')
previous=json.loads(out.read_text()) if out.is_file() else []
out.write_text(json.dumps(sorted(set(previous+changed)),indent=2))
print('Normalized CRLF source files:',len(changed),flush=True)
