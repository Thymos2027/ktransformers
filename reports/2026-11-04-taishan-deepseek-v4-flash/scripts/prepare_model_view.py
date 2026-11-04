import hashlib
import json
from pathlib import Path

source=Path('/models/DeepSeek-V4-Flash-W8A8')
target=Path('/workspace/models/DeepSeek-V4-Flash-W8A8')
target.mkdir(parents=True,exist_ok=True)
for p in source.iterdir():
    if p.name.startswith('.') or p.name=='config.json':continue
    q=target/p.name
    if not q.exists():q.symlink_to(p)
config=json.loads((source/'config.json').read_text())
assert config['quantization_config']['quant_method']=='compressed-tensors'
ratios=config.get('compress_ratios')
assert isinstance(ratios,list),ratios
config['compress_ratios']=[0 if x==1 else x for x in ratios]
(target/'config.json').write_text(json.dumps(config,indent=2))
record={'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),'runtime_config_sha256':hashlib.sha256((target/'config.json').read_bytes()).hexdigest(),'change':'compress_ratios: non-compression sentinel 1 -> 0; original files untouched','original_compress_ratios':ratios}
Path('/workspace/artifacts/model-view.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
