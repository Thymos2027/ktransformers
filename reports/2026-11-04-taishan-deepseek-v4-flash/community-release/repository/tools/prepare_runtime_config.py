# SPDX-License-Identifier: Apache-2.0
import argparse
import json
from pathlib import Path

ap=argparse.ArgumentParser(description='Create separate W8A8 runtime view; preserve source files.')
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
source=args.source.resolve()
target=args.output.resolve()
if source==target or source in target.parents or target in source.parents:
    raise SystemExit('Use separate non-nested source and output directories')
if target.exists() and any(target.iterdir()):raise SystemExit('Output must be empty')
config=json.loads((source/'config.json').read_text())
if config.get('quantization_config',{}).get('quant_method')!='compressed-tensors':raise SystemExit('Expected compressed-tensors W8A8')
ratios=config.get('compress_ratios')
if not isinstance(ratios,list):raise SystemExit('Missing compress_ratios')
config['compress_ratios']=[0 if x==1 else x for x in ratios]
target.mkdir(parents=True,exist_ok=True)
for p in source.iterdir():
    if p.name=='config.json' or p.name.startswith('.'):continue
    (target/p.name).symlink_to(p,target_is_directory=p.is_dir())
(target/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
