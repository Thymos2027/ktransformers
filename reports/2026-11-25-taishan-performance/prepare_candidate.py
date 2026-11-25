"""Reuse EP2 ownership with selectable attention TP2 or TP1/DP2."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

import attn_prepare as attn
import ep_prepare as ep


def build_wrapper(text):
    text = ep.build_wrapper(text)
    anchor = '            if self.num_gpu_experts != 64 or self.gpu_experts_mask.numel() != 256:\n'
    return ep.replace(text, anchor, '''            attention_dp = os.environ.get("KT_NPU_EXPERIMENTAL_ATTN_TP1") == "1"
            expected_attention = (1, 2) if attention_dp else (2, 1)
            if (parallel.attn_tp_size, parallel.attn_dp_size) != expected_attention:
                raise ValueError("Attention topology differs from selected performance phase")
''' + anchor)


def main():
    root = Path(sys.argv[1]).resolve()
    if hashlib.sha256((root/ep.WRAPPER).read_bytes()).hexdigest() != ep.EXPECTED:
        raise RuntimeError('Pinned wrapper identity changed')
    differences, identities = [], {}
    for relative, builder in ((ep.WRAPPER, build_wrapper), (ep.LAYER, ep.build_layer),
                              (attn.MODEL, attn.build_model), (attn.MOE_UTILS, attn.build_moe_utils)):
        path = root/relative
        before = path.read_text(encoding='utf-8')
        after = builder(before)
        compile(after, str(path), 'exec')
        path.write_text(after,encoding='utf-8')
        identities[relative] = dict(base_sha256=hashlib.sha256(before.encode()).hexdigest(),
                                   candidate_sha256=hashlib.sha256(after.encode()).hexdigest())
        differences.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile='a/'+relative, tofile='b/'+relative))
    (root.parent/'candidate.diff').write_text(''.join(differences),encoding='utf-8')
    (root.parent/'candidate.json').write_text(json.dumps(dict(files=identities,
        scope='Prior EP2 residency and HCCL DP combine; attention TP2/DP1 or TP1/DP2; performance research'), indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
