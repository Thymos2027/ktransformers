"""Add success-only streaming evidence to an isolated, pinned SGLang copy."""
import hashlib
import json
from pathlib import Path
import sys

STREAM = 'python/sglang/srt/layers/moe/kt_stream_prefill.py'
IDENTITIES = {
    STREAM: 'cbe7904dd2ddd87f4e94c2a7fda56af7f84be9d54a80117e564e54a318047dd2',
    'python/sglang/srt/layers/moe/kt_ep_wrapper.py':
        'ac454c4aa5410181625a8c9aa520b7c18b6ab210e2706586e0fc7d64aad4dd8f',
}
OLD = '        return _streaming_forward(layer_idx, hidden_states, topk_output, top_k, E)'
NEW = '''        out = _streaming_forward(layer_idx, hidden_states, topk_output, top_k, E)
        logger.info("[KT PREFILL AUDIT] layer=%d tokens=%d handled=True",
                    layer_idx, hidden_states.shape[0])
        return out'''


def patch(text):
    if text.count(OLD) != 1:
        raise ValueError('Pinned streaming return site changed')
    return text.replace(OLD, NEW)


def main(root):
    before = {}
    for name, expected in IDENTITIES.items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned source identity mismatch: '+name)
        before[name] = actual
    path = root/STREAM
    text = patch(path.read_text())
    compile(text, str(path), 'exec')
    path.write_text(text)
    record = dict(before=before, after={STREAM:hashlib.sha256(path.read_bytes()).hexdigest()},
                  change='Log layer/token count after successful streaming; no tensor sync or timing calls',
                  compute_implementation_unchanged=True, tp=1, ep=1)
    (root.parent/'candidate.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
