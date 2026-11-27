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


SIDE_OLD = '        self._ascend_pending_join = (join_event, side_stream, compute_stream)'
SIDE_NEW = SIDE_OLD + '\n        if not getattr(self, "_kt_side_audit_logged", False):\n            logger.info("[KT SIDE AUDIT] layer=%d forked=True", self.kt_config.layer_idx)\n            self._kt_side_audit_logged = True'


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
    side_path = root/'python/sglang/srt/layers/moe/kt_ep_wrapper.py'
    side = side_path.read_text()
    if side.count(SIDE_OLD) != 1:
        raise ValueError('Pinned side-stream fork site changed')
    side = side.replace(SIDE_OLD, SIDE_NEW)
    compile(side, str(side_path), 'exec')
    side_path.write_text(side)
    path = root/STREAM
    text = patch(path.read_text())
    compile(text, str(path), 'exec')
    path.write_text(text)
    record = dict(before=before, after={STREAM:hashlib.sha256(path.read_bytes()).hexdigest(),
                  'python/sglang/srt/layers/moe/kt_ep_wrapper.py':hashlib.sha256(side_path.read_bytes()).hexdigest()},
                  change='Success-only streaming logs and one side-fork log per layer at graph capture; no tensor synchronization or timing changes',
                  compute_implementation_unchanged=True, tp=1, ep=1)
    (root.parent/'candidate.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
