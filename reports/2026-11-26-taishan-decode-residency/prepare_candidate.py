"""Install the bounded residency port only into a verified isolated SGLang tree."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
BASE = 'python/sglang/srt/'
IDENTITIES = {
    BASE+'layers/moe/kt_stream_prefill.py': 'cbe7904dd2ddd87f4e94c2a7fda56af7f84be9d54a80117e564e54a318047dd2',
    BASE+'layers/moe/kt_ep_wrapper.py': 'ac454c4aa5410181625a8c9aa520b7c18b6ab210e2706586e0fc7d64aad4dd8f',
    BASE+'model_executor/model_runner.py': 'aaeb661f2b6b385f6fe667f96039e569b6ff933a33530fe6d559675ea31f6f76',
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned integration site changed: '+old[:90])
    return text.replace(old, new)


def patched(name, text):
    if name.endswith('kt_stream_prefill.py'):
        old = '        return _streaming_forward(layer_idx, hidden_states, topk_output, top_k, E)'
        return replace_once(text, old, '''        out = _streaming_forward(layer_idx, hidden_states, topk_output, top_k, E)
        logger.info("[KT PREFILL AUDIT] layer=%d tokens=%d handled=True", layer_idx, hidden_states.shape[0])
        return out''')
    if name.endswith('kt_ep_wrapper.py'):
        text = replace_once(text, '    wrapper.run_pinned_forward_sync(hidden_states, stream_handle)',
                            '''    wrapper.run_pinned_forward_sync(hidden_states, stream_handle)
    from sglang.srt.layers.moe.kt_decode_residency import record_decode
    record_decode(wrapper, hidden_states)''')
        text = replace_once(text, '        maybe_reserve_slot(self, layer.w13_weight.device, layer)',
                            '''        maybe_reserve_slot(self, layer.w13_weight.device, layer)
        from sglang.srt.layers.moe.kt_decode_residency import register
        register(self, layer)''')
        old = '        self._ascend_pending_join = (join_event, side_stream, compute_stream)'
        return replace_once(text, old, old+'''
        if not getattr(self, "_kt_side_audit_logged", False):
            logger.info("[KT SIDE AUDIT] layer=%d forked=True", self.kt_config.layer_idx)
            self._kt_side_audit_logged = True''')
    old = '        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)'
    return replace_once(text, old, old+'''
        if not self.is_draft_worker:
            from sglang.srt.layers.moe.kt_decode_residency import before_forward
            before_forward(forward_batch)''')


def main(root):
    before, contents = {}, {}
    for name, expected in IDENTITIES.items():
        data = (root/name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError('Pinned source identity mismatch: '+name)
        before[name] = actual
        contents[name] = patched(name, data.decode())
        compile(contents[name], name, 'exec')
    for name, text in contents.items():
        (root/name).write_text(text, encoding='utf-8', newline='\n')
    names = list(contents)
    for name in ('kt_decode_residency.py', 'residency_policy.py'):
        destination = root/BASE/'layers/moe'/name
        if destination.exists():
            raise ValueError('Refuse to overwrite an existing integration module')
        compile((HERE/name).read_text(), name, 'exec')
        shutil.copyfile(HERE/name, destination)
        names.append(str(destination.relative_to(root)).replace('\\', '/'))
    record = dict(before=before, after={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
                  feature='Bounded decode promotion with native prepared-byte cache and CPU fallback',
                  reference='vllm-ascend PR18604 f687e5ba2d672a9f10eebc6b516ddcc160060f05',
                  enabled_by_default=False, tp=1, ep=1, max_running_requests=1)
    (root.parent/'candidate.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
