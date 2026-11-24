"""Extend the bounded EP2 trial with independent full attention replicas."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

import ep_prepare as ep

MODEL = 'python/sglang/srt/models/deepseek_v4.py'
MOE_UTILS = 'python/sglang/srt/layers/moe/utils.py'


def build_wrapper(text):
    text = ep.build_wrapper(text)
    text = ep.replace(text, '            and server_args.tp_size == 2 and server_args.ep_size == 2\n',
        '''            and os.environ.get("KT_NPU_EXPERIMENTAL_ATTN_TP1") == "1"
            and server_args.tp_size == 2 and server_args.ep_size == 2
            and server_args.dp_size == 2 and server_args.enable_dp_attention
''')
    return ep.replace(text,
        '            if (parallel.tp_size, parallel.moe_ep_size, parallel.moe_tp_size) != (2, 2, 1):\n',
        '''            if (parallel.tp_size, parallel.attn_tp_size, parallel.attn_dp_size,
                parallel.moe_ep_size, parallel.moe_tp_size) != (2, 1, 2, 2, 1):
''').replace('EP2 requires TP2 attention / EP2 experts / MoE TP1',
                   'Trial requires attention TP1/DP2, EP2 experts / MoE TP1')


build_layer = ep.build_layer


def build_model(text):
    text = ep.replace(text, 'import logging\n', 'import logging\nimport os\n')
    anchor = '        from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis\n'
    audit = '''        if os.environ.get("KT_NPU_EXPERIMENTAL_ATTN_TP1") == "1":
            parallel = get_parallel()
            if (parallel.tp_size, parallel.attn_tp_size, parallel.attn_dp_size,
                parallel.moe_ep_size, parallel.moe_tp_size) != (2, 1, 2, 2, 1):
                raise ValueError("Resolved attention/MoE topology differs from TP1/DP2/EP2")
            if self.attn_tp_size != 1 or self.n_local_heads != self.n_heads or self.n_local_groups != self.n_groups:
                raise ValueError("Attention projections remain tensor sharded")
            logger.warning(
                "[KT ATTN TP1 AUDIT] layer=%s rank=%s attn_tp=1 attn_dp=2 "
                "local_heads=%s total_heads=%s wq_b_shape=%s wo_b_shape=%s",
                layer_id, parallel.tp_rank, self.n_local_heads, self.n_heads,
                list(self.wq_b.weight.shape), list(self.wo_b.weight.shape),
            )

'''
    audit = audit.replace('            logger.warning(\n', '''            from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
            if should_use_dp_reduce_scatterv():
                raise ValueError("NPU trial must use post-experts AllReduce and DP scatter")
            logger.warning(
''').replace('total_heads=%s wq_b_shape=',
             'total_heads=%s combine=hccl_allreduce_scatter wq_b_shape=')
    return ep.replace(text, anchor, audit + anchor)


def build_moe_utils(text):
    anchor = '''    return (
        not should_use_flashinfer_cutlass_moe_fp4_allgather()
        and get_moe_a2a_backend().is_none()
        and is_dp_attention_enabled()
        and get_parallel().attn_dp_size > 1
        and get_parallel().tp_size == get_parallel().attn_dp_size
        and get_parallel().moe_ep_size == get_parallel().attn_dp_size
    )'''
    return ep.replace(text, anchor, '''    # Bounded NPU DP-attention trial: reduce_scatterv requires a PyNCCL
    # communicator, absent on HCCL. Keep the same decision for both the
    # post-experts all-reduce skip gate and the decoder's DP combine gate.
    if _is_npu and os.environ.get("KT_NPU_EXPERIMENTAL_ATTN_TP1") == "1":
        return False
''' + anchor)


def main():
    root = Path(sys.argv[1]).resolve()
    if hashlib.sha256((root/ep.WRAPPER).read_bytes()).hexdigest() != ep.EXPECTED:
        raise ValueError('Pinned KT wrapper identity changed')
    differences, identities = [], {}
    for relative, builder in ((ep.WRAPPER,build_wrapper),(ep.LAYER,build_layer),
                              (MODEL,build_model),(MOE_UTILS,build_moe_utils)):
        path=root/relative
        before=path.read_text()
        after=builder(before)
        compile(after,str(path),'exec')
        path.write_text(after)
        identities[relative]=dict(base_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(after.encode()).hexdigest())
        differences.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile='a/'+relative,tofile='b/'+relative))
    (root.parent/'candidate.diff').write_text(''.join(differences))
    (root.parent/'candidate.json').write_text(json.dumps(dict(files=identities,
        scope='Attention TP1/DP2, MoE EP2/TP1, disjoint 32+32 full experts; numerical correctness unaccepted'),indent=2)+'\n')


if __name__=='__main__':main()
