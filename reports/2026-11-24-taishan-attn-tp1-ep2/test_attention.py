"""Verify full attention geometry and startup rejection of incomplete topology."""
import ast
from pathlib import Path
import re
import unittest

from graph_gate import check_decode_graph
from prepare_candidate import build_model,build_wrapper
from request_budget_probe import slots

HERE=Path(__file__).resolve().parent


def startup():
    path=HERE/('attention_controller.py' if (HERE/'attention_controller.py').exists() else 'remote_controller.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='check_startup')
    scope=dict(re=re,check_decode_graph=check_decode_graph)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'startup','exec'),scope)
    return scope['check_startup']


def log():
    lines=['tp_size=2 ep_size=2 kt_num_gpu_experts=64 max_total_tokens=16384 dp_size=2 enable_dp_attention=True max_running_requests=2']
    lines+=['Initialize DeepSeekV4TokenToKVPool with max_num_reqs=1']*2
    for layer in range(43):
        for rank in (0,1):
            lines.append(f'[KT EP2 AUDIT] layer={layer} rank={rank} start={rank*32} stop={rank*32+32} local=32 global=64 cpu_experts=192 moe_tp=1 intermediate=2048 cpu_owner={rank==0}')
            lines.append(f'[KT ATTN TP1 AUDIT] layer={layer} rank={rank} attn_tp=1 attn_dp=2 local_heads=64 total_heads=64 combine=hccl_allreduce_scatter')
    lines+=['TP0 EP0] Capture target decode NPU graph end','TP1 EP1] Capture target decode NPU graph end']
    return '\n'.join(lines)


class AttentionTests(unittest.TestCase):
    def test_complete_topology(self):
        self.assertEqual(startup()(log(),2,64,True)['attention_tp'],1)

    def test_sharded_heads_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'still sharded'):
            startup()(log().replace('local_heads=64','local_heads=32'),2,64,True)

    def test_missing_attention_rank_rejected(self):
        text=log().replace('[KT ATTN TP1 AUDIT] layer=0 rank=1','[MISSING] layer=0 rank=1')
        with self.assertRaisesRegex(RuntimeError,'Incomplete attention'):
            startup()(text,2,64,True)

    def test_missing_graph_rank_rejected(self):
        with self.assertRaises(RuntimeError):
            startup()(log().replace('TP1 EP1] Capture target decode NPU graph end',''),2,64,True)

    def test_non_dp_attention_rejected(self):
        with self.assertRaises(RuntimeError):
            startup()(log().replace('enable_dp_attention=True','enable_dp_attention=False'),2,64,True)

    def test_zero_request_pool_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'one request slot'):
            startup()(log().replace('max_num_reqs=1','max_num_reqs=0'),2,64,True)

    def test_wrong_combine_policy_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'policy audit'):
            startup()(log().replace('combine=hccl_allreduce_scatter','combine=reduce_scatterv'),2,64,True)

    def test_global_request_budget_matches_dp(self):
        text=(HERE/'launch.sh').read_text(encoding='utf-8')
        self.assertEqual(int(re.search(r'--max-running-requests (\d+)',text)[1]),2)
        self.assertEqual(int(re.search(r'--data-parallel-size (\d+)',text)[1]),2)

    def test_actual_pool_formula_when_snapshot_available(self):
        path=HERE/'source-snapshot/pool_configurator.py'
        if not path.exists():self.skipTest('Requires pinned pool source snapshot')
        source=path.read_text(encoding='utf-8')
        self.assertEqual(slots(source,1,2),0)
        self.assertEqual(slots(source,2,2),1)

    def test_real_source_patch(self):
        snapshot=HERE/'source-snapshot'
        if not (snapshot/'model.py').exists():self.skipTest('Requires pinned model source snapshot')
        model=build_model((snapshot/'model.py').read_text(encoding='utf-8'))
        ast.parse(model)
        self.assertIn('self.n_local_heads != self.n_heads',model)
        wrapper=build_wrapper((snapshot/'wrapper.py').read_text(encoding='utf-8'))
        self.assertIn('server_args.dp_size == 2 and server_args.enable_dp_attention',wrapper)
        self.assertIn('(2, 1, 2, 2, 1)',wrapper)


if __name__=='__main__':unittest.main()
