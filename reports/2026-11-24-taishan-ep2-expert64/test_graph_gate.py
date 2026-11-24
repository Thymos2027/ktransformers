"""Cover rank-qualified capture labels, missing ranks and actual EP2 logs."""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest

import ep_checkpoint
from graph_gate import check_decode_graph

HERE=Path(__file__).resolve().parent
SOURCE=(HERE/'runs/ep2-expert64-20261124-device-fix/remote' if os.name=='nt' else
        Path('/data/deploy/dsv4-tp2-ep2-expert64-20261124-device-fix'))


def startup_function():
    tree=ast.parse((HERE/'remote_controller.py').read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='check_startup')
    scope=dict(re=re,check_decode_graph=check_decode_graph)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'remote_controller.py','exec'),scope)
    return scope['check_startup']


class GraphTests(unittest.TestCase):
    def test_tp_only_labels(self):
        check_decode_graph('TP0] Capture target decode NPU graph end\nTP1] Capture target decode NPU graph end')

    def test_tp_ep_labels(self):
        check_decode_graph('TP0 EP0] Capture target decode NPU graph end\nTP1 EP1] Capture target decode NPU graph end')

    def test_missing_rank_rejected(self):
        with self.assertRaises(RuntimeError):check_decode_graph('TP0 EP0] Capture target decode NPU graph end')

    def test_start_marker_does_not_prove_success(self):
        with self.assertRaises(RuntimeError):check_decode_graph('TP0 EP0] Capture target decode NPU graph start\nTP1 EP1] Capture target decode NPU graph end')

    def test_wrong_ep_rank_rejected(self):
        with self.assertRaises(RuntimeError):check_decode_graph('TP0 EP1] Capture target decode NPU graph end\nTP1 EP0] Capture target decode NPU graph end')

    @unittest.skipUnless((SOURCE/'logs/ep2graph.log').exists(),'Requires archived EP2 model log')
    def test_real_ep2_startup_and_complete_residency_audit(self):
        log=(SOURCE/'logs/ep2graph.log').read_text(encoding='utf-8').partition('SIGTERM received.')[0]
        self.assertEqual(startup_function()(log,2,64,True)['audit_records'],86)


@unittest.skipUnless((SOURCE/'ep2eager-smoke.json').exists(),'Requires archived EP2 eager evidence')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target=Path(self.temp.name)
        for name in ('benchmark-config.json','candidate.json','environment-preflight.json','weight-preflight.json',
                     'source.json','launch.sh','workload.py','requests.json'):
            shutil.copyfile(SOURCE/name,self.target/name)

    def test_actual_eager_phase_is_reusable(self):
        record=ep_checkpoint.validate(SOURCE,self.target,startup_function())
        self.assertEqual(record['reused_phases'],['ep2eager'])
        self.assertNotIn('requests.jsonl',record['copied_files'])

    def test_changed_native_backend_rejected(self):
        p=self.target/'environment-preflight.json'
        d=json.loads(p.read_text(encoding='utf-8'));d['extension_sha256']='different'
        p.write_text(json.dumps(d),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'identity changed'):
            ep_checkpoint.validate(SOURCE,self.target,startup_function())

    def test_changed_candidate_rejected(self):
        p=self.target/'candidate.json';p.write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'identity changed'):
            ep_checkpoint.validate(SOURCE,self.target,startup_function())


if __name__=='__main__':unittest.main()
