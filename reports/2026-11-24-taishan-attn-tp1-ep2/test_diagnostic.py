"""Preserve the failed JSON requirement and route each request explicitly."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import diagnostic_workload as client

HERE=Path(__file__).resolve().parent


class DiagnosticTests(unittest.TestCase):
    def test_invalid_json_remains_failed(self):
        tree=ast.parse((HERE/'diagnostic_controller.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='assess')
        scope=dict(json=json)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'diagnostic assessment','exec'),scope)
        check=scope['assess']
        self.assertTrue(check('[1,2,5,9]',[1,2,5,9]))
        self.assertFalse(check('["json_array": [1, 2, 5, 9]]',[1,2,5,9]))
        self.assertFalse(check('{"json_array": [1, 2, 5, 9]}',[1,2,5,9]))

    def test_original_prompt_unchanged(self):
        tree=ast.parse((HERE/'diagnostic_controller.py').read_text(encoding='utf-8'))
        cases=ast.literal_eval(next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets)).value)
        path=HERE/'workload.py'
        if not path.exists():path=HERE.parent/'2026-11-11-taishan-tp-speed/workload.py'
        source=ast.parse(path.read_text(encoding='utf-8'))
        gate=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='gate')
        frozen=ast.literal_eval(next(n for n in gate.body if isinstance(n,ast.Assign)).value)
        self.assertEqual(cases[0][1],next(c[1] for c in frozen if c[0]=='sorting'))

    def test_request_routes_and_preserves_sampling(self):
        captured=[]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __iter__(self):
                row=dict(choices=[dict(delta=dict(content='42'),finish_reason='stop')],
                    usage=dict(prompt_tokens=5,completion_tokens=1))
                return iter([b'data: '+json.dumps(row).encode()+b'\n',b'data: [DONE]\n'])
        def open_request(request,timeout):
            captured.append(json.loads(request.data))
            return Response()
        core=SimpleNamespace(HTTP=SimpleNamespace(open=open_request),now=lambda:'test',safe=lambda:None)
        for rank in (0,1):
            result=client.stream(core,'model',[dict(role='user',content='prompt')],64,routed_dp_rank=rank)
            client.check_transport(result)
            self.assertEqual(captured[-1]['routed_dp_rank'],rank)
            self.assertEqual(captured[-1]['seed'],20261111)
            self.assertEqual(captured[-1]['temperature'],0)
            self.assertEqual(captured[-1]['max_tokens'],64)
            self.assertFalse(captured[-1]['ignore_eos'])


if __name__=='__main__':unittest.main()
