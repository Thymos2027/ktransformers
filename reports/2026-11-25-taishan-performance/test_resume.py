"""Protect pre-profiler measurements from accidental reuse of changed results."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import resume_performance as resume
import startup_gate

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'runs/performance-20261125-093427-first/remote'


def example():
    frozen=dict(short=[dict(name='input_128',token_ids=[1,2],prompt_tokens=2)])
    rows=[]
    for phase in resume.REUSED_PHASES:
        for i in range(4):
            rows.append(dict(phase=phase,case='input_128',iteration=i,dp_rank=i%2,
                kind='warmup' if i==0 else 'formal',
                payload=dict(input_ids=[1,2],routed_dp_rank=i%2,sampling_params=dict(
                    temperature=0,sampling_seed=20261111,max_new_tokens=512,ignore_eos=True)),
                usage=dict(prompt_tokens=2,completion_tokens=512),done=True,finish_reason='length',
                content='example',content_sha256=hashlib.sha256(b'example').hexdigest(),
                token_events=[dict(completion_tokens=1,seconds=1),dict(completion_tokens=512,seconds=52.1)],
                ttft_seconds=1,total_seconds=53,tpot_ms=100,decode_tokens_per_second=10,
                meta_info=dict(prefill_finished_time=102,forward_entry_time=100),
                server_prefill_seconds=2,server_prefill_tokens_per_second=1))
    return rows,frozen


class ResumeTests(unittest.TestCase):
    def test_recompute_valid_rows(self):
        self.assertEqual(resume.verify_rows(*example()),8)

    def test_reject_changed_rows(self):
        for field,value in [('content','corrupt'),('tpot_ms',99),('server_prefill_seconds',0),
                             ('dp_rank',1),('done',False)]:
            rows,frozen=example()
            rows[0][field]=value
            with self.assertRaises(ValueError):resume.verify_rows(rows,frozen)
        rows,frozen=example()
        with self.assertRaises(ValueError):resume.verify_rows(rows[:-1],frozen)
        rows[0]['payload']['sampling_params']['sampling_seed']=0
        with self.assertRaises(ValueError):resume.verify_rows(rows,frozen)

    def test_no_profiler_entrypoint_in_performance_controller(self):
        source=(HERE/'remote_controller.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        self.assertNotIn('def profile(',source)
        self.assertFalse(any(isinstance(x,ast.Constant) and isinstance(x.value,str)
            and ('/start_profile' in x.value or '/stop_profile' in x.value) for x in ast.walk(tree)))

    @unittest.skipUnless((SOURCE/'requests.jsonl').is_file(),'Original checkpoint evidence only available on local workspace')
    def test_actual_checkpoint_and_rejection(self):
        frozen=resume.json_read(SOURCE/'frozen-workloads.json')
        rows=[json.loads(x) for x in (SOURCE/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(resume.verify_rows(rows,frozen),16)
        with tempfile.TemporaryDirectory(prefix='taishan-resume-test-') as folder:
            target=Path(folder)
            for name in (*resume.IDENTITY_FILES,'candidate.json','weight-preflight.json','benchmark-config.json','logs/environment.log'):
                dest=target/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes((SOURCE/name).read_bytes())
            record=resume.restore(SOURCE,target,startup_gate.check)
            self.assertEqual(record['requests'],16)
            self.assertEqual(record['formal_requests'],12)
            self.assertEqual((target/'requests.jsonl').read_bytes(),(SOURCE/'requests.jsonl').read_bytes())
            (target/'launch.sh').write_bytes(b'changed launch parameters')
            with self.assertRaisesRegex(ValueError,'runtime identity'):
                resume.restore(SOURCE,target,startup_gate.check)


if __name__=='__main__':
    unittest.main()
