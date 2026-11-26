"""Check public prompt identity and incomplete/miscounted result rejection."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'2026-11-25-taishan-prefill-streaming'
if PREVIOUS.is_dir():
    sys.path.insert(1,str(PREVIOUS))
import benchmark
import tutorial_gate


class BenchmarkTests(unittest.TestCase):
    def test_tokenize_uses_prompt_field_without_chat_template(self):
        from unittest.mock import Mock
        core=Mock()
        core.request.return_value={'count':2,'tokens':[1,2]}
        self.assertEqual(benchmark.tokenize_prompt(core,'raw prompt')['count'],2)
        core.request.assert_called_once_with(benchmark.workload.BASE,'/tokenize',
                                             {'prompt':'raw prompt'},timeout=30)

    def test_extract_exact_upstream_prompt_builder(self):
        text=(HERE/'decode_throughput_test.upstream.sh').read_text()
        code=benchmark.prompt_builder(text)
        with tempfile.TemporaryDirectory() as directory:
            paths=[str(Path(directory)/n) for n in ('prompt','probe','request')]
            with patch.object(sys,'argv',['builder','130',paths[0],'4.6','1000',paths[1],paths[2]]):
                exec(compile(code,'upstream-builder','exec'),{})
            import json
            request=json.loads(Path(paths[2]).read_text())
            self.assertTrue(request['text'].startswith('A logistics company'))
            self.assertEqual(request['sampling_params'],dict(max_new_tokens=1000,temperature=0,ignore_eos=True))
            self.assertNotIn('messages',request)
        with self.assertRaises(ValueError):
            benchmark.prompt_builder(text+text)

    def test_reject_wrong_output_budget(self):
        row=dict(usage=dict(prompt_tokens=801,completion_tokens=1000),
                 finish_reason='length',done=True,output_ids=list(range(1000)))
        benchmark.validate_result(row,801)
        row['output_ids'].pop()
        with self.assertRaises(RuntimeError):
            benchmark.validate_result(row,801)

    def test_partial_matrix_cannot_pass(self):
        with self.assertRaises(ValueError):
            benchmark.summarize([])


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.log=(HERE/'startup-reference.txt').read_text(encoding='utf-8')

    def test_actual_first_run_model_default_passes(self):
        result=tutorial_gate.startup(self.log,'table12')
        self.assertEqual(result['swa_ratio'],0.1)

    def test_wrong_ratio_and_unexpected_value_suffix_rejected(self):
        for value in ('0.5','0.15'):
            with self.assertRaisesRegex(RuntimeError,'swa_full_tokens_ratio'):
                tutorial_gate.startup(self.log.replace('swa_full_tokens_ratio=0.1',
                                                      'swa_full_tokens_ratio='+value),'table12')

    def test_16k_override_and_32k_memory_expert_settings_checked(self):
        log16=self.log.replace('swa_full_tokens_ratio=0.1','swa_full_tokens_ratio=0.35')
        self.assertEqual(tutorial_gate.startup(log16,'table19')['swa_ratio'],0.35)
        with self.assertRaises(RuntimeError):
            tutorial_gate.startup(self.log,'table19')
        log32=self.log.replace('kt_num_gpu_experts=32','kt_num_gpu_experts=28').replace(
            'mem_fraction_static=0.81','mem_fraction_static=0.775')
        self.assertEqual(tutorial_gate.startup(log32,'table39')['experts'],28)

    def test_kv_cap_fallback_and_disabled_optimization_rejected(self):
        for text in (self.log.replace('max_total_tokens=None','max_total_tokens=16384'),
                     self.log+'\nstreaming failed -> hybrid fallback',
                     self.log.replace('"KT_SIDE_STREAM": "1"','"KT_SIDE_STREAM": "0"')):
            with self.assertRaises(RuntimeError):
                tutorial_gate.startup(text,'table12')

    def test_hardware_profile_explicit_capacity_is_audited(self):
        settings=dict(experts=28,mem_fraction=0.775,swa_ratio=0.5,max_total_tokens=73728)
        text=self.log.replace('kt_num_gpu_experts=32','kt_num_gpu_experts=28').replace(
            'mem_fraction_static=0.81','mem_fraction_static=0.775').replace(
            'swa_full_tokens_ratio=0.1','swa_full_tokens_ratio=0.5').replace(
            'max_total_tokens=None','max_total_tokens=73728')
        self.assertTrue(tutorial_gate.startup(text,'hardwarelong',settings)['passed'])
        with self.assertRaises(RuntimeError):
            tutorial_gate.startup(self.log,'hardwarelong',settings)


class CapacityTests(unittest.TestCase):
    def test_actual_capped_initialization_overrides_estimate(self):
        log=(HERE/'capacity-reference.txt').read_text(encoding='utf-8')
        result=tutorial_gate.capacity(log,[dict(prompt_tokens=31540)],
                                      server_info={'max_total_num_tokens':73728})
        self.assertEqual((result['actual_full'],result['actual_swa']),(73728,36864))
        self.assertEqual(result['estimated_and_final_size_records'],2)
        self.assertTrue(result['initialized_pool_checked'])
        for text,info in ((log,{'max_total_num_tokens':120192}),
                          (log.replace('swa_size=36864','swa_size=60032'),{'max_total_num_tokens':73728}),
                          (log.split('Initialize DeepSeekV4TokenToKVPool')[0],{'max_total_num_tokens':73728})):
            with self.assertRaises(RuntimeError):
                tutorial_gate.capacity(text,[dict(prompt_tokens=31540)],server_info=info)

    def test_previous_16k_failure_rejected_before_request(self):
        with self.assertRaisesRegex(RuntimeError,'Actual pool insufficient'):
            tutorial_gate.capacity('DSV4 pool sizes: full=40832, swa=14208,',
                                   [dict(prompt_tokens=15568)])

    def test_32k_needs_page_and_extra_margin(self):
        cases=[dict(prompt_tokens=15568),dict(prompt_tokens=31540)]
        result=tutorial_gate.capacity('DSV4 pool sizes: full=73728, swa=36864,',cases)
        self.assertEqual(result['required_swa'],32768)
        self.assertEqual(result['required_full'],33768)
        for log in ('DSV4 pool sizes: full=73728, swa=31616,',
                    'DSV4 pool sizes: full=32768, swa=36864,',''):
            with self.assertRaises(RuntimeError):
                tutorial_gate.capacity(log,cases)


if __name__=='__main__':
    unittest.main()
