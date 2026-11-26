"""Check that fallback, incomplete token evidence and partial matrices cannot pass."""
import json
import unittest

import prefill_gate as gate
import prepare_candidate
import workload


def frame(count, finish=None, text='A normal engineering answer.', ids=None, **extra):
    meta = dict(completion_tokens=count,prompt_tokens=1024,forward_entry_time=100.,
                prefill_finished_time=102.,finish_reason=None if finish is None else dict(type=finish))
    meta.update(extra)
    return ('data: '+json.dumps(dict(text=text,output_ids=list(range(count)) if ids is None else ids,
                                     meta_info=meta))+'\n').encode()


class ProtocolTests(unittest.TestCase):
    def consume(self, frames):
        times = iter(range(1,100))
        return workload.consume(frames,0.,clock=lambda:next(times))

    def test_empty_first_text_still_times_first_token_and_preserves_ids(self):
        result = self.consume([frame(1,text=''),frame(2),frame(3,'length'),b'data: [DONE]\n'])
        self.assertEqual(result['ttft_seconds'],1)
        self.assertEqual(result['tpot_ms'],1000)
        self.assertEqual(result['output_ids'],[0,1,2])
        self.assertEqual(result['server_prefill_seconds'],2)

    def test_missing_done_or_prefill_timestamps_rejected(self):
        for frames in ([frame(1,'stop')],
                       [frame(1,'stop',prefill_finished_time=0),b'data: [DONE]\n']):
            with self.assertRaises(RuntimeError):
                self.consume(frames)

    def test_missing_or_incomplete_output_ids_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Raw output IDs'):
            self.consume([frame(1),frame(2,'length',ids=[0]),b'data: [DONE]\n'])

    def test_short_body_is_flagged_without_discarding_token_timing(self):
        result = self.consume([frame(1),frame(64,'length',text='This'),b'data: [DONE]\n'])
        self.assertTrue(result['content_anomaly'])
        self.assertEqual(len(result['output_ids']),64)
        self.assertIsNotNone(result['tpot_ms'])


class StreamingTests(unittest.TestCase):
    def audit_log(self, tokens=4096):
        return '\n'.join(f'[KT PREFILL AUDIT] layer={i} tokens={tokens} handled=True' for i in range(43))

    def test_all_layers_must_stream(self):
        self.assertEqual(gate.request_audit(self.audit_log(),'stream',4096)['streamed_layers'],43)
        for log in ('',self.audit_log().replace('layer=42','layer=41'),self.audit_log(1024)):
            with self.assertRaises(RuntimeError):
                gate.request_audit(log,'stream',4096)

    def test_silent_fallback_rejected_even_with_success_records(self):
        for failure in ('streaming failed (OOM) -> hybrid fallback','inline resident L2 failed (error)'):
            with self.assertRaisesRegex(RuntimeError,'failed'):
                gate.request_audit(self.audit_log()+'\n'+failure,'bundle',4096)

    def test_control_and_short_prompts_must_not_stream(self):
        self.assertEqual(gate.request_audit('','baseline',4096)['streamed_layers'],0)
        self.assertEqual(gate.request_audit('','stream',144)['streamed_layers'],0)
        with self.assertRaises(RuntimeError):
            gate.request_audit(self.audit_log(),'baseline',4096)

    def startup_log(self,phase):
        active, bundle = int(phase!='baseline'),int(phase=='bundle')
        config = dict(PREFILL_PHASE=phase,ASCEND_RT_VISIBLE_DEVICES='0',TASK_QUEUE_ENABLE='1',
            KT_PREFILL_STREAM=str(active),KT_DYNAMIC_RESIDENT=str(bundle),KT_SIDE_STREAM=str(bundle),
            KT_SHARED_EXPERTS_STREAM='0',KT_MXFP4_DEPOOL=str(active),KT_MXFP4_GGUF_DEDUP=str(active),
            KT_PREFILL_STREAM_THRESHOLD='512',KT_STREAM_WARMUP='1')
        log = '[PREFILL CONFIG] '+json.dumps(config)+'\n'
        log += ('tp_size=1 ep_size=1 dp_size=1 kt_num_gpu_experts=32 kt_cpuinfer=96 '
                'kt_threadpool_count=4 max_total_tokens=16384 max_running_requests=1 '
                'chunked_prefill_size=8192 swa_full_tokens_ratio=0.5 enable_metrics=True '
                'stream_interval=1 disable_cuda_graph=False\nCapture target decode NPU graph end\n')
        if active:
            log += '[KT_STREAM][depool] reserved ND streaming slot\n'
        return log

    def test_startup_requires_correct_switches_and_hbm_slot(self):
        for phase in gate.PHASES:
            self.assertTrue(gate.startup(self.startup_log(phase),phase)['passed'])
        with self.assertRaises(RuntimeError):
            gate.startup(self.startup_log('stream').replace('reserved ND streaming slot','unreserved'),'stream')
        with self.assertRaises(RuntimeError):
            gate.startup(self.startup_log('stream'),'baseline')

    def test_candidate_instrumentation_requires_exact_return_site(self):
        self.assertIn('[KT PREFILL AUDIT]',prepare_candidate.patch(prepare_candidate.OLD))
        for text in ('',prepare_candidate.OLD*2):
            with self.assertRaises(ValueError):
                prepare_candidate.patch(text)


class SummaryTests(unittest.TestCase):
    def matrix(self):
        cases = [dict(name=f'input_{n}',prompt_tokens=n) for n in (144,1024,4096,7168)]
        rows = []
        for phase in gate.PHASES:
            for case in cases:
                for iteration in range(4):
                    row = dict(phase=phase,case=case['name'],iteration=iteration,
                        kind='warmup' if iteration==0 else 'formal',
                        usage=dict(prompt_tokens=case['prompt_tokens'],completion_tokens=64),
                        output_ids=list(range(64)),streaming_audit=dict(passed=True),content_anomaly=False)
                    row.update({metric:2. if phase=='baseline' else 1. for metric in gate.METRICS})
                    rows.append(row)
        return rows,cases

    def test_warmups_excluded_and_anomalous_decode_separately_qualified(self):
        rows,cases = self.matrix()
        rows[1]['content_anomaly'] = True
        result = gate.summarize(rows,cases)
        self.assertEqual(result['formal_requests'],36)
        self.assertEqual(result['groups'][0]['normal_content_requests'],2)
        self.assertEqual(result['comparisons'][0]['prefill_speedup'],2.)

    def test_incomplete_duplicate_or_wrong_budget_rejected(self):
        rows,cases = self.matrix()
        for invalid in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                gate.summarize(invalid,cases)
        rows[0]['usage']['completion_tokens']=63
        with self.assertRaises(ValueError):
            gate.summarize(rows,cases)


if __name__ == '__main__':
    unittest.main()
