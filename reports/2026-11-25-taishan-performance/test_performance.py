"""Protocol checks protect measured timing, token counts and ownership."""
import json
import unittest

import workload
import startup_gate


def frame(count, finish=None, text='156', **changes):
    meta=dict(completion_tokens=count,prompt_tokens=144,forward_entry_time=100.0,
              prefill_finished_time=102.0,finish_reason=None if finish is None else dict(type=finish))
    meta.update(changes)
    return ('data: '+json.dumps(dict(text=text,meta_info=meta))+'\n').encode()


class ProtocolTests(unittest.TestCase):
    def consume(self,frames):
        times=iter([1.,2.,3.,4.,5.,6.,7.,8.])
        return workload.consume(frames,0.,clock=lambda:next(times))

    def test_token_events_are_not_text_chunk_counts(self):
        result=self.consume([frame(1,text=''),frame(2),frame(3,'length'),frame(3,'length'),b'data: [DONE]\n'])
        self.assertEqual(result['ttft_seconds'],1)
        self.assertEqual(result['tpot_ms'],1000)
        self.assertEqual(result['server_prefill_seconds'],2)
        self.assertEqual(result['server_prefill_tokens_per_second'],72)
        self.assertEqual(len(result['token_events']),3)

    def test_missing_done_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            self.consume([frame(1,'stop')])

    def test_missing_prefill_timestamps_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'timestamps'):
            self.consume([frame(1,'stop',prefill_finished_time=0),b'data: [DONE]\n'])

    def test_counter_regression_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'decreased'):
            self.consume([frame(2),frame(1,'stop'),b'data: [DONE]\n'])

    def test_aggregated_first_event_does_not_fabricate_tpot(self):
        result=self.consume([frame(2),frame(3,'length'),b'data: [DONE]\n'])
        self.assertIsNone(result['tpot_ms'])
        self.assertEqual(result['event_token_jumps'],1)

    def test_server_error_is_not_a_completion(self):
        with self.assertRaisesRegex(RuntimeError,'generation error'):
            self.consume([b'data: {"error":"scheduler failed"}\n'])


class StartupTests(unittest.TestCase):
    def log(self,dpa,graph,wide):
        text=('tp_size=2 ep_size=2 kt_num_gpu_experts=64 max_total_tokens=16384 '
              'kt_cpuinfer=96 enable_metrics=True stream_interval=1 '
              f'chunked_prefill_size={8192 if wide else 512} '
              f'swa_full_tokens_ratio={0.5 if wide else 0.1} '
              f'dp_size={2 if dpa else 1} enable_dp_attention={dpa} '
              f'max_running_requests={2 if dpa else 1} disable_cuda_graph={not graph}\n')
        for layer in range(43):
            for rank in (0,1):
                text+=(f'[KT EP2 AUDIT] layer={layer} rank={rank} start={rank*32} stop={(rank+1)*32} '
                       f'local=32 global=64 cpu_experts=192 moe_tp=1 intermediate=2048 cpu_owner={rank==0}\n')
                if dpa:
                    text+=(f'[KT ATTN TP1 AUDIT] layer={layer} rank={rank} attn_tp=1 attn_dp=2 '
                           'local_heads=64 total_heads=64 combine=hccl_allreduce_scatter\n')
        text+='Initialize DeepSeekV4TokenToKVPool with max_num_reqs=1\n'*2
        if graph:
            text+='TP0 EP0] Capture target decode NPU graph end\nTP1 EP1] Capture target decode NPU graph end\n'
        return text

    def test_all_five_phase_shapes(self):
        for dpa,graph,wide in ((True,False,False),(True,True,False),(False,True,False),
                               (False,True,True),(True,True,True)):
            text=self.log(dpa,graph,wide)
            self.assertEqual(startup_gate.check(text,dpa,graph,wide)['expert_audit_records'],86)
            for bad in (text.replace('max_num_reqs=1','max_num_reqs=0'),
                        text.replace('cpu_owner=False','cpu_owner=True',1),
                        text.replace('layer=0 rank=0 start=','layer=43 rank=0 start=',1)):
                with self.assertRaises(RuntimeError):
                    startup_gate.check(bad,dpa,graph,wide)


if __name__=='__main__':
    unittest.main()
