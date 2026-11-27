"""Reject incorrect execution paths, truncated evidence and broad device cleanup."""
import json
import unittest
from unittest.mock import patch

import prefill_gate as gate
import prepare_candidate
import runtime_support
import workload


def startup_log(phase):
    active, side = int(phase in gate.STREAM), int(phase in gate.SIDE)
    config = dict(PREFILL_PHASE=phase, ASCEND_RT_VISIBLE_DEVICES='0', TASK_QUEUE_ENABLE='1',
        SGLANG_SET_CPU_AFFINITY='0', KT_PREFILL_STREAM=str(active), KT_DYNAMIC_RESIDENT='0',
        KT_SIDE_STREAM=str(side), KT_SHARED_EXPERTS_STREAM='0', KT_MXFP4_DEPOOL=str(active),
        KT_MXFP4_GGUF_DEDUP=str(active), KT_PREFILL_STREAM_THRESHOLD='512', KT_STREAM_WARMUP='1',
        KT_HOT_TAIL_TOKENS='0')
    text = '[PREFILL CONFIG] '+json.dumps(config)+'\n'
    text += ('tp_size=1 ep_size=1 dp_size=1 kt_num_gpu_experts=32 kt_cpuinfer=96 '
             'kt_threadpool_count=4 max_total_tokens=16384 max_running_requests=1 '
             'chunked_prefill_size=8192 swa_full_tokens_ratio=0.5 enable_metrics=True '
             'stream_interval=1 disable_cuda_graph=False\nCapture target decode NPU graph end\n')
    if active:
        text += '[KT_STREAM][depool] reserved ND streaming slot\n'
    if side:
        text += '\n'.join(f'[KT SIDE AUDIT] layer={i} forked=True' for i in range(43))
    return text


def valid_row(phase='h0', case='short', prompt=144, iteration=1):
    row = dict(phase=phase, case=case, iteration=iteration,
        kind='warmup' if iteration == 0 else 'formal',
        usage=dict(prompt_tokens=prompt, completion_tokens=1000),
        output_ids=list(range(1000)), done=True, finish_reason='length', content_anomaly=False,
        token_events=[dict(seconds=i, completion_tokens=i) for i in range(1,1001)],
        ttft_seconds=1., total_seconds=1000.1, tpot_ms=1000., decode_tokens_per_second=1.,
        server_prefill_seconds=.9, server_prefill_tokens_per_second=prompt/.9,
        streaming_audit=dict(passed=True))
    row.update(gate.validate_request(row,prompt))
    return row


class ExecutionTests(unittest.TestCase):
    def test_phase_flags_and_actual_side_forks(self):
        for phase in gate.PHASES:
            self.assertTrue(gate.startup(startup_log(phase),phase)['passed'])
        for bad in (startup_log('h1').replace('layer=42','layer=41'),
                    startup_log('h1').replace('forked=True','forked=False'),
                    startup_log('s1').replace('"KT_DYNAMIC_RESIDENT": "0"','"KT_DYNAMIC_RESIDENT": "1"')):
            with self.assertRaises(RuntimeError):
                gate.startup(bad,'h1' if '"PREFILL_PHASE": "h1"' in bad else 's1')

    def test_stream_layers_control_and_fallback(self):
        audit = '\n'.join(f'[KT PREFILL AUDIT] layer={i} tokens=4096 handled=True' for i in range(43))
        self.assertEqual(gate.request_audit(audit,'s0',4096)['streamed_layers'],43)
        self.assertEqual(gate.request_audit('','h1',4096)['streamed_layers'],0)
        self.assertEqual(gate.request_audit('','s1',144)['streamed_layers'],0)
        for text,phase in ((audit,'h0'),('', 's1'),(audit+'\nstreaming failed -> hybrid fallback','s0')):
            with self.assertRaises(RuntimeError):
                gate.request_audit(text,phase,4096)

    def test_capacity_uses_finalized_allocation(self):
        log = 'DSV4 pool sizes: full=310912, swa=155392\nDSV4 pool sizes: full=16384, swa=8192'
        info = dict(context_length=8192,max_total_num_tokens=16384)
        self.assertEqual(gate.capacity(log,info)['swa'],8192)
        with self.assertRaises(RuntimeError):
            gate.capacity(log.replace('swa=8192','swa=4096'),info)
        with self.assertRaises(RuntimeError):
            gate.capacity(log,dict(context_length=8192,max_total_num_tokens=8192))

    def test_exact_pinned_patch_sites(self):
        self.assertIn('[KT PREFILL AUDIT]',prepare_candidate.patch(prepare_candidate.OLD))
        for text in ('',prepare_candidate.OLD*2):
            with self.assertRaises(ValueError):
                prepare_candidate.patch(text)


class EvidenceTests(unittest.TestCase):
    def test_complete_long_output_and_windows(self):
        checked = gate.validate_request(valid_row(),144)
        self.assertEqual(checked['window_tpot_ms']['513-1000'],1000.)
        self.assertTrue(checked['timing_audit']['per_token_arrivals_complete'])

    def test_interior_notification_gap_keeps_exact_endpoint_metrics(self):
        row=valid_row()
        row['token_events'].pop(524)  # Token 525 is not separately notified.
        row['event_token_jumps']=1
        checked=gate.validate_request(row,144)
        self.assertEqual(checked['window_tpot_ms']['513-1000'],1000.)
        self.assertEqual(checked['timing_audit']['notification_count'],999)
        self.assertEqual(checked['timing_audit']['cumulative_count_jumps'][0]['unobserved_counts'],[525])
        self.assertFalse(checked['timing_audit']['per_token_arrivals_complete'])
        self.assertFalse(checked['timing_audit']['interpolated_timestamps'])

    def test_unobserved_window_boundary_is_null_not_interpolated(self):
        row=valid_row()
        row['token_events'].pop(511)  # Missing token 512 timing.
        checked=gate.validate_request(row,144)
        self.assertIsNone(checked['window_tpot_ms']['257-512'])
        self.assertEqual(checked['window_tpot_ms']['513-1000'],1000.)
        self.assertEqual(checked['timing_audit']['missing_window_boundaries'],[512])

    def test_missing_endpoints_or_non_monotonic_counts_still_fail(self):
        for change in ('first','last','duplicate','reverse','nan'):
            row=valid_row()
            if change=='first': row['token_events'].pop(0)
            elif change=='last': row['token_events'].pop()
            elif change=='duplicate': row['token_events'][10]['completion_tokens']=10
            elif change=='reverse': row['token_events'][10]['completion_tokens']=9
            else: row['token_events'][10]['seconds']=float('nan')
            with self.assertRaises(RuntimeError):
                gate.validate_request(row,144)

    def test_missing_ids_events_and_anomalies_rejected(self):
        for field,value in (('output_ids',list(range(999))),('token_events',[]),('content_anomaly',True)):
            row=valid_row()
            row[field]=value
            with self.assertRaises(RuntimeError):
                gate.validate_request(row,144)

    def test_full_matrix_and_warmup_count(self):
        cases=[dict(name='short',prompt_tokens=144),dict(name='long',prompt_tokens=4096)]
        rows=[valid_row(phase,c['name'],c['prompt_tokens'],i)
              for phase in gate.PHASES for c in cases for i in range(4)]
        result=gate.summarize(rows,cases)
        self.assertEqual((result['formal_requests'],result['warmup_requests'],len(result['groups'])),(24,8,8))
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                gate.summarize(bad,cases)

    def test_summary_exposes_missing_window_samples(self):
        cases=[dict(name='short',prompt_tokens=144),dict(name='long',prompt_tokens=4096)]
        rows=[valid_row(phase,c['name'],c['prompt_tokens'],i)
              for phase in gate.PHASES for c in cases for i in range(4)]
        for row in rows:
            if row['phase']=='s1' and row['case']=='long' and row['kind']=='formal':
                row['token_events'].pop(511)
                row.update(gate.validate_request(row,4096))
        group=gate.summarize(rows,cases)['groups'][-1]
        self.assertIsNone(group['window_tpot_ms']['257-512'])
        self.assertEqual(group['window_sample_counts']['257-512'],0)
        self.assertEqual(group['requests_with_count_jumps'],3)

    def test_native_protocol_preserves_ids_and_empty_first_token_timing(self):
        frames=[]
        for count,text in ((1,''),(2,'A normal model answer.')):
            frames.append(('data: '+json.dumps(dict(text=text,output_ids=list(range(count)),
                meta_info=dict(completion_tokens=count,prompt_tokens=144,forward_entry_time=1.,
                    prefill_finished_time=2.,finish_reason=dict(type='length') if count==2 else None)))+'\n').encode())
        frames.append(b'data: [DONE]\n')
        clock=iter((1.,2.,3.,4.))
        row=workload.consume(frames,0.,clock=lambda:next(clock))
        self.assertEqual(row['ttft_seconds'],1.)
        self.assertEqual(row['output_ids'],[0,1])
        with self.assertRaises(RuntimeError):
            workload.consume(frames[:-1],0.,clock=lambda:1.)


class ScopeTests(unittest.TestCase):
    def test_remote_stop_cannot_include_original(self):
        with patch.object(runtime_support,'_stop_all') as stop:
            runtime_support.scoped_stop(include_original=True)
            stop.assert_called_once_with(include_original=False)

    def test_container_only_maps_npu2(self):
        with patch.object(runtime_support.core,'safe'), patch.object(runtime_support.core,'docker',return_value='id') as docker:
            runtime_support.create('h0')
            args=docker.call_args.args
            self.assertIn('/dev/davinci2',args)
            self.assertNotIn('/dev/davinci5',args)
            self.assertIn(runtime_support.core.ORIGINAL+':ro',args)


if __name__ == '__main__':
    unittest.main()
