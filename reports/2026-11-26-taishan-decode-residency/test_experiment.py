"""Actual-path and bounded-update evidence gates; independent of NPU availability."""
import json
import unittest
import prefill_gate as gate
from verify_results import measurement_log


def log(candidate=True,promotions=2):
    value='\n'.join(f'[KT PREFILL AUDIT] layer={i} tokens=4096 handled=True' for i in range(43))
    value+='\n[KT_STREAM] inline resident: top-32 x 43 layers folded into prefill, share=.5\n'
    if candidate:
        value+='[KT RESIDENCY] '+json.dumps(dict(event='update',completed_decodes=32,epoch=1,
                   promotions=promotions,promotions_total=promotions,route_choices=32*43*6,
                   cache_bytes=50*1024**2))
    return value


class GateTests(unittest.TestCase):
    def test_shutdown_tracebacks_are_retained_separately_from_measurement_errors(self):
        active,shutdown=measurement_log('measurements complete\nSIGTERM received.\nTraceback\nCancelledError')
        self.assertEqual(active,'measurements complete\n')
        self.assertTrue(shutdown['cancelled_error_after_sigterm'])
        active,_=measurement_log('Traceback\nSIGTERM received.\nCancelledError')
        self.assertIn('Traceback',active)
        with self.assertRaises(ValueError): measurement_log('SIGTERM received.\nOutOfMemoryError')

    def test_actual_path_and_observation_counts(self):
        result=gate.request_audit(log(),'b0',4096)
        self.assertTrue(result['candidate_triggered'])
        self.assertEqual(result['promotions'],2)
        self.assertFalse(gate.request_audit(log(promotions=0),'b0',4096)['candidate_triggered'])
        self.assertEqual(gate.request_audit(log(False),'a0',4096)['promotions'],0)

    def test_wrong_arm_missing_layers_and_partial_observation_fail(self):
        for text,phase in ((log(),'a0'),(log(False),'b0'),
                           (log().replace('layer=42','layer=41'),'b0'),
                           (log().replace(str(32*43*6),str(32*42*6)),'b0'),
                           (log()+'\nstreaming failed -> hybrid fallback','b0')):
            with self.assertRaises(RuntimeError): gate.request_audit(text,phase,4096)

    def test_budget_and_cache_capacity_fail_closed(self):
        with self.assertRaises(RuntimeError): gate.request_audit(log(promotions=9),'b0',4096)
        too_big=log().replace(str(50*1024**2),str(2*1024**3))
        with self.assertRaises(RuntimeError): gate.request_audit(too_big,'b0',4096)


if __name__=='__main__':
    unittest.main()
