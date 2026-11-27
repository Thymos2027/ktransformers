"""Policy, partial failure and exact pinned-source integration checks (CPU only)."""
import hashlib
import tempfile
from pathlib import Path
import unittest

import prepare_candidate as prepare
from residency_policy import Policy, Promotion, transact


class PolicyTests(unittest.TestCase):
    def test_hot_misses_replace_cold_residents_without_changing_routes(self):
        p = Policy(experts=8, budget=1)
        p.reset({0:[0,1],1:[0,1]})
        for _ in range(8):
            p.observe(0,[1,4])
            p.observe(1,[0,5])
        plan = p.plan()
        self.assertEqual(plan,[Promotion(0,0,0,4,8.)])
        self.assertEqual(p.slots[0],[0,1])
        self.assertEqual((p.routes,p.hits),(32,16))
        p.accept(plan)
        self.assertEqual(p.slots[0],[4,1])

    def test_hysteresis_and_sparse_samples_prevent_churn(self):
        p=Policy(experts=8)
        p.reset({0:[0,1]})
        for _ in range(3): p.observe(0,[4])
        self.assertEqual(p.plan(),[])
        p.reset({0:[0,1]})
        for _ in range(8): p.observe(0,[0,1,4])
        self.assertEqual(p.plan(),[])

    def test_global_budget_and_one_slot_per_layer(self):
        p=Policy(experts=8,budget=2)
        p.reset({i:[0,1] for i in range(4)})
        for _ in range(10):
            for layer in range(4): p.observe(layer,[4,5])
        result=p.plan()
        self.assertEqual(len(result),2)
        self.assertEqual(len({i.layer for i in result}),2)

    def test_reset_clears_request_history_and_invalid_ids_fail(self):
        p=Policy(experts=8)
        p.reset({0:[1,2]})
        p.observe(0,[3,4])
        p.reset({0:[3,4]})
        self.assertEqual((p.routes,p.hits,p.steps[0]),(0,0,0))
        for ids in ([8],[-1],[0.5]):
            with self.assertRaises(ValueError): p.observe(0,ids)
        with self.assertRaises(ValueError): p.reset({0:[1,1]})

    def test_stale_slot_generation_is_rejected(self):
        p=Policy(experts=8)
        p.reset({0:[0,1]})
        with self.assertRaises(RuntimeError): p.accept([Promotion(0,0,2,4,1)])


class TransactionTests(unittest.TestCase):
    def test_prepares_every_bundle_before_overwrite_and_publishes_after_fence(self):
        events=[]
        transact([1,2],lambda i: events.append(('prepare',i)) or i,
                 lambda i,b: events.append(('copy',i)),lambda:events.append(('fence',)),
                 lambda p:events.append(('publish',)))
        self.assertEqual(events,[('prepare',1),('prepare',2),('copy',1),('copy',2),('fence',),('publish',)])

    def test_partial_copy_failure_fences_and_never_publishes(self):
        events=[]
        def copy(i,b):
            events.append(('copy',i))
            if i==2: raise RuntimeError('injected partial DMA failure')
        with self.assertRaises(RuntimeError):
            transact([1,2],lambda i:i,copy,lambda:events.append(('fence',)),
                     lambda p:events.append(('publish',)))
        self.assertEqual(events,[('copy',1),('copy',2),('fence',)])

    def test_prepare_failure_never_changes_resident_weights(self):
        events=[]
        def prepare(i):
            if i==2: raise RuntimeError('conversion failed')
            return i
        with self.assertRaises(RuntimeError):
            transact([1,2],prepare,lambda *a:events.append('copy'),lambda:events.append('fence'),lambda p:events.append('publish'))
        self.assertEqual(events,[])


class SourceTests(unittest.TestCase):
    def test_full_installer_verifies_and_records_all_five_deployed_files(self):
        cache=Path(__file__).parent/'source-cache'
        if not cache.exists(): self.skipTest('Pinned baseline is checked by remote preparer')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'sglang'
            for name in prepare.IDENTITIES:
                target=root/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((cache/Path(name).name).read_bytes())
            prepare.main(root)
            import json
            manifest=json.loads((root.parent/'candidate.json').read_text())
            self.assertEqual(len(manifest['after']),5)
            for name,digest in manifest['after'].items():
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)
                compile((root/name).read_text(encoding='utf-8'),name,'exec')

    def test_pinned_source_patch_compiles_and_off_switch_is_present(self):
        cache=Path(__file__).parent/'source-cache'
        if not cache.exists(): self.skipTest('Source cache is local only; deployed tree is verified by preparer')
        for name, expected in prepare.IDENTITIES.items():
            data=(cache/Path(name).name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),expected)
            text=prepare.patched(name,data.decode())
            compile(text,name,'exec')
            if name.endswith('model_runner.py'): self.assertIn('before_forward(forward_batch)',text)
            if name.endswith('kt_ep_wrapper.py'):
                self.assertIn('record_decode(wrapper, hidden_states)',text)
                self.assertIn('register(self, layer)',text)

    def test_changed_or_duplicate_integration_site_fails(self):
        for text in ('missing','a a'):
            with self.assertRaises(ValueError): prepare.replace_once(text,'a','b')


if __name__ == '__main__':
    unittest.main()
