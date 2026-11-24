"""Disjoint ownership, CPU complement and weighted sum checks."""
import ast
from pathlib import Path
import unittest
from ep_layout import layout
from prepare_candidate import build_wrapper, build_layer


class LayoutTests(unittest.TestCase):
    def test_exact_disjoint_ownership(self):
        a,b=layout(0),layout(1)
        self.assertEqual([i for i,x in enumerate(a[1]) if x],list(range(32)))
        self.assertEqual([i for i,x in enumerate(b[1]) if x],list(range(32,64)))
        self.assertEqual(a[0],b[0])
        self.assertEqual(sum(a[0]),64)
        for i in range(256):
            self.assertEqual(int(a[1][i])+int(b[1][i]),int(a[0][i]))

    def test_loader_slots_match_global_ids(self):
        for rank in (0,1):
            _,mask,slots=layout(rank)
            self.assertEqual([slots[i] for i in range(256) if mask[i]],list(range(32)))
            self.assertTrue(all(slots[i]==-1 for i in range(256) if not mask[i]))

    def test_every_expert_computed_exactly_once(self):
        a,b=layout(0),layout(1)
        for i in range(256):
            cpu=not a[0][i]
            self.assertEqual(int(cpu)+int(a[1][i])+int(b[1][i]),1)

    def test_weighted_sum_with_cpu_added_only_on_rank_zero(self):
        a,b=layout(0),layout(1)
        for ids in ([1,3,6,10,20,31],[32,33,40,44,50,63],[64,85,107,128,215,255],[3,27,44,107,126,215]):
            weights=[.1,.15,.2,.25,.12,.18]
            reference=sum(w*(i+1)**2 for i,w in zip(ids,weights))
            rank0=sum(w*(i+1)**2 for i,w in zip(ids,weights) if a[1][i] or not a[0][i])
            rank1=sum(w*(i+1)**2 for i,w in zip(ids,weights) if b[1][i])
            self.assertAlmostEqual(rank0+rank1,reference)

    def test_invalid_rank_rejected(self):
        with self.assertRaises(ValueError):layout(2)

    def test_patch_rejects_unexpected_source(self):
        with self.assertRaises(ValueError):build_wrapper('unexpected')
        with self.assertRaises(ValueError):build_layer('unexpected')

    def test_patched_real_sources_compile_when_snapshot_available(self):
        root=Path(__file__).parent/'source-snapshot'
        if not root.exists():self.skipTest('Pinned source snapshot is local preparation evidence')
        for name,builder in [('wrapper.py',build_wrapper),('layer.py',build_layer)]:
            source=(root/name).read_text(encoding='utf-8')
            result=builder(source)
            ast.parse(result)
            self.assertNotEqual(source,result)

    def test_new_load_time_tensor_factories_pin_cpu_device(self):
        root=Path(__file__).parent/'source-snapshot'
        if not root.exists():self.skipTest('Requires pinned source snapshot')
        source=(root/'wrapper.py').read_text(encoding='utf-8')
        patched=build_wrapper(source)
        before=ast.parse(source)
        after=ast.parse(patched)
        def factories(tree):
            return {ast.unparse(node) for node in ast.walk(tree) if isinstance(node,ast.Call)
                    and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id=='torch' and node.func.attr in ('tensor','arange')}
        introduced=factories(after)-factories(before)
        self.assertEqual(len(introduced),5)
        for expression in introduced:
            node=ast.parse(expression,mode='eval').body
            device=next((kw.value for kw in node.keywords if kw.arg=='device'),None)
            self.assertIsNotNone(device,expression)
            if node.func.attr=='tensor':self.assertEqual(ast.literal_eval(device),'cpu')


if __name__=='__main__':unittest.main()
