"""Keep NPU combine selection and the post-experts reduction skip in sync."""
from pathlib import Path
import unittest

from combine_probe import policy
from prepare_candidate import build_moe_utils

SOURCE=Path(__file__).resolve().parent/'source-snapshot/moe_utils.py'


@unittest.skipUnless(SOURCE.exists(),'Requires pinned MoE policy snapshot; actual remote probe runs before model load')
class CombineTests(unittest.TestCase):
    def setUp(self):
        self.original=SOURCE.read_text(encoding='utf-8')
        self.patched=build_moe_utils(self.original)

    def test_original_npu_ep2_selects_unsupported_reduce_scatterv(self):
        self.assertEqual(policy(self.original),(True,True))

    def test_fixed_npu_keeps_allreduce_and_scatter(self):
        self.assertEqual(policy(self.patched),(False,False))

    def test_non_npu_policy_unchanged(self):
        self.assertEqual(policy(self.patched,npu=False),policy(self.original,npu=False))

    def test_other_npu_runs_unchanged(self):
        self.assertEqual(policy(self.patched,enabled=False),policy(self.original,enabled=False))

    def test_fused_reduction_still_skips(self):
        self.assertEqual(policy(self.patched,fuse=True),(False,True))


class SourceTests(unittest.TestCase):
    def test_unexpected_policy_source_rejected(self):
        with self.assertRaises(ValueError):build_moe_utils('unexpected source')


if __name__=='__main__':unittest.main()
