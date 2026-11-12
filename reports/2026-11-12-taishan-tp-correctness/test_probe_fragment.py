"""Exercise the isolated probe with SGLang's array-backed output history."""
from array import array
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


HERE = Path(__file__).resolve().parent


class CustomLogitProcessor:
    pass


class Request:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = array("i", [3, 4, 5])
        self.output_ids = array("i")


class ProbeTest(unittest.TestCase):
    def test_array_history_and_capture_toggle(self):
        namespace = {"CustomLogitProcessor": CustomLogitProcessor,
                     "torch": torch, "json": json}
        source = (HERE / "probe_processor_fragment.py.txt").read_text(encoding="utf-8")
        exec(compile(source, "probe_processor_fragment.py.txt", "exec"), namespace)
        probe = namespace["KTFrozenHistoryProbe"]()
        with tempfile.TemporaryDirectory() as temp:
            previous = os.environ.get("KT_PROBE_DIR")
            os.environ["KT_PROBE_DIR"] = temp
            try:
                request = Request("array-history")
                for step in range(3):
                    params = {"__req__": request, "kt_forced_ids": [1, 2],
                              "kt_capture": step != 1,
                              "__kt_probe_mode__": "EXTEND" if step == 0 else "DECODE",
                              "__kt_probe_graph_replay__": False,
                              "__kt_probe_seq_len__": 3 + step,
                              "__kt_probe_input_id__": None if step == 0 else step}
                    logits = torch.arange(10, dtype=torch.float32)[None, :]
                    probe(logits, [params])
                    if step < 2:
                        self.assertEqual(int(logits.argmax()), step + 1)
                        # The overlap scheduler may not publish the latest token yet.
                        if step == 1:
                            request.output_ids.append(1)
                    else:
                        self.assertEqual(int(logits.argmax()), 9)
                rows = [json.loads(line) for line in
                        (Path(temp) / "rank0.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["step"] for r in rows], [0, 1, 2])
                self.assertEqual([r["capture"] for r in rows], [True, False, True])
                self.assertEqual([r["scheduler_history_len"] for r in rows], [0, 0, 1])
                self.assertTrue((Path(temp) / "array-history-rank0-step000.pt").is_file())
                request.output_ids.append(2)
                params["__kt_probe_seq_len__"] = 6
                params["__kt_probe_input_id__"] = 9
                probe(torch.arange(10, dtype=torch.float32)[None, :], [params])
                extra = json.loads((Path(temp) / "rank0-extra.jsonl").read_text(
                    encoding="utf-8").splitlines()[0])
                self.assertEqual(extra["step"], 3)
                self.assertFalse((Path(temp) / "array-history-rank0-step003.pt").exists())
                partial = Request("chunked-prefill")
                partial.origin_input_ids = array("i", range(792))
                partial_params = {"__req__": partial, "kt_forced_ids": [1],
                                  "kt_capture": True, "__kt_probe_mode__": "EXTEND",
                                  "__kt_probe_graph_replay__": False,
                                  "__kt_probe_seq_len__": 512,
                                  "__kt_probe_input_id__": None}
                probe(torch.arange(10, dtype=torch.float32)[None, :],
                      [partial_params])
                partial_log = json.loads((Path(temp) / "rank0-partial.jsonl").read_text(
                    encoding="utf-8").splitlines()[0])
                self.assertEqual(partial_log["seq_len"], 512)
                self.assertFalse((Path(temp) / "chunked-prefill-rank0-step000.pt").exists())
                partial_params["__kt_probe_seq_len__"] = 792
                probe(torch.arange(10, dtype=torch.float32)[None, :],
                      [partial_params])
                partial_params["__kt_probe_seq_len__"] = 793
                partial_params["__kt_probe_mode__"] = "DECODE"
                partial_params["__kt_probe_input_id__"] = 1
                probe(torch.arange(10, dtype=torch.float32)[None, :],
                      [partial_params])
                import probe_client
                probe_client.check_capture(Path(temp), "chunked-prefill", 792, [1], 1, False)
                duplicate = Request("duplicate-position")
                duplicate_params = {"__req__": duplicate, "kt_forced_ids": [1],
                                    "kt_capture": True, "__kt_probe_mode__": "EXTEND",
                                    "__kt_probe_graph_replay__": False,
                                    "__kt_probe_seq_len__": 3,
                                    "__kt_probe_input_id__": None}
                for _ in range(2):
                    probe(torch.arange(10, dtype=torch.float32)[None, :],
                          [duplicate_params])
                rows = [json.loads(line) for line in
                        (Path(temp) / "rank0.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([r["call_for_position"] for r in rows[-2:]], [0, 1])
                self.assertTrue((Path(temp) / "duplicate-position-rank0-step000-call1.pt").is_file())
            finally:
                if previous is None:
                    os.environ.pop("KT_PROBE_DIR", None)
                else:
                    os.environ["KT_PROBE_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
