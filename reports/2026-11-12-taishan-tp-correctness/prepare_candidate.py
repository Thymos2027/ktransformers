"""Patch only an identity-checked SGLang archive for a bounded TP probe."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
import gate_candidate as gate  # noqa: E402


RUNNER = "python/sglang/srt/model_executor/model_runner.py"
PROCESSOR = "python/sglang/srt/sampling/custom_logit_processor.py"
EXPECTED = {
    RUNNER: "aaeb661f2b6b385f6fe667f96039e569b6ff933a33530fe6d559675ea31f6f76",
    PROCESSOR: "e0e0ab65e2ded8975a9d161a52c92cb61f2a8a02a6af485b37912cad1171e23d",
}


def one(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one anchor: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    gate.main()
    diffs = [(root.parent / "candidate.diff").read_text(encoding="utf-8")]
    identities = {}
    for relative, expected in EXPECTED.items():
        path = root / relative
        before = path.read_bytes()
        actual = hashlib.sha256(before).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source identity mismatch: {relative}: {actual}")
        original = before.decode("utf-8")
        if relative == RUNNER:
            changed = one(
                original,
                "        self._preprocess_logits(logits_output, forward_batch.sampling_info)\n\n        # Sample the next tokens\n",
                """        self._preprocess_logits(logits_output, forward_batch.sampling_info)
        # Probe metadata is attached after the model forward and outside graph capture.
        if forward_batch.sampling_info.custom_params is not None:
            for i, params in enumerate(forward_batch.sampling_info.custom_params):
                if params is not None and "kt_forced_ids" in params:
                    params["__kt_probe_graph_replay__"] = bool(
                        getattr(forward_batch, "_kt_probe_graph_replay", False)
                    )
                    params["__kt_probe_mode__"] = forward_batch.forward_mode.name
                    params["__kt_probe_seq_len__"] = int(forward_batch.seq_lens[i].item())
                    params["__kt_probe_input_id__"] = (
                        int(forward_batch.input_ids[i].item())
                        if forward_batch.forward_mode.is_decode() else None
                    )

        # Sample the next tokens
""",
            )
            changed = one(
                changed,
                "            # Replay cuda graph if applicable\n            if can_run_graph:\n",
                """            # The flag records the branch actually taken, not graph eligibility.
            forward_batch._kt_probe_graph_replay = False
            # Replay cuda graph if applicable
            if can_run_graph:
                forward_batch._kt_probe_graph_replay = True
""",
            )
        else:
            addition = (HERE / "probe_processor_fragment.py.txt").read_text(encoding="utf-8")
            changed = original + "\n\n" + addition
        compile(changed, str(path), "exec")
        path.write_text(changed, encoding="utf-8")
        diffs.append("".join(difflib.unified_diff(
            original.splitlines(True), changed.splitlines(True),
            fromfile="a/" + relative, tofile="b/" + relative,
        )))
        identities[relative] = {"base_sha256": expected,
                                "candidate_sha256": hashlib.sha256(changed.encode()).hexdigest()}
    (root.parent / "candidate.diff").write_text("".join(diffs), encoding="utf-8")
    (root.parent / "probe-source.json").write_text(
        json.dumps(identities, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
