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
LAYER = "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
LAYER_SHA256 = "53ac234dce482d88feb19cacf3a4e375a2b0c8b8396e4dac1b2719a5ba3dffde"
MODEL = "python/sglang/srt/models/deepseek_v2.py"
MODEL_SHA256 = "1ba2ffe6bf7537d0fcef6945f44b239883125a8bc382e86fa57885c135bed21f"
WRAPPER = "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
TRACE = "python/sglang/srt/layers/moe/kt_localization_trace.py"


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
    path = root / LAYER
    before = path.read_bytes()
    if hashlib.sha256(before).hexdigest() != LAYER_SHA256:
        raise RuntimeError("MoE layer identity changed")
    original = before.decode("utf-8")
    old = "        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):\n            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)\n\n        return final_hidden_states\n"
    new = '''        # Local diagnostic only: last logical token at each MoE layer call.
        trace_root = os.environ.get("KT_LAYER_TRACE_DIR") if self.layer_id == 0 else None
        if trace_root:
            from pathlib import Path
            from torch import distributed as dist
            trace_dir = Path(trace_root)
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_call = getattr(self, "_kt_trace_call", 0)
            self._kt_trace_call = trace_call + 1
            trace_rank = dist.get_rank() if dist.is_initialized() else 0
            trace_prefix = f"layer{self.layer_id}-rank{trace_rank}-call{trace_call:03d}"
            pre_path = trace_dir / f"{trace_prefix}-pre.pt"
            if pre_path.exists():
                raise RuntimeError(f"Duplicate layer trace: {pre_path}")
            torch.save(final_hidden_states[-1].detach().cpu(), pre_path)

        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        if trace_root:
            post_path = trace_dir / f"{trace_prefix}-post.pt"
            if post_path.exists():
                raise RuntimeError(f"Duplicate layer trace: {post_path}")
            torch.save(final_hidden_states[-1].detach().cpu(), post_path)
        return final_hidden_states
'''
    changed = one(original, old, new)
    if "import os\n" not in changed:
        changed = one(changed, "import logging\n", "import logging\nimport os\n")
    changed = one(
        changed,
        "        origin_hidden_states_dim = hidden_states.shape[-1]\n",
        '''        if self.layer_id == 0:
            from sglang.srt.layers.moe.kt_localization_trace import save as kt_trace_save
            kt_trace_save(
                "moe_input", self.layer_id, getattr(self, "_kt_trace_call", 0),
                hidden=hidden_states, topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
            )
        origin_hidden_states_dim = hidden_states.shape[-1]
''',
    )
    compile(changed, str(path), "exec")
    path.write_text(changed, encoding="utf-8")
    diffs.append("".join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile="a/" + LAYER, tofile="b/" + LAYER,
    )))
    identities[LAYER] = {"base_sha256": LAYER_SHA256,
                         "candidate_sha256": hashlib.sha256(changed.encode()).hexdigest()}
    path = root / WRAPPER
    original = path.read_text(encoding="utf-8")
    changed = one(
        original,
        "        del dispatcher\n        use_graph = (\n",
        '''        del dispatcher
        if self.kt_config.layer_idx == 0:
            from sglang.srt.layers.moe.kt_localization_trace import save as kt_trace_save
            call = getattr(self, "_kt_component_call", 0)
            self._kt_component_call = call + 1
            kt_trace_save("kt_pre", 0, call, hidden=hidden_states,
                          topk_ids=topk_output.topk_ids,
                          topk_weights=topk_output.topk_weights)
        use_graph = (
''',
    )
    changed = one(
        changed,
        "        return hidden_states + cpu_output\n",
        '''        combined = hidden_states + cpu_output
        if self.kt_config.layer_idx == 0:
            from sglang.srt.layers.moe.kt_localization_trace import save as kt_trace_save
            kt_trace_save("kt_post", 0, self._kt_component_call - 1,
                          npu=hidden_states, cpu=cpu_output, combined=combined)
        return combined
''',
    )
    compile(changed, str(path), "exec")
    path.write_text(changed, encoding="utf-8")
    diffs.append("".join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile="a/" + WRAPPER, tofile="b/" + WRAPPER,
    )))
    identities[WRAPPER] = {"candidate_sha256": hashlib.sha256(changed.encode()).hexdigest()}
    path = root / MODEL
    before = path.read_bytes()
    if hashlib.sha256(before).hexdigest() != MODEL_SHA256:
        raise RuntimeError("DeepSeek V2 MoE identity changed")
    original = before.decode("utf-8")
    start = original.index("    def forward_normal(\n", original.index("class DeepseekV2MoE"))
    end = original.index("    def forward_cpu(\n", start)
    section = original[start:end]
    old = '''        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
'''
    new = '''        reduce_needed = self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        )
        if self.layer_id == 0:
            from sglang.srt.layers.moe.kt_localization_trace import save as kt_trace_save
            call = getattr(self, "_kt_v2_trace_call", 0)
            self._kt_v2_trace_call = call + 1
            kt_trace_save("v2_pre_reduce", 0, call,
                          hidden=final_hidden_states, reduce_needed=reduce_needed)
        if reduce_needed:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
'''
    section = one(section, old, new)
    section = one(
        section,
        '''        if shared_output is not None and self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states
''',
        '''        if shared_output is not None and self._shared_expert_tp1:
            final_hidden_states += shared_output
        if self.layer_id == 0:
            kt_trace_save("v2_after", 0, self._kt_v2_trace_call - 1,
                          hidden=final_hidden_states)
        return final_hidden_states
''',
    )
    changed = original[:start] + section + original[end:]
    compile(changed, str(path), "exec")
    path.write_text(changed, encoding="utf-8")
    diffs.append("".join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile="a/" + MODEL, tofile="b/" + MODEL,
    )))
    identities[MODEL] = {"base_sha256": MODEL_SHA256,
                         "candidate_sha256": hashlib.sha256(changed.encode()).hexdigest()}
    path = root / TRACE
    if path.exists():
        raise RuntimeError("Localization helper already exists")
    helper = '''"""Bounded tensor capture for an isolated layer-0 diagnostic."""
import os
from pathlib import Path
import torch
from torch import distributed as dist


def save(kind, layer, call, **values):
    root = os.environ.get("KT_COMPONENT_TRACE_DIR")
    if root is None or layer != 0:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"layer{layer}-rank{rank}-call{call:03d}-{kind}.pt"
    if path.exists():
        raise RuntimeError(f"Duplicate component trace: {path}")
    sample = {name: (value[-1].detach().cpu().contiguous()
                     if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > 0
                     else value.detach().cpu().contiguous() if isinstance(value, torch.Tensor)
                     else value)
              for name, value in values.items()}
    torch.save(sample, path)
'''
    compile(helper, str(path), "exec")
    path.write_text(helper, encoding="utf-8")
    diffs.append("".join(difflib.unified_diff(
        [], helper.splitlines(True), fromfile="/dev/null", tofile="b/" + TRACE,
    )))
    identities[TRACE] = {"candidate_sha256": hashlib.sha256(helper.encode()).hexdigest()}
    (root.parent / "candidate.diff").write_text("".join(diffs), encoding="utf-8")
    (root.parent / "probe-source.json").write_text(
        json.dumps(identities, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
