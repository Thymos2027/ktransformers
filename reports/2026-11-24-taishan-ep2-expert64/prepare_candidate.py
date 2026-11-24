"""Opt in to disjoint full-expert ownership with one global CPU executor."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

WRAPPER = 'python/sglang/srt/layers/moe/kt_ep_wrapper.py'
LAYER = 'python/sglang/srt/layers/moe/fused_moe_triton/layer.py'
EXPECTED = 'ac454c4aa5410181625a8c9aa520b7c18b6ab210e2706586e0fc7d64aad4dd8f'


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source anchor mismatch: ' + old[:100])
    return text.replace(old, new, 1)


def build_wrapper(text):
    text = replace(text, '        if server_args.tp_size != 1 or server_args.ep_size != 1:\n', '''        ep2_trial = (
            os.environ.get("KT_NPU_EXPERIMENTAL_EP2") == "1"
            and server_args.tp_size == 2 and server_args.ep_size == 2
            and server_args.kt_num_gpu_experts == 64
            and server_args.kt_expert_placement_strategy == "prefix"
            and not server_args.enable_eplb
        )
        if (server_args.tp_size != 1 or server_args.ep_size != 1) and not ep2_trial:
''')
    text = replace(text, '        self.override_num_local_experts = True\n', '''        self.npu_ep2 = os.environ.get("KT_NPU_EXPERIMENTAL_EP2") == "1"
        self.cpu_gpu_experts_mask = self.gpu_experts_mask
        if self.npu_ep2:
            from ep_layout import layout
            parallel = get_parallel()
            if (parallel.tp_size, parallel.moe_ep_size, parallel.moe_tp_size) != (2, 2, 1):
                raise ValueError("EP2 requires TP2 attention / EP2 experts / MoE TP1")
            if self.num_gpu_experts != 64 or self.gpu_experts_mask.numel() != 256:
                raise ValueError("EP2 requires 64 globally resident experts out of 256")
            cpu_skip, local_mask, slots = layout(parallel.moe_ep_rank)
            if not torch.equal(self.gpu_experts_mask.cpu(), torch.tensor(cpu_skip, dtype=torch.bool, device="cpu")):
                raise ValueError("EP2 requires prefix residency and identity expert placement")
            self.gpu_experts_mask = torch.tensor(local_mask, dtype=torch.bool, device="cpu")
            self.logical_to_gpu_index = torch.tensor(slots, dtype=torch.long, device="cpu")
            self.num_gpu_experts = 32
        self.override_num_local_experts = True
''')
    text = replace(text, '        self.global_num_experts = num_experts\n', '''        if self.npu_ep2:
            if num_experts != 128 or intermediate_size_per_partition != 2048:
                raise ValueError("Unexpected EP2 model geometry")
            num_experts = 256  # CPU routes use global expert IDs, not an EP-local shard.
        self.global_num_experts = num_experts
''')
    text = replace(text, '                gpu_experts_mask=self.gpu_experts_mask,\n',
                   '                gpu_experts_mask=self.cpu_gpu_experts_mask if self.npu_ep2 else self.gpu_experts_mask,\n')
    text = replace(text, '            self.wrapper.load_weights(physical_to_logical_map_cpu)\n', '''            if self.npu_ep2 and not torch.equal(
                physical_to_logical_map_cpu.cpu().to(torch.long), torch.arange(256, device="cpu")
            ):
                raise ValueError("EP2 trial does not support expert placement remapping")
            self.wrapper.load_weights(physical_to_logical_map_cpu)
''')
    text = replace(text, '        return int(self.logical_to_gpu_index[logical_expert_id].item())\n', '''        slot = int(self.logical_to_gpu_index[logical_expert_id].item())
        if self.npu_ep2 and slot >= 0:
            if "_ep2_loaded_ids" not in self.__dict__:
                self._ep2_loaded_ids = set()
            self._ep2_loaded_ids.add(logical_expert_id)
        return slot
''')
    anchor = '        # 3. Load CPU weights using KT wrapper\n'
    audit = '''        if self.npu_ep2:
            start = get_parallel().moe_ep_rank * 32
            expected = torch.arange(256, device=self.gpu_experts_mask.device)
            if not torch.equal(self.gpu_experts_mask, (expected >= start) & (expected < start + 32)):
                raise ValueError("EP2 local residency changed")
            if (self.wrapper is not None) != (self.tp_rank == 0):
                raise ValueError("EP2 CPU executor duplicated or missing")
            if self.__dict__.get("_ep2_loaded_ids", set()) != set(range(start, start + 32)):
                raise ValueError("EP2 checkpoint loader did not visit the assigned 32 global experts")
            shapes = {key: list(getattr(layer, key).shape) for key in ("w13_weight", "w2_weight")}
            if shapes != {"w13_weight": [32, 4096, 4096], "w2_weight": [32, 2048, 4096]}:
                raise ValueError(f"EP2 full-expert weight shapes differ: {shapes}")
            logger.warning(
                "[KT EP2 AUDIT] layer=%s rank=%s start=%s stop=%s local=32 global=64 "
                "cpu_experts=192 moe_tp=1 intermediate=2048 cpu_owner=%s shapes=%s",
                self.kt_config.layer_idx, self.tp_rank, start, start + 32,
                self.tp_rank == 0, shapes,
            )

'''
    return replace(text, anchor, audit + anchor)


def build_layer(text):
    anchor = '    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:\n'
    return replace(text, anchor, anchor + '''        if isinstance(self.quant_method, KTEPWrapperMethod) and getattr(self.quant_method, "npu_ep2", False):
            # Preserve the global ID until the KT resident-slot mapper. Standard
            # EP ownership (0..127 / 128..255) differs from this bounded trial.
            return expert_id if 0 <= expert_id < 64 else -1
''')


def main():
    root = Path(sys.argv[1]).resolve()
    if hashlib.sha256((root / WRAPPER).read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Pinned KT wrapper identity changed')
    differences, identities = [], {}
    for relative, builder in ((WRAPPER, build_wrapper), (LAYER, build_layer)):
        path = root / relative
        before = path.read_text()
        after = builder(before)
        compile(after, str(path), 'exec')
        path.write_text(after)
        identities[relative] = dict(base_sha256=hashlib.sha256(before.encode()).hexdigest(),
                                   candidate_sha256=hashlib.sha256(after.encode()).hexdigest())
        differences.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile='a/'+relative, tofile='b/'+relative))
    (root.parent / 'candidate.diff').write_text(''.join(differences))
    (root.parent / 'candidate.json').write_text(json.dumps(dict(files=identities,
        scope='Experimental EP2 prefix64, 32 full experts per rank, CPU complement on rank zero; not accepted'), indent=2)+'\n')


if __name__ == '__main__':
    main()
