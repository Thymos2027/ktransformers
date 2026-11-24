"""Apply the original TP2 opt-in patch, then audit actual resident tensors."""
from pathlib import Path
import difflib
import hashlib
import json
import sys

import base_prepare


def main():
    root = Path(sys.argv[1]).resolve()
    path = root / base_prepare.RELATIVE
    original = path.read_text()
    base_prepare.main()
    candidate = path.read_text()
    anchor = "        # 3. Load CPU weights using KT wrapper\n"
    audit = '''        # Startup-only audit, before graph capture and request timing.
        mask = self.gpu_experts_mask
        expected = torch.arange(mask.numel(), device=mask.device) < self.num_gpu_experts
        if mask.numel() != self.global_num_experts or not torch.equal(mask, expected):
            raise RuntimeError("Resident expert mask is not the requested contiguous prefix")
        if (self.wrapper is not None) != (self.tp_rank == 0):
            raise RuntimeError("CPU executor ownership differs from rank-zero-only contract")
        shapes = {key: list(getattr(layer, key).shape)
                  for key in ("w13_weight", "w2_weight") if hasattr(layer, key)}
        if len(shapes) != 2 or any(s[0] != self.num_gpu_experts for s in shapes.values()):
            raise RuntimeError(f"Resident weight tensor expert axes mismatch: {shapes}")
        logger.warning(
            "[KT RESIDENCY AUDIT] layer=%s rank=%s resident=%s cpu_experts=%s "
            "cpu_owner=%s weight_shapes=%s",
            self.kt_config.layer_idx, self.tp_rank, self.num_gpu_experts,
            self.global_num_experts - self.num_gpu_experts, self.tp_rank == 0, shapes,
        )

'''
    if candidate.count(anchor) != 1:
        raise RuntimeError("Expected one pre-capture CPU weight-loading anchor")
    candidate = candidate.replace(anchor, audit + anchor, 1)
    compile(candidate, str(path), "exec")
    path.write_bytes(candidate.encode())
    diff = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                      fromfile='a/' + base_prepare.RELATIVE,
                                      tofile='b/' + base_prepare.RELATIVE))
    (root.parent / 'candidate.diff').write_text(diff)
    (root.parent / 'candidate.json').write_text(json.dumps(dict(
        base_sha256=base_prepare.EXPECTED,
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        scope='Original TP2 opt-in patch plus startup-only residency tensor/mask audit; no forward-path changes',
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
