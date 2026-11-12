"""Enable a bounded TP2 experiment in an isolated, identity-checked source tree."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import sys


EXPECTED = "ac454c4aa5410181625a8c9aa520b7c18b6ab210e2706586e0fc7d64aad4dd8f"
RELATIVE = "python/sglang/srt/layers/moe/kt_ep_wrapper.py"


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    path = root / RELATIVE
    before = path.read_bytes()
    if hashlib.sha256(before).hexdigest() != EXPECTED:
        raise RuntimeError("KT wrapper identity changed; review before patching")
    original = before.decode()
    old = "        if server_args.tp_size != 1 or server_args.ep_size != 1:\n"
    new = '''        experimental_tp2 = (
            os.environ.get("KT_NPU_EXPERIMENTAL_TP2") == "1"
            and server_args.tp_size == 2
            and server_args.ep_size == 1
        )
        if (server_args.tp_size != 1 or server_args.ep_size != 1) and not experimental_tp2:
'''
    if original.count(old) != 1:
        raise RuntimeError("Expected one NPU TP guard")
    candidate = original.replace(old, new, 1)
    anchor = "        # 3. Initialize KT wrapper for CPU experts\n"
    addition = '''        if os.environ.get("KT_NPU_EXPERIMENTAL_TP2") == "1":
            if layer.moe_tp_size != 2:
                raise ValueError("TP2 trial requires MoE tensor parallel size 2")
            logger.warning(
                "[KT TP2 EXPERIMENT] layer=%s rank=%s moe_tp=%s local_intermediate=%s "
                "full_intermediate=%s resident_experts=%s cpu_owner=%s",
                self.kt_config.layer_idx, self.tp_rank, layer.moe_tp_size,
                intermediate_size_per_partition, intermediate_size_full,
                self.num_gpu_experts, self.tp_rank == 0,
            )

'''
    if candidate.count(anchor) != 1:
        raise RuntimeError("Expected one CPU wrapper creation anchor")
    candidate = candidate.replace(anchor, addition + anchor, 1)
    compile(candidate, str(path), "exec")
    path.write_bytes(candidate.encode())
    diff = "".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                       fromfile="a/" + RELATIVE, tofile="b/" + RELATIVE))
    (root.parent / "candidate.diff").write_text(diff, encoding="utf-8")
    (root.parent / "candidate.json").write_text(json.dumps({
        "base_sha256": EXPECTED,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "scope": "opt-in TP2/EP1 gate and diagnostics only; runtime support unverified",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
