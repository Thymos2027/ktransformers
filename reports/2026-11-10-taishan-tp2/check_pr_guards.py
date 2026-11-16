"""Exercise the actual guard AST without importing accelerator dependencies."""
import ast
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

source = Path(sys.argv[1]).read_text(encoding="utf-8")
tree = ast.parse(source)
compile(source, sys.argv[1], "exec")
config = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_kt_config_from_server_args")
npu_guard = next(n for n in config.body if isinstance(n, ast.If) and ast.unparse(n.test) == "server_args.device == 'npu'")
guard_code = compile(ast.fix_missing_locations(ast.Module(body=[npu_guard], type_ignores=[])), "config-guard", "exec")
count = 0
for switch in (None, "0", "1"):
    if switch is None:
        os.environ.pop("KT_NPU_EXPERIMENTAL_TP2", None)
    else:
        os.environ["KT_NPU_EXPERIMENTAL_TP2"] = switch
    for device in ("npu", "cuda"):
        for tp in (1, 2, 4):
            for ep in (1, 2):
                for experts in (0, 32):
                    for backend in ("none", "deepep"):
                        args = NS(device=device, tp_size=tp, ep_size=ep, kt_num_gpu_experts=experts, moe_a2a_backend=backend)
                        expected = device != "npu" or (experts > 0 and backend == "none" and ep == 1 and (tp == 1 or (tp == 2 and switch == "1")))
                        try:
                            exec(guard_code, {"server_args": args, "os": os})
                            accepted = True
                        except ValueError:
                            accepted = False
                        assert accepted == expected, (switch, vars(args), expected)
                        count += 1

cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KTEPWrapperMethod") if any(isinstance(n, ast.ClassDef) and n.name == "KTEPWrapperMethod" for n in tree.body) else next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(isinstance(m, ast.FunctionDef) and m.name == "create_weights" for m in n.body))
weights = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "create_weights")
moe_guard = next(n for n in weights.body if isinstance(n, ast.If) and "KT_NPU_EXPERIMENTAL_TP2" in ast.unparse(n.test))
moe_code = compile(ast.fix_missing_locations(ast.Module(body=[moe_guard], type_ignores=[])), "moe-guard", "exec")
logging.disable(logging.CRITICAL)
for device in ("npu", "cuda"):
    for switch in ("0", "1"):
        os.environ["KT_NPU_EXPERIMENTAL_TP2"] = switch
        for tp in (1, 2):
            for moe_tp in (1, 2, 4):
                expected = not (device == "npu" and switch == "1" and tp == 2 and moe_tp != 2)
                env = {"os": os, "layer": NS(w13_weight=NS(device=NS(type=device)), moe_tp_size=moe_tp), "get_parallel": lambda: NS(tp_size=tp), "logger": logging.getLogger("guard"), "self": NS(kt_config=NS(layer_idx=0), tp_rank=0, num_gpu_experts=32), "intermediate_size_per_partition": 1024, "intermediate_size_full": 2048}
                try:
                    exec(moe_code, env)
                    accepted = True
                except ValueError:
                    accepted = False
                assert accepted == expected, (device, switch, tp, moe_tp)
                count += 1
print(f"PASS: syntax and {count} configuration/MoE guard cases (AST-isolated; no accelerator execution)")
