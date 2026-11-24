"""Execute generated EP2 initialization under a non-CPU torch default device."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace


def execute_probe(torch, source):
    tree = ast.parse(source)
    wrapper = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'KTEPWrapperMethod')
    constructor = next(x for x in wrapper.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
    # Select the innermost matching if, rather than the containing CPU-owner if.
    candidates = [x for x in ast.walk(wrapper) if isinstance(x, ast.If)
                  and 'EP2 trial does not support expert placement remapping' in ast.unparse(x)]
    mapping_gate = min(candidates, key=lambda x: len(ast.unparse(x)))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             constructor], type_ignores=[])
    ast.fix_missing_locations(module)
    os.environ['KT_NPU_EXPERIMENTAL_EP2'] = '1'
    checked = []
    for rank in (0, 1):
        parallel = SimpleNamespace(tp_size=2, tp_rank=rank, moe_ep_size=2, moe_ep_rank=rank, moe_tp_size=1)
        namespace = dict(torch=torch, os=os, KTRANSFORMERS_AVAILABLE=True, get_parallel=lambda: parallel)
        exec(compile(module, '<patched EP2 constructor>', 'exec'), namespace)
        global_mask = torch.arange(256, device='cpu') < 64
        config = SimpleNamespace(gpu_experts_mask=global_mask, logical_to_gpu_index=None, num_gpu_experts=64)
        instance, gpu_method = SimpleNamespace(), SimpleNamespace()
        physical_map = torch.arange(256, device='cpu')
        with torch.device('meta'):
            namespace['__init__'](instance, gpu_method, config)
            gate_scope = dict(torch=torch, self=instance, physical_to_logical_map_cpu=physical_map)
            exec(compile(ast.Module(body=[mapping_gate], type_ignores=[]), '<patched CPU mapping gate>', 'exec'), gate_scope)
        if instance.gpu_experts_mask.device.type != 'cpu' or instance.logical_to_gpu_index.device.type != 'cpu':
            raise RuntimeError('Load-time EP tables followed the accelerator default device')
        if instance.cpu_gpu_experts_mask is not global_mask or instance.num_gpu_experts != 32:
            raise RuntimeError('CPU union mask or local expert count differs')
        actual = torch.nonzero(instance.gpu_experts_mask).flatten().tolist()
        if actual != list(range(rank*32, (rank+1)*32)):
            raise RuntimeError('Rank-local expert ownership changed')
        checked.append(dict(rank=rank, expert_ids=actual, tables_device='cpu', default_device='meta'))
    return checked


def main():
    import json
    import torch
    source = Path('/experiment/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py').read_text()
    broken = source.replace('torch.tensor(cpu_skip, dtype=torch.bool, device="cpu")', 'torch.tensor(cpu_skip)')
    if broken == source:
        raise RuntimeError('Expected constructor regression anchor is absent')
    try:
        execute_probe(torch, broken)
    except (RuntimeError, NotImplementedError):
        reproduced = True
    else:
        raise RuntimeError('The historical device-placement bug was not reproduced')
    results = execute_probe(torch, source)
    print('EP2 device probe: ' + json.dumps(dict(passed=True, historical_bug_reproduced=reproduced, checks=results)), flush=True)
    return results


if __name__ == '__main__':
    main()
