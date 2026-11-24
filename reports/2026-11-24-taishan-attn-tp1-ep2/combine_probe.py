"""Check the patched policy and real HCCL sum / DP scatter before model load."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def policy(source, npu=True, enabled=True, fuse=False):
    names={'should_use_dp_reduce_scatterv','should_skip_post_experts_all_reduce'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=2:raise ValueError('MoE combine policy source changed')
    parallel=SimpleNamespace(attn_dp_size=2,tp_size=2,moe_ep_size=2,dwdp_size=1)
    backend=SimpleNamespace(is_none=lambda:True,is_flashinfer=lambda:False,is_pplx=lambda:False)
    scope=dict(_is_npu=npu,os=SimpleNamespace(environ={'KT_NPU_EXPERIMENTAL_ATTN_TP1':'1' if enabled else '0'}),
        should_use_flashinfer_cutlass_moe_fp4_allgather=lambda:False,
        get_moe_a2a_backend=lambda:backend,is_dp_attention_enabled=lambda:True,
        get_parallel=lambda:parallel,should_skip_mlp_all_reduce=lambda:fuse)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual MoE policy','exec'),scope)
    return (scope['should_use_dp_reduce_scatterv'](),
            scope['should_skip_post_experts_all_reduce'](is_tp_path=True))


def check(rank):
    import torch
    import torch.distributed as dist
    from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton

    root=Path('/experiment')
    source=(root/'sglang/python/sglang/srt/layers/moe/utils.py').read_text()
    if policy(source)!=(False,False):
        raise RuntimeError('NPU trial must retain exactly one post-experts AllReduce')
    if policy(source,npu=False)!=(True,True):
        raise RuntimeError('Non-NPU combine policy changed')
    dp_source=(root/'sglang/python/sglang/srt/layers/dp_attention.py').read_text()
    scatter=next(n for n in ast.parse(dp_source).body if isinstance(n,ast.FunctionDef) and n.name=='dp_scatter')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.Module(body=[future,scatter],type_ignores=[])
    ast.fix_missing_locations(module)
    device=f'npu:{rank}'
    records=[]
    for sizes in ((3,0),(0,3),(2,3),(1,1)):
        rows=sum(sizes)
        offset=sum(sizes[:rank])
        count=sizes[rank]
        base=torch.arange(rows*4,dtype=torch.float32,device=device).view(rows,4)
        # CPU contributes 10 once on rank zero; disjoint expert partials are
        # 2+base and 4+2*base. Exact representable values expose missing/double sums.
        contribution=(12+base if rank==0 else 4+2*base).to(torch.bfloat16)
        expected=(16+3*base).to(torch.bfloat16)[offset:offset+count].cpu()
        value=torch.empty_like(contribution)
        output=torch.empty((count,4),dtype=torch.bfloat16,device=device)
        start=torch.tensor(offset,dtype=torch.int64,device=device)
        size=torch.tensor(count,dtype=torch.int64,device=device)
        scope=dict(torch=torch,memcpy_triton=memcpy_triton,get_dp_local_info=lambda _: (start,size))
        exec(compile(module,'actual DP scatter','exec'),scope)

        def combine():
            dist.all_reduce(value)
            scope['dp_scatter'](output,value,None)

        for _ in range(2):
            value.copy_(contribution)
            combine()
            torch.npu.synchronize()
            if not torch.equal(output.cpu(),expected):raise RuntimeError('Eager HCCL/DP scatter sum differs')
        graph=torch.npu.NPUGraph()
        value.copy_(contribution)
        torch.npu.synchronize()
        dist.barrier()
        with torch.npu.graph(graph):combine()
        for _ in range(2):
            value.copy_(contribution)
            graph.replay()
            torch.npu.synchronize()
            if not torch.equal(output.cpu(),expected):raise RuntimeError('Graph HCCL/DP scatter sum differs')
        records.append(dict(sizes=sizes,local_rows=count,eager=True,graph_replays=2,passed=True))
    result=dict(rank=rank,passed=True,combine='hccl_allreduce_scatter',checks=records)
    (root/f'combine-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('NPU combine probe: '+json.dumps(result),flush=True)
    return result
