"""Real 910B native-byte transfers and fixed-address graph replay before model A/B."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time


def main():
    import torch
    import torch_npu
    from sglang.srt.layers.moe import kt_stream_prefill as stream
    from sglang.srt.layers.moe.kt_decode_residency import NAMES, PreparedBundle, convert_expert
    device=torch.device('npu:0')
    torch.npu.set_device(device)
    cfg=json.loads((Path(os.environ['DSV4_MODEL_PATH'])/'config.json').read_text())
    H,I=cfg['hidden_size'],cfg['moe_intermediate_size']
    E=cfg.get('n_routed_experts',cfg.get('num_experts'))
    layers=cfg['num_hidden_layers']
    if (E,layers)!=(256,43): raise RuntimeError('Unexpected model geometry')
    stream._remember_dims(E,H,I,layers)
    layer=torch.nn.Module()
    for name,shape in (('w13_weight',(2,H,2*I)),('w2_weight',(2,I,H)),
                       ('w13_weight_scale',(2,2*I)),('w2_weight_scale',(2,H))):
        dtype=torch.int8 if name.endswith('weight') else torch.bfloat16
        tensor=torch.empty(shape,dtype=dtype,device=device)
        if dtype==torch.int8: tensor=torch_npu.npu_format_cast(tensor,29)
        setattr(layer,name,torch.nn.Parameter(tensor,requires_grad=False))
    prepared, references, cold_ms={}, {}, {}
    for expert in (0,5):
        started=time.perf_counter()
        tensors=convert_expert(0,expert,layer)
        prepared[expert]=PreparedBundle(tensors)
        prepared[expert].validate(layer)
        cold_ms[str(expert)]=1000*(time.perf_counter()-started)
        references[expert]=tensors
    prepared[0].copy_to(layer,0)
    prepared[0].copy_to(layer,1)
    torch.npu.synchronize()
    pointers={n:getattr(layer,n).data_ptr() for n in NAMES}
    torch.manual_seed(20261126)
    hidden=torch.randn((1,H),dtype=torch.bfloat16,device=device)*0.01
    weights=torch.ones((1,1),device=device,dtype=torch.bfloat16)
    ids=torch.ones((1,1),device=device,dtype=torch.int32)

    def forward(tensors,route,count):
        return stream._streaming_fused_experts(hidden,tensors['w13_weight'],tensors['w13_weight_scale'],
                                              tensors['w2_weight'],tensors['w2_weight_scale'],
                                              weights,route,1,count)

    resident={n:getattr(layer,n) for n in NAMES}
    # Warm both expert-count variants and all routing/GMM workspaces before capture.
    expected={e:forward(t,torch.zeros_like(ids),1).clone() for e,t in references.items()}
    forward(resident,ids,2)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    capture_stream=torch.npu.Stream(device=device)
    with torch.npu.graph(graph,stream=capture_stream):
        output=forward(resident,ids,2)
    errors=[]
    for expert in (5,0,5,0,5):
        prepared[expert].copy_to(layer,1)
        torch.npu.synchronize()
        # Compare logical contents after copying full padded physical NZ slices.
        for name in NAMES:
            torch.testing.assert_close(getattr(layer,name).cpu()[1:2],references[expert][name].cpu(),rtol=0,atol=0)
        graph.replay()
        torch.npu.synchronize()
        if not torch.isfinite(output).all(): raise RuntimeError('Non-finite replay output')
        torch.testing.assert_close(output,expected[expert],rtol=0.01,atol=0.01)
        errors.append(float((output-expected[expert]).abs().max().item()))
        if pointers!={n:getattr(layer,n).data_ptr() for n in NAMES}:
            raise RuntimeError('Captured addresses changed during expert replacement')
    copy_ms=[]
    for _ in range(10):
        started=time.perf_counter()
        prepared[5].copy_to(layer,1)
        torch.npu.synchronize()
        copy_ms.append(1000*(time.perf_counter()-started))
    result=dict(passed=True,geometry=dict(hidden=H,intermediate=I,experts=E,layers=layers),
                expert_bytes=prepared[5].bytes,cold_convert_prepare_ms=cold_ms,hot_copy_fence_ms=copy_ms,
                actual_graph_replays=5,max_abs_errors=errors,reference_tolerance=dict(rtol=.01,atol=.01),
                copied_weights_scales_exact=True,stable_addresses=True,
                scope='Layer0 two-slot real-MXFP4 conversion; graph replay after external step-boundary updates')
    Path('/experiment/residency-hardware-gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
