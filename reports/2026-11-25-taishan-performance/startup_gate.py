"""Require real EP ownership, attention topology, request slots and graph capture."""
import re
from graph_gate import check_decode_graph


def check(log, dpa, graph, wide):
    required = ['tp_size=2','ep_size=2','kt_num_gpu_experts=64','max_total_tokens=16384',
                'kt_cpuinfer=96','enable_metrics=True','stream_interval=1']
    required += [f'chunked_prefill_size={8192 if wide else 512}',
                 f'swa_full_tokens_ratio={0.5 if wide else 0.1}']
    required += ['dp_size=2','enable_dp_attention=True','max_running_requests=2'] if dpa else [
                'dp_size=1','enable_dp_attention=False','max_running_requests=1']
    for marker in required:
        if marker not in log:
            raise RuntimeError('Missing resolved performance option: '+marker)
    records = re.findall(r'\[KT EP2 AUDIT\] layer=(\d+) rank=(\d+) start=(\d+) stop=(\d+) '
                         r'local=32 global=64 cpu_experts=192 moe_tp=1 intermediate=2048 cpu_owner=(True|False)',log)
    expected = {(l,r) for l in range(43) for r in (0,1)}
    if len(records)!=86 or {(int(x[0]),int(x[1])) for x in records}!=expected:
        raise RuntimeError('Incomplete expert ownership audit')
    for layer,rank,start,stop,owner in records:
        if (int(start),int(stop),owner=='True')!=(int(rank)*32,(int(rank)+1)*32,rank=='0'):
            raise RuntimeError('Expert ownership differs from 32+32 / CPU rank zero')
    pools = re.findall(r'Initialize DeepSeekV4TokenToKVPool with max_num_reqs=(\d+)\b',log)
    if pools!=['1','1']:
        raise RuntimeError('Expected one request slot on each rank')
    if dpa:
        attention = re.findall(r'\[KT ATTN TP1 AUDIT\] layer=(\d+) rank=(\d+) attn_tp=1 attn_dp=2 local_heads=(\d+) total_heads=(\d+)',log)
        if len(attention)!=86 or {(int(x[0]),int(x[1])) for x in attention}!=expected:
            raise RuntimeError('Incomplete full-attention audit')
        if any(x[2]!=x[3] for x in attention) or log.count('combine=hccl_allreduce_scatter')!=86:
            raise RuntimeError('Attention topology or HCCL combine mismatch')
    elif '[KT ATTN TP1 AUDIT]' in log:
        raise RuntimeError('Control accidentally enables attention DP')
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception',log):
        raise RuntimeError('Runtime exception before startup acceptance')
    if graph:
        check_decode_graph(log)
    elif 'disable_cuda_graph=True' not in log:
        raise RuntimeError('Expected eager mode')
    return dict(expert_audit_records=86, attention_dp=dpa, graph=graph,
                actual_prefill_chunk=8192 if wide else 512,
                swa_ratio=0.5 if wide else 0.1, quality_accepted=False)
