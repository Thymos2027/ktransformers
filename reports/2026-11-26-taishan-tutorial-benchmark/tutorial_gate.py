"""Audit the pinned fork's resolved model defaults, rather than generic defaults."""
import json
import re

import prefill_gate


def startup(log,phase,settings=None):
    if phase not in ('table12','table19','table39','hardwarelong'):
        raise ValueError('Unknown tutorial phase')
    experts=28 if phase=='table39' else 32
    fraction=0.775 if phase=='table39' else 0.81
    # arg_groups/overrides.py sets 0.1 for DeepseekV4 when no flag was supplied.
    swa=0.35 if phase=='table19' else 0.1
    cap='None'
    if settings:
        experts,fraction,swa,cap=(settings[k] for k in ('experts','mem_fraction','swa_ratio','max_total_tokens'))
    for marker in ('tp_size=1','ep_size=1','dp_size=1','kt_cpuinfer=32','kt_threadpool_count=1',
            'context_length=65536','chunked_prefill_size=32768','max_running_requests=1',
            f'max_total_tokens={cap}',f'kt_num_gpu_experts={experts}',f'mem_fraction_static={fraction}',
            f'swa_full_tokens_ratio={swa}','disable_cuda_graph=False'):
        if not re.search(r'(?<!\w)'+re.escape(marker)+r'(?=[,\s)]|$)',log):
            raise RuntimeError('Missing tutorial option/graph: '+marker)
    for marker in ('Capture target decode NPU graph end',
                   '[KT_STREAM][depool] reserved ND streaming slot'):
        if marker not in log:
            raise RuntimeError('Missing tutorial option/graph: '+marker)
    match=re.findall(r'\[TUTORIAL CONFIG\] (\{[^\n]+\})',log)
    if len(match)!=1:
        raise RuntimeError('Missing resolved environment')
    env=json.loads(match[0])
    expected={key:'1' for key in ('KT_PREFILL_STREAM','KT_MXFP4_DEPOOL','KT_MXFP4_GGUF_DEDUP',
                                  'KT_DYNAMIC_RESIDENT','KT_SIDE_STREAM','SGLANG_SET_CPU_AFFINITY')}
    expected.update(KT_PREFILL_STREAM_THRESHOLD='512',ASCEND_RT_VISIBLE_DEVICES='0')
    for key,value in expected.items():
        if env.get(key)!=value:
            raise RuntimeError('Tutorial environment mismatch: '+key)
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception|'+prefill_gate.FALLBACK,log):
        raise RuntimeError('Startup failed/fell back')
    return dict(passed=True,env=env,experts=experts,mem_fraction=fraction,swa_ratio=swa,
                swa_source='explicit hardware profile' if settings else (
                    'explicit tutorial override' if phase=='table19' else 'pinned DeepSeek-V4 model default'))


def capacity(log,cases,output_tokens=1000,margin=1024,server_info=None):
    records=list(re.finditer(r'DSV4 pool sizes: full=(\d+), swa=(\d+),',log))
    initialized=list(re.finditer(r'Initialize DeepSeekV4TokenToKVPool with [^\n]*?swa_size=(\d+)',log))
    if initialized:
        if len(initialized)!=1:
            raise RuntimeError('Ambiguous actual DSV4 initialization')
        previous=[m for m in records if m.start()<initialized[0].start()]
        if not previous:
            raise RuntimeError('Missing final DSV4 pool sizes before initialization')
        full,swa=map(int,previous[-1].groups())
        if swa!=int(initialized[0].group(1)):
            raise RuntimeError('Final DSV4 sizes disagree with initialized SWA pool')
    elif len(records)==1 and server_info is None:
        full,swa=map(int,records[0].groups())
    else:
        raise RuntimeError('Missing or ambiguous actual DSV4 pool sizes')
    if server_info is not None:
        if not initialized or server_info.get('max_total_num_tokens')!=full:
            raise RuntimeError('Initialized full capacity disagrees with server_info')
    largest=max(c['prompt_tokens'] for c in cases)
    paged=(largest+127)//128*128
    required_swa=paged+128+margin
    required_full=paged+output_tokens+128+margin
    if swa<required_swa or full<required_full:
        raise RuntimeError(f'Actual pool insufficient: full={full}/{required_full}, swa={swa}/{required_swa}')
    return dict(passed=True,actual_full=full,actual_swa=swa,largest_prompt=largest,
                required_full=required_full,required_swa=required_swa,margin_tokens=margin,
                estimated_and_final_size_records=len(records),initialized_pool_checked=bool(initialized),
                server_info_checked=server_info is not None,
                policy='Single-chunk prefill with page-alignment and extra capacity margin')
