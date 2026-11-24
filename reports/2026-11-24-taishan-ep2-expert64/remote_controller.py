"""Bounded eager/graph trial of disjoint 32+32 expert residency."""
import json
import re
import statistics
from graph_gate import check_decode_graph
import ep_checkpoint

import tp2_controller as trial

core = trial.core
trial.PHASES = (('ep2eager', 2, 64, False), ('ep2graph', 2, 64, True))
core.NAMES = {phase: f'dsv4-tp2-{core.RUN_ID}-{phase}'
              for phase in ('environment', 'hccl', 'ep2eager', 'ep2graph')}


def check_startup(log, tp, experts, graph):
    for marker in ('tp_size=2', 'ep_size=2', 'kt_num_gpu_experts=64', 'max_total_tokens=16384'):
        if marker not in log:
            raise RuntimeError('Missing EP2 configuration marker: '+marker)
    records = re.findall(r'\[KT EP2 AUDIT\] layer=(\d+) rank=(\d+) start=(\d+) stop=(\d+) '
                         r'local=32 global=64 cpu_experts=192 moe_tp=1 intermediate=2048 cpu_owner=(True|False)', log)
    if len(records) != 86 or {(int(x[0]), int(x[1])) for x in records} != {(l,r) for l in range(43) for r in (0,1)}:
        raise RuntimeError('Incomplete EP2 residency audit')
    for layer, rank, start, stop, owner in records:
        if int(start) != int(rank)*32 or int(stop) != int(start)+32 or (owner=='True') != (rank=='0'):
            raise RuntimeError('EP2 ownership mismatch')
    if re.search(r'Traceback|OutOfMemoryError|Scheduler hit an exception',log):
        raise RuntimeError('EP2 startup exception')
    if graph:
        check_decode_graph(log)
    elif 'disable_cuda_graph=True' not in log:
        raise RuntimeError('Eager mode missing')
    return dict(layers=43, ranks=2, expert_parallel=2, moe_tensor_parallel=1,
                rank_experts={'0':list(range(32)), '1':list(range(32,64))},
                cpu_experts=list(range(64,256)), audit_records=86, graph=graph)


def summarize():
    rows=[json.loads(x) for x in (core.ROOT/'requests.jsonl').read_text().splitlines()]
    if len(rows)!=8 or len({(x['case'],x['iteration']) for x in rows})!=8:
        raise RuntimeError('Incomplete EP2 graph performance matrix')
    output=dict(formal_requests=6,warmup_requests=2,configuration='TP2 attention, EP2 MoE, 32 full experts per rank',
                correctness_accepted=False,cases=[],checkpoint_reuse=json.loads((core.ROOT/'checkpoint-reuse.json').read_text())
                if (core.ROOT/'checkpoint-reuse.json').exists() else None)
    for case in ('input_128','input_768'):
        group=[x for x in rows if x['case']==case and x['kind']=='formal']
        if len(group)!=3:raise RuntimeError('Missing EP2 repeats')
        for row in group:
            trial.workload.check_transport(row)
            if row['usage']['completion_tokens']!=512:raise RuntimeError('Incomplete output')
        output['cases'].append(dict(case=case,metrics={key:dict(median=statistics.median(x[key] for x in group),
            values=[x[key] for x in group]) for key in ('ttft_seconds','decode_tokens_per_second','total_seconds')}))
    core.atomic(core.ROOT/'summary.json',output)


trial.check_startup=check_startup
trial.summarize=summarize
trial.resume_checkpoint.validate=ep_checkpoint.validate
if __name__=='__main__':
    raise SystemExit(trial.main())
