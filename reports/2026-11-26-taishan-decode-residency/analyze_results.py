"""Offline update-cost, route and output-history review of the saved requests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def analyze(run):
    root=run/'remote'
    rows=[json.loads(s) for s in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=[]
    for phase in ('a0','b0','b1','a1'):
        for case in ('prefill_1024','prefill_4096'):
            selected=[r for r in rows if r['phase']==phase and r['case']==case and r['kind']=='formal']
            detail=[]
            for r in selected:
                updates=r['streaming_audit']['updates']
                if not updates: continue
                last=updates[-1]
                detail.append(dict(update_seconds=sum(u['update_ms'] for u in updates)/1000,
                    drain_seconds=sum(u['drain_ms'] for u in updates)/1000,
                    promotions=last['promotions_total'],cache_hits=last['cache_hits'],cache_misses=last['cache_misses'],
                    prepared_cache_hit_rate=last['cache_hits']/(last['cache_hits']+last['cache_misses']),
                    cache_mib=last['cache_bytes']/1024**2,transfer_gib=last['transfer_bytes']/1024**3,
                    sampled_decode_steps=last['completed_decodes'],
                    route_coverage=last['resident_hits']/last['route_choices'],
                    first_window_coverage=updates[0]['resident_hits']/updates[0]['route_choices']))
            groups.append(dict(phase=phase,case=case,formal_requests=len(selected),
                distinct_output_histories=len({tuple(r['output_ids']) for r in selected}),
                median_update_stats={k:statistics.median(d[k] for d in detail) for k in detail[0]} if detail else {},
                median_window_tpot_ms={k:statistics.median(r['window_tpot_ms'][k] for r in selected
                                            if r['window_tpot_ms'][k] is not None)
                                      for k in selected[0]['window_tpot_ms']}))
    functions=[json.loads(s) for s in (root/'functional-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    external=Counter()
    for path in root.glob('*-resources.json'):
        for name in json.loads(path.read_text())['running_containers']:
            if not name.startswith('dsv4-tp2-'+run.name+'-'): external[name]+=1
    result=dict(groups=groups,performance_requests=len(rows),formal_requests=sum(r['kind']=='formal' for r in rows),
                output_ids=sum(len(r['output_ids']) for r in rows),notifications=sum(len(r['token_events']) for r in rows),
                notification_jumps=sum(r['event_token_jumps'] for r in rows),
                functional_observations=len(functions),functional_answer_matches=sum(r['answer_matches'] for r in functions),
                external_container_snapshot_occurrences=dict(external),
                caveats=['Drain intervals include previously queued inference and are not independently added overhead.',
                         'Candidate route coverage has no matched baseline-route measurement.',
                         'Free-running output histories may differ across arms.'])
    (run/'update-cost-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    print(json.dumps(analyze(parser.parse_args().run),ensure_ascii=False,indent=2))
