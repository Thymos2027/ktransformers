"""Exercise the actual DSV4 request-pool formula without loading the model."""
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace


def slots(source, global_requests, dp_size):
    assignments=[node for node in ast.walk(ast.parse(source))
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Attribute)
            and target.attr=='requested_max_running_requests_per_worker' for target in node.targets)]
    if len(assignments)!=1:
        raise RuntimeError('DSV4 request pool source anchor changed')
    kvc=SimpleNamespace(server_args=SimpleNamespace(max_running_requests=global_requests),
                        ps=SimpleNamespace(attn_dp_size=dp_size))
    return eval(compile(ast.Expression(assignments[0].value),'DSV4 request pool','eval'),{'kvc':kvc})


def main():
    root=Path('/experiment')
    source=(root/'sglang/python/sglang/srt/model_executor/pool_configurator.py').read_text()
    launch=(root/'launch.sh').read_text()
    requests=int(re.search(r'--max-running-requests (\d+)',launch)[1])
    dp=int(re.search(r'--data-parallel-size (\d+)',launch)[1])
    if slots(source,1,2)!=0 or slots(source,requests,dp)!=1 or (requests,dp)!=(2,2):
        raise RuntimeError('Request budget must resolve to one slot per attention rank')
    print('Attention request budget: '+json.dumps(dict(passed=True,global_requests=requests,
        attention_dp=dp,per_worker=1,historical_zero_slot_reproduced=True)),flush=True)


if __name__=='__main__':main()
