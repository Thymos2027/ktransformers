"""Sequential LAN streaming benchmark; no server configuration changes."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import requests

ROOT = Path(__file__).resolve().parent
BASE = 'http://192.168.137.11:18080'
SESSION = requests.Session()
SESSION.trust_env = False
RESULTS = []


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def write(name, data):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def status(phase, **extra):
    data = dict(phase=phase, updated_at=now(), pid=os.getpid(), completed_trials=len(RESULTS), **extra)
    write('status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def trial(name, prompt, tokens, repetition, warmup=False):
    status('running', trial=name, repetition=repetition, requested_output_tokens=tokens)
    started = now()
    start = time.perf_counter()
    first = last = None
    first_count = last_count = 0
    done = False
    final = None
    with SESSION.post(BASE + '/generate', json={
        'text': prompt,
        'sampling_params': {'temperature': 0, 'max_new_tokens': tokens, 'ignore_eos': True},
        'stream': True,
    }, stream=True, timeout=(15, 900)) as response:
        response.raise_for_status()
        response.encoding = 'utf-8'
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                done = True
                break
            event = json.loads(payload)
            if 'error' in event:
                raise RuntimeError(str(event['error']))
            final = event
            count = event['meta_info']['completion_tokens']
            if count > last_count:
                elapsed = time.perf_counter() - start
                if first is None:
                    first, first_count = elapsed, count
                last, last_count = elapsed, count
    total = time.perf_counter() - start
    if not done or final is None or first is None or last_count != tokens:
        raise RuntimeError(f'Incomplete stream: done={done}, tokens={last_count}/{tokens}')
    meta = final['meta_info']
    result = dict(
        case=name, repetition=repetition, warmup=warmup, started_at=started,
        completed_at=now(), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_tokens=meta['prompt_tokens'], completion_tokens=last_count,
        first_event_tokens=first_count, ttft_seconds=first, total_seconds=total,
        decode_tokens_per_second=(last_count-first_count)/(last-first) if last > first else None,
        output_tokens_per_total_second=last_count/total,
        last_token_seconds=last, server_meta=meta, output=final.get('text', ''),
    )
    write(f'{name}-{repetition}.json', result)
    if not warmup:
        RESULTS.append(result)
        write('results.json', RESULTS)
    print(json.dumps({k: v for k, v in result.items() if k not in ('output', 'server_meta')}, ensure_ascii=False), flush=True)


def main():
    status('starting')
    SESSION.get(BASE + '/health', timeout=15).raise_for_status()
    response = SESSION.get(BASE + '/get_server_info', timeout=30)
    response.raise_for_status()
    info = response.json()
    keys = ['model_path', 'device', 'quantization', 'context_length', 'max_running_requests',
            'disable_radix_cache', 'stream_interval', 'kt_num_gpu_experts',
            'kt_threadpool_count', 'kt_cpuinfer', 'mem_fraction_static', 'chunked_prefill_size']
    write('environment.json', dict(captured_at=now(), endpoint=BASE,
        client='Windows LAN; requests trust_env=False',
        server={key: info.get(key) for key in keys},
        background_activity='Full GGUF audit passed at 11:15; ModelScope upload active at initial check. No background services paused.'))
    short = 'Explain how a CPU and an accelerator cooperate to run a large language model. Give a detailed technical explanation with examples.\n'
    long = ('Reference notes:\n' +
            'The server uses CPU expert offload and accelerator attention. Memory bandwidth, thread scheduling, and data movement affect latency.\n' * 80 +
            '\nUsing these notes, explain the performance tradeoffs in detail.\n')
    write('prompts.json', {'short': short, 'long': long})
    trial('warmup', short, 32, 1, warmup=True)
    for repetition in range(1, 4):
        trial('short-256', short, 256, repetition)
        trial('long-256', long, 256, repetition)
    trial('short-512', short, 512, 1)
    summaries = []
    for name in ('short-256', 'long-256', 'short-512'):
        rows = [r for r in RESULTS if r['case'] == name]
        summaries.append(dict(case=name, repetitions=len(rows),
            prompt_tokens=sorted({r['prompt_tokens'] for r in rows}),
            completion_tokens=rows[0]['completion_tokens'],
            **{metric: {'median': statistics.median(r[metric] for r in rows),
                        'min': min(r[metric] for r in rows), 'max': max(r[metric] for r in rows)}
               for metric in ('ttft_seconds', 'decode_tokens_per_second', 'total_seconds', 'output_tokens_per_total_second')}))
    write('summary.json', summaries)
    lines = ['# 单并发流式速度测试', '',
        f'完成时间：{now()}。接口：`{BASE}/generate`。预热 32 token 后测量，温度 0，`ignore_eos=True`。', '',
        '| 输入 / 输出 token | 次数 | 首 token 延迟中位数（秒） | 生成速度中位数（token/s） | 总耗时中位数（秒） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in summaries:
        lines.append(f"| {row['prompt_tokens']} / {row['completion_tokens']} | {row['repetitions']} | {row['ttft_seconds']['median']:.3f} | {row['decode_tokens_per_second']['median']:.3f} | {row['total_seconds']['median']:.3f} |")
    lines += ['', '生成速度 =（最终输出 token 数 − 首个流事件 token 数）/（最后新增 token 到达时间 − 首个 token 到达时间）。首 token 延迟和总耗时包含局域网及客户端开销。输入 token 数取服务端返回值；未把流事件数量当作 token 数。', '',
        '当前仅测单并发、两种输入长度和固定长度输出；强制忽略 EOS，用于测吞吐，不代表自然回答长度或质量。全量审计已结束，但后台 ModelScope 上传仍可能共享磁盘、内存和网络资源。没有暂停服务或调整运行配置。', '',
        '逐次结果及原始末次事件信息见各 JSON，`results.json` 不含预热，`summary.json` 保存统计。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    status('completed', summary_file=str(ROOT/'summary.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise SystemExit(1)
