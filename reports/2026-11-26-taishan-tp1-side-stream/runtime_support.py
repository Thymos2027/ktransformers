"""NPU2-only containers; preserve concurrent NPU5 workloads and original service."""
import hashlib
import json
from pathlib import Path
import socket
import time

import tp_runtime as core
import prefill_gate

IMAGE = 'sha256:f03917b88b3696659b70e2e11c3190978af4732d87be05fc624245b9bb6f6b3e'
core.IMAGE = IMAGE
core.NAMES = {phase: f'dsv4-tp2-{core.RUN_ID}-{phase}'
              for phase in ('environment', *prefill_gate.PHASES)}
core.DEADLINE = time.monotonic() + 3*3600
_stop_all = core.stop_all


def scoped_stop(include_original=False):
    _stop_all(include_original=False)


core.stop_all = scoped_stop


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', 18082)) == 0


def snapshot(phase, stage):
    core.atomic(core.ROOT/f'{phase}-{stage}-resources.json', dict(
        at=core.now(), raw=core.run(['bash', '-c',
            'npu-smi info; free -b; numastat -m; ps -eo pid,pcpu,pmem,args --sort=-pcpu | head -12']),
        running_containers=core.docker('ps', '--format', '{{.Names}}').splitlines(),
        loadavg=Path('/proc/loadavg').read_text(), meminfo=Path('/proc/meminfo').read_text(),
        interpretation='Shared host; NPU5 workloads retained; npu-smi snapshots are not peaks'))


def preflight():
    raw = core.run(['npu-smi', 'info'])
    initial = dict(at=core.now(), original_running=core.docker(
        'inspect', '-f', '{{.State.Running}}', core.ORIGINAL), npu_smi=raw,
        running_containers=core.docker('ps', '--format', '{{.Names}}').splitlines(),
        shared_host_authorized=True, physical_npu=2)
    core.atomic(core.ROOT/'initial-state.json', initial)
    if 'No running processes found in NPU 2' not in raw or port_open():
        raise RuntimeError('NPU2 must be idle and port18082 free')
    if initial['original_running'] != 'false':
        raise RuntimeError('Original DS-V4 service is running; refuse to stop it')
    if set(core.NAMES.values()).intersection(core.docker('ps', '-a', '--format', '{{.Names}}').splitlines()):
        raise RuntimeError('Experiment container names already exist')
    if core.docker('image', 'inspect', '-f', '{{.Id}}', IMAGE) != IMAGE:
        raise RuntimeError('Pinned image missing')
    mem = {line.split(':')[0]: int(line.split()[1])
           for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable'] < 220*1024**2:
        raise RuntimeError('Less than 220 GiB available RAM')
    reference = json.loads((core.ROOT/'weight-reference.json').read_text())
    model = Path('/data/llm/DeepSeek-V4-Flash-W8A8')
    metadata = [model/'config.json', model/'model.safetensors.index.json']
    shards = [model/name for name in set(json.loads(metadata[1].read_text())['weight_map'].values())]
    gguf = sorted(Path('/data/deploy/dsv4-20261104/gguf').glob('*.gguf'))
    if not shards or len(gguf) != 43 or any(not p.is_file() for p in shards+gguf):
        raise RuntimeError('Model files incomplete')
    identity = dict(metadata_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata},
                    files={str(p): p.stat().st_size for p in shards+gguf})
    if any(identity[key] != reference[key] for key in identity):
        raise RuntimeError('Weight metadata/sizes differ from verified prior deployment')
    core.atomic(core.ROOT/'weight-preflight.json', dict(identity,
        verification='Reuse prior complete weight integrity audit; current metadata and every file size match'))
    if hashlib.sha256((core.ROOT/'requests.json').read_bytes()).hexdigest() != (
            'da4c23cec376ecb3b5db41ac0550bd6f0c3bdfa167c0c721f7657f2812b02aa1'):
        raise RuntimeError('Historical workload identity changed')
    core.read_cards()
    if time.time() - (core.ROOT/'heartbeat').stat().st_mtime > 45:
        raise RuntimeError('Cooling observer heartbeat stale')
    snapshot('initial', 'before')


def create(phase):
    core.safe()
    args = ['run', '-d', '--name', core.NAMES[phase], '--restart', 'no', '--network', 'host',
            '--ipc', 'host', '--privileged', '--security-opt', 'seccomp=unconfined',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '--volumes-from', core.ORIGINAL+':ro', '-v', str(core.ROOT)+':/experiment:rw',
            '-w', '/experiment']
    for device in ('davinci2', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
        args += ['--device', '/dev/'+device]
    args += ['--entrypoint', '/bin/bash', IMAGE, '/experiment/launch.sh', phase]
    result = core.docker(*args, timeout=60)
    core.safe()
    return result


core.create_container = create


def cleanup():
    core.DEADLINE = max(core.DEADLINE, time.monotonic()+600)
    core.stop_all()
    states = {}
    for phase, name in core.NAMES.items():
        raw = core.docker('inspect', '-f', '{{json .State}}', name, check=False)
        if raw:
            states[name] = json.loads(raw)
            core.capture_logs(phase)
    end = time.monotonic()+45
    while True:
        raw = core.run(['npu-smi', 'info'])
        idle = 'No running processes found in NPU 2' in raw
        if idle or time.monotonic() >= end:
            break
        time.sleep(3)
    original = core.docker('inspect', '-f', '{{.State.Running}}', core.ORIGINAL)
    passed = (all(not s['Running'] and s['Pid'] == 0 for s in states.values())
              and idle and not port_open() and original == 'false')
    core.atomic(core.ROOT/'cleanup.json', dict(at=core.now(), passed=passed, states=states,
        npu_smi=raw, original_running=original, port_open=port_open(),
        other_workloads_never_stopped=True,
        final_running_containers=core.docker('ps', '--format', '{{.Names}}').splitlines()))
    return passed
