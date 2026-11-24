"""Detached trial launcher; observe the existing cooling service without controlling it."""
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from thermal_policy import POLICY, SustainedTemperature
from transport_policy import TransportGrace

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / '2026-11-10-taishan-tp2'
SPEED = HERE.parent / '2026-11-11-taishan-tp-speed'
COOLING_STATUS = Path.home() / 'AppData/Local/taishan/npu-cooling/status.json'
PHASES = ('environment', 'hccl', 'tp2k64eager', 'tp1k32graph', 'tp2k32graph', 'tp2k64graph')
BUNDLE = ('run_guarded.py', 'prepare_candidate.py', 'base_prepare.py', 'launch.sh', 'remote_controller.py',
          'tp_runtime.py', 'base_runtime.py', 'thermal_policy.py', 'test_thermal_policy.py',
          'transport_policy.py', 'resume_checkpoint.py', 'test_transport_resume.py',
          'workload.py', 'hccl_probe.py', 'requests.json', 'experiment-plan.md',
          'environment_probe.py', 'policy-alignment.json')
SOURCES = {'base_prepare.py': SUPPORT / 'prepare_candidate.py',
           'base_runtime.py': SUPPORT / 'remote_controller.py',
           'workload.py': SPEED / 'workload.py', 'hccl_probe.py': SUPPORT / 'hccl_probe.py',
           'requests.json': SPEED / 'requests.json'}
TERMINAL = {'complete', 'failed', 'aborted_thermal_or_telemetry'}

spec = importlib.util.spec_from_file_location('tp_guard_helpers', SUPPORT / 'run_guarded.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def cooling_sample():
    value = json.loads(COOLING_STATUS.read_text(encoding='utf-8'))
    now = dt.datetime.now(dt.timezone.utc)
    sampled = dt.datetime.fromisoformat(value['sample']['timestamp_utc'])
    age = (now - sampled).total_seconds()
    if (value['status'] != 'running' or not value['sample_valid'] or not 0 <= age < 30
            or value.get('controls_workloads') is not False or value.get('full_at_c') != 70):
        raise RuntimeError(f'Existing cooling service invalid/stale: status={value.get("status")}, age={age:.1f}s')
    return value, dict(at=helpers.now(), cards=value['sample']['npus'],
                       fans=value['sample']['fans_rpm'], sampled_at=sampled.timestamp(),
                       cooling_pid=value['pid'], fan_target=value.get('target'),
                       thermal_policy=POLICY)


def emergency_stop(reason):
    helpers.atomic(helpers.RUN / 'alarm.json', dict(at=helpers.now(), reason=reason))
    try:
        helpers.ssh('taishan', f'touch {helpers.REMOTE}/abort.json')
    finally:
        quoted = ' '.join(helpers.NAMES)
        command = ('for name in ' + quoted + '; do '
                   'if sudo -n docker inspect "$name" >/dev/null 2>&1; then '
                   'sudo -n docker stop -t 3 "$name"; fi; done')
        helpers.ssh('taishan', command, timeout=120)


def heartbeat_status(timeout):
    code = ("import json;from pathlib import Path;p=Path(" + repr(helpers.REMOTE) + ");"
            "(p/'heartbeat').touch();q=p/'status.json';"
            "print(json.dumps({'heartbeat_written':True,'remote_status':"
            "json.loads(q.read_text()) if q.exists() else {'phase':'starting'}}))")
    value = json.loads(helpers.ssh('taishan', 'python3 -c ' + shlex.quote(code), timeout=timeout))
    if value.get('heartbeat_written') is not True:
        raise RuntimeError('Remote heartbeat was not acknowledged')
    return value['remote_status']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True,
                        help='Run label; adds YYYYMMDD-HHMMSS in Asia/Shanghai when absent')
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', args.run_id):
        parser.error('run-id must contain lowercase letters, digits or hyphens')
    args.run_id = helpers.make_run_id(args.run_id)
    if args.resume_from and not re.fullmatch(r'/data/deploy/dsv4-tp2-[a-z0-9-]+', args.resume_from):
        parser.error('resume-from must be an existing trial directory under /data/deploy')
    helpers.RUN = HERE / 'runs' / args.run_id
    helpers.REMOTE = '/data/deploy/dsv4-tp2-' + args.run_id
    helpers.NAMES = [f'dsv4-tp2-{args.run_id}-{phase}' for phase in PHASES]
    helpers.RUN.mkdir(parents=True, exist_ok=False)
    launched, misses, count = False, 0, 0
    policy = SustainedTemperature()
    helpers.status('preflight', thermal_policy=POLICY, cooling_status=str(COOLING_STATUS))
    try:
        before, first = cooling_sample()
        reason = policy.check(first['cards'], first['sampled_at'])
        if reason:
            raise RuntimeError(reason)
        helpers.atomic(helpers.RUN / 'cooling-service-before.json', before)
        helpers.atomic(helpers.RUN / 'thermal-policy.json', POLICY)
        if helpers.ssh('taishan', f'test -e {helpers.REMOTE} && echo exists || echo absent').strip() != 'absent':
            raise RuntimeError('Remote trial directory already exists')
        helpers.ssh('taishan', f'mkdir {helpers.REMOTE}')
        for name in BUNDLE:
            content = Path(SOURCES.get(name, HERE / name)).read_bytes().replace(b'\r\n', b'\n')
            staged = helpers.RUN / name
            staged.write_bytes(content)
            result = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(staged),
                                     'taishan:' + helpers.REMOTE + '/' + name],
                                    capture_output=True, timeout=40)
            if result.returncode:
                raise RuntimeError('Upload failed: ' + result.stderr.decode(errors='replace'))
        tests = helpers.ssh('taishan', f'cd {helpers.REMOTE} && python3 -m unittest discover -p "test_*.py" -v 2>&1', timeout=30)
        (helpers.RUN / 'remote-tests.stdout.log').write_text(tests, encoding='utf-8')
        helpers.ssh('taishan', f'touch {helpers.REMOTE}/heartbeat')
        helpers.status('launching_remote')
        launched = True
        resume_option = ' --resume-from ' + shlex.quote(args.resume_from) if args.resume_from else ''
        remote_pid = helpers.ssh('taishan',
            f'nohup python3 -u {helpers.REMOTE}/remote_controller.py{resume_option} > {helpers.REMOTE}/controller.stdout.log '
            f'2> {helpers.REMOTE}/controller.stderr.log < /dev/null & echo $!').strip()
        helpers.atomic(helpers.RUN / 'launch.json', dict(at=helpers.now(), remote_pid=remote_pid,
                                                       remote=helpers.REMOTE, thermal_policy=POLICY))
        deadline = time.monotonic() + 3.5 * 3600 + 180
        transport = TransportGrace(time.monotonic())
        remote = {'phase': 'starting'}
        while time.monotonic() < deadline:
            try:
                service, reading = cooling_sample()
                count += 1
                reason = policy.check(reading['cards'], reading['sampled_at'])
                with (helpers.RUN / 'thermal.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(reading) + '\n')
                with (helpers.RUN / 'cooling-service-samples.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(dict(at=helpers.now(), service=service)) + '\n')
                if reason:
                    emergency_stop(reason)
                    helpers.status('aborted_thermal_or_telemetry', reason=reason, samples=count)
                    try:
                        helpers.collect()
                    except Exception as exc:
                        print(f'Evidence collection: {exc}', flush=True)
                    return 1
                misses = 0
            except Exception as exc:
                policy.invalidate()
                misses += 1
                with (helpers.RUN / 'thermal.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(dict(at=helpers.now(), error=str(exc), misses=misses)) + '\n')
                if misses >= 2:
                    raise RuntimeError(f'Two actual cooling telemetry reads failed: {exc}') from exc
                time.sleep(5)
                continue
            # Connection failures have their own bounded retry budget. They do
            # not invalidate a fresh cooling sample or manufacture a thermal alarm.
            try:
                remote = heartbeat_status(timeout=transport.timeout(time.monotonic()))
                transport.success(time.monotonic())
            except Exception as exc:
                at = time.monotonic()
                expired = transport.failure(at)
                fields = dict(at=helpers.now(), error=str(exc), failures=transport.failures,
                              seconds_since_heartbeat=at-transport.last_success)
                with (helpers.RUN / 'transport.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(fields) + '\n')
                helpers.status('reconnecting', samples=count, transport=fields, remote_status=remote)
                if expired:
                    raise RuntimeError('No acknowledged SSH heartbeat for 75 seconds; remote 90-second watchdog remains active') from exc
                time.sleep(2)
                continue
            helpers.atomic(helpers.RUN / 'remote-status.json', remote)
            helpers.status('running', samples=count, remote_status=remote, last=reading)
            if remote['phase'] in TERMINAL:
                retrieval_error = None
                try:
                    helpers.collect()
                except Exception as exc:
                    retrieval_error = str(exc)
                helpers.status(remote['phase'], samples=count, remote_status=remote,
                               results_downloaded=retrieval_error is None, retrieval_error=retrieval_error)
                return 0 if remote['phase'] == 'complete' else 1
            time.sleep(5)
        raise TimeoutError('Detached experiment exceeded time budget')
    except Exception as exc:
        if launched:
            try:
                emergency_stop(str(exc))
            except Exception as stop_error:
                print(f'Emergency stop could not reach host: {stop_error}', flush=True)
        helpers.status('failed', error=str(exc), samples=count)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
