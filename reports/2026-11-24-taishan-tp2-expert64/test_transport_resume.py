"""Transport recovery and checkpoint rejection tests using the actual completed baseline."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import types
import unittest

import resume_checkpoint as checkpoint
from transport_policy import TransportGrace

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'runs/expert64-20261124-cooling90/remote' if os.name == 'nt' else
          Path('/data/deploy/dsv4-tp2-expert64-20261124-cooling90'))
METADATA = ('benchmark-config.json', 'candidate.json', 'source.json', 'weight-preflight.json',
            'environment-preflight.json', 'requests.json', 'launch.sh', 'workload.py')


def function_from_controller(name, namespace):
    tree = ast.parse((HERE / 'remote_controller.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'remote_controller.py', 'exec'), namespace)
    return namespace[name]


def startup_check():
    path = HERE / 'workload.py'
    if not path.exists():
        path = HERE.parent / '2026-11-11-taishan-tp-speed/workload.py'
    spec = importlib.util.spec_from_file_location('_checkpoint_workload', path)
    workload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workload)
    return function_from_controller('check_startup', dict(re=re, workload=workload))


class TransportTests(unittest.TestCase):
    def test_two_twenty_second_timeouts_allow_recovery(self):
        policy = TransportGrace(0)
        self.assertFalse(policy.failure(20))
        self.assertFalse(policy.failure(42))
        policy.success(44)
        self.assertEqual(policy.failures, 0)
        self.assertFalse(policy.failure(64))

    def test_seventy_five_second_outage_expires(self):
        policy = TransportGrace(0)
        self.assertFalse(policy.failure(74.999))
        self.assertTrue(policy.failure(75))

    def test_success_renews_budget(self):
        policy = TransportGrace(0)
        policy.success(70)
        self.assertFalse(policy.failure(100))
        self.assertTrue(policy.failure(145))

    def test_call_timeout_does_not_exceed_remaining_grace(self):
        policy = TransportGrace(0)
        self.assertEqual(policy.timeout(0), 20)
        self.assertEqual(policy.timeout(70), 5)
        self.assertEqual(policy.timeout(75), 1)

    def test_one_poll_updates_heartbeat_and_reads_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'status.json').write_text('{"phase":"loading"}', encoding='utf-8')
            calls = []
            def ssh(alias, command, timeout):
                calls.append((alias, timeout))
                code = shlex.split(command)[2]
                from contextlib import redirect_stdout
                import io
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exec(code, {})
                return buffer.getvalue()
            helper = types.SimpleNamespace(REMOTE=str(root), ssh=ssh)
            tree = ast.parse((HERE / 'run_guarded.py').read_text(encoding='utf-8'))
            node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'heartbeat_status')
            namespace = dict(helpers=helper, json=json, shlex=shlex)
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'run_guarded.py', 'exec'), namespace)
            self.assertEqual(namespace['heartbeat_status'](timeout=20), dict(phase='loading'))
            self.assertTrue((root / 'heartbeat').is_file())
            self.assertEqual(calls, [('taishan', 20)])


@unittest.skipUnless((SOURCE / 'requests.jsonl').is_file(), 'Requires archived cooling90 run evidence')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.target = Path(self.temp.name) / 'target'
        self.source.mkdir()
        self.target.mkdir()
        names = list(METADATA) + ['cleanup.json', 'status.json', 'requests.jsonl']
        for phase in checkpoint.REUSED_PHASES:
            names += [f'logs/{phase}.log', f'{phase}-startup-audit.json',
                      f'{phase}-before-resources.json', f'{phase}-after-resources.json']
        names += ['tp2k64eager-smoke.json', 'tp1k32graph-gate.json']
        for name in names:
            destination = self.source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, destination)
        for name in METADATA:
            shutil.copyfile(self.source / name, self.target / name)
        self.startup = startup_check()

    def validate(self):
        return checkpoint.validate(self.source, self.target, self.startup)

    def change(self, root, name, callback):
        value = checkpoint.read(root, name)
        callback(value)
        (root / name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def test_actual_checkpoint_has_six_complete_formal_requests(self):
        result = self.validate()
        self.assertEqual(result['baseline_formal_requests'], 6)
        self.assertEqual(result['reused_phases'], list(checkpoint.REUSED_PHASES))

    def test_copy_preserves_all_hashes_and_does_not_copy_abort_marker(self):
        record = self.validate()
        checkpoint.copy_verified(record, self.target)
        for name in record['copied_files']:
            self.assertEqual(checkpoint.digest(self.target / name), record['source_sha256'][name])
        self.assertFalse((self.target / 'abort.json').exists())
        self.assertFalse((self.target / 'alarm.json').exists())

    def test_missing_request_is_rejected(self):
        path = self.source / 'requests.jsonl'
        path.write_text('\n'.join(path.read_text(encoding='utf-8').splitlines()[:-1]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '8 completed'):
            self.validate()

    def test_changed_prompt_is_rejected(self):
        with (self.target / 'requests.json').open('a', encoding='utf-8') as output:
            output.write(' ')
        with self.assertRaisesRegex(ValueError, 'requests.json'):
            self.validate()

    def test_changed_image_is_rejected(self):
        self.change(self.target, 'benchmark-config.json', lambda x: x.update(image='different'))
        with self.assertRaisesRegex(ValueError, 'image'):
            self.validate()

    def test_changed_forward_candidate_is_rejected(self):
        self.change(self.target, 'candidate.json', lambda x: x.update(candidate_sha256='different'))
        with self.assertRaisesRegex(ValueError, 'forward implementation'):
            self.validate()

    def test_changed_native_backend_is_rejected(self):
        self.change(self.target, 'environment-preflight.json', lambda x: x.update(extension_sha256='different'))
        with self.assertRaisesRegex(ValueError, 'native backend'):
            self.validate()

    def test_source_change_after_validation_is_rejected(self):
        record = self.validate()
        with (self.source / 'requests.jsonl').open('a', encoding='utf-8') as output:
            output.write('\n')
        with self.assertRaisesRegex(ValueError, 'source changed after validation'):
            checkpoint.copy_verified(record, self.target)


if __name__ == '__main__':
    unittest.main()
