"""Exercise stop boundaries and the real runtime adapter without NPU hardware."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from thermal_policy import POLICY, SustainedTemperature


def cards(a=50, b=50):
    return {'2': dict(health='OK', temperature_c=a),
            '5': dict(health='OK', temperature_c=b)}


class TemperatureTests(unittest.TestCase):
    def test_70_and_89_never_stop(self):
        guard = SustainedTemperature()
        for now in range(0, 100, 5):
            self.assertIsNone(guard.check(cards(70, 89), now))

    def test_exact_90_requires_full_ten_seconds(self):
        guard = SustainedTemperature()
        for now in (0, 5, 9.999):
            self.assertIsNone(guard.check(cards(90), now))
        self.assertIn('sustained=10.0s', guard.check(cards(90), 10))

    def test_spike_does_not_stop(self):
        guard = SustainedTemperature()
        for now, temp in [(0, 95), (5, 89), (10, 95), (15, 89)]:
            self.assertIsNone(guard.check(cards(temp), now))

    def test_drop_below_90_restarts_timer(self):
        guard = SustainedTemperature()
        for now, temp in [(0, 91), (9, 89), (10, 91), (19, 91)]:
            self.assertIsNone(guard.check(cards(temp), now))
        self.assertIsNotNone(guard.check(cards(91), 20))

    def test_both_cards_have_separate_timers(self):
        guard = SustainedTemperature()
        for now, pair in [(0, (90, 50)), (5, (50, 90)), (10, (90, 50)), (15, (50, 90))]:
            self.assertIsNone(guard.check(cards(*pair), now))
        self.assertIsNone(guard.check(cards(50, 90), 20))
        self.assertIn('NPU 5', guard.check(cards(50, 90), 25))

    def test_repeated_cached_sample_does_not_advance_time(self):
        guard = SustainedTemperature()
        for _ in range(20):
            self.assertIsNone(guard.check(cards(91), 100))
        self.assertIsNotNone(guard.check(cards(91), 110))

    def test_telemetry_gap_does_not_count_as_hot_time(self):
        guard = SustainedTemperature()
        self.assertIsNone(guard.check(cards(91), 0))
        self.assertIsNone(guard.check(cards(91), 20))
        self.assertIsNotNone(guard.check(cards(91), 30))

    def test_clock_regression_resets_timer(self):
        guard = SustainedTemperature()
        self.assertIsNone(guard.check(cards(91), 100))
        self.assertIsNone(guard.check(cards(91), 90))
        self.assertIsNotNone(guard.check(cards(91), 100))

    def test_health_failure_is_immediate(self):
        value = cards()
        value['5']['health'] = 'Alarm'
        self.assertIn('health=Alarm', SustainedTemperature().check(value, 0))

    def test_missing_card_is_immediate(self):
        value = cards()
        del value['2']
        self.assertIn('Missing NPU', SustainedTemperature().check(value, 0))

    def test_nan_is_rejected(self):
        self.assertIn('invalid temperature', SustainedTemperature().check(cards(float('nan')), 0))

    def test_invalid_sample_explicitly_resets(self):
        guard = SustainedTemperature()
        guard.check(cards(91), 0)
        guard.invalidate()
        self.assertIsNone(guard.check(cards(91), 10))


class AdapterTests(unittest.TestCase):
    def test_runtime_overrides_legacy_70_stop_and_keeps_shared_module_state(self):
        legacy = types.ModuleType('base_runtime')
        legacy.read_cards = lambda: self.fail('Legacy 70C check was used')
        def raw(temperature):
            return (f'| 2 910B4-1 | OK | 100.0 {temperature} 0 / 0 |\n'
                    '| 5 910B4-1 | OK | 90.0 50 0 / 0 |')
        samples = iter([70, 90, 90, 90])
        legacy.run = lambda *a, **kw: raw(next(samples))
        name = '_thermal_runtime_under_test'
        with patch.dict(sys.modules, {'base_runtime': legacy}):
            spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('tp_runtime.py'))
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                self.assertIs(sys.modules[name], legacy)
                self.assertEqual(legacy.THERMAL_POLICY, POLICY)
                legacy.NAMES = {'test': 'container'}
                self.assertEqual(module.core.NAMES['test'], 'container')
                with patch.object(module.time, 'monotonic', side_effect=[0, 1, 6, 11]):
                    self.assertEqual(legacy.read_cards()['2']['temperature_c'], 70)
                    legacy.read_cards()
                    legacy.read_cards()
                    with self.assertRaisesRegex(RuntimeError, 'sustained=10.0s'):
                        legacy.read_cards()
            finally:
                sys.modules.pop(name, None)


if __name__ == '__main__':
    unittest.main()
