"""Workload stop rule matching TaiShan: >=90 C on one card for 10 seconds."""
import math

LIMIT_C = 90
HOLD_SECONDS = 10
MAX_SAMPLE_GAP_SECONDS = 15
POLICY = dict(temperature_limit_c=LIMIT_C, temperature_comparison='>=',
              temperature_hold_seconds=HOLD_SECONDS,
              max_sample_gap_seconds=MAX_SAMPLE_GAP_SECONDS,
              fan_control='existing Taishan-NpuCooling service; read only')


class SustainedTemperature:
    def __init__(self):
        self.since = {}
        self.last_sample = None

    def invalidate(self):
        self.since.clear()
        self.last_sample = None

    def check(self, cards, sampled_at):
        if set(cards) != {'2', '5'}:
            self.invalidate()
            return f'Missing NPU telemetry: {cards}'
        if (self.last_sample is not None and
                (sampled_at < self.last_sample or
                 sampled_at - self.last_sample > MAX_SAMPLE_GAP_SECONDS)):
            self.invalidate()
        self.last_sample = sampled_at
        for device, card in cards.items():
            if card['health'] != 'OK':
                self.invalidate()
                return f'NPU {device} health={card["health"]}'
            temperature = card['temperature_c']
            if not math.isfinite(temperature):
                self.invalidate()
                return f'NPU {device} has invalid temperature'
            if temperature >= LIMIT_C:
                self.since.setdefault(device, sampled_at)
            else:
                self.since.pop(device, None)
            duration = sampled_at - self.since.get(device, sampled_at)
            card['over_limit_seconds'] = duration
            if temperature >= LIMIT_C and duration >= HOLD_SECONDS:
                return (f'NPU {device} temperature={temperature}C >= {LIMIT_C}C '
                        f'sustained={duration:.1f}s required={HOLD_SECONDS}s')
        return None
