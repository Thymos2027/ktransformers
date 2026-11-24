"""Reuse frozen experiment helpers with the current, sustained temperature rule."""
import re
import sys
import threading
import time

import base_runtime as core
from thermal_policy import POLICY, SustainedTemperature


temperature_guard = SustainedTemperature()
temperature_lock = threading.Lock()


def read_cards():
    raw = core.run(['npu-smi', 'info'], timeout=12)
    cards = {}
    pattern = r'\|\s*(\d+)\s+910B4-1\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s+(\d+)\s+'
    for device, health, power, temperature in re.findall(pattern, raw):
        cards[device] = dict(health=health, power_w=float(power), temperature_c=int(temperature))
    with temperature_lock:
        reason = temperature_guard.check(cards, time.monotonic())
    if reason:
        raise RuntimeError('NPU guard triggered: ' + reason)
    return cards


# Keep the original module object: its helper functions and the controller must
# share mutable ROOT/NAMES/IMAGE/DEADLINE state. Historical source stays intact.
core.read_cards = read_cards
core.THERMAL_POLICY = POLICY
sys.modules[__name__] = core
