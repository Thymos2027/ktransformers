"""Recognize successful decode capture with TP-only or TP/EP logger labels."""
import re


def check_decode_graph(log):
    records = re.findall(r'\bTP(\d+)(?:\s+EP(\d+))?\]\s+Capture target decode NPU graph end\b', log)
    if {int(tp) for tp, ep in records} != {0, 1}:
        raise RuntimeError('Missing successful decode graph capture for one of TP0/TP1')
    if any(ep and int(ep) != int(tp) for tp, ep in records):
        raise RuntimeError('Unexpected TP/EP rank association in graph capture log')
    return records
