# SPDX-License-Identifier: Apache-2.0
"""Bounded decode promotions; CPU remains the exact-routing fallback.

The prepare/copy/fence/publish contract follows vLLM-Ascend PR #18604.
The frequency policy and CPU fallback integration are specific to this experiment.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Promotion:
    layer: int
    slot: int
    old: int
    new: int
    gain: float


class Policy:
    def __init__(self, experts=256, budget=8, decay=0.5, ratio=1.5, minimum=4):
        if experts < 1 or budget < 1 or not 0 <= decay < 1 or ratio < 1 or minimum < 1:
            raise ValueError('Invalid residency policy')
        self.experts, self.budget = experts, budget
        self.decay, self.ratio, self.minimum = decay, ratio, minimum
        self.slots, self.counts, self.scores, self.steps = {}, {}, {}, {}
        self.routes = self.hits = 0

    def reset(self, slots):
        if not slots or any(not ids or len(set(ids)) != len(ids) or
                            any(not 0 <= e < self.experts for e in ids) for ids in slots.values()):
            raise ValueError('Invalid per-layer resident slots')
        self.slots = {layer: list(ids) for layer, ids in slots.items()}
        self.counts = {layer: [0] * self.experts for layer in slots}
        self.scores = {layer: [0.] * self.experts for layer in slots}
        self.steps = {layer: 0 for layer in slots}
        self.routes = self.hits = 0

    def observe(self, layer, ids):
        if layer not in self.slots:
            raise ValueError('Unregistered layer')
        resident = set(self.slots[layer])
        for raw in ids:
            expert = int(raw)
            if expert != raw or not 0 <= expert < self.experts:
                raise ValueError('Invalid routed expert')
            self.counts[layer][expert] += 1
            self.routes += 1
            self.hits += expert in resident
        self.steps[layer] += 1

    def plan(self):
        candidates = []
        for layer, slots in self.slots.items():
            counts, scores = self.counts[layer], self.scores[layer]
            for expert in range(self.experts):
                scores[expert] = self.decay * scores[expert] + counts[expert]
                counts[expert] = 0
            resident = set(slots)
            slot = min(range(len(slots)), key=lambda i: (scores[slots[i]], i))
            cold = slots[slot]
            hot = max((e for e in range(self.experts) if e not in resident),
                      key=lambda e: (scores[e], -e), default=None)
            if hot is not None and scores[hot] >= self.minimum and scores[hot] > self.ratio * scores[cold] + 1:
                candidates.append(Promotion(layer, slot, cold, hot, scores[hot] - scores[cold]))
        # One replacement per layer, with a global bound on work at this boundary.
        return sorted(candidates, key=lambda p: (-p.gain, p.layer))[:self.budget]

    def accept(self, plan):
        for item in plan:
            if self.slots[item.layer][item.slot] != item.old:
                raise RuntimeError('Resident generation changed before publication')
        for item in plan:
            self.slots[item.layer][item.slot] = item.new


def transact(plan, prepare, copy, fence, publish):
    """Never publish maps after a partial copy. Caller must stop on an exception.

    Prepared bundles stay owned until the transfer fence, including failure paths.
    This is not rollback: a failed in-place copy may have overwritten old weights.
    """
    prepared = [(item, prepare(item)) for item in plan]
    try:
        for item, bundle in prepared:
            copy(item, bundle)
    finally:
        fence()
    publish(plan)
