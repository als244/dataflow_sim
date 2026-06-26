"""Shared interval model for PressureFit."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from dataflow_sim.policies._common import _object_sizes
from dataflow_sim.core.schema import TaskChain


@dataclass(frozen=True)
class _Facts:
    n: int
    sizes: dict[str, int]
    producer: dict[str, int]
    uses: dict[str, list[int]]
    mutators: dict[str, set[int]]
    backing_ids: set[str]
    compute_ids: set[str]
    final_locations: dict[str, str]
    task_start: list[int]
    task_end: list[int]
    next_outputs: list[int]


def _build_facts(chain: TaskChain) -> _Facts:
    sizes = _object_sizes(chain)
    n = len(chain.tasks)
    producer = {o.id: -1 for o in chain.initial_memory}
    for i, task in enumerate(chain.tasks):
        for out in task.outputs:
            producer[out.id] = i

    uses: dict[str, list[int]] = defaultdict(list)
    mutators: dict[str, set[int]] = defaultdict(set)
    for i, task in enumerate(chain.tasks):
        for inp in task.inputs:
            uses[inp].append(i)
        for oid in task.mutates_inputs:
            mutators[oid].add(i)

    task_start: list[int] = []
    task_end: list[int] = []
    t = 0
    for task in chain.tasks:
        task_start.append(t)
        t += task.runtime
        task_end.append(t)

    next_outputs = [0] * (n + 1)
    for b in range(-1, n - 1):
        task = chain.tasks[b + 1]
        next_outputs[b + 1] = sum(
            out.size for out in task.outputs if out.location == "fast"
        )

    return _Facts(
        n=n,
        sizes=sizes,
        producer=producer,
        uses={k: sorted(v) for k, v in uses.items()},
        mutators=mutators,
        backing_ids={o.id for o in chain.initial_memory if o.location == "backing"},
        compute_ids={o.id for o in chain.initial_memory if o.location == "fast"},
        final_locations=dict(chain.final_locations),
        task_start=task_start,
        task_end=task_end,
        next_outputs=next_outputs,
    )


def _transfer_time(size: int, bandwidth: int | None) -> int:
    if bandwidth is None or bandwidth <= 0:
        return 0
    return max(1, math.ceil(size / bandwidth))


def _anchors(oid: str, facts: _Facts) -> list[int]:
    out: set[int] = set()
    if oid in facts.compute_ids:
        out.add(-1)
    p = facts.producer.get(oid, -1)
    if p >= 0:
        out.add(p)
    for u in facts.uses.get(oid, []):
        out.add(u - 1)
    return sorted(out)


def _effective_a(a: int, producer: int) -> int:
    if a > -1 and a != producer:
        return a - 1
    return a


def _pool_size(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
) -> list[int]:
    pool = [0] * (facts.n + 1)
    for oid, ivs in intervals.items():
        p = facts.producer.get(oid, -1)
        for a, b in ivs:
            real_a = _effective_a(a, p)
            for k in range(max(-1, real_a), min(facts.n - 1, b) + 1):
                pool[k + 1] += facts.sizes[oid]
    return pool


def _fire_task_for_interval(
    oid: str,
    a: int,
    b: int,
    facts: _Facts,
) -> int | None:
    cands: list[int] = []
    p = facts.producer.get(oid, -1)
    if p >= 0 and a <= p <= b:
        cands.append(p)
    for u in facts.uses.get(oid, []):
        if a <= u - 1 <= b:
            cands.append(u)
    if not cands:
        return None
    return min(facts.n - 1, max(cands))


def _first_use_in_interval(
    oid: str,
    a: int,
    b: int,
    facts: _Facts,
) -> int | None:
    for u in facts.uses.get(oid, []):
        if a <= u - 1 <= b:
            return u
    return None


def _departing_before_next(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    idx: int,
) -> int:
    boundary = idx - 1
    if boundary < 0 or boundary >= facts.n - 1:
        return 0
    total = 0
    for oid, ivs in intervals.items():
        p = facts.producer.get(oid, -1)
        for a, b in ivs:
            if not (_effective_a(a, p) <= boundary <= b):
                continue
            if _fire_task_for_interval(oid, a, b, facts) == boundary:
                total += facts.sizes[oid]
    return total


def _nonblocking_arrivals_before_next(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    idx: int,
) -> int:
    boundary = idx - 1
    if boundary < 0 or boundary >= facts.n - 1:
        return 0
    total = 0
    for oid, ivs in intervals.items():
        p = facts.producer.get(oid, -1)
        for a, b in ivs:
            if a <= -1 or a == p or _effective_a(a, p) != boundary:
                continue
            first_use = _first_use_in_interval(oid, a, b, facts)
            if first_use is None or first_use > boundary + 1:
                total += facts.sizes[oid]
    return total


def _modeled_boundary_need(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    idx: int,
    pool: list[int] | None = None,
) -> int:
    if pool is None:
        pool = _pool_size(facts, intervals)
    return (
        pool[idx]
        - _departing_before_next(facts, intervals, idx)
        - _nonblocking_arrivals_before_next(facts, intervals, idx)
        + facts.next_outputs[idx]
    )
