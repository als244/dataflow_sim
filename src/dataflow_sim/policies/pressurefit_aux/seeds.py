"""Initial residency and seed interval construction for PressureFit."""
from __future__ import annotations

import math
from dataclasses import replace

from dataflow_sim.policies._common import _UseEvent
from dataflow_sim.policies.pressurefit_aux.core import _Facts, _build_facts
from dataflow_sim.policies.pressurefit_aux.types import _IntervalSet
from dataflow_sim.core.schema import TaskChain


def _pressure_initial_placement(
    bare: TaskChain,
    device_capacity: int | None,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
) -> set[str]:
    """Select host-initial objects to add to device at t=0."""
    if not bare.tasks:
        return set()

    host_objs = {o.id: o for o in bare.initial_memory if o.location == "host"}
    already_device = {o.id for o in bare.initial_memory if o.location == "device"}
    if device_capacity is None:
        return {oid for oid in host_objs if uses_by_task.get(oid)}

    cap = device_capacity
    task0 = bare.tasks[0]
    placement = {
        oid for oid in task0.inputs
        if oid in host_objs and oid not in already_device
    }
    initial_device_bytes = sum(
        o.size for o in bare.initial_memory if o.location == "device"
    )
    task0_outputs = sum(o.size for o in task0.outputs if o.location == "device")
    must_bytes = sum(sizes[o] for o in placement)
    if initial_device_bytes + must_bytes + task0_outputs > cap:
        raise ValueError(
            "infeasible: task 0 inputs plus device output reservation exceed "
            f"device_capacity ({initial_device_bytes}+{must_bytes}+"
            f"{task0_outputs}>{cap})"
        )

    facts = _build_facts(replace(bare, device_capacity=cap))
    inbound_bw = bare.bandwidth_h2d
    used = initial_device_bytes + must_bytes

    def cold_deadline_misses() -> dict[str, int]:
        if inbound_bw is None or inbound_bw <= 0:
            return {}
        jobs: list[tuple[int, int, str, int]] = []
        for oid, obj in host_objs.items():
            if oid in placement:
                continue
            events = uses_by_task.get(oid, [])
            if not events:
                continue
            first = events[0].task_idx
            if first == 0:
                continue
            tau = max(1, math.ceil(obj.size / inbound_bw))
            jobs.append((facts.task_start[first], first, oid, tau))

        jobs.sort(key=lambda j: (j[0], j[1], j[2]))
        cursor = facts.task_end[0]
        misses: dict[str, int] = {}
        for deadline, _first, oid, tau in jobs:
            end = cursor + tau
            miss = max(0, end - deadline)
            if miss:
                misses[oid] = miss
            cursor = end
        return misses

    miss_by_oid = cold_deadline_misses()
    remaining: list[tuple[int, int, int, int, str]] = []
    for oid, obj in host_objs.items():
        if oid in placement or not uses_by_task.get(oid):
            continue
        first = uses_by_task[oid][0]
        tau = (
            max(1, math.ceil(obj.size / inbound_bw))
            if inbound_bw is not None and inbound_bw > 0
            else 0
        )
        slack = max(0, first.ideal_start - facts.task_end[0] - tau)
        remaining.append((
            first.task_idx,
            slack,
            -miss_by_oid.get(oid, 0),
            -sizes[oid],
            oid,
        ))

    for _first, _slack, _neg_miss, _neg_size, oid in sorted(remaining):
        if used + sizes[oid] + task0_outputs <= cap:
            placement.add(oid)
            used += sizes[oid]

    return placement


def _initial_residency(
    facts: _Facts,
    initial_device: set[str],
) -> _IntervalSet:
    intervals: _IntervalSet = {}
    all_ids = set(facts.sizes) | set(facts.uses) | set(facts.producer)
    for oid in all_ids:
        p = facts.producer.get(oid, -1)
        uses = facts.uses.get(oid, [])
        if oid in facts.host_ids:
            if not uses:
                continue
            a = -1 if oid in initial_device else uses[0] - 1
            b = uses[-1] - 1
        elif oid in facts.device_ids:
            a = -1
            b = uses[-1] - 1 if uses else -1
        else:
            if p < 0:
                continue
            a = p
            b = uses[-1] - 1 if uses else p
        intervals[oid] = [(a, b)]
    return intervals


def _copy_intervals(seed: _IntervalSet) -> _IntervalSet:
    return {oid: list(ivs) for oid, ivs in seed.items()}
