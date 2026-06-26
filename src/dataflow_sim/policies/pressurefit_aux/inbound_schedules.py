"""Inbound prefetch scheduling helpers for PressureFit."""
from __future__ import annotations

import math
from dataclasses import dataclass

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _fire_task_for_interval,
    _first_use_in_interval,
    _pool_size,
)


@dataclass(frozen=True)
class _PrefetchJob:
    oid: str
    earliest: int
    latest: int
    deadline: int
    tau: int
    first_use: int
    # Interval entry boundary `a`. The analytic model counts the object's
    # bytes from boundary `a - 1`; firing the trigger on an earlier task
    # materializes bytes at boundaries the model never charged.
    entry_a: int
    size: int


def _extend_inbound_lead_time(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    cap: int | None,
    inbound_bw: int | None,
    extra_pressure: list[int] | None = None,
) -> None:
    """Move prefetch interval entries left when strict capacity permits."""
    if cap is None or inbound_bw is None or inbound_bw <= 0:
        return
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)

    prefetches: list[tuple[int, str, int, int, int, int, int]] = []
    for oid, ivs in intervals.items():
        p = facts.producer.get(oid, -1)
        for idx, (a, b) in enumerate(ivs):
            if idx == 0 and (a == -1 or (p >= 0 and a == p)):
                continue
            if a <= 0:
                continue
            first_use = _first_use_in_interval(oid, a, b, facts)
            if first_use is None:
                continue

            earliest_a = 1
            if p >= 0:
                earliest_a = max(earliest_a, p + 1)
            if idx > 0:
                earliest_a = max(earliest_a, ivs[idx - 1][1] + 2)
            if earliest_a >= a:
                continue

            tau = max(1, math.ceil(facts.sizes[oid] / inbound_bw))
            deadline = facts.task_start[first_use]
            prefetches.append((deadline, oid, idx, tau, a, b, earliest_a))

    if not prefetches:
        return

    pool = _pool_size(facts, intervals)
    next_start_t = math.inf
    for deadline, oid, idx, tau, current_a, _b, earliest_a in sorted(
        prefetches, key=lambda item: (-item[0], item[1], item[2])
    ):
        ivs = intervals.get(oid)
        if ivs is None or idx >= len(ivs) or ivs[idx][0] != current_a:
            continue

        latest_finish = deadline if math.isinf(next_start_t) else min(deadline, next_start_t)
        ideal_start_t = max(0, latest_finish - tau)
        ideal_fire = -1
        for task_idx, end_t in enumerate(facts.task_end):
            if end_t <= ideal_start_t:
                ideal_fire = task_idx
            else:
                break

        ideal_a = max(ideal_fire + 1, earliest_a)
        if ideal_a >= current_a:
            next_start_t = facts.task_end[current_a - 1]
            continue

        size = facts.sizes[oid]
        chosen_a = current_a
        old_eff_lo = current_a - 1
        for try_a in range(ideal_a, current_a):
            new_eff_lo = try_a - 1
            ok = True
            for boundary in range(new_eff_lo, old_eff_lo):
                idx_in_pool = boundary + 1
                if (
                    pool[idx_in_pool]
                    + size
                    + facts.next_outputs[idx_in_pool]
                    + extra_pressure[idx_in_pool]
                    > cap
                ):
                    ok = False
                    break
            if ok:
                chosen_a = try_a
                break

        if chosen_a < current_a:
            new_eff_lo = chosen_a - 1
            for boundary in range(new_eff_lo, old_eff_lo):
                pool[boundary + 1] += size
            a, b = ivs[idx]
            ivs[idx] = (chosen_a, b)
            next_start_t = facts.task_end[chosen_a - 1]
        else:
            next_start_t = facts.task_end[current_a - 1]


def _prefetch_fire_task(
    oid: str,
    interval_idx: int,
    intervals: list[tuple[int, int]],
    facts: _Facts,
    inbound_bw: int | None,
    *,
    respect_interval_start: bool = False,
) -> int:
    a, b = intervals[interval_idx]
    first_use = _first_use_in_interval(oid, a, b, facts)
    if first_use is None:
        return max(0, min(facts.n - 1, a - 1))

    earliest = 0
    if interval_idx > 0:
        prev = intervals[interval_idx - 1]
        prev_fire = _fire_task_for_interval(oid, prev[0], prev[1], facts)
        if prev_fire is not None:
            earliest = prev_fire
    p = facts.producer.get(oid, -1)
    if p >= 0:
        earliest = max(earliest, p)

    latest = max(0, first_use - 1)
    if respect_interval_start:
        latest = min(latest, max(0, a - 1))
    if inbound_bw is None or inbound_bw <= 0:
        return latest
    tau = max(1, math.ceil(facts.sizes[oid] / inbound_bw))
    deadline = facts.task_start[first_use]
    for t in range(latest, earliest - 1, -1):
        if facts.task_end[t] + tau <= deadline:
            return t
    return latest


def _prefetch_job(
    oid: str,
    interval_idx: int,
    intervals: list[tuple[int, int]],
    facts: _Facts,
    inbound_bw: int | None,
    *,
    respect_interval_start: bool = False,
) -> _PrefetchJob | None:
    a, b = intervals[interval_idx]
    first_use = _first_use_in_interval(oid, a, b, facts)
    if first_use is None:
        return None

    earliest = 0
    if interval_idx > 0:
        prev = intervals[interval_idx - 1]
        prev_fire = _fire_task_for_interval(oid, prev[0], prev[1], facts)
        if prev_fire is not None:
            earliest = prev_fire
    p = facts.producer.get(oid, -1)
    if p >= 0:
        earliest = max(earliest, p)

    latest = max(0, first_use - 1)
    if respect_interval_start:
        latest = min(latest, max(0, a - 1))
    tau = (
        max(1, math.ceil(facts.sizes[oid] / inbound_bw))
        if inbound_bw is not None and inbound_bw > 0
        else 0
    )
    return _PrefetchJob(
        oid=oid,
        earliest=earliest,
        latest=latest,
        deadline=facts.task_start[first_use],
        tau=tau,
        first_use=first_use,
        entry_a=a,
        size=facts.sizes[oid],
    )


def _assign_prefetch_jobs(
    jobs: list[_PrefetchJob],
    facts: _Facts,
    *,
    pool: list[int] | None = None,
    cap: int | None = None,
    extra_pressure: list[int] | None = None,
) -> tuple[list[list[str]], list[dict[str, int]]]:
    """Pack inbound jobs backward from their deadlines as one FIFO queue.

    When `pool` and `cap` are given, packing is pressure-aware: a job may
    not fire earlier than compute pressure allows. Firing on task `t`
    materializes destination bytes at boundaries `[t, entry_a - 2]` that the
    interval model counts only from `entry_a - 1`; the clamp slides the
    trigger later until every newly covered boundary still satisfies the
    strict capacity inequality, and commits the accepted coverage to `pool`
    so subsequent jobs see it.

    The clamp deliberately charges from the trigger boundary, not from a
    queue-aware estimate of the actual transfer start. A queue-aware variant
    (charge from `max(enqueue, stream cursor)`) was measured and rejected:
    it interpolates between unclamped packing and this clamp — inheriting
    unclamped packing's repair divergence on long tight chains while losing
    this clamp's conservative-rescuer wins — instead of dominating either.
    """
    prefetches: list[list[str]] = [[] for _ in range(facts.n)]
    prefetch_order: list[dict[str, int]] = [dict() for _ in range(facts.n)]
    if not jobs:
        return prefetches, prefetch_order
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)

    next_start = math.inf
    assignments: list[tuple[int, _PrefetchJob]] = []
    for job in sorted(jobs, key=lambda j: (j.deadline, j.latest, j.oid), reverse=True):
        latest_finish = (
            job.deadline if math.isinf(next_start)
            else min(job.deadline, next_start)
        )
        desired_start = latest_finish - job.tau
        for t in range(job.latest, job.earliest - 1, -1):
            if facts.task_end[t] <= desired_start:
                fire = t
                break
        else:
            fire = job.earliest
        if pool is not None and cap is not None:
            fire = _pressure_clamped_fire(
                job, fire, pool, cap, extra_pressure, facts,
            )
        assignments.append((fire, job))
        next_start = max(facts.task_end[fire], desired_start)

    for fire, job in assignments:
        prefetches[fire].append(job.oid)
        prefetch_order[fire][job.oid] = job.first_use
    return prefetches, prefetch_order


def _pressure_clamped_fire(
    job: _PrefetchJob,
    fire: int,
    pool: list[int],
    cap: int,
    extra_pressure: list[int],
    facts: _Facts,
) -> int:
    """Slide `fire` later until its newly covered boundaries fit the cap."""
    model_entry = job.entry_a - 1
    if fire >= model_entry:
        return fire
    clamped = fire
    for x in range(model_entry - 1, fire - 1, -1):
        idx = x + 1
        if (
            pool[idx]
            + job.size
            + facts.next_outputs[idx]
            + extra_pressure[idx]
            > cap
        ):
            clamped = x + 1
            break
    for x in range(clamped, model_entry):
        pool[x + 1] += job.size
    return clamped
