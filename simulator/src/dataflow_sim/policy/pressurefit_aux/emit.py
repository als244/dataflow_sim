"""Interval-to-trigger emission for PressureFit."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dataflow_sim.policy.pressurefit_aux.core import (
    _Facts,
    _fire_task_for_interval,
    _first_use_in_interval,
    _pool_size,
)
from dataflow_sim.schema import Object, Task, TaskChain, TransferTrigger


@dataclass(frozen=True)
class _PrefetchJob:
    oid: str
    earliest: int
    latest: int
    deadline: int
    tau: int
    first_use: int


def _emit_chain(
    bare: TaskChain,
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    *,
    pack_h2d: bool = True,
    respect_interval_start: bool = False,
    latest_h2d: bool = False,
) -> TaskChain:
    releases: list[list[str]] = [[] for _ in range(facts.n)]
    offloads: list[list[str]] = [[] for _ in range(facts.n)]
    unscheduled_prefetches: list[list[str]] = [[] for _ in range(facts.n)]
    unscheduled_order: list[dict[str, int]] = [dict() for _ in range(facts.n)]
    prefetch_jobs: list[_PrefetchJob] = []
    pre_placed: set[str] = set()

    for oid, ivs in intervals.items():
        ivs = sorted(ivs)
        p = facts.producer.get(oid, -1)
        has_host = oid in facts.host_ids

        for idx, (a, b) in enumerate(ivs):
            if idx == 0 and a == -1:
                if has_host:
                    pre_placed.add(oid)
            elif idx == 0 and p == a and p >= 0:
                pass
            else:
                job = _prefetch_job(
                    oid, idx, ivs, facts, bare.bandwidth_h2d,
                    respect_interval_start=respect_interval_start,
                )
                if job is None or not pack_h2d:
                    fire = _prefetch_fire_task(
                        oid, idx, ivs, facts, bare.bandwidth_h2d,
                        respect_interval_start=respect_interval_start,
                    )
                    unscheduled_prefetches[fire].append(oid)
                    unscheduled_order[fire][oid] = (
                        job.first_use if job is not None else facts.n
                    )
                else:
                    prefetch_jobs.append(job)

            fire_task = _fire_task_for_interval(oid, a, b, facts)
            if fire_task is None:
                continue
            mutated = any(a <= m - 1 <= b for m in facts.mutators.get(oid, set()))
            is_last = idx == len(ivs) - 1
            final_location = facts.final_locations.get(oid)
            if final_location == "device" and is_last:
                continue
            if final_location == "host" and is_last:
                if mutated or not has_host:
                    offloads[fire_task].append(oid)
                else:
                    releases[fire_task].append(oid)
            elif mutated and not is_last:
                offloads[fire_task].append(oid)
            elif mutated:
                releases[fire_task].append(oid)
            elif (not is_last) and (oid not in facts.host_ids):
                offloads[fire_task].append(oid)
            else:
                releases[fire_task].append(oid)

    prefetches, prefetch_order = _assign_prefetch_jobs(
        prefetch_jobs,
        facts,
        latest_only=latest_h2d,
    )
    for i, oids in enumerate(unscheduled_prefetches):
        if oids:
            prefetches[i].extend(oids)
            prefetch_order[i].update(unscheduled_order[i])

    for i in range(facts.n):
        if prefetches[i]:
            prefetches[i] = sorted(
                dict.fromkeys(prefetches[i]),
                key=lambda oid: (prefetch_order[i].get(oid, facts.n), oid),
            )
        if releases[i]:
            releases[i] = list(dict.fromkeys(releases[i]))
        if offloads[i]:
            offloads[i] = list(dict.fromkeys(offloads[i]))

        wasteful = (set(releases[i]) | set(offloads[i])) & set(prefetches[i])
        if wasteful:
            releases[i] = [o for o in releases[i] if o not in wasteful]
            offloads[i] = [o for o in offloads[i] if o not in wasteful]
            prefetches[i] = [o for o in prefetches[i] if o not in wasteful]

    host_objs = {o.id: o for o in bare.initial_memory if o.location == "host"}
    new_initial = list(bare.initial_memory)
    for oid in sorted(pre_placed):
        src = host_objs[oid]
        new_initial.append(Object(
            id=src.id,
            size=src.size,
            location="device",
            type=src.type,
        ))

    new_tasks: list[Task] = []
    for i, task in enumerate(bare.tasks):
        new_tasks.append(replace(
            task,
            releases_after=releases[i],
            offload_after=[TransferTrigger(obj_id=o) for o in offloads[i]],
            prefetch_after=[TransferTrigger(obj_id=o) for o in prefetches[i]],
        ))

    return replace(bare, initial_memory=new_initial, tasks=new_tasks)




def _prefetch_fire_task(
    oid: str,
    interval_idx: int,
    intervals: list[tuple[int, int]],
    facts: _Facts,
    bw_h2d: int | None,
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
    if bw_h2d is None or bw_h2d <= 0:
        return latest
    tau = max(1, math.ceil(facts.sizes[oid] / bw_h2d))
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
    bw_h2d: int | None,
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
        max(1, math.ceil(facts.sizes[oid] / bw_h2d))
        if bw_h2d is not None and bw_h2d > 0
        else 0
    )
    return _PrefetchJob(
        oid=oid,
        earliest=earliest,
        latest=latest,
        deadline=facts.task_start[first_use],
        tau=tau,
        first_use=first_use,
    )


def _extend_h2d_lead_time(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    cap: int | None,
    bw_h2d: int | None,
    extra_pressure: list[int] | None = None,
) -> None:
    """Move prefetch interval entries left when strict capacity permits."""
    if cap is None or bw_h2d is None or bw_h2d <= 0:
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

            tau = max(1, math.ceil(facts.sizes[oid] / bw_h2d))
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


def _assign_prefetch_jobs(
    jobs: list[_PrefetchJob],
    facts: _Facts,
    *,
    latest_only: bool = False,
) -> tuple[list[list[str]], list[dict[str, int]]]:
    prefetches: list[list[str]] = [[] for _ in range(facts.n)]
    prefetch_order: list[dict[str, int]] = [dict() for _ in range(facts.n)]
    if not jobs:
        return prefetches, prefetch_order

    next_start = math.inf
    assignments: list[tuple[int, _PrefetchJob]] = []
    for job in sorted(jobs, key=lambda j: (j.deadline, j.latest, j.oid), reverse=True):
        fire = job.latest
        if not latest_only:
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
        assignments.append((fire, job))
        if not latest_only:
            next_start = max(facts.task_end[fire], desired_start)

    for fire, job in assignments:
        prefetches[fire].append(job.oid)
        prefetch_order[fire][job.oid] = job.first_use
    return prefetches, prefetch_order
