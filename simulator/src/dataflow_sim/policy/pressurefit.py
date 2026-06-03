"""PressureFit policy.

Standalone fast planner:
  1. choose pressure-gated initial residency;
  2. build continuous residency intervals from liveness anchors;
  3. split intervals at overloaded boundaries until the cap fits;
  4. optionally extend H2D interval entries earlier when strict cap permits;
  5. emit release/offload/prefetch triggers with deadline-aware H2D ordering;
  6. verify with the simulator.

The policy is name-agnostic. It uses object source availability, size, uses,
producer, and explicit mutation metadata.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace

from dataflow_sim.policy._common import (
    _UseEvent,
    _compute_ideal_starts,
    _object_sizes,
    _object_uses_by_task_idx,
)
from dataflow_sim.schema import Object, Task, TaskChain, TransferTrigger
from dataflow_sim.simulator import run as simulator_run


@dataclass(frozen=True)
class _Facts:
    n: int
    sizes: dict[str, int]
    producer: dict[str, int]
    uses: dict[str, list[int]]
    mutators: dict[str, set[int]]
    host_ids: set[str]
    device_ids: set[str]
    final_locations: dict[str, str]
    task_start: list[int]
    task_end: list[int]
    next_outputs: list[int]


@dataclass(frozen=True)
class _PrefetchJob:
    oid: str
    earliest: int
    latest: int
    deadline: int
    tau: int
    first_use: int


@dataclass(frozen=True)
class _InitialProtectionJob:
    oid: str
    release_t: int
    deadline: int
    tau: int
    first_use: int
    size: int
    residency_cost: int


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
            out.size for out in task.outputs if out.location == "device"
        )

    return _Facts(
        n=n,
        sizes=sizes,
        producer=producer,
        uses={k: sorted(v) for k, v in uses.items()},
        mutators=mutators,
        host_ids={o.id for o in chain.initial_memory if o.location == "host"},
        device_ids={o.id for o in chain.initial_memory if o.location == "device"},
        final_locations=dict(chain.final_locations),
        task_start=task_start,
        task_end=task_end,
        next_outputs=next_outputs,
    )


def _pressure_initial_placement(
    bare: TaskChain,
    device_capacity: int | None,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
) -> set[str]:
    """Select host-initial objects to add to device at t=0.

    Initial placement is only an initial-boundary decision. Objects chosen
    here may still be split/released later by the interval reducer if a future
    boundary is tight; requiring them to fit continuously until first use would
    make the initial pool unnecessarily cold.
    """
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
    bw = bare.bandwidth_h2d
    used = initial_device_bytes + must_bytes

    def cold_deadline_misses() -> dict[str, int]:
        if bw is None or bw <= 0:
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
            tau = max(1, math.ceil(obj.size / bw))
            jobs.append((facts.task_start[first], first, oid, tau))

        jobs.sort(key=lambda j: (j[0], j[1], j[2]))
        cursor = facts.task_end[0]
        misses: dict[str, int] = {}
        for deadline, first, oid, tau in jobs:
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
            max(1, math.ceil(obj.size / bw))
            if bw is not None and bw > 0
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


def _initial_residency(facts: _Facts, initial_device: set[str]) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
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


def _anchors(oid: str, facts: _Facts) -> list[int]:
    out: set[int] = set()
    if oid in facts.device_ids:
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


def _pool_size(facts: _Facts, intervals: dict[str, list[tuple[int, int]]]) -> list[int]:
    pool = [0] * (facts.n + 1)
    for oid, ivs in intervals.items():
        p = facts.producer.get(oid, -1)
        for a, b in ivs:
            real_a = _effective_a(a, p)
            for k in range(max(-1, real_a), min(facts.n - 1, b) + 1):
                pool[k + 1] += facts.sizes[oid]
    return pool


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


def _reduce_to_fit(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    cap: int | None,
    extra_pressure: list[int] | None = None,
    protected_initial: set[str] | None = None,
) -> None:
    if cap is None:
        return
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)
    if protected_initial is None:
        protected_initial = set()
    anchors_by_oid = {oid: _anchors(oid, facts) for oid in intervals}
    pool = _pool_size(facts, intervals)

    def static_overflow(pool: list[int], idx: int) -> int:
        return pool[idx] + facts.next_outputs[idx] + extra_pressure[idx] - cap

    def relaxed_overflow(pool: list[int], idx: int) -> int:
        return _modeled_boundary_need(facts, intervals, idx, pool) + extra_pressure[idx] - cap

    def candidates_for(
        pool: list[int],
        worst_idx: int,
        *,
        allow_boundary_relief: bool,
    ) -> list[tuple[tuple[int, int, int, int, int], str, int, int | None, int | None]]:
        del pool
        worst_b = worst_idx - 1
        out: list[tuple[tuple[int, int, int, int, int], str, int, int | None, int | None]] = []
        for oid, ivs in intervals.items():
            p = facts.producer.get(oid, -1)
            for idx, (a, b) in enumerate(ivs):
                if not (_effective_a(a, p) <= worst_b <= b):
                    continue
                anchors = anchors_by_oid.get(oid)
                if anchors is None:
                    anchors = _anchors(oid, facts)
                    anchors_by_oid[oid] = anchors
                anchors_in = [x for x in anchors if a <= x <= b]
                if worst_b in anchors:
                    if not allow_boundary_relief:
                        continue
                    right = [x for x in anchors_in if x >= worst_b + 1]
                    if not right:
                        continue
                    left_end = worst_b
                    right_start = min(right)
                    if _fire_task_for_interval(oid, a, left_end, facts) != worst_b:
                        continue
                else:
                    left = [x for x in anchors_in if x <= worst_b - 1]
                    right = [x for x in anchors_in if x >= worst_b + 1]
                    left_end = max(left) if left else None
                    right_start = min(right) if right else None
                left_b = left_end if left_end is not None else a - 1
                right_a = right_start if right_start is not None else b + 1
                gap_len = right_a - left_b - 1
                if gap_len <= 0:
                    continue
                drops_init = left_end is None and a == -1
                if drops_init and oid in protected_initial:
                    continue
                left_dirty = (
                    left_end is not None
                    and any(a <= m - 1 <= left_end for m in facts.mutators.get(oid, set()))
                )
                release_eligible = oid in facts.host_ids and not left_dirty
                stream_cost = 0 if (drops_init or release_eligible) else 1
                first_use = facts.uses.get(oid, [facts.n])[0]
                key = (
                    stream_cost,
                    0 if drops_init else 1,
                    -first_use,
                    -facts.sizes[oid],
                    -gap_len,
                )
                out.append((key, oid, idx, left_end, right_start))
        return out

    iterations = 0
    max_iterations = max(1, 2 * (facts.n + 2) * max(1, len(facts.sizes)))
    while True:
        iterations += 1
        if iterations > max_iterations:
            raise ValueError(
                "infeasible: pressurefit pressure reduction exceeded "
                f"{max_iterations} split attempts"
            )
        worst_idx = max(range(len(pool)), key=lambda i: static_overflow(pool, i))
        if static_overflow(pool, worst_idx) <= 0:
            return
        candidates = candidates_for(pool, worst_idx, allow_boundary_relief=False)
        if not candidates:
            worst_idx = max(range(len(pool)), key=lambda i: relaxed_overflow(pool, i))
            if relaxed_overflow(pool, worst_idx) <= 0:
                return
            candidates = candidates_for(pool, worst_idx, allow_boundary_relief=True)

        if not candidates:
            worst_b = worst_idx - 1
            raise ValueError(
                f"infeasible: pressurefit cannot reduce boundary {worst_b} "
                f"under device_capacity={cap}"
            )

        _key, oid, idx, left_end, right_start = min(candidates, key=lambda c: c[0])
        a, b = intervals[oid][idx]
        pieces: list[tuple[int, int]] = []
        if left_end is not None:
            pieces.append((a, left_end))
        if right_start is not None:
            pieces.append((right_start, b))
        if pieces == [(a, b)]:
            raise ValueError(
                "infeasible: pressurefit pressure reduction selected a "
                "non-progressing split"
            )
        _subtract_removed_interval_pressure(facts, pool, oid, (a, b), pieces)
        intervals[oid][idx:idx + 1] = pieces
        if not intervals[oid]:
            del intervals[oid]


def _subtract_removed_interval_pressure(
    facts: _Facts,
    pool: list[int],
    oid: str,
    old: tuple[int, int],
    new_pieces: list[tuple[int, int]],
) -> None:
    """Update a precomputed pool after splitting one interval.

    `_pool_size` counts interval residency on boundary index `boundary + 1`.
    A split only removes residency from the old interval's gap, so we can
    subtract those boundaries in place instead of rebuilding the full pool.
    """
    p = facts.producer.get(oid, -1)
    old_a, old_b = old
    old_start = max(-1, _effective_a(old_a, p))
    old_end = min(facts.n - 1, old_b)
    if old_start > old_end:
        return

    normalized_pieces: list[tuple[int, int]] = []
    for a, b in new_pieces:
        start = max(-1, _effective_a(a, p))
        end = min(facts.n - 1, b)
        if start <= end:
            normalized_pieces.append((start, end))

    size = facts.sizes[oid]
    for boundary in range(old_start, old_end + 1):
        if any(start <= boundary <= end for start, end in normalized_pieces):
            continue
        pool[boundary + 1] -= size


def _fire_task_for_interval(oid: str, a: int, b: int, facts: _Facts) -> int | None:
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


def _first_use_in_interval(oid: str, a: int, b: int, facts: _Facts) -> int | None:
    for u in facts.uses.get(oid, []):
        if a <= u - 1 <= b:
            return u
    return None


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
    """Move prefetch interval entries left when strict capacity permits.

    The pressure reducer creates feasible residency intervals but may leave H2D
    arrivals just-in-time. This pass packs known prefetch jobs backward from
    their consumer deadlines, then extends each interval left only across
    boundaries where the strict footprint still fits.
    """
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


def _initial_protection_headroom(facts: _Facts, cap: int | None) -> int:
    if cap is None or facts.n == 0:
        return 0
    initial_device_bytes = sum(facts.sizes[oid] for oid in facts.device_ids)
    mandatory_host_inputs = {
        oid for oid in facts.host_ids
        if facts.uses.get(oid) and facts.uses[oid][0] == 0
    }
    mandatory_host_bytes = sum(facts.sizes[oid] for oid in mandatory_host_inputs)
    return cap - initial_device_bytes - mandatory_host_bytes - facts.next_outputs[0]


def _pressure_probe_intervals(
    facts: _Facts,
    cap: int | None,
) -> dict[str, list[tuple[int, int]]]:
    if cap is None:
        return {}
    all_host = {oid for oid in facts.host_ids if facts.uses.get(oid)}
    if not all_host:
        return {}

    probe = _initial_residency(facts, all_host)
    try:
        _reduce_to_fit(facts, probe, cap)
    except ValueError:
        return {}
    return probe


def _initial_protection_jobs_from_probe(
    facts: _Facts,
    cap: int | None,
    bw_h2d: int | None,
) -> list[_InitialProtectionJob]:
    """Build H2D jobs created when initial host residency is cut by pressure."""
    if cap is None or bw_h2d is None or bw_h2d <= 0 or facts.n == 0:
        return []

    probe = _pressure_probe_intervals(facts, cap)
    if not probe:
        return []

    jobs: list[_InitialProtectionJob] = []
    for oid in sorted(facts.host_ids):
        uses = facts.uses.get(oid)
        if not uses:
            continue
        first_use = uses[0]
        if first_use == 0:
            continue
        ivs = sorted(probe.get(oid, []))
        if any(a == -1 for a, _b in ivs):
            continue

        first_anchor = first_use - 1
        containing = next(
            ((a, b) for a, b in ivs if a <= first_anchor <= b),
            None,
        )
        if containing is None:
            continue

        a, _b = containing
        fire_task = max(0, min(facts.n - 1, a - 1))
        release_t = facts.task_end[fire_task]
        deadline = facts.task_start[first_use]
        tau = max(1, math.ceil(facts.sizes[oid] / bw_h2d))
        residency_span = max(1, deadline)
        jobs.append(_InitialProtectionJob(
            oid=oid,
            release_t=release_t,
            deadline=deadline,
            tau=tau,
            first_use=first_use,
            size=facts.sizes[oid],
            residency_cost=facts.sizes[oid] * residency_span,
        ))
    return jobs


def _source_initial_protection_jobs(
    facts: _Facts,
    bw_h2d: int | None,
    *,
    clean_only: bool,
) -> list[_InitialProtectionJob]:
    """Build source-object jobs for initial-protection frontier candidates."""
    if bw_h2d is None or bw_h2d <= 0 or facts.n == 0:
        return []

    jobs: list[_InitialProtectionJob] = []
    for oid in sorted(facts.host_ids):
        uses = facts.uses.get(oid)
        if not uses:
            continue
        if clean_only and facts.mutators.get(oid):
            continue
        first_use = uses[0]
        deadline = facts.task_start[first_use]
        tau = max(1, math.ceil(facts.sizes[oid] / bw_h2d))
        jobs.append(_InitialProtectionJob(
            oid=oid,
            release_t=0,
            deadline=deadline,
            tau=tau,
            first_use=first_use,
            size=facts.sizes[oid],
            residency_cost=facts.sizes[oid] * max(1, deadline),
        ))
    return jobs


def _h2d_deadline_misses(
    jobs: list[_InitialProtectionJob],
    protected: set[str],
) -> list[tuple[int, _InitialProtectionJob]]:
    cursor = 0
    misses: list[tuple[int, _InitialProtectionJob]] = []
    for job in sorted(jobs, key=lambda j: (j.deadline, j.release_t, j.oid)):
        if job.oid in protected:
            continue
        start = max(cursor, job.release_t)
        end = start + job.tau
        miss = max(0, end - job.deadline)
        if miss:
            misses.append((miss, job))
        cursor = end
    return misses


def _select_initial_protection_set(
    jobs: list[_InitialProtectionJob],
    headroom: int,
) -> set[str]:
    """Protect enough initial residency to remove predicted H2D deadline misses."""
    if headroom <= 0:
        return set()

    protected: set[str] = set()
    remaining = headroom
    for _ in range(len(jobs)):
        misses = _h2d_deadline_misses(jobs, protected)
        if not misses:
            break

        _miss, first_missed = min(
            misses,
            key=lambda item: (item[1].deadline, -item[0], item[1].oid),
        )
        eligible = [
            job for job in jobs
            if (
                job.oid not in protected
                and job.deadline <= first_missed.deadline
                and job.size <= remaining
            )
        ]
        if not eligible:
            break

        chosen = min(
            eligible,
            key=lambda job: (
                job.residency_cost / max(1, job.tau),
                -job.tau,
                job.deadline,
                job.size,
                job.oid,
            ),
        )
        protected.add(chosen.oid)
        remaining -= chosen.size

    return protected


def _initial_protection_prefix_sets(
    jobs: list[_InitialProtectionJob],
    headroom: int,
    *,
    group_key,
) -> list[set[str]]:
    """Return prefix sets at H2D-work frontier points.

    The frontier is measured in transfer time, not object count. It starts with
    the first urgency group in the supplied order, then records prefixes when
    cumulative protected H2D work reaches successive powers of that first
    group's work. The first following urgency group is also recorded so the
    portfolio can test the immediate neighbor of the first congestion point.
    """
    if not jobs or headroom <= 0:
        return []

    first_key = group_key(jobs[0])
    first_group_work = 0
    first_group_count = 0
    for job in jobs:
        if group_key(job) != first_key:
            break
        first_group_work += max(1, job.tau)
        first_group_count += 1
    next_work_target = max(1, first_group_work)
    expand_work_frontier = first_group_count == 1
    total_work = sum(max(1, job.tau) for job in jobs)
    work_horizon = max(
        next_work_target,
        math.ceil(math.sqrt(next_work_target * max(1, total_work))),
    )

    protected: set[str] = set()
    protected_bytes = 0
    protected_work = 0
    out: list[set[str]] = []
    recorded: set[frozenset[str]] = set()
    groups_seen = 0
    prev_key = None

    def record() -> None:
        if not protected:
            return
        frozen = frozenset(protected)
        if frozen in recorded:
            return
        recorded.add(frozen)
        out.append(set(protected))

    for i, job in enumerate(jobs):
        key = group_key(job)
        if key != prev_key:
            groups_seen += 1
            prev_key = key

        if protected_bytes + job.size > headroom:
            break

        protected.add(job.oid)
        protected_bytes += job.size
        protected_work += max(1, job.tau)

        next_key = group_key(jobs[i + 1]) if i + 1 < len(jobs) else None
        group_finished = next_key != key
        if group_finished and groups_seen <= 2:
            record()

        if not expand_work_frontier:
            continue

        while (
            next_work_target <= work_horizon
            and protected_work >= next_work_target
        ):
            record()
            next_work_target *= 2
        if protected_work >= work_horizon:
            record()
            break

    return out


def _initial_protection_sets(
    facts: _Facts,
    cap: int | None,
    bw_h2d: int | None,
) -> list[set[str]]:
    jobs = _initial_protection_jobs_from_probe(facts, cap, bw_h2d)
    headroom = _initial_protection_headroom(facts, cap)
    out: list[set[str]] = []
    seen: set[frozenset[str]] = set()

    def add_many(sets: list[set[str]]) -> None:
        for protected in sets:
            frozen = frozenset(protected)
            if not frozen or frozen in seen:
                continue
            seen.add(frozen)
            out.append(protected)

    selected = _select_initial_protection_set(jobs, headroom)
    add_many([selected])

    deadline_jobs = sorted(
        jobs,
        key=lambda job: (job.deadline, job.first_use, -job.size, job.oid),
    )
    add_many(_initial_protection_prefix_sets(
        deadline_jobs,
        headroom,
        group_key=lambda job: job.deadline,
    ))

    clean_tail_jobs = sorted(
        _source_initial_protection_jobs(facts, bw_h2d, clean_only=True),
        key=lambda job: (-job.first_use, -job.size, job.oid),
    )
    add_many(_initial_protection_prefix_sets(
        clean_tail_jobs,
        headroom,
        group_key=lambda job: job.first_use,
    ))

    tail_jobs = sorted(
        _source_initial_protection_jobs(facts, bw_h2d, clean_only=False),
        key=lambda job: (-job.first_use, -job.size, job.oid),
    )
    add_many(_initial_protection_prefix_sets(
        tail_jobs,
        headroom,
        group_key=lambda job: job.first_use,
    ))

    return out


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
                    unscheduled_order[fire][oid] = job.first_use if job is not None else facts.n
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
        new_initial.append(Object(id=src.id, size=src.size, location="device", type=src.type))

    new_tasks: list[Task] = []
    for i, task in enumerate(bare.tasks):
        new_tasks.append(replace(
            task,
            releases_after=releases[i],
            offload_after=[TransferTrigger(obj_id=o) for o in offloads[i]],
            prefetch_after=[TransferTrigger(obj_id=o) for o in prefetches[i]],
        ))

    return replace(bare, initial_memory=new_initial, tasks=new_tasks)


_DEADLOCK_RE = re.compile(
    r"task '([^']+)' deadlocked .* device (\d+)\+(\d+) bytes > cap (\d+)"
)
_OUTPUT_NEED_RE = re.compile(
    r"task '([^']+)' cannot satisfy device memory need of (\d+) bytes "
    r"\(current free \+ all scheduled offloads = (\d+), capacity=(\d+)\)"
)
_MISSING_INPUTS_RE = re.compile(
    r"task '([^']+)' deadlocked .* inputs \[([^\]]+)\] not live on device"
)


def _physical_pressure_from_error(
    msg: str,
    facts: _Facts,
    bare: TaskChain,
) -> tuple[int, int, int] | None:
    """Translate simulator output-cap deadlocks into extra boundary pressure."""
    m = _DEADLOCK_RE.search(msg)
    if m:
        task_id, used_s, need_s, cap_s = m.groups()
        actual_need = int(used_s) + int(need_s)
        over = actual_need - int(cap_s)
    else:
        m = _OUTPUT_NEED_RE.search(msg)
        if m:
            task_id, need_s, free_s, cap_s = m.groups()
            over = int(need_s) - int(free_s)
            actual_need = int(cap_s) + over
        else:
            m = _MISSING_INPUTS_RE.search(msg)
            if not m or bare.device_capacity is None:
                return None
            task_id, inputs_s = m.groups()
            missing = re.findall(r"'([^']+)'", inputs_s)
            over = sum(facts.sizes.get(oid, 0) for oid in missing)
            if over <= 0:
                return None
            actual_need = bare.device_capacity + over

    task_idx = next((i for i, task in enumerate(bare.tasks) if task.id == task_id), None)
    if task_idx is None:
        return None
    arr_idx = task_idx  # boundary (task_idx - 1) stored as +1.
    if not (0 <= arr_idx <= facts.n):
        return None
    return arr_idx, actual_need, max(1, over)


def apply_pressurefit_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
    refinement_iters: int = 0,
) -> TaskChain:
    """Return an annotated chain using the standalone PressureFit policy."""
    del refinement_iters  # kept for API symmetry; this policy does not search.
    if device_capacity is not None:
        bare = replace(bare, device_capacity=device_capacity)

    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)
    initial_device = _pressure_initial_placement(
        bare, bare.device_capacity, sizes, uses_by_task,
    )
    facts = _build_facts(bare)
    base_intervals = _initial_residency(facts, initial_device)

    def verify_candidate_plan(
        *,
        pack_h2d: bool,
        extend_h2d: bool = False,
        respect_interval_start: bool = False,
        latest_h2d: bool = False,
        protected_initial: set[str] | None = None,
        reserve_pressure: int = 0,
        interval_seed: dict[str, list[tuple[int, int]]] | None = None,
    ) -> tuple[int, TaskChain]:
        if protected_initial is None:
            protected_initial = set()
        if protected_initial:
            all_initial_host = {oid for oid in facts.host_ids if facts.uses.get(oid)}
            intervals = _initial_residency(facts, all_initial_host)
        elif interval_seed is not None:
            intervals = {oid: list(ivs) for oid, ivs in interval_seed.items()}
        else:
            intervals = {oid: list(ivs) for oid, ivs in base_intervals.items()}
        extra_pressure = [reserve_pressure] * (facts.n + 1)
        _reduce_to_fit(
            facts, intervals, bare.device_capacity, extra_pressure,
            protected_initial=protected_initial,
        )
        if extend_h2d:
            _extend_h2d_lead_time(
                facts, intervals, bare.device_capacity, bare.bandwidth_h2d,
                extra_pressure,
            )

        for _ in range(12):
            annotated = _emit_chain(
                bare, facts, intervals,
                pack_h2d=pack_h2d,
                respect_interval_start=respect_interval_start,
                latest_h2d=latest_h2d,
            )
            try:
                log = simulator_run(annotated, snapshots=False)
                return max(iv.end for iv in log.task_intervals), annotated
            except ValueError as e:
                physical = _physical_pressure_from_error(str(e), facts, bare)
                if physical is None or bare.device_capacity is None:
                    raise
                idx, actual_need, over = physical
                model_total = _modeled_boundary_need(facts, intervals, idx)
                required_extra = actual_need - model_total
                if required_extra <= extra_pressure[idx]:
                    required_extra = extra_pressure[idx] + over
                extra_pressure[idx] = max(reserve_pressure, 1, required_extra)
                _reduce_to_fit(
                    facts, intervals, bare.device_capacity, extra_pressure,
                    protected_initial=protected_initial,
                )
                if extend_h2d:
                    _extend_h2d_lead_time(
                        facts, intervals, bare.device_capacity, bare.bandwidth_h2d,
                        extra_pressure,
                    )

        annotated = _emit_chain(
            bare, facts, intervals,
            pack_h2d=pack_h2d,
            respect_interval_start=respect_interval_start,
            latest_h2d=latest_h2d,
        )
        log = simulator_run(annotated, snapshots=False)
        return max(iv.end for iv in log.task_intervals), annotated

    results: list[tuple[int, TaskChain]] = []
    first_error: Exception | None = None
    base_candidate_specs = (
        {"pack_h2d": True},
        {"pack_h2d": False},
        {
            "pack_h2d": False,
            "extend_h2d": True,
            "respect_interval_start": True,
        },
        {
            "pack_h2d": True,
            "latest_h2d": True,
        },
    )
    for candidate_spec in base_candidate_specs:
        try:
            results.append(verify_candidate_plan(**candidate_spec))
        except Exception as e:
            if first_error is None:
                first_error = e

    reserve_pressure = max(facts.next_outputs) if bare.device_capacity is not None else 0
    if reserve_pressure > 0:
        try:
            results.append(verify_candidate_plan(
                pack_h2d=True,
                reserve_pressure=reserve_pressure,
            ))
        except Exception as e:
            if first_error is None:
                first_error = e

    if bare.device_capacity is not None:
        cold_admission_cap = bare.device_capacity // 2
        if cold_admission_cap > 0:
            try:
                cold_initial = _pressure_initial_placement(
                    bare, cold_admission_cap, sizes, uses_by_task,
                )
                if cold_initial != initial_device:
                    results.append(verify_candidate_plan(
                        pack_h2d=True,
                        interval_seed=_initial_residency(facts, cold_initial),
                    ))
            except Exception as e:
                if first_error is None:
                    first_error = e

    protected_sets = _initial_protection_sets(
        facts, bare.device_capacity, bare.bandwidth_h2d,
    )
    if _use_fast_portfolio(facts):
        protected_sets = []

    seen_protected_sets: set[frozenset[str]] = set()
    max_source_object_size = max(
        (facts.sizes[oid] for oid in facts.host_ids if facts.uses.get(oid)),
        default=0,
    )
    for protected_idx, protected in enumerate(protected_sets):
        frozen = frozenset(protected)
        if not frozen or frozen in seen_protected_sets:
            continue
        seen_protected_sets.add(frozen)
        protected_bytes = sum(facts.sizes[oid] for oid in protected)
        if protected_idx == 0 or protected_bytes <= max_source_object_size:
            try:
                results.append(verify_candidate_plan(
                    pack_h2d=True,
                    extend_h2d=True,
                    protected_initial=protected,
                ))
            except Exception as e:
                if first_error is None:
                    first_error = e
        try:
            results.append(verify_candidate_plan(
                pack_h2d=False,
                extend_h2d=True,
                protected_initial=protected,
            ))
        except Exception as e:
            if first_error is None:
                first_error = e
        try:
            results.append(verify_candidate_plan(
                pack_h2d=False,
                extend_h2d=True,
                respect_interval_start=True,
                protected_initial=protected,
            ))
        except Exception as e:
            if first_error is None:
                first_error = e

    if not results:
        assert first_error is not None
        raise first_error
    results.sort(key=lambda x: x[0])
    return results[0][1]


def _use_fast_portfolio(facts: _Facts) -> bool:
    """Use a smaller candidate portfolio for very long chains.

    The full portfolio is valuable for short canonical sweeps, where trying a
    dozen initial-protection frontiers can pick up small wins. Repeated
    training-step chains multiply the task/object count and make each frontier
    much more expensive. Keep the full portfolio for small research sweeps and
    use the bounded portfolio for interactive-scale chains.
    """
    return facts.n > 256 or len(facts.sizes) > 512
