"""Candidate portfolio construction for PressureFit."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from dataflow_sim.policy._common import _UseEvent
from dataflow_sim.policy.pressurefit_aux.core import (
    _Facts,
    _anchors,
    _build_facts,
    _fire_task_for_interval,
    _first_use_in_interval,
    _transfer_time,
)
from dataflow_sim.policy.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.schema import TaskChain

PressureFitPortfolioMode = Literal["auto", "fast", "full"]


@dataclass(frozen=True)
class _CandidateSpec:
    """One bounded alternative passed through the shared PressureFit pipeline."""
    name: str
    family: str
    seed_key: str = "base"
    seed: str = "base"
    pack_h2d: bool = False
    extend_h2d: bool = False
    respect_interval_start: bool = False
    latest_h2d: bool = False
    reserve_pressure: int = 0
    protected_initial: frozenset[str] = frozenset()
    skip_reason: str | None = None
    pre_error: Exception | None = None
    fallback_only: bool = False


@dataclass(frozen=True)
class _PortfolioPlan:
    requested_mode: PressureFitPortfolioMode
    effective_mode: str
    fast_portfolio: bool
    specs: list[_CandidateSpec]


@dataclass(frozen=True)
class _InitialProtectionJob:
    oid: str
    release_t: int
    deadline: int
    tau: int
    first_use: int
    size: int
    residency_cost: int


def _build_candidate_portfolio(
    bare: TaskChain,
    facts: _Facts,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    initial_device: set[str],
    seeds: dict[str, dict[str, list[tuple[int, int]]]],
    requested_mode: PressureFitPortfolioMode,
) -> _PortfolioPlan:
    """Construct the bounded candidate list."""
    effective_mode, fast_portfolio, minimal_fast = _resolve_portfolio_mode(
        facts, requested_mode,
    )
    specs: list[_CandidateSpec] = []

    _add_base_specs(specs, minimal_fast)
    _add_reserve_spec(specs, facts, bare.device_capacity, minimal_fast)
    _add_cold_admission_spec(
        specs, bare, facts, sizes, uses_by_task, initial_device, seeds,
        minimal_fast,
    )
    _add_initial_protection_specs(
        specs, facts, bare.device_capacity, bare.bandwidth_h2d,
        fast_portfolio, effective_mode,
    )

    return _PortfolioPlan(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        fast_portfolio=fast_portfolio,
        specs=specs,
    )


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


def _initial_residency(
    facts: _Facts,
    initial_device: set[str],
) -> dict[str, list[tuple[int, int]]]:
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


def _trim_source_idle_gaps(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    bw_h2d: int | None,
    bw_d2h: int | None,
) -> dict[str, list[tuple[int, int]]]:
    """Build the source-gap seed by splitting long source-state no-use gaps."""
    out: dict[str, list[tuple[int, int]]] = {}
    source_ids = facts.host_ids | facts.device_ids
    for oid, ivs in intervals.items():
        if oid not in source_ids:
            out[oid] = list(ivs)
            continue
        anchors = _anchors(oid, facts)
        size = facts.sizes[oid]
        h2d_tau = _transfer_time(size, bw_h2d)
        new_ivs: list[tuple[int, int]] = []
        for a, b in sorted(ivs):
            anchors_in = [x for x in anchors if a <= x <= b]
            if len(anchors_in) < 2:
                new_ivs.append((a, b))
                continue

            start = a
            for left_anchor, right_anchor in zip(anchors_in, anchors_in[1:]):
                if right_anchor - left_anchor <= 1:
                    continue
                first_use = _first_use_in_interval(oid, right_anchor, b, facts)
                left_fire = _fire_task_for_interval(oid, start, left_anchor, facts)
                if first_use is None or left_fire is None:
                    continue

                dirty = any(
                    start <= m - 1 <= left_anchor
                    for m in facts.mutators.get(oid, set())
                )
                needs_writeback = dirty or oid not in facts.host_ids
                if not needs_writeback:
                    continue
                d2h_tau = _transfer_time(size, bw_d2h)
                gap_time = facts.task_start[first_use] - facts.task_end[left_fire]
                if gap_time >= d2h_tau + h2d_tau:
                    new_ivs.append((start, left_anchor))
                    start = right_anchor
            new_ivs.append((start, b))
        out[oid] = new_ivs
    return out


def _copy_intervals(
    seed: dict[str, list[tuple[int, int]]],
) -> dict[str, list[tuple[int, int]]]:
    return {oid: list(ivs) for oid, ivs in seed.items()}


def _build_candidate_seeds(
    facts: _Facts,
    base_intervals: dict[str, list[tuple[int, int]]],
    bw_h2d: int | None,
    bw_d2h: int | None,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Build seed interval sets shared by candidate specs."""
    all_initial_host = {oid for oid in facts.host_ids if facts.uses.get(oid)}
    return {
        "base": base_intervals,
        "source-gap": _trim_source_idle_gaps(
            facts, base_intervals, bw_h2d, bw_d2h,
        ),
        "all-host": _initial_residency(facts, all_initial_host),
    }


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
    """Return prefix sets at H2D-work frontier points."""
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


def _resolve_portfolio_mode(
    facts: _Facts,
    requested_mode: PressureFitPortfolioMode,
) -> tuple[str, bool, bool]:
    auto_would_use_fast = _use_fast_portfolio(facts)
    fast_requested = (
        requested_mode == "fast"
        or (requested_mode == "auto" and auto_would_use_fast)
    )
    minimal_fast = fast_requested and _use_minimal_fast_portfolio(facts)
    effective_mode = (
        "fast-minimal" if minimal_fast
        else "fast" if fast_requested
        else "full"
    )
    return effective_mode, effective_mode != "full", minimal_fast


def _skip_spec(
    name: str,
    family: str,
    reason: str,
    *,
    seed: str = "base",
    seed_key: str = "base",
    **kwargs,
) -> _CandidateSpec:
    return _CandidateSpec(
        name=name,
        family=family,
        seed=seed,
        seed_key=seed_key,
        skip_reason=reason,
        **kwargs,
    )


def _add_base_specs(specs: list[_CandidateSpec], minimal_fast: bool) -> None:
    if minimal_fast:
        specs.extend([
            _CandidateSpec("base-unpacked", "base", pack_h2d=False),
            _CandidateSpec(
                "source-gap-unpacked",
                "source-gap-trim",
                seed_key="source-gap",
                seed="source-gap-trim",
                pack_h2d=False,
            ),
            _CandidateSpec(
                "base-latest-h2d",
                "base",
                pack_h2d=True,
                latest_h2d=True,
                fallback_only=True,
            ),
            _skip_spec(
                "base-packed-fifo",
                "base",
                "fast-minimal portfolio skips secondary H2D schedules",
                pack_h2d=True,
            ),
            _skip_spec(
                "base-interval-entry",
                "base",
                "fast-minimal portfolio skips secondary H2D schedules",
                extend_h2d=True,
                respect_interval_start=True,
            ),
        ])
        return

    specs.extend([
        _CandidateSpec("base-packed-fifo", "base", pack_h2d=True),
        _CandidateSpec("base-unpacked", "base", pack_h2d=False),
        _CandidateSpec(
            "source-gap-unpacked",
            "source-gap-trim",
            seed_key="source-gap",
            seed="source-gap-trim",
            pack_h2d=False,
        ),
        _CandidateSpec(
            "base-interval-entry",
            "base",
            pack_h2d=False,
            extend_h2d=True,
            respect_interval_start=True,
        ),
        _CandidateSpec(
            "base-latest-h2d",
            "base",
            pack_h2d=True,
            latest_h2d=True,
        ),
    ])


def _add_reserve_spec(
    specs: list[_CandidateSpec],
    facts: _Facts,
    device_capacity: int | None,
    minimal_fast: bool,
) -> None:
    reserve_pressure = max(facts.next_outputs) if device_capacity is not None else 0
    if minimal_fast:
        specs.append(_skip_spec(
            "reserve-next-output",
            "slack-reserve",
            "fast-minimal portfolio skips secondary pressure seeds",
        ))
    elif reserve_pressure > 0:
        specs.append(_CandidateSpec(
            "reserve-next-output",
            "slack-reserve",
            pack_h2d=True,
            reserve_pressure=reserve_pressure,
        ))
    else:
        specs.append(_skip_spec(
            "reserve-next-output",
            "slack-reserve",
            "no finite-cap next-output reserve pressure",
        ))


def _add_cold_admission_spec(
    specs: list[_CandidateSpec],
    bare: TaskChain,
    facts: _Facts,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    initial_device: set[str],
    seeds: dict[str, dict[str, list[tuple[int, int]]]],
    minimal_fast: bool,
) -> None:
    if minimal_fast:
        specs.append(_skip_spec(
            "cold-admission-packed",
            "cold-admission",
            "fast-minimal portfolio skips secondary pressure seeds",
            seed="cold-admission",
            seed_key="cold-admission",
        ))
        return
    if bare.device_capacity is None:
        specs.append(_skip_spec(
            "cold-admission-packed",
            "cold-admission",
            "unlimited capacity",
            seed="cold-admission",
            seed_key="cold-admission",
        ))
        return

    cold_admission_cap = bare.device_capacity // 2
    if cold_admission_cap <= 0:
        specs.append(_skip_spec(
            "cold-admission-packed",
            "cold-admission",
            "cold admission cap is zero",
            seed="cold-admission",
            seed_key="cold-admission",
        ))
        return

    try:
        cold_initial = _pressure_initial_placement(
            bare, cold_admission_cap, sizes, uses_by_task,
        )
        if cold_initial == initial_device:
            specs.append(_skip_spec(
                "cold-admission-packed",
                "cold-admission",
                "cold initial set equals base initial set",
                seed="cold-admission",
                seed_key="cold-admission",
            ))
        else:
            seeds["cold-admission"] = _initial_residency(facts, cold_initial)
            specs.append(_CandidateSpec(
                "cold-admission-packed",
                "cold-admission",
                seed_key="cold-admission",
                seed="cold-admission",
                pack_h2d=True,
            ))
    except Exception as e:
        specs.append(_CandidateSpec(
            "cold-admission-packed",
            "cold-admission",
            seed_key="cold-admission",
            seed="cold-admission",
            pack_h2d=True,
            pre_error=e,
        ))


def _add_initial_protection_specs(
    specs: list[_CandidateSpec],
    facts: _Facts,
    device_capacity: int | None,
    bw_h2d: int | None,
    fast_portfolio: bool,
    effective_mode: str,
) -> None:
    protected_sets = _initial_protection_sets(facts, device_capacity, bw_h2d)
    if fast_portfolio:
        if protected_sets:
            specs.append(_skip_spec(
                "initial-protection-frontiers",
                "initial-protection",
                f"{effective_mode} portfolio omits {len(protected_sets)} frontier sets",
                seed="protected-initial",
                seed_key="all-host",
            ))
        return

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
        common = {
            "seed_key": "all-host",
            "seed": "protected-initial",
            "protected_initial": frozen,
            "extend_h2d": True,
        }
        if protected_idx == 0 or protected_bytes <= max_source_object_size:
            specs.append(_CandidateSpec(
                f"protected-{protected_idx}-packed-extended",
                "initial-protection",
                pack_h2d=True,
                **common,
            ))
        else:
            specs.append(_skip_spec(
                f"protected-{protected_idx}-packed-extended",
                "initial-protection",
                "protected set exceeds max single source-object size",
                pack_h2d=True,
                **common,
            ))
        specs.append(_CandidateSpec(
            f"protected-{protected_idx}-unpacked-extended",
            "initial-protection",
            pack_h2d=False,
            **common,
        ))
        specs.append(_CandidateSpec(
            f"protected-{protected_idx}-interval-entry",
            "initial-protection",
            pack_h2d=False,
            respect_interval_start=True,
            **common,
        ))


def _use_fast_portfolio(facts: _Facts) -> bool:
    """Use a smaller candidate portfolio for very long chains."""
    return facts.n > 256 or len(facts.sizes) > 512


def _use_minimal_fast_portfolio(facts: _Facts) -> bool:
    """Use the primary fast candidate for very large interactive chains."""
    return facts.n > 4096 or len(facts.sizes) > 4096
