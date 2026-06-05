"""Initial-residency protection frontiers for PressureFit."""
from __future__ import annotations

import math

from dataflow_sim.policies.pressurefit_aux.core import _Facts
from dataflow_sim.policies.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.policies.pressurefit_aux.seeds import _initial_residency
from dataflow_sim.policies.pressurefit_aux.types import (
    _InitialProtectionJob,
    _IntervalSet,
)


def _initial_protection_sets(
    facts: _Facts,
    cap: int | None,
    inbound_bw: int | None,
) -> list[set[str]]:
    jobs = _initial_protection_jobs_from_probe(facts, cap, inbound_bw)
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
        _source_initial_protection_jobs(facts, inbound_bw, clean_only=True),
        key=lambda job: (-job.first_use, -job.size, job.oid),
    )
    add_many(_initial_protection_prefix_sets(
        clean_tail_jobs,
        headroom,
        group_key=lambda job: job.first_use,
    ))

    tail_jobs = sorted(
        _source_initial_protection_jobs(facts, inbound_bw, clean_only=False),
        key=lambda job: (-job.first_use, -job.size, job.oid),
    )
    add_many(_initial_protection_prefix_sets(
        tail_jobs,
        headroom,
        group_key=lambda job: job.first_use,
    ))

    return out


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


def _initial_protection_jobs_from_probe(
    facts: _Facts,
    cap: int | None,
    inbound_bw: int | None,
) -> list[_InitialProtectionJob]:
    """Build inbound jobs created when initial host residency is cut by pressure."""
    if cap is None or inbound_bw is None or inbound_bw <= 0 or facts.n == 0:
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
        tau = max(1, math.ceil(facts.sizes[oid] / inbound_bw))
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


def _select_initial_protection_set(
    jobs: list[_InitialProtectionJob],
    headroom: int,
) -> set[str]:
    """Protect enough initial residency to remove predicted inbound deadline misses."""
    if headroom <= 0:
        return set()

    protected: set[str] = set()
    remaining = headroom
    for _ in range(len(jobs)):
        misses = _inbound_deadline_misses(jobs, protected)
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


def _pressure_probe_intervals(
    facts: _Facts,
    cap: int | None,
) -> _IntervalSet:
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


def _source_initial_protection_jobs(
    facts: _Facts,
    inbound_bw: int | None,
    *,
    clean_only: bool,
) -> list[_InitialProtectionJob]:
    """Build source-object jobs for initial-protection frontier candidates."""
    if inbound_bw is None or inbound_bw <= 0 or facts.n == 0:
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
        tau = max(1, math.ceil(facts.sizes[oid] / inbound_bw))
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


def _inbound_deadline_misses(
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


def _initial_protection_prefix_sets(
    jobs: list[_InitialProtectionJob],
    headroom: int,
    *,
    group_key,
) -> list[set[str]]:
    """Return prefix sets at inbound-work frontier points."""
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
