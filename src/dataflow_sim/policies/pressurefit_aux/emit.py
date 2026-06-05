"""Interval-to-trigger emission for PressureFit."""
from __future__ import annotations

from dataclasses import replace

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _fire_task_for_interval,
)
from dataflow_sim.policies.pressurefit_aux.inbound_schedules import (
    _PrefetchJob,
    _assign_prefetch_jobs,
    _prefetch_fire_task,
    _prefetch_job,
)
from dataflow_sim.core.schema import Object, Task, TaskChain, TransferTrigger


def _emit_chain(
    bare: TaskChain,
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    *,
    pack_inbound: bool = True,
    respect_interval_start: bool = False,
    latest_inbound: bool = False,
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
                if job is None or not pack_inbound:
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
        latest_only=latest_inbound,
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
