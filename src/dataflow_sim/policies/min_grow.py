"""min_grow auto-policy: MAX→shrink with simulator-as-oracle.

See docs/policy/other_policies/min-grow.md for the design doc. Two-phase pipeline:

  Phase A — Plan search: start at MAX (every object resident from its
            earliest legal boundary to its last use), then iteratively
            SHRINK (introduce releases/offloads/prefetches) until the
            simulator reports peak fast memory <= cap. Each candidate
            shrink is scored by full simulator replay (makespan + peak).

  Phase B — Schedule derivation: given a finalized plan, deterministically
            emit releases_after / offload_after / prefetch_after triggers
            and an updated initial_memory.

For unlimited cap, MAX is immediately optimal (zero search). For tight
cap, beam search shrinks until cap-feasible; once feasible, prefers lowest
makespan.

The ONLY infeasibility min_grow raises is when even the MIN plan (forced
residency at use) exceeds fast_memory_capacity at some boundary — the chain
literally cannot run.

Boundary convention (matching the dataflow_sim simulator convention): boundary k = snapshot AFTER task k's
triggers fire. Boundary -1 = initial state. Plan intervals are HALF-OPEN
[a, b) — object in pool at boundaries a, a+1, ..., b-1; gone at b.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

from dataflow_sim.core.schema import Object, Task, TaskChain, TransferTrigger
from dataflow_sim.engine.simulator import run as simulator_run


# ============================================================================
# Plan data model (docs/policy/other_policies/min-grow.md §4)
# ============================================================================

@dataclass(frozen=True)
class Interval:
    """Half-open residency interval. Object is in pool at boundaries [a, b)."""
    a: int
    b: int


@dataclass(frozen=True)
class Plan:
    intervals: dict[str, tuple[Interval, ...]]


# ============================================================================
# Workload-derived facts
# ============================================================================

@dataclass(frozen=True)
class _Facts:
    n: int
    sizes: dict[str, int]
    producer: dict[str, int]                  # -1 for initial_memory entries
    inputs_by_task: tuple[frozenset[str], ...]
    outputs_by_task: tuple[frozenset[str], ...]
    mutators_by_task: tuple[frozenset[str], ...]
    uses_by_obj: dict[str, tuple[int, ...]]
    mutator_tasks_of: dict[str, tuple[int, ...]]
    backing_init_ids: frozenset[str]
    compute_init_ids: frozenset[str]
    runtimes: tuple[int, ...]
    next_outputs_size: tuple[int, ...]        # bytes reserved by task k at start (length n+1)
    cap: int | None
    bw_from_slow: int | None
    bw_to_slow: int | None


def _build_facts(bare: TaskChain) -> _Facts:
    n = len(bare.tasks)
    sizes: dict[str, int] = {o.id: o.size for o in bare.initial_memory}
    for t in bare.tasks:
        for a in t.outputs:
            sizes[a.id] = a.size

    producer: dict[str, int] = {o.id: -1 for o in bare.initial_memory}
    for i, t in enumerate(bare.tasks):
        for a in t.outputs:
            producer[a.id] = i

    inputs_by_task = tuple(frozenset(t.inputs) for t in bare.tasks)
    outputs_by_task = tuple(frozenset(a.id for a in t.outputs) for t in bare.tasks)
    mutators_by_task = tuple(frozenset(t.mutates_inputs) for t in bare.tasks)

    uses: dict[str, list[int]] = defaultdict(list)
    muts: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(bare.tasks):
        for inp in t.inputs:
            uses[inp].append(i)
        for m in t.mutates_inputs:
            muts[m].append(i)

    outputs_size = [sum(a.size for a in t.outputs) for t in bare.tasks]
    next_outputs = tuple(outputs_size + [0])

    return _Facts(
        n=n,
        sizes=sizes,
        producer=producer,
        inputs_by_task=inputs_by_task,
        outputs_by_task=outputs_by_task,
        mutators_by_task=mutators_by_task,
        uses_by_obj={k: tuple(sorted(v)) for k, v in uses.items()},
        mutator_tasks_of={k: tuple(sorted(v)) for k, v in muts.items()},
        backing_init_ids=frozenset(o.id for o in bare.initial_memory if o.location == "backing"),
        compute_init_ids=frozenset(o.id for o in bare.initial_memory if o.location == "fast"),
        runtimes=tuple(t.runtime for t in bare.tasks),
        next_outputs_size=next_outputs,
        cap=bare.fast_memory_capacity,
        bw_from_slow=bare.bandwidth_from_slow,
        bw_to_slow=bare.bandwidth_to_slow,
    )


# ============================================================================
# MIN plan (used only as infeasibility check and worst-case fallback)
# ============================================================================

def _min_plan(facts: _Facts) -> Plan:
    """Forced residency only.

    Per object reference:
      - input at task i  → interval [i-1, i)
      - output at task i → interval [i, i+1)

    Adjacent / overlapping intervals are coalesced. Compute-init objects
    get an extra [-1, 0) so the plan reflects their initial residency.
    """
    raw: dict[str, list[Interval]] = defaultdict(list)
    for i in range(facts.n):
        for inp in facts.inputs_by_task[i]:
            raw[inp].append(Interval(i - 1, i))
        for out in facts.outputs_by_task[i]:
            raw[out].append(Interval(i, i + 1))

    for oid in facts.compute_init_ids:
        raw[oid].append(Interval(-1, 0))

    out: dict[str, tuple[Interval, ...]] = {}
    for oid, ivs in raw.items():
        ivs.sort(key=lambda iv: iv.a)
        merged: list[Interval] = [ivs[0]]
        for iv in ivs[1:]:
            if iv.a <= merged[-1].b:
                merged[-1] = Interval(merged[-1].a, max(merged[-1].b, iv.b))
            else:
                merged.append(iv)
        out[oid] = tuple(merged)
    return Plan(intervals=out)


# ============================================================================
# MAX plan — min_grow's starting point
# ============================================================================

def _max_plan(facts: _Facts) -> Plan:
    """Build the MAX plan: each object kept resident continuously from its
    earliest legal boundary to its last use's exit boundary.

    For each object o:
      - earliest_a = -1 if o is backing-init or compute-init (pre-placed)
                   = producer(o) otherwise
      - last_b = max(b across all forced intervals in MIN) — i.e., the
                 boundary AT WHICH the object exits (= last_use task index)
      - MAX interval = (earliest_a, last_b)  (single interval)

    Mutated backing-init objects (e.g., dW_i) get exit at the mutator task,
    so derive_schedule emits the offload trigger right after the mutation
    (not delayed to end-of-step). The user's "release/offload ASAP after
    last use" intuition is naturally satisfied by this construction.

    Objects with zero uses get [-1, 0) (backing/compute-init) or are omitted
    entirely (produced and never used — they'd be dead outputs).
    """
    min_p = _min_plan(facts)
    intervals: dict[str, tuple[Interval, ...]] = {}
    for oid, ivs in min_p.intervals.items():
        last_b = max(iv.b for iv in ivs)
        is_pre_placeable = (oid in facts.backing_init_ids) or (oid in facts.compute_init_ids)
        if is_pre_placeable:
            intervals[oid] = (Interval(-1, last_b),)
        else:
            prod = facts.producer.get(oid, -1)
            # First MIN interval's `a` is the earliest constraint for produced objects
            first_a = ivs[0].a
            intervals[oid] = (Interval(max(prod, first_a), last_b),)
    return Plan(intervals=intervals)


# ============================================================================
# Static cap pre-filter (still useful as cheap rejection)
# ============================================================================

def _boundary_pool(plan: Plan, facts: _Facts) -> list[int]:
    """pool[k] for k in [0, n+1], where pool[k] = bytes resident at boundary k-1."""
    pool = [0] * (facts.n + 1)
    for oid, ivs in plan.intervals.items():
        sz = facts.sizes[oid]
        for iv in ivs:
            lo = iv.a + 1
            hi = iv.b + 1
            for k in range(lo, hi):
                pool[k] += sz
    return pool


def _static_peak(plan: Plan, facts: _Facts) -> int:
    """Max over all boundaries of (pool[b] + next_task_output_reservation)."""
    pool = _boundary_pool(plan, facts)
    worst = 0
    for b in range(-1, facts.n):
        idx = b + 1
        next_task = b + 1
        next_out = facts.next_outputs_size[next_task] if next_task < facts.n else 0
        worst = max(worst, pool[idx] + next_out)
    return worst


def _respects_static_cap(plan: Plan, facts: _Facts) -> bool:
    if facts.cap is None:
        return True
    return _static_peak(plan, facts) <= facts.cap


# ============================================================================
# Reduction enumeration
# ============================================================================

def _enumerate_extensions(plan: Plan, facts: _Facts) -> list[tuple[Plan, str, int]]:
    """Inverse of reductions: grow residency. Returns (plan, oid, first_use)
    where first_use is the earliest task that uses the object (used for
    Belady-style ranking — pre-placing low-first-use objects helps makespan
    most, since their from_slow can't hide behind compute).
    """
    out: list[tuple[Plan, str, int]] = []
    for oid, ivs in plan.intervals.items():
        prod = facts.producer.get(oid, -1)
        is_pre_placeable = (oid in facts.backing_init_ids) or (oid in facts.compute_init_ids)
        uses = facts.uses_by_obj.get(oid, ())
        first_use = uses[0] if uses else facts.n + 1

        for idx, iv in enumerate(ivs):
            prev_b = ivs[idx - 1].b if idx > 0 else -1
            next_a = ivs[idx + 1].a if idx + 1 < len(ivs) else facts.n + 1

            if idx == 0:
                earliest = -1 if is_pre_placeable else max(prod, 0) if prod >= 0 else 0
            else:
                earliest = prev_b

            latest_b = next_a

            if iv.a > earliest:
                new_a = earliest
                new_ivs = list(ivs)
                new_ivs[idx] = Interval(new_a, iv.b)
                if idx > 0 and new_a <= prev_b:
                    merged = Interval(ivs[idx - 1].a, iv.b)
                    new_ivs = list(ivs[:idx - 1]) + [merged] + list(ivs[idx + 1:])
                out.append((_with_intervals(plan, oid, tuple(new_ivs)), oid, first_use))

            if iv.b < latest_b and iv.b < facts.n + 1:
                new_b = min(latest_b, facts.n)
                new_ivs = list(ivs)
                new_ivs[idx] = Interval(iv.a, new_b)
                if idx + 1 < len(ivs) and new_b >= next_a:
                    merged = Interval(iv.a, ivs[idx + 1].b)
                    new_ivs = list(ivs[:idx]) + [merged] + list(ivs[idx + 2:])
                out.append((_with_intervals(plan, oid, tuple(new_ivs)), oid, first_use))

    seen: set[tuple] = set()
    uniq: list[tuple[Plan, str, int]] = []
    for p, oid, fu in out:
        key = _plan_key(p)
        if key not in seen:
            seen.add(key)
            uniq.append((p, oid, fu))
    return uniq


def _enumerate_reductions(plan: Plan, facts: _Facts) -> list[tuple[Plan, str, int]]:
    """Generate candidate REDUCTIONS of `plan`. Each result is
    (new_plan, modified_obj_id, next_use_time) where next_use_time is the
    earliest task index that uses the object after the reduction takes
    effect — used for Belady-style ranking (higher = safer to evict,
    because more compute can hide the re-prefetch).

    Reduction kinds:
      - Shrink leftward to first forced boundary (un-pre-place / drop prefix)
      - Shrink rightward past last forced boundary (release earlier)
      - Split at any free gap between forced uses (introduces an
        offload+re-prefetch pair, freeing the boundaries in between)
      - Remove interval entirely if it has no forced uses (rare)
    """
    out: list[tuple[Plan, str, int]] = []
    for oid, ivs in plan.intervals.items():
        for idx, iv in enumerate(ivs):
            forced = _forced_boundaries_in(oid, iv, facts)
            if not forced:
                new_ivs = list(ivs[:idx]) + list(ivs[idx + 1:])
                new_intervals = dict(plan.intervals)
                if new_ivs:
                    new_intervals[oid] = tuple(new_ivs)
                else:
                    del new_intervals[oid]
                out.append((Plan(intervals=new_intervals), oid, facts.n + 1))
                continue

            min_forced = forced[0]
            max_forced = forced[-1]

            # Shrink leftward: new a = min_forced
            if iv.a < min_forced:
                new_ivs = list(ivs)
                new_ivs[idx] = Interval(min_forced, iv.b)
                # next use after shrink = the first task that uses o at boundary >= min_forced
                fut = _first_use_after(oid, min_forced, facts)
                out.append((_with_intervals(plan, oid, tuple(new_ivs)), oid, fut))

            # Shrink rightward
            new_b_right = max_forced + 1
            if new_b_right < iv.b:
                new_ivs = list(ivs)
                new_ivs[idx] = Interval(iv.a, new_b_right)
                # next use after rightward shrink = first use AT OR AFTER new_b
                fut = _first_use_after(oid, new_b_right, facts)
                out.append((_with_intervals(plan, oid, tuple(new_ivs)), oid, fut))

            # Split at each free gap
            for i in range(len(forced) - 1):
                k1 = forced[i]
                k2 = forced[i + 1]
                if k2 - k1 < 2:
                    continue
                left_iv = Interval(iv.a, k1 + 1)
                right_iv = Interval(k2, iv.b)
                new_ivs = list(ivs[:idx]) + [left_iv, right_iv] + list(ivs[idx + 1:])
                # Belady score: the gap's "next use" = k2 (= boundary of next use)
                fut = _first_use_after(oid, k2, facts)
                out.append((_with_intervals(plan, oid, tuple(new_ivs)), oid, fut))

    seen: set[tuple] = set()
    uniq: list[tuple[Plan, str, int]] = []
    for p, oid, fut in out:
        key = _plan_key(p)
        if key not in seen:
            seen.add(key)
            uniq.append((p, oid, fut))
    return uniq


def _first_use_after(oid: str, boundary: int, facts: _Facts) -> int:
    """First task index t such that t-1 >= boundary AND t uses oid (as input
    or output). Returns facts.n+1 if no such use.
    """
    uses = facts.uses_by_obj.get(oid, ())
    for u in uses:
        if u - 1 >= boundary:
            return u
    prod = facts.producer.get(oid, -1)
    if prod >= 0 and prod >= boundary:
        return prod
    return facts.n + 1


def _forced_boundaries_in(oid: str, iv: Interval, facts: _Facts) -> list[int]:
    """Boundaries within [iv.a, iv.b) that MUST stay covered for the
    forced-residency invariant. A boundary k is forced if some task t
    uses o at t-1 == k or t == k AND iv is the interval covering it.

    Simplification: returns all forced boundaries from uses + outputs
    that fall within [iv.a, iv.b).
    """
    forced: set[int] = set()
    # Input uses
    for t in facts.uses_by_obj.get(oid, ()):
        # Task t input requires boundary t-1 (just before task t)
        if iv.a <= t - 1 < iv.b:
            forced.add(t - 1)
    # Production: produces at boundary = producer (== producer task idx)
    prod = facts.producer.get(oid, -1)
    if prod >= 0 and iv.a <= prod < iv.b:
        forced.add(prod)
    # Compute-init: initial state IS a forced boundary if iv covers -1.
    # (No, actually if compute_init, boundary -1 has the object regardless
    # of plan; but if we shrink to NOT include -1, we'd need an offload
    # at task 0 and re-prefetch later. That's allowed for min_grow search.)
    # For simplicity treat -1 as forced for compute-init iff first input use:
    if oid in facts.compute_init_ids and iv.a <= -1 < iv.b:
        uses = facts.uses_by_obj.get(oid, ())
        if uses and uses[0] == 0:
            # First use is task 0 — we need -1 to be covered (no time to prefetch).
            forced.add(-1)
    return sorted(forced)


def _with_intervals(plan: Plan, oid: str, new_ivs: tuple[Interval, ...]) -> Plan:
    new = dict(plan.intervals)
    new[oid] = new_ivs
    return Plan(intervals=new)


def _plan_key(plan: Plan) -> tuple:
    return tuple(sorted(plan.intervals.items()))


# ============================================================================
# Analytic ranking (used to prioritize candidates for simulator scoring)
# ============================================================================

def _analytic_cost(plan: Plan, facts: _Facts) -> tuple[int, int]:
    """Return (from_slow_bytes + to_slow_bytes, static_peak) for ranking.

    Lower from_slow+to_slow = fewer transfers = better for makespan (loosely).
    Lower static_peak = closer to cap-feasibility.
    """
    from_slow_bytes = 0
    to_slow_bytes = 0
    for oid, ivs in plan.intervals.items():
        prod = facts.producer.get(oid, -1)
        is_backing_src = oid in facts.backing_init_ids
        mut_tasks = facts.mutator_tasks_of.get(oid, ())
        for idx, iv in enumerate(ivs):
            if iv.a > -1 and iv.a != prod:
                from_slow_bytes += facts.sizes[oid]
            mutated_in = any(iv.a < mt <= iv.b for mt in mut_tasks)
            is_last = (idx == len(ivs) - 1)
            if iv.b < facts.n:
                if mutated_in:
                    to_slow_bytes += facts.sizes[oid]
                elif is_backing_src:
                    pass  # release
                elif is_last:
                    pass  # release (dead)
                else:
                    to_slow_bytes += facts.sizes[oid]  # preserve for re-prefetch
            else:
                if is_backing_src and any(iv.a < mt <= facts.n for mt in mut_tasks):
                    to_slow_bytes += facts.sizes[oid]
    static_peak = _static_peak(plan, facts)
    return from_slow_bytes + to_slow_bytes, static_peak


# ============================================================================
# Phase B — schedule derivation (unchanged from MIN→grow version)
# ============================================================================

def _derive_schedule(plan: Plan, bare: TaskChain, facts: _Facts) -> TaskChain:
    """Walk the plan and emit triggers; return an annotated TaskChain."""
    n = facts.n
    releases: list[list[str]] = [[] for _ in range(n)]
    offloads: list[list[str]] = [[] for _ in range(n)]
    prefetches: list[list[str]] = [[] for _ in range(n)]
    pre_placed: set[str] = set()

    prefetch_keys: list[dict[str, int]] = [dict() for _ in range(n)]
    offload_keys: list[dict[str, int]] = [dict() for _ in range(n)]

    # Tentative task start times (compute-only, no transfer stalls).
    # Used for smart prefetch placement to avoid attaching prefetches to
    # zero-runtime tasks that give no lead time.
    tentative_start = [0] * n
    t_now = 0
    for i in range(n):
        tentative_start[i] = t_now
        t_now += facts.runtimes[i]
    tentative_end = [tentative_start[i] + facts.runtimes[i] for i in range(n)]

    for oid, ivs in plan.intervals.items():
        prod = facts.producer.get(oid, -1)
        is_backing_src = oid in facts.backing_init_ids
        is_dev_src = oid in facts.compute_init_ids
        mut_tasks = facts.mutator_tasks_of.get(oid, ())

        for idx, iv in enumerate(ivs):
            # Entry
            if iv.a == -1:
                if is_backing_src:
                    pre_placed.add(oid)
            elif iv.a == prod:
                pass
            elif is_backing_src or is_dev_src:
                if 0 <= iv.a < n:
                    # Smart prefetch placement: find latest task t <= iv.a
                    # such that from_slow fired at t.end has enough time to complete
                    # before the consuming task. Avoids attaching to
                    # zero-runtime tasks that would cause guaranteed stalls.
                    fire_task = _smart_prefetch_task(
                        oid, iv, facts, tentative_start, tentative_end
                    )
                    prefetches[fire_task].append(oid)
                    prefetch_keys[fire_task][oid] = _first_use_in_interval(facts, oid, iv)
            else:
                # Produced object re-entry (after a split)
                if 0 <= iv.a < n:
                    fire_task = _smart_prefetch_task(
                        oid, iv, facts, tentative_start, tentative_end
                    )
                    prefetches[fire_task].append(oid)
                    prefetch_keys[fire_task][oid] = _first_use_in_interval(facts, oid, iv)

            # Exit: fire trigger right after the last use within (iv.a, iv.b],
            # not at iv.b. Avoids delaying offloads/releases past the last use.
            # E.g., for A_0 produced at f_0 with interval [0, 1) and no use in
            # (0, 1], trigger fires at task 0 (production task) so to_slow starts
            # immediately — not at task 1 which wastes a task's worth of time.
            if iv.b == n:
                continue  # end-of-step handled below
            mutated_in = any(iv.a < mt <= iv.b for mt in mut_tasks)
            # Find last use within (iv.a, iv.b]: input use OR production
            last_use_in_iv = iv.a  # default: production boundary (if produced)
            for u in facts.uses_by_obj.get(oid, ()):
                if iv.a < u <= iv.b and u > last_use_in_iv:
                    last_use_in_iv = u
            # Include mutation as a "use" — mutation must complete before offload
            for mt in mut_tasks:
                if iv.a < mt <= iv.b and mt > last_use_in_iv:
                    last_use_in_iv = mt
            exit_task = max(0, min(n - 1, last_use_in_iv))
            is_last = (idx == len(ivs) - 1)
            if mutated_in:
                offloads[exit_task].append(oid)
                offload_keys[exit_task][oid] = iv.a
            elif is_backing_src:
                releases[exit_task].append(oid)
            elif is_last:
                releases[exit_task].append(oid)
            else:
                offloads[exit_task].append(oid)
                offload_keys[exit_task][oid] = iv.a

    # End-of-step writebacks for mutated backing-source objects still resident at n
    for oid, ivs in plan.intervals.items():
        if oid not in facts.backing_init_ids:
            continue
        last = ivs[-1]
        if last.b != n:
            continue
        mut_tasks = facts.mutator_tasks_of.get(oid, ())
        ever_mutated = any(last.a < mt <= facts.n for mt in mut_tasks)
        if ever_mutated:
            offloads[n - 1].append(oid)
            offload_keys[n - 1][oid] = last.a

    # Elide same-task release+prefetch pairs of the same object: a release
    # followed by re-prefetch of the same id on the same task is wasteful
    # (the bytes are freed then immediately re-acquired via from_slow) — the only
    # effect is paying from_slow time for content that's already on backing. The net
    # pool effect is zero, so eliding both is strictly better. Same for
    # offload+prefetch pairs (offload then re-prefetch on same task — net no-op
    # on the backing copy but wastes a to_slow+from_slow round-trip).
    for i in range(n):
        rel_set = set(releases[i])
        off_set = set(offloads[i])
        pre_set = set(prefetches[i])
        wasteful_rel = rel_set & pre_set
        wasteful_off = off_set & pre_set
        wasteful = wasteful_rel | wasteful_off
        if wasteful:
            releases[i] = [o for o in releases[i] if o not in wasteful]
            offloads[i] = [o for o in offloads[i] if o not in wasteful]
            prefetches[i] = [o for o in prefetches[i] if o not in wasteful]

    for i in range(n):
        if prefetches[i]:
            unique = list(dict.fromkeys(prefetches[i]))
            unique.sort(key=lambda o: prefetch_keys[i].get(o, 0))
            prefetches[i] = unique
        if offloads[i]:
            unique = list(dict.fromkeys(offloads[i]))
            unique.sort(key=lambda o: offload_keys[i].get(o, 0))
            offloads[i] = unique
        if releases[i]:
            releases[i] = list(dict.fromkeys(releases[i]))

    backing_objs = {o.id: o for o in bare.initial_memory if o.location == "backing"}
    new_initial = list(bare.initial_memory)
    for oid in pre_placed:
        src = backing_objs[oid]
        new_initial.append(Object(id=src.id, size=src.size, location="fast", type=src.type))

    new_tasks: list[Task] = []
    for i, task in enumerate(bare.tasks):
        new_tasks.append(replace(
            task,
            releases_after=list(releases[i]),
            offload_after=[TransferTrigger(obj_id=o) for o in offloads[i]],
            prefetch_after=[TransferTrigger(obj_id=o) for o in prefetches[i]],
        ))
    return replace(bare, initial_memory=new_initial, tasks=new_tasks)


def _first_use_in_interval(facts: _Facts, oid: str, iv: Interval) -> int:
    for u in facts.uses_by_obj.get(oid, ()):
        if iv.a < u <= iv.b:
            return u
    return iv.b


def _smart_prefetch_task(
    oid: str,
    iv: Interval,
    facts: _Facts,
    t_start: list[int],
    t_end: list[int],
) -> int:
    """Choose the task to attach a prefetch trigger to for object `oid`
    with plan interval `iv`. Returns a task index in [0, n).

    Default: iv.a (= the interval's entry boundary task). But if attaching
    there gives no lead time (e.g., the task is zero-runtime or the next
    use is its immediate successor), walk backward to find a task whose
    end is at least D(o) ticks before the use task's start.

    Trade-off: firing earlier than plan's `a` makes the object enter pool
    earlier than the static cap check accounts for. We bound how far back
    we walk to limit this discrepancy.
    """
    if facts.bw_from_slow is None or facts.bw_from_slow <= 0:
        return iv.a if 0 <= iv.a < facts.n else 0
    D = max(1, math.ceil(facts.sizes[oid] / facts.bw_from_slow))
    use_task = _first_use_in_interval(facts, oid, iv)
    if not (0 <= use_task < facts.n):
        return iv.a if 0 <= iv.a < facts.n else 0
    deadline = t_start[use_task]
    # Walk back from iv.a looking for sufficient lead time. Stop walking
    # as soon as we find a task whose end + D <= deadline.
    best = iv.a if 0 <= iv.a < facts.n else 0
    for t in range(min(iv.a, facts.n - 1), -1, -1):
        if t_end[t] + D <= deadline:
            return t
        best = t
    return best


# ============================================================================
# Phase A — MAX→shrink beam search
# ============================================================================

def _greedy_shrink_to_static_cap(plan: Plan, facts: _Facts) -> Plan:
    """Legacy MAX→shrink path. Kept for reference but no longer min_grow's default
    pre-pass — see _lead_time_grow_from_min."""
    if facts.cap is None:
        return plan
    while _static_peak(plan, facts) > facts.cap:
        reductions = _enumerate_reductions(plan, facts)
        if not reductions:
            return plan
        cur_peak = _static_peak(plan, facts)
        useful = [t for t in reductions if _static_peak(t[0], facts) < cur_peak]
        if not useful:
            return plan
        best = min(useful, key=lambda t: (_static_peak(t[0], facts), -t[2]))
        plan = best[0]
    return plan


def _forward_output_accumulation(facts: _Facts) -> int:
    """Estimate bytes of long-lived forward outputs (e.g., activations) that
    accumulate in pool during forward and must coexist with pre-placements.

    For each task t whose outputs are USED LATER (after some threshold gap),
    those outputs need pool space from production until they're either
    offloaded OR consumed. The to_slow stream can drain SOME of them in parallel
    with forward compute, but if to_slow is the bottleneck, residual accumulates.

    Returns: bytes that we expect to be IN POOL at end of forward
    (= production - to_slow_drained_during_forward).
    """
    n = facts.n
    bw_to_slow = facts.bw_to_slow or 1
    # Long-lived output = produced AND last use is "much later" (threshold = n/4).
    # This excludes y_i (used in next task) but includes A_i (used in backward).
    threshold = max(1, n // 4)
    fwd_outputs: list[tuple[int, int]] = []  # (producer_task, size)
    for i in range(n):
        for oid in facts.outputs_by_task[i]:
            uses = facts.uses_by_obj.get(oid, ())
            if uses and (uses[-1] - i) > threshold:
                fwd_outputs.append((i, facts.sizes[oid]))
    if not fwd_outputs:
        return 0
    earliest_prod = min(p for p, _ in fwd_outputs)
    latest_prod = max(p for p, _ in fwd_outputs)
    fwd_compute = sum(facts.runtimes[i] for i in range(earliest_prod, latest_prod + 1))
    total_bytes = sum(s for _, s in fwd_outputs)
    to_slow_capacity = fwd_compute * bw_to_slow
    return max(0, total_bytes - to_slow_capacity)


def _lead_time_grow_from_min(facts: _Facts) -> Plan:
    """Lead-time-driven analytic pre-pass: start from MIN, add pre-placements
    in lead-time-ascending order (smallest lead time first = highest
    pre-placement value) until cap is exhausted.

    Rationale (from stall-analysis insight):
      - Forward stalls are dominated by insufficient_memory (pool can't fit
        output reservation). Caused by over-pre-placement.
      - Backward stalls are dominated by dependency_missing (from_slow queue
        congestion). Caused by under-pre-placement of critical-path objects.
      - The trade-off is: each pre-placement saves a potential from_slow stall
        (if lead_time < D(o)), but adds pool pressure throughout its
        residency.

    Lead-time-ascending order naturally selects objects whose re-prefetch
    CAN'T hide behind compute (W_0 first — no lead time; W_1 next, etc.).
    Objects with massive lead time (dW_i, used at b_i in backward) come
    LAST in priority and typically aren't pre-placed at all — matching
    belady_reactive's empirically-optimal pattern.

    For each candidate, try (1) full MAX-style extension (interval to
    last-use boundary), then (2) first-interval-only pre-placement.
    """
    plan = _min_plan(facts)
    if facts.cap is None:
        return _max_plan(facts)

    # Tentative task start times (no transfer stalls assumed)
    t_start = [0] * facts.n
    t_now = 0
    for i in range(facts.n):
        t_start[i] = t_now
        t_now += facts.runtimes[i]

    # Pre-place set: walk tasks in order, accumulate sizes of NEW objects
    # referenced (inputs ∪ outputs). Stop when total > cap. Pre-place all
    # backing-init/compute-init objects referenced in the prefix.
    #
    # Intuition: pre-place objects in the "warm prefix" — those that fit
    # along with the produced outputs during the first several tasks.
    # Objects referenced only LATER (e.g., dW_i at b_i in training) don't
    # fit in the warm prefix and shouldn't be pre-placed (cap already full
    # by then).
    # Workload-agnostic: no notion of forward/backward/activation.
    cap = facts.cap
    accumulated_bytes = 0
    accumulated_oids: set[str] = set()
    pre_place_set: set[str] = set()
    for ti in range(facts.n):
        task_oids = (
            list(facts.inputs_by_task[ti])
            + [a for a in facts.outputs_by_task[ti]]
        )
        # Compute new bytes this task would add
        new_oids = [o for o in task_oids if o not in accumulated_oids]
        new_bytes = sum(facts.sizes[o] for o in new_oids)
        if accumulated_bytes + new_bytes > cap:
            break
        accumulated_oids.update(new_oids)
        accumulated_bytes += new_bytes
        for o in new_oids:
            if o in facts.backing_init_ids or o in facts.compute_init_ids:
                pre_place_set.add(o)

    # Apply pre-placements to the plan in lead-time-ascending order
    # (preferred for trigger placement in derive_schedule).
    candidates: list[tuple[int, str]] = []
    for oid in pre_place_set:
        uses = facts.uses_by_obj.get(oid, ())
        if not uses:
            candidates.append((0, oid))
        else:
            candidates.append((t_start[uses[0]], oid))
    candidates.sort()

    for _, oid in candidates:
        cur_ivs = list(plan.intervals.get(oid, ()))
        if not cur_ivs or cur_ivs[0].a == -1:
            continue
        last_b = max(iv.b for iv in cur_ivs)
        try_full = _with_intervals(plan, oid, (Interval(-1, last_b),))
        if _respects_static_cap(try_full, facts):
            plan = try_full
            continue
        new_first = Interval(-1, cur_ivs[0].b)
        try_first = _with_intervals(plan, oid, (new_first,) + tuple(cur_ivs[1:]))
        if _respects_static_cap(try_first, facts):
            plan = try_first
    return plan


def _find_pending_outbound_stalls(
    plan: Plan, bare: TaskChain, facts: _Facts,
) -> list[tuple[str, str]]:
    """Run sim; identify (cur_task_id, offloaded_obj_id) pairs where the
    compute task stalls waiting for an input that's in pending_outbound state.

    These represent UNNECESSARY offload+re-prefetch round trips: the object
    was on compute, got offloaded, now compute waits for it to come back.
    Strictly better to have kept it resident.
    """
    try:
        annotated = _derive_schedule(plan, bare, facts)
        log = simulator_run(annotated)
    except Exception:
        return []
    compute = sorted(
        [iv for iv in log.task_intervals if iv.track == 'compute'],
        key=lambda iv: iv.start,
    )
    stalls: list[tuple[str, str]] = []
    for prev, cur in zip(compute, compute[1:]):
        gap = cur.start - prev.end
        if gap <= 0:
            continue
        # Snapshot at prev.end
        snap = None
        for ev in log.events:
            if ev.t >= prev.end:
                snap = ev.snapshot
                break
        if snap is None:
            continue
        cur_task = next(t for t in annotated.tasks if t.id == cur.task_id)
        dev_state = {m.id: m.state for m in snap.memory if m.location == 'compute'}
        for inp in cur_task.inputs:
            if dev_state.get(inp) == 'pending_outbound':
                stalls.append((cur.task_id, inp))
    return stalls


def _repair_pending_outbound_stalls(
    plan: Plan, bare: TaskChain, facts: _Facts,
    best_makespan: float,
    time_budget_end: float,
) -> tuple[Plan, float]:
    """Generalized interval-merge repair (formerly only pending_outbound).

    For any split interval pair in the plan, try merging if cap-feasible
    and sim-confirmed to improve makespan. Catches BOTH:
      - pending_outbound stalls (offload + re-prefetch round trip)
      - silent release+re-prefetch waste (backing-init multi-use objects
        where cap has headroom to keep them resident)
      - any other interval split that turns out to be unnecessary

    Iteratively: for each object with multiple intervals, try merging
    adjacent pairs. Keep the best non-worse merge per iteration. Stop
    when no merge improves.
    """
    import time as _time
    cap = facts.cap
    if cap is None:
        return plan, best_makespan

    cur_plan = plan
    cur_makespan = best_makespan
    while _time.monotonic() < time_budget_end:
        improved_this_iter = False
        # PRIORITY 1: pending_outbound stalls (most impactful — eliminates
        # both to_slow + from_slow + the stall they cause). Try these first.
        stalls = _find_pending_outbound_stalls(cur_plan, bare, facts)
        unique_oids = list(dict.fromkeys(o for _, o in stalls))
        for oid in unique_oids:
            if _time.monotonic() >= time_budget_end:
                break
            ivs = list(cur_plan.intervals.get(oid, ()))
            if len(ivs) < 2:
                continue
            merged = Interval(ivs[0].a, ivs[1].b)
            new_ivs = (merged,) + tuple(ivs[2:])
            new_plan = _with_intervals(cur_plan, oid, new_ivs)
            if not _respects_static_cap(new_plan, facts):
                continue
            result = _score_with_peak(new_plan, bare, facts)
            if result is None or result[1] > cap:
                continue
            if result[0] < cur_makespan:
                cur_plan = new_plan
                cur_makespan = result[0]
                improved_this_iter = True

        # PRIORITY 2: ANY split-interval merge that improves makespan
        # (catches silent release+re-prefetch waste, not just stalls).
        # Iterate all objects with multiple intervals; try merging each
        # adjacent pair; apply if improves.
        for oid, ivs_tuple in list(cur_plan.intervals.items()):
            ivs = list(ivs_tuple)
            if len(ivs) < 2:
                continue
            i = 0
            while i < len(ivs) - 1:
                if _time.monotonic() >= time_budget_end:
                    break
                if ivs[i].b >= ivs[i + 1].a:
                    i += 1
                    continue
                merged = Interval(ivs[i].a, ivs[i + 1].b)
                new_ivs = tuple(ivs[:i]) + (merged,) + tuple(ivs[i + 2:])
                new_plan = _with_intervals(cur_plan, oid, new_ivs)
                if not _respects_static_cap(new_plan, facts):
                    i += 1
                    continue
                result = _score_with_peak(new_plan, bare, facts)
                if result is None or result[1] > cap:
                    i += 1
                    continue
                if result[0] < cur_makespan:
                    cur_plan = new_plan
                    cur_makespan = result[0]
                    improved_this_iter = True
                    ivs = list(new_ivs)  # restart with new shorter list
                    # don't increment i — the new merged interval is at i
                else:
                    i += 1

        if not improved_this_iter:
            break
    return cur_plan, cur_makespan


def _score_with_peak(
    plan: Plan, bare: TaskChain, facts: _Facts
) -> tuple[int, int] | None:
    """Run simulator, return (makespan, peak_fast_memory_bytes), or None if the
    simulator rejected the plan (cap violation).
    """
    try:
        annotated = _derive_schedule(plan, bare, facts)
        log = simulator_run(annotated, snapshots=False)
    except Exception:
        return None
    makespan = max(iv.end for iv in log.task_intervals)
    return makespan, log.peak_fast_memory_bytes


def _v5_plan_search(
    bare: TaskChain,
    facts: _Facts,
    *,
    k_beam: int,
    k_sim: int,
    time_budget_s: float,
    patience: int = 3,
) -> Plan:
    start = time.monotonic()

    # Infeasibility check: if even MIN exceeds cap statically, give up.
    min_plan = _min_plan(facts)
    if not _respects_static_cap(min_plan, facts):
        pool = _boundary_pool(min_plan, facts)
        worst_idx = max(
            range(len(pool)),
            key=lambda k: pool[k] + (facts.next_outputs_size[k] if k < facts.n else 0),
        )
        worst_b = worst_idx - 1
        worst_pool = pool[worst_idx]
        worst_out = (
            facts.next_outputs_size[worst_idx] if worst_idx < facts.n else 0
        )
        raise ValueError(
            f"infeasible: forced pool {worst_pool}+{worst_out}={worst_pool + worst_out} "
            f"> cap {facts.cap} at boundary {worst_b}"
        )

    # Build MAX and check if it's already feasible (the easy / loose-cap case).
    max_plan = _max_plan(facts)
    cap = facts.cap

    if cap is None:
        # Unlimited cap: MAX is optimal (no transfers needed beyond writeback).
        return max_plan

    # Try MAX
    if _respects_static_cap(max_plan, facts):
        result = _score_with_peak(max_plan, bare, facts)
        if result is not None and result[1] <= cap:
            # MAX is feasible — this is optimal-or-near-optimal.
            return max_plan

    # MIN is the worst-case fallback if search fails.
    min_result = _score_with_peak(min_plan, bare, facts)
    min_plan_makespan: float = math.inf
    if min_result is not None and min_result[1] <= cap:
        min_plan_makespan = min_result[0]

    best_plan: Plan = min_plan
    best_makespan: float = math.inf

    # =====================================================================
    # Phase A0 — Analytic pre-pass: build plan by lead-time-driven growth.
    # Start from MIN, add pre-placements in lead-time-ascending order
    # (objects whose re-prefetch CAN'T hide behind compute go first).
    # dWs naturally get LAST priority (huge lead time = re-prefetch hides
    # behind backward compute). Matches belady_reactive's empirical pattern.
    # If MAX fits, lead_time_grow returns MAX (unlimited cap case).
    # =====================================================================
    if cap is not None and _static_peak(max_plan, facts) > cap:
        start_plan = _lead_time_grow_from_min(facts)
    else:
        start_plan = max_plan

    # =====================================================================
    # Phase A1 — Simulator-driven over-shrink: from the static-feasible plan,
    # iteratively try Belady-ranked further shrinks. Accept the best
    # non-WORSE shrink at each step (allows walking through plateaus, since
    # one un-pre-placement may not improve makespan alone but two-three
    # together break through pool pressure). Tracks the BEST makespan
    # encountered (best_plan); stops after `shrink_patience` consecutive
    # steps without strict improvement.
    # =====================================================================
    if cap is not None:
        sp = _score_with_peak(start_plan, bare, facts)
        if sp is not None and sp[1] <= cap:
            # Initialize best with analytic pre-pass plan (so over-shrink can
            # only improve on it; if all shrinks degrade, we keep this plan).
            if sp[0] < best_makespan:
                best_makespan = sp[0]
                best_plan = start_plan
            cur_plan = start_plan
            cur_makespan = sp[0]
            shrink_patience = 6
            stale_shrink = 0
            t1_budget_end = start + time_budget_s * 0.6  # 60% of total budget
            while time.monotonic() < t1_budget_end and stale_shrink < shrink_patience:
                reductions = _enumerate_reductions(cur_plan, facts)
                reductions = [t for t in reductions if _respects_static_cap(t[0], facts)]
                if not reductions:
                    break
                reductions.sort(key=lambda t: -t[2])  # Belady-latest first
                # Score top K_sim; pick best NON-WORSE (allows plateau walk).
                best_step = None
                for c, _, _ in reductions[:k_sim]:
                    if time.monotonic() >= t1_budget_end:
                        break
                    r = _score_with_peak(c, bare, facts)
                    if r is None:
                        continue
                    if r[1] <= cap and r[0] <= cur_makespan:
                        if best_step is None or r[0] < best_step[1]:
                            best_step = (c, r[0], r[1])
                if best_step is None:
                    break  # no non-worse shrink found
                cur_plan, cur_makespan, _ = best_step
                if cur_makespan < best_makespan:
                    best_makespan = cur_makespan
                    best_plan = cur_plan
                    stale_shrink = 0
                else:
                    stale_shrink += 1
            start_plan = best_plan  # carry BEST (not last) into beam search

    # =====================================================================
    # Phase A1.5 — Stall repair: any pending_outbound stall is a strict
    # plan suboptimality (object was offloaded then needed back = wasted
    # to_slow+from_slow round trip). Fix by merging the relevant intervals.
    # =====================================================================
    if cap is not None and best_makespan < math.inf:
        repaired_plan, repaired_ms = _repair_pending_outbound_stalls(
            best_plan, bare, facts, best_makespan, start + time_budget_s * 0.8,
        )
        if repaired_ms < best_makespan:
            best_plan = repaired_plan
            best_makespan = repaired_ms
            start_plan = repaired_plan

    # =====================================================================
    # Beam search: starts from start_plan (feasible-or-MAX). Reductions and
    # extensions. Belady ranking. Simulator scores actual makespan.
    # =====================================================================
    initial_score = _score_with_peak(start_plan, bare, facts)
    if initial_score is None:
        beam: list[tuple[Plan, float, int]] = [(start_plan, math.inf, _static_peak(start_plan, facts))]
    else:
        beam = [(start_plan, initial_score[0], initial_score[1])]
        if initial_score[1] <= cap and initial_score[0] < best_makespan:
            best_makespan = initial_score[0]
            best_plan = start_plan
    seen: set[tuple] = {_plan_key(start_plan)}
    stale = 0

    while time.monotonic() - start < time_budget_s:
        if best_makespan < math.inf and stale >= patience:
            break

        ext_cands: list[tuple[Plan, str, int]] = []
        red_cands: list[tuple[Plan, str, int]] = []
        for (p, _, _) in beam:
            for c_tuple in _enumerate_reductions(p, facts):
                c = c_tuple[0]
                key = _plan_key(c)
                if key in seen:
                    continue
                seen.add(key)
                red_cands.append(c_tuple)
            for c_tuple in _enumerate_extensions(p, facts):
                c = c_tuple[0]
                if cap is not None and _static_peak(c, facts) > cap:
                    continue
                key = _plan_key(c)
                if key in seen:
                    continue
                seen.add(key)
                ext_cands.append(c_tuple)
        if not ext_cands and not red_cands:
            break

        # Belady ranking:
        # - Reductions: prefer evicting objects with LATEST next use (re-prefetch
        #   hides behind more compute).
        # - Extensions: prefer pre-placing objects with EARLIEST first use
        #   (from_slow most likely to stall otherwise).
        red_cands.sort(key=lambda t: (-t[2], _static_peak(t[0], facts)))
        ext_cands.sort(key=lambda t: (t[2], _static_peak(t[0], facts)))

        # Split simulator budget between directions. If no feasible plan yet,
        # all budget goes to reductions (we need to shrink).
        if best_makespan == math.inf:
            to_score = [c for c, _, _ in red_cands[:k_sim]]
        else:
            half = max(1, k_sim // 2)
            to_score = (
                [c for c, _, _ in red_cands[:k_sim - half]]
                + [c for c, _, _ in ext_cands[:half]]
            )

        scored: list[tuple[Plan, float, int]] = []
        for c in to_score:
            if time.monotonic() - start >= time_budget_s:
                break
            r = _score_with_peak(c, bare, facts)
            if r is None:
                scored.append((c, math.inf, _static_peak(c, facts)))
            else:
                scored.append((c, r[0], r[1]))
        if not scored:
            break

        improved = False
        for c, ms, peak in scored:
            if peak <= cap and ms < best_makespan:
                best_makespan = ms
                best_plan = c
                improved = True

        if best_makespan < math.inf:
            if improved:
                stale = 0
            else:
                stale += 1

        # Beam advance: keep best_plan + top K-1 candidates.
        # If not yet feasible, prioritize lower static_peak.
        # If feasible, prioritize lower makespan (best frontier).
        def beam_key(entry):
            _, ms, peak = entry
            feasible = peak <= cap
            return (not feasible, peak if not feasible else 0, ms)
        scored.sort(key=beam_key)
        new_beam: list[tuple[Plan, float, int]] = []
        if best_makespan < math.inf:
            new_beam.append((best_plan, best_makespan, 0))  # always carry best
        for e in scored:
            if len(new_beam) >= k_beam:
                break
            if e[0] is not (new_beam[0][0] if new_beam else None):
                new_beam.append(e)
        beam = new_beam

    # Fallback to MIN if search found nothing better.
    if best_makespan > min_plan_makespan:
        return min_plan
    return best_plan


# ============================================================================
# Public entry point
# ============================================================================

def apply_min_grow_policy(
    bare: TaskChain,
    *,
    k_beam: int = 5,
    k_sim: int = 15,
    time_budget_s: float = 3.0,
) -> TaskChain:
    """min_grow auto policy: MAX→shrink with simulator-as-oracle.

    Raises ValueError("infeasible: ...") iff the MIN-plan footprint exceeds
    fast_memory_capacity at some boundary (chain literally cannot run). Otherwise
    returns an annotated TaskChain.
    """
    for t in bare.tasks:
        if t.releases_after or t.offload_after or t.prefetch_after:
            raise ValueError(
                f"apply_min_grow_policy: task {t.id!r} is not bare (has existing triggers)"
            )

    facts = _build_facts(bare)
    plan = _v5_plan_search(
        bare, facts,
        k_beam=k_beam, k_sim=k_sim, time_budget_s=time_budget_s,
    )
    return _derive_schedule(plan, bare, facts)
