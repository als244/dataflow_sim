"""max_reduce — clean general formulation auto policy.

See docs/policy/other_policies/max-reduce.md for the design doc. Three phases:
  1. Residency  — decide which boundaries each object is on compute. Pure
                  memory accounting; no streams, no times.
  2. Triggers   — derive initial-placement + per-task triggers from the
                  residency intervals.
  3. (Phase 3 — stream-slack reporting — is informational and lives outside
     the planner; the simulator is the actual arbiter of timing.)

Outputs an annotated TaskChain. Raises ValueError("infeasible: ...") if no
plan can fit the chain at the given cap. Never falls back to other planners.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from dataflow_sim.core.schema import Object, Task, TaskChain, TransferTrigger

# A residency interval is a half-open boundary range; we use [a, b] inclusive
# of both ends. Boundary `-1` is the initial snapshot (before task 0 runs);
# boundary `k` for k >= 0 is the snapshot just after task k's triggers fire.


# ---------- per-object derived facts ----------

def _producer_idx(bare: TaskChain) -> dict[str, int]:
    """obj_id -> task index that produces it (-1 if in initial_memory)."""
    out: dict[str, int] = {o.id: -1 for o in bare.initial_memory}
    for i, task in enumerate(bare.tasks):
        for o in task.outputs:
            out[o.id] = i
    return out


def _uses_by_obj(bare: TaskChain) -> dict[str, list[int]]:
    """obj_id -> sorted list of task indices that consume it as input."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, task in enumerate(bare.tasks):
        for inp in task.inputs:
            out[inp].append(i)
    return {k: sorted(v) for k, v in out.items()}


def _object_sizes(bare: TaskChain) -> dict[str, int]:
    sizes: dict[str, int] = {o.id: o.size for o in bare.initial_memory}
    for task in bare.tasks:
        for o in task.outputs:
            sizes[o.id] = o.size
    return sizes


def _object_types(bare: TaskChain) -> dict[str, str]:
    types: dict[str, str] = {o.id: o.type for o in bare.initial_memory}
    for task in bare.tasks:
        for o in task.outputs:
            types[o.id] = o.type
    return types


# ---------- Phase 1: residency ----------

def _initial_residency(bare: TaskChain) -> dict[str, list[tuple[int, int]]]:
    """MAX residency for each object: a single interval covering its entire
    active span. The exact shape depends on the object's role:

      - compute-initial: [-1, last_use_boundary] (already on compute, kept live)
      - backing-initial:   [-1, last_use_boundary] (pre-placed, kept live)
      - task output:    [producer_idx, last_use_boundary]

    For gradients (backing-initial, type=gradient), the last_use_boundary is
    extended by 1 (= the writeback boundary, where the offload trigger fires
    to flush updated bytes back to backing). For non-gradients, last_use_boundary
    is `last_use_task_idx - 1` (the obj is consumed during the last-use task
    and structurally dies at boundary `last_use_task_idx`).

    For an object with no use AT ALL (rare; only "dead" outputs), the
    residency is a single point at its appearance boundary.
    """
    producer = _producer_idx(bare)
    uses = _uses_by_obj(bare)
    types = _object_types(bare)
    backing_init_ids = {o.id for o in bare.initial_memory if o.location == "backing"}

    residency: dict[str, list[tuple[int, int]]] = {}
    all_ids: set[str] = set(producer) | set(uses)
    for oid in all_ids:
        u = uses.get(oid, [])
        appears = producer[oid] if producer[oid] >= 0 else -1
        if not u:
            # Dead object — just hold for one boundary then drop.
            residency[oid] = [(appears, appears)]
            continue
        last_use = u[-1]
        # Residency interval [a, b]: obj live at boundaries a, a+1, ..., b.
        # The departure trigger fires at task (b + 1), removing the obj at
        # boundary b + 1. So for obj with last use at task u, we set b = u-1
        # (obj live at boundary u-1 = start of task u, consumed during u,
        # trigger fires at end of task u = boundary u). This is the SAME
        # b for gradients vs non-gradients — only the trigger TYPE differs
        # (writeback offload vs release).
        live_until = last_use - 1
        residency[oid] = [(appears, live_until)]
    return residency


def _next_outputs_size(bare: TaskChain) -> list[int]:
    """For each boundary k, the compute bytes reserved by task[k+1]'s outputs.
    By boundary semantics, outputs of task[k+1] are reserved at its START,
    which is boundary k. So we charge them against pool[k]."""
    n = len(bare.tasks)
    out = [0] * (n + 1)  # boundaries -1..n-1 (index 0..n in this array)
    # boundary k → array index k + 1
    for k in range(-1, n - 1):
        nxt = bare.tasks[k + 1]
        out[k + 1] = sum(o.size for o in nxt.outputs if o.location == "fast")
    return out


def _task_end_times(bare: TaskChain) -> list[int]:
    """Cumulative ideal task-end times (no stalls)."""
    n = len(bare.tasks)
    out = [0] * n
    cum = 0
    for i in range(n):
        cum += bare.tasks[i].runtime
        out[i] = cum
    return out


def _departure_type(
    oid: str,
    interval_idx: int,
    n_intervals: int,
    backing_source: bool,
    is_gradient: bool,
) -> str:
    """Returns 'offload' or 'release' for this interval's departure."""
    is_last = (interval_idx == n_intervals - 1)
    if is_last:
        return "offload" if is_gradient else "release"
    # Mid-interval: must preserve bytes for next interval.
    return "release" if (backing_source and not is_gradient) else "offload"


def _effective_residency_ends(
    bare: TaskChain,
    residency: dict[str, list[tuple[int, int]]],
    sizes: dict[str, int],
) -> dict[tuple[str, int], int]:
    """For each (oid, interval_idx), the EFFECTIVE last boundary at which the
    object is in the compute pool. For intervals ending in release this is
    just `b` (release is instant). For intervals ending in offload it's
    extended forward by however many task boundaries the to_slow takes to drain
    given FIFO contention with all other offloads.

    Simulates the to_slow FIFO in fire-order to assign each offload an actual
    completion time, then maps that time to a task boundary.
    """
    n = len(bare.tasks)
    task_end = _task_end_times(bare)
    bw = bare.bandwidth_to_slow
    backing_init_ids = {o.id for o in bare.initial_memory if o.location == "backing"}
    types = _object_types(bare)

    # Collect offload events in fire order.
    events: list[tuple[int, str, int, int]] = []  # (fire_time, oid, idx, size)
    eff_ends: dict[tuple[str, int], int] = {}
    for oid, intervals in residency.items():
        backing_source = oid in backing_init_ids
        is_grad = backing_source and types.get(oid) == "gradient"
        for idx, (a, b) in enumerate(intervals):
            dep = _departure_type(oid, idx, len(intervals), backing_source, is_grad)
            if dep == "release":
                eff_ends[(oid, idx)] = b
                continue
            if b + 1 >= n:
                # Offload would fire past end of chain; treat as live until end.
                eff_ends[(oid, idx)] = n - 1
                continue
            fire_t = task_end[b]  # offload trigger fires at end of task b
            # Actually triggers fire AT task (b+1)? Re-check: a departure
            # at end of interval [a,b] (= obj NOT live at b+1) fires at task
            # b+1. So fire time = task_end[b + 1] in absolute µs. But that's
            # AFTER task b+1 has run. The to_slow starts AT that moment.
            # Actually the simulator fires offload at end of producing task —
            # which for a departure between intervals [a,b] and [c,d] means
            # the trigger is on task (b+1) and fires at task_end[b+1].
            fire_t = task_end[b + 1] if bw else 0
            events.append((fire_t, oid, idx, sizes[oid]))

    events.sort(key=lambda e: (e[0], e[1]))  # tiebreak by oid for determinism

    if bw is None or bw <= 0:
        # No bandwidth → instant model fallback (could happen at unlimited cap).
        for oid, intervals in residency.items():
            for idx, (a, b) in enumerate(intervals):
                eff_ends.setdefault((oid, idx), b)
        return eff_ends

    to_slow_busy = 0
    for fire_t, oid, idx, sz in events:
        start = max(to_slow_busy, fire_t)
        tau = max(1, math.ceil(sz / bw))
        end = start + tau
        to_slow_busy = end
        # Find max boundary k such that task_end[k] < end.
        # boundary k = end of task k = task_end[k]. If end > task_end[k],
        # the to_slow still hasn't finished as of boundary k → obj still in pool.
        eff_b = -1
        for k in range(n):
            if task_end[k] < end:
                eff_b = k
            else:
                break
        eff_ends[(oid, idx)] = max(eff_b, residency[oid][idx][1])

    return eff_ends


def _effective_a(a: int, producer_idx: int) -> int:
    """Boundary at which the simulator first counts a prefetched object in
    the compute pool. For an interval starting at `a`, if the source is a
    prefetch (a > -1 and a is not the production boundary), then from_slow STARTS
    at boundary (a - 1) and the simulator's compute entry exists from then
    onward (state=inbound → live by boundary a). For pre-placed (a == -1)
    or task-output (a == producer) intervals, no such extension applies."""
    if a > -1 and a != producer_idx:
        return a - 1
    return a


def _compute_to_slow_tails(
    bare: TaskChain,
    residency: dict[str, list[tuple[int, int]]],
    sizes: dict[str, int],
    mutators: dict[str, set[int]],
) -> dict[tuple[str, int], int]:
    """For each interval that ends in an OFFLOAD trigger, simulate the to_slow
    FIFO to determine the actual last boundary the object occupies. Returns
    a dict (obj_id, interval_idx) -> effective_b extended past the logical
    interval end by to_slow transit time. For RELEASE departures, no extension.

    The simulation is conservative: it walks offload events in fire-time
    order, assigns each one a slot in the to_slow queue (max of fire_time and
    cumulative to_slow-busy), and computes its completion time. The object's
    effective last boundary is the largest k with task_end[k] < completion."""
    n = len(bare.tasks)
    bw = bare.bandwidth_to_slow
    if bw is None or bw <= 0:
        return {}
    task_end = _task_end_times(bare)
    backing_init_ids = {o.id for o in bare.initial_memory if o.location == "backing"}
    uses_by_obj = _uses_by_obj(bare)
    producer = _producer_idx(bare)

    # Collect (fire_time, oid, interval_idx, size) for each interval whose
    # departure trigger is an OFFLOAD (not a release).
    events: list[tuple[int, str, int, int]] = []
    for oid, intervals in residency.items():
        has_backing_source = oid in backing_init_ids
        obj_mutators = mutators.get(oid, set())
        ever_mutated = bool(obj_mutators)
        for idx, (a, b) in enumerate(intervals):
            is_last = (idx == len(intervals) - 1)
            uses_in = [u for u in uses_by_obj.get(oid, []) if a <= u - 1 <= b]
            p = producer.get(oid, -1)
            production_in = [p] if (p >= 0 and a <= p <= b) else []
            fire_cands = uses_in + production_in
            if not fire_cands:
                continue
            fire_task = max(fire_cands)
            if fire_task >= n:
                continue
            mutated_in = any(m in obj_mutators for m in uses_in)
            dirty = mutated_in
            # Same trigger-type decision as in _emit_triggers:
            if dirty:
                trigger = "offload"
            elif not is_last and not has_backing_source:
                trigger = "offload"
            else:
                trigger = "release"
            if trigger == "offload":
                events.append((task_end[fire_task], oid, idx, sizes[oid]))

    events.sort(key=lambda e: (e[0], e[1]))
    eff_ends: dict[tuple[str, int], int] = {}
    to_slow_busy = 0
    for fire_t, oid, idx, sz in events:
        start = max(to_slow_busy, fire_t)
        tau = max(1, math.ceil(sz / bw))
        end = start + tau
        to_slow_busy = end
        eff_b = -1
        for k in range(n):
            if task_end[k] < end:
                eff_b = k
            else:
                break
        # Don't shrink below the logical end (eff_b should EXTEND, not retract).
        logical_b = residency[oid][idx][1]
        eff_ends[(oid, idx)] = max(eff_b, logical_b)
    return eff_ends


def _pool_size_per_boundary(
    bare: TaskChain,
    residency: dict[str, list[tuple[int, int]]],
    sizes: dict[str, int],
    eff_ends: dict[tuple[str, int], int] | None = None,
) -> list[int]:
    """pool[k] = sum of sizes of objects whose effective residency includes
    boundary k, for k in [-1, n-1]. Indexed as array[k + 1]. Accounts for
    prefetch arrival on the left edge AND to_slow transit tail on the right edge
    (objects in pending_outbound state still occupy bytes)."""
    n = len(bare.tasks)
    producer = _producer_idx(bare)
    pool = [0] * (n + 1)
    for oid, intervals in residency.items():
        sz = sizes[oid]
        p = producer.get(oid, -1)
        for idx, (a, b) in enumerate(intervals):
            real_a = _effective_a(a, p)
            real_b = eff_ends.get((oid, idx), b) if eff_ends else b
            for k in range(real_a, real_b + 1):
                pool[k + 1] += sz
    return pool


def _reduce_to_fit_cap(
    bare: TaskChain,
    residency: dict[str, list[tuple[int, int]]],
    sizes: dict[str, int],
) -> None:
    """Phase 1 reduction loop: while any boundary's pool + next-output
    reservation exceeds cap, split a residency interval to evict one object
    from the overloaded boundary. Mutates `residency` in place.

    Raises ValueError("infeasible: ...") if even the minimum-residency plan
    (everything just-in-time prefetched) exceeds cap at some boundary.
    """
    cap = bare.fast_memory_capacity
    if cap is None:
        return  # no constraint
    n = len(bare.tasks)
    next_outputs = _next_outputs_size(bare)
    uses = _uses_by_obj(bare)
    producer = _producer_idx(bare)
    compute_init_ids = {o.id for o in bare.initial_memory if o.location == "fast"}

    def _anchors(oid: str) -> list[int]:
        """Mandatory boundaries that ANY residency interval of `oid` must
        cover collectively (across all intervals). Each anchor must lie in
        SOME interval; splitting can move it between intervals but can't
        drop it. For oid:
          - compute-initial: boundary -1 is anchor (already on compute at start)
          - task output: producer(oid) is anchor (must be live when produced)
          - each use u: boundary (u - 1) is anchor (must be live for task u)
        """
        out: list[int] = []
        if oid in compute_init_ids:
            out.append(-1)
        p = producer.get(oid, -1)
        if p >= 0:
            out.append(p)
        for u in uses.get(oid, []):
            out.append(u - 1)
        return sorted(set(out))

    # Phase 1 pool tracking accounts for the prefetch left-edge (from_slow-start)
    # but NOT for to_slow transit on the right edge. Modeling to_slow tails would
    # make max_reduce declare configs infeasible whenever the to_slow queue can't fully
    # drain by an arbitrary boundary — but the simulator's drain loop will
    # stall compute to wait for to_slow naturally, which is the right physical
    # behavior. max_reduce instead emits a logically-correct plan; any tail-induced
    # pressure manifests as small inter-task gaps in the simulator's timeline.
    pool = _pool_size_per_boundary(bare, residency, sizes)
    eff_ends: dict[tuple[str, int], int] = {}  # unused but kept for API

    def overflow_at(k_arr: int) -> int:
        return pool[k_arr] + next_outputs[k_arr] - cap

    while True:
        worst_arr = max(range(n + 1), key=lambda i: overflow_at(i))
        if overflow_at(worst_arr) <= 0:
            return  # all boundaries fit
        worst_k = worst_arr - 1  # actual boundary index

        # Find an eligible victim. An object can be evicted from boundary k
        # if k itself is NOT one of its anchors (mandatory boundaries) AND
        # the residency interval covering k can be trimmed/split to exclude
        # k while keeping all of o's anchors covered.
        #
        # Procedure: for o's interval [a, b] containing k, compute:
        #   - anchors_in_interval = o's anchors that fall in [a, b]
        #   - anchors_le_k_minus_1 = anchors with c <= k - 1
        #   - anchors_ge_k_plus_1  = anchors with c >= k + 1
        # If k is an anchor itself, skip (mandatory).
        # Otherwise the new intervals are:
        #   - [a, max(anchors_le_k_minus_1)] if non-empty, else dropped
        #   - [min(anchors_ge_k_plus_1), b]  if non-empty, else dropped
        # The dropped middle range frees boundaries from fast memory.
        # Eviction ranking. Three orthogonal cost axes drive the choice:
        #
        # (1) STREAM COST per byte freed depends on the eviction TYPE, not
        #     just the obj's mutation status:
        #     - drop_init (= un-pre-place by removing the obj's [-1, …]
        #       portion): only adds 1 from_slow (the just-in-time prefetch when
        #       the obj is first needed). The writeback offload for mutated
        #       objs happens REGARDLESS of pre-placement, so it doesn't
        #       count as eviction cost here.
        #     - mid-life split (cut between two uses): if the obj is
        #       release-eligible (backing source exists AND never mutated),
        #       cost = 1 from_slow. Else round-trip cost = 1 to_slow + 1 from_slow.
        #
        # (2) FIRST-USE TIME: pre-placing an obj that's used SOON avoids
        #     forward from_slow contention; pre-placing one used FAR can be
        #     deferred to backward where from_slow has slack. Prefer evicting
        #     LATE-first-use objs first (= keep early-use objs pre-placed).
        #
        # (3) drop_init vs mid-life: drop_init frees more boundaries (all
        #     of [-1, first_anchor)) and is cheaper for many cases. Prefer it.
        #
        # Combined key (lower = better): (stream_cost, not_drop_init,
        # -first_use, -size, -gap_length).
        backing_init_ids = {o.id for o in bare.initial_memory if o.location == "backing"}
        uses_by_obj = _uses_by_obj(bare)
        candidates: list[tuple[tuple, str, int, int | None, int | None]] = []
        for oid, intervals in residency.items():
            for idx, (a, b) in enumerate(intervals):
                p = producer.get(oid, -1)
                real_a = _effective_a(a, p)
                if not (real_a <= worst_k <= b):
                    continue
                anchors_all = _anchors(oid)
                if worst_k in anchors_all:
                    continue  # mandatory at this boundary
                anchors_in = [c for c in anchors_all if a <= c <= b]
                anchors_le = [c for c in anchors_in if c <= worst_k - 1]
                anchors_ge = [c for c in anchors_in if c >= worst_k + 1]
                left_end = max(anchors_le) if anchors_le else None
                right_start = min(anchors_ge) if anchors_ge else None
                left_b = left_end if left_end is not None else a - 1
                right_a = right_start if right_start is not None else b + 1
                gap_len = right_a - left_b - 1
                if gap_len <= 0:
                    continue
                drops_init = (left_end is None and a == -1)
                has_backing_source = oid in backing_init_ids
                obj_ever_mutated = any(
                    oid in t.mutates_inputs for t in bare.tasks
                )
                if drops_init:
                    # Un-pre-place: 1 from_slow. Writeback happens regardless.
                    stream_cost = 0
                else:
                    release_eligible = has_backing_source and not obj_ever_mutated
                    stream_cost = 0 if release_eligible else 1
                # First-use time across ENTIRE chain (not just this interval).
                # Pre-placing an early-use obj is more valuable than pre-
                # placing a late-use obj — late uses have from_slow slack to load
                # them on demand. So prefer evicting LATE-first-use objs.
                obj_uses = uses_by_obj.get(oid, [])
                first_use = obj_uses[0] if obj_uses else n  # never used → max idx
                key = (
                    stream_cost,
                    0 if drops_init else 1,
                    -first_use,
                    -sizes[oid],
                    -gap_len,
                )
                candidates.append((key, oid, idx, left_end, right_start))

        if not candidates:
            live_here = [
                (oid, sizes[oid])
                for oid, intervals in residency.items()
                for a, b in intervals
                if a <= worst_k <= b
            ]
            live_str = ", ".join(
                f"{oid}({sz})" for oid, sz in sorted(live_here, key=lambda x: -x[1])[:10]
            )
            raise ValueError(
                f"infeasible at boundary {worst_k}: pool={pool[worst_arr]} + "
                f"reserved_outputs={next_outputs[worst_arr]} > cap={cap}; "
                f"all live objects are mandatory at this boundary "
                f"(top 10 by size: {live_str})"
            )

        # Pick the best candidate.
        candidates.sort(key=lambda c: c[0])
        _key, oid, idx, left_end, right_start = candidates[0]
        a, b = residency[oid][idx]
        new_pieces: list[tuple[int, int]] = []
        if left_end is not None:
            new_pieces.append((a, left_end))
        if right_start is not None:
            new_pieces.append((right_start, b))
        new_intervals = list(residency[oid])
        new_intervals[idx:idx + 1] = new_pieces
        residency[oid] = new_intervals
        # Recompute pool — splitting can change a piece's prefetch-extension
        # status (e.g., dropping the leftmost pre-placed portion turns the
        # remaining piece into a prefetched one, shifting effective_a).
        pool = _pool_size_per_boundary(bare, residency, sizes)


def _check_min_pool_feasibility(
    bare: TaskChain,
    sizes: dict[str, int],
) -> None:
    """Quick check that the minimum-residency plan (every object on compute
    ONLY at its use boundaries, plus production boundary for outputs) fits
    cap. If this fails, no plan can fit — error out before the reduction
    loop hits the same conclusion the slow way."""
    cap = bare.fast_memory_capacity
    if cap is None:
        return
    n = len(bare.tasks)
    producer = _producer_idx(bare)
    uses = _uses_by_obj(bare)
    next_outputs = _next_outputs_size(bare)
    # min_pool[k]: sizes of objs MANDATORILY live at boundary k.
    # Mandatory: production boundary (producer == k), use boundaries (u - 1 == k).
    min_pool = [0] * (n + 1)
    for oid, sz in sizes.items():
        for u in uses.get(oid, []):
            min_pool[u - 1 + 1] += sz  # boundary u-1
        if producer[oid] >= 0:
            min_pool[producer[oid] + 1] += sz  # boundary producer[oid]
    # compute-initial objects also pinned at boundary -1
    for o in bare.initial_memory:
        if o.location == "fast":
            # Live at all boundaries until last use - 1 (or boundary -1 if no use)
            u = uses.get(o.id, [])
            last_b = (u[-1] - 1) if u else -1
            for k in range(-1, last_b + 1):
                if k == -1 or u and k == u[-1] - 1:
                    # Already counted at use boundary; skip.
                    if k != -1:
                        continue
                    min_pool[k + 1] += o.size

    for k in range(-1, n):
        if min_pool[k + 1] + next_outputs[k + 1] > cap:
            raise ValueError(
                f"infeasible at boundary {k}: minimum pool size "
                f"({min_pool[k + 1]} bytes) + next-task output reservation "
                f"({next_outputs[k + 1]} bytes) exceeds capacity ({cap}). "
                f"No memory schedule can fit this chain at this cap."
            )


# ---------- Phase 2: triggers ----------

def _emit_triggers(
    bare: TaskChain,
    residency: dict[str, list[tuple[int, int]]],
) -> tuple[set[str], dict[int, dict[str, list[str]]]]:
    """Returns (initial_compute, annotations) where:
      - initial_compute: set of backing obj_ids to pre-place on compute
      - annotations: task_idx -> {"release": [...], "offload": [...], "prefetch": [...]}
    """
    producer = _producer_idx(bare)
    uses = _uses_by_obj(bare)
    backing_init_ids = {o.id for o in bare.initial_memory if o.location == "backing"}
    n = len(bare.tasks)

    # Per-object set of task indices that mutate this object. Generalizes
    # the prior "type=='gradient'" special case: any input listed in a task's
    # `mutates_inputs` is treated as mutated by that task. The backing copy of
    # such an object goes stale at the mutating task and must be flushed
    # back via an offload before being safely dropped.
    mutators: dict[str, set[int]] = defaultdict(set)
    for i, task in enumerate(bare.tasks):
        for oid in task.mutates_inputs:
            mutators[oid].add(i)

    initial_compute: set[str] = set()
    annotations: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: {"release": [], "offload": [], "prefetch": []}
    )

    for oid, intervals in residency.items():
        intervals = sorted(intervals)
        has_backing_source = oid in backing_init_ids
        obj_mutators = mutators.get(oid, set())

        for i, (a, b) in enumerate(intervals):
            is_first = (i == 0)
            is_last = (i == len(intervals) - 1)
            p = producer.get(oid, -1)

            # --- Arrival ---
            if is_first and a == -1:
                if has_backing_source:
                    initial_compute.add(oid)
                # compute-initial: already in pool, no trigger
            elif is_first and a == p and p >= 0:
                # Natural production — no trigger needed
                pass
            else:
                # Need a prefetch to deliver oid at boundary a.
                # Trigger fires at end of task (a - 1).
                if a < 0:
                    raise ValueError(
                        f"max_reduce bug: prefetched interval starts at boundary {a} "
                        f"for {oid!r}; can't prefetch before chain begins"
                    )
                annotations[a - 1]["prefetch"].append(oid)

            # --- Departure ---
            # fire_task is the EARLIEST task at which oid is no longer needed
            # in this interval. Compute as the latest "anchor task":
            #   - For each use u with u-1 in [a, b]: the obj is needed
            #     through task u (consumed during u). Trigger fires at end
            #     of task u (= step 8/9, after u runs).
            #   - For production p in [a, b]: the obj appears at end of task
            #     p (step 7). Trigger can fire on the SAME task p (step 9
            #     runs after step 7, so the obj is live then immediately
            #     queued for to_slow).
            # Take the max so the trigger fires only after all needs are met.
            uses_in = [u for u in uses.get(oid, []) if a <= u - 1 <= b]
            production_in = [p] if (p >= 0 and a <= p <= b) else []
            fire_candidates = uses_in + production_in
            if not fire_candidates:
                # Degenerate interval with no anchor (shouldn't happen if
                # Phase 1 preserved use coverage).
                continue
            fire_task = max(fire_candidates)
            if fire_task >= n:
                # Obj outlives the chain (the last use is the final task and
                # there's no post-chain trigger slot). Acceptable for objects
                # that should remain on compute at end-of-run (e.g. the very
                # last output). Mutated objects ending here would silently
                # lose their writeback — flag it.
                if obj_mutators and is_last and has_backing_source:
                    raise ValueError(
                        f"max_reduce bug: mutated backing-initial {oid!r} has its last "
                        f"residency interval ending at boundary {b} (post-chain) "
                        f"— no place to fire the writeback offload"
                    )
                continue

            # Decide trigger type. "dirty" = a mutation in this interval
            # OR carried-over dirt from a prior interval that wasn't flushed.
            # In max_reduce's plan, every non-last interval ends in either release
            # (clean) or offload (preserves bytes), so dirt never carries
            # across intervals — we only need to check THIS interval.
            mutated_in_interval = any(m in obj_mutators for m in uses_in)
            dirty = mutated_in_interval

            if dirty:
                # Must offload to preserve the updated bytes (writeback if
                # backing source exists, or just preserve-for-next-prefetch if
                # not). Either way: offload.
                annotations[fire_task]["offload"].append(oid)
            elif not is_last and not has_backing_source:
                # No backing source AND another interval will re-use o; must
                # offload to keep the bytes accessible for re-prefetch.
                annotations[fire_task]["offload"].append(oid)
            else:
                # Clean and (has backing source OR is the last interval). Just
                # drop the compute entry; backing copy is identical (or obj is
                # structurally dead).
                annotations[fire_task]["release"].append(oid)

    return initial_compute, dict(annotations)


def _build_chain(
    bare: TaskChain,
    initial_compute: set[str],
    annotations: dict[int, dict[str, list[str]]],
) -> TaskChain:
    """Construct the annotated TaskChain from initial_compute + per-task triggers."""
    backing_objs = {o.id: o for o in bare.initial_memory if o.location == "backing"}
    new_initial = list(bare.initial_memory)
    for oid in initial_compute:
        src = backing_objs[oid]
        new_initial.append(
            Object(id=src.id, size=src.size, location="fast", type=src.type)
        )

    new_tasks: list[Task] = []
    for i, task in enumerate(bare.tasks):
        ann = annotations.get(i, {"release": [], "offload": [], "prefetch": []})
        # Dedup while preserving order.
        rel = list(dict.fromkeys(ann["release"]))
        off = list(dict.fromkeys(ann["offload"]))
        pre = list(dict.fromkeys(ann["prefetch"]))
        new_tasks.append(replace(
            task,
            releases_after=rel,
            offload_after=[TransferTrigger(obj_id=o) for o in off],
            prefetch_after=[TransferTrigger(obj_id=o) for o in pre],
        ))

    return replace(bare, initial_memory=new_initial, tasks=new_tasks)


# ---------- public entry point ----------

# ---------- Phase 3: EDF prefetch scheduling on the from_slow FIFO ----------

def _schedule_prefetches_from_slow(
    bare: TaskChain,
    residency: dict[str, list[tuple[int, int]]],
    sizes: dict[str, int],
) -> None:
    """Extend prefetched residency intervals LEFT to give each prefetch
    enough from_slow lead time, packed EDF-backward into the FIFO.

    Without this, every prefetch fires one task before its consumer
    (=just-in-time), so a contended from_slow queue causes the consumer to stall
    waiting for the transfer. The general principle: when transfers share
    a FIFO stream and each has a known deadline, issue them in deadline
    order as early as the DAG and memory cap allow. The FIFO itself
    orders delivery; the planner's job is to push enough work in early
    enough that the queue is never empty when a deadline arrives.

    Process latest-deadline first, packing backward:
      end[i]   = min(deadline[i], start[i+1])         # i+1 = next-later-deadline
      start[i] = end[i] - tau[i]
      fire[i]  = latest task k with task_end[k] <= start[i]
      new_a[i] = fire[i] + 1  # interval's logical start boundary

    Each extension is cap-checked: if extending to `new_a` would push pool
    over cap at any boundary in [new_a-1 .. current_a-1), bound the
    extension to the latest fitting candidate. Falls back to the original
    just-in-time placement if no extension fits.
    """
    if bare.bandwidth_from_slow is None or bare.bandwidth_from_slow <= 0:
        return
    cap = bare.fast_memory_capacity
    n = len(bare.tasks)
    bw = bare.bandwidth_from_slow
    task_end = _task_end_times(bare)
    producer = _producer_idx(bare)
    next_outputs = _next_outputs_size(bare)
    pool = _pool_size_per_boundary(bare, residency, sizes)

    # Enumerate prefetched intervals: ones that fire a prefetch trigger in
    # Phase 2 (not pre-placed, not natural production).
    Prefetch = tuple  # (deadline, oid, idx, tau, current_a, b, earliest_a)
    prefetches: list[Prefetch] = []
    for oid, intervals in residency.items():
        p = producer.get(oid, -1)
        for idx, (a, b) in enumerate(intervals):
            is_first = (idx == 0)
            # Skip pre-placed initial-pool interval and natural production
            if is_first and (a == -1 or a == p):
                continue
            if a <= 0:
                continue  # can't fire prefetch before task 0
            deadline = task_end[a - 1]
            tau = max(1, math.ceil(sizes[oid] / bw))
            # Earliest valid a:
            #   - backing-initial / compute-initial first interval: any a >= 1
            #     (= fire at task 0 or later; can't fire before chain starts)
            #   - task output: a >= producer + 1 (obj doesn't exist before)
            #   - non-first interval: a >= prev_interval.end + 2
            #     (leave at least 1 boundary gap so we're not contiguous
            #     with prior interval's departure trigger)
            earliest_a = 1
            if p >= 0:
                earliest_a = max(earliest_a, p + 1)
            if not is_first:
                prev_b = intervals[idx - 1][1]
                earliest_a = max(earliest_a, prev_b + 2)
            prefetches.append((deadline, oid, idx, tau, a, b, earliest_a))

    if not prefetches:
        return

    # EDF-backward: process latest-deadline first, pack into from_slow
    prefetches.sort(key=lambda e: (-e[0], e[1]))

    next_start_t = float('inf')
    for deadline, oid, idx, tau, current_a, b, earliest_a in prefetches:
        # Ideal placement: end as late as possible (= deadline or next event's start)
        end_t = min(deadline, next_start_t)
        ideal_start_t = end_t - tau
        if ideal_start_t < 0:
            ideal_start_t = 0

        # Largest task k with task_end[k] <= ideal_start_t (== ideal fire task)
        ideal_fire = -1
        for k in range(n):
            if task_end[k] <= ideal_start_t:
                ideal_fire = k
            else:
                break
        ideal_new_a = max(ideal_fire + 1, earliest_a)

        if ideal_new_a >= current_a:
            # Existing placement is already at/later than EDF target — no
            # extension needed. Just update FIFO cursor for the next event.
            next_start_t = task_end[current_a - 1]
            continue

        # Cap-bounded extension: find the LARGEST allowed new_a in
        # [ideal_new_a, current_a). Walk forward from ideal_new_a — first one
        # whose extension doesn't overflow wins (i.e., the EARLIEST safe
        # extension target).
        sz = sizes[oid]
        chosen_new_a = current_a  # default: no extension
        for try_a in range(ideal_new_a, current_a):
            try_eff_lo = try_a - 1  # effective_a for the extended interval
            old_eff_lo = current_a - 1
            # Extension adds sz to pool at boundaries [try_eff_lo .. old_eff_lo - 1]
            ok = True
            if cap is not None:
                for c in range(try_eff_lo, old_eff_lo):
                    if pool[c + 1] + sz + next_outputs[c + 1] > cap:
                        ok = False
                        break
            if ok:
                chosen_new_a = try_a
                break

        if chosen_new_a < current_a:
            # Commit the extension
            old_eff_lo = current_a - 1
            new_eff_lo = chosen_new_a - 1
            for c in range(new_eff_lo, old_eff_lo):
                pool[c + 1] += sz
            residency[oid][idx] = (chosen_new_a, b)
            next_start_t = task_end[chosen_new_a - 1]
        else:
            # Couldn't extend at all
            next_start_t = task_end[current_a - 1]


def apply_max_reduce_policy(bare: TaskChain) -> TaskChain:
    """max_reduce auto policy. See docs/policy/other_policies/max-reduce.md for the full spec.

    Raises ValueError("infeasible: ...") if no memory schedule can fit the
    chain at the configured `fast_memory_capacity`. Otherwise returns an annotated
    chain satisfying all invariants listed in the doc.
    """
    sizes = _object_sizes(bare)
    _check_min_pool_feasibility(bare, sizes)
    residency = _initial_residency(bare)
    _reduce_to_fit_cap(bare, residency, sizes)
    _schedule_prefetches_from_slow(bare, residency, sizes)
    initial_compute, annotations = _emit_triggers(bare, residency)
    return _build_chain(bare, initial_compute, annotations)
