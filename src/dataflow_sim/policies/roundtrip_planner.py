"""roundtrip_planner proactive round-trip planner auto-policy.

Constructive planner that explicitly enumerates and packs offload+prefetch
round-trips for objects whose use-to-use (or production-to-first-use) gap is
wide enough to amortize the to_slow+from_slow transfer cost. First-use prefetches for
backing-resident inputs are mandatory and packed second; round-trips are
optional optimizations packed in a demand-driven pass that only commits
trips that actually reduce capacity pressure at some boundary.

Unlike the belady_reactive reactive planner this never simulates a shadow forward — it
operates entirely on per-boundary byte-pool projections (`bps[k]`) and two
stream-slot timelines.
"""
from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, replace

from dataflow_sim.core.schema import TaskChain
from dataflow_sim.policies._common import (
    TriggerKind,
    _UseEvent,
    _add_gradient_writebacks,
    _apply_annotations,
    _compute_ideal_starts,
    _compute_uses,
    _last_use_task_idx,
    _object_sizes,
    _object_uses_by_task_idx,
    _verify_and_refine,
)


# ---------- legacy greedy initial placement (used by roundtrip_planner driver) ----------

def _initial_placement(
    bare: TaskChain,
    fast_memory_capacity: int | None,
    uses: dict[str, list[int]],
    sizes: dict[str, int],
) -> set[str]:
    """Decide which backing-resident objects to also pre-place on compute.

    Strategy: must-place inputs and compute-located outputs of T_1; then greedy
    fill remaining capacity by `first_use` ascending.
    Returns the set of object ids to ADD to the compute pool (not the union).

    roundtrip_planner explicitly schedules offload+prefetch round-trips for activations whose
    first-use is far from their production, so leaving room for activations in
    init is the planner's job, not init's. A conservative init means fewer
    forced first-use prefetches crowding the from_slow stream.
    """
    # Already compute-resident in the bare chain (e.g., `input`)
    already_compute = {o.id for o in bare.initial_memory if o.location == "fast"}
    backing_objs = {o.id: o for o in bare.initial_memory if o.location == "backing"}

    if not bare.tasks:
        return set()
    t1 = bare.tasks[0]
    must_place = set(t1.inputs)
    for out in t1.outputs:
        if out.location == "fast":
            must_place.add(out.id)
    must_place = {oid for oid in must_place if oid not in already_compute}
    # Only objects that exist on backing can be promoted into fast memory.
    must_place = {oid for oid in must_place if oid in backing_objs}

    # Verify widest-T_1 feasibility (we'll catch other tasks' widest separately)
    if fast_memory_capacity is not None:
        usage = sum(sizes.get(o, 0) for o in already_compute | must_place)
        if usage > fast_memory_capacity:
            raise ValueError(
                f"widest-task infeasibility at T_1: needs {usage} compute bytes, "
                f"capacity is {fast_memory_capacity}"
            )

    placement = set(must_place)

    # Reserve room for T_1's compute-located outputs (they reserve at task start
    # so they consume capacity alongside the initial pool).
    t1_outputs_size = sum(
        out.size for out in t1.outputs if out.location == "fast"
    )

    # Greedy fill remaining capacity, sorted by first-use ascending. At finite
    # capacity, leave slack equal to the widest single-task footprint MINUS
    # T_1's pinned set (which is already on compute). This prevents the initial
    # pool from crowding out prefetches/cascade evictions the planner needs
    # later. Without slack, tight caps fail to fit `dW_head` etc.
    usage = sum(sizes.get(o, 0) for o in already_compute | placement)
    candidates = sorted(
        (oid for oid in backing_objs if oid not in placement and uses.get(oid)),
        key=lambda o: uses[o][0],
    )
    if fast_memory_capacity is not None:
        widest_task = max(
            sum(sizes.get(i, 0) for i in t.inputs)
            + sum(o.size for o in t.outputs if o.location == "fast")
            for t in bare.tasks
        )
        t1_pinned = t1_outputs_size + sum(sizes.get(i, 0) for i in t1.inputs)
        slack = max(0, widest_task - t1_pinned)
        effective_cap = fast_memory_capacity - slack
    else:
        effective_cap = None
    for oid in candidates:
        if effective_cap is None:
            placement.add(oid)
            usage += sizes[oid]
        elif usage + sizes[oid] + t1_outputs_size <= effective_cap:
            placement.add(oid)
            usage += sizes[oid]

    return placement


# ---------- roundtrip_planner — proactive round-trip planning ----------

@dataclass
class _RoundTrip:
    """A candidate offload-then-prefetch trip that frees an object's compute
    bytes during the gap between two consecutive uses."""
    obj_id: str
    size: int
    prev_use_task_idx: int    # offload boundary must be > this
    next_use_task_idx: int    # prefetch boundary must be < this
    gap_start: int            # = ideal end of prev_use's task
    gap_end: int              # = ideal_starts of next_use's task
    tau_off: int
    tau_pre: int
    # filled in by the packer:
    offload_boundary_idx: int | None = None
    prefetch_boundary_idx: int | None = None
    offload_end_t: int | None = None
    prefetch_start_t: int | None = None


@dataclass
class _StreamSlot:
    """An occupied interval on a stream's timeline. Slots on the same stream
    are sorted by `start` and never overlap."""
    start: int
    end: int
    obj_id: str
    kind: TriggerKind  # "offload" or "prefetch"


def _enumerate_roundtrips(
    bare: TaskChain,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    ideal_starts: dict[str, int],
) -> list[_RoundTrip]:
    """Enumerate every (obj, gap) where the gap is wide enough to fit an
    offload + prefetch round-trip on the streams (ignoring contention —
    that's checked at packing time). Gaps come in two flavors:

      (a) **production -> first-use** for task outputs (e.g. activations
          produced by f_i and not consumed until r_i much later). This is
          typically the LARGEST gap an object has — failing to enumerate
          it forces the planner to fall back on belady_reactive's reactive eviction,
          which fires offloads only when memory pressure binds (~tens of
          ms late on a 32-layer config), wasting the early forward window
          when to_slow would otherwise be idle.

      (b) **use_i -> use_{i+1}** for consecutive-use pairs (e.g. weights
          used at f_i then again at b_i). Always enumerated.
    """
    if bare.bandwidth_to_slow is None or bare.bandwidth_from_slow is None:
        return []
    bw_d, bw_h = bare.bandwidth_to_slow, bare.bandwidth_from_slow

    # Find producer task index + production end-time for each compute output.
    producer_idx: dict[str, int] = {}
    production_t: dict[str, int] = {}
    for i, task in enumerate(bare.tasks):
        end_t = ideal_starts[task.id] + task.runtime
        for out in task.outputs:
            if out.location == "fast":
                producer_idx[out.id] = i
                production_t[out.id] = end_t

    out: list[_RoundTrip] = []
    for obj_id, events in uses_by_task.items():
        if not events:
            continue
        size = sizes[obj_id]
        tau_off = max(1, math.ceil(size / bw_d))
        tau_pre = max(1, math.ceil(size / bw_h))

        # (a) production -> first-use gap (task outputs only).
        if obj_id in producer_idx:
            first = events[0]
            gap_start = production_t[obj_id]
            gap_end = first.ideal_start
            if gap_end - gap_start >= tau_off + tau_pre:
                out.append(_RoundTrip(
                    obj_id=obj_id, size=size,
                    prev_use_task_idx=producer_idx[obj_id],
                    next_use_task_idx=first.task_idx,
                    gap_start=gap_start, gap_end=gap_end,
                    tau_off=tau_off, tau_pre=tau_pre,
                ))

        # (b) consecutive-use pairs.
        for i in range(len(events) - 1):
            u_prev, u_next = events[i], events[i + 1]
            prev_task = bare.tasks[u_prev.task_idx]
            gap_start = ideal_starts[prev_task.id] + prev_task.runtime
            gap_end = u_next.ideal_start
            if gap_end - gap_start < tau_off + tau_pre:
                continue
            out.append(_RoundTrip(
                obj_id=obj_id, size=size,
                prev_use_task_idx=u_prev.task_idx,
                next_use_task_idx=u_next.task_idx,
                gap_start=gap_start, gap_end=gap_end,
                tau_off=tau_off, tau_pre=tau_pre,
            ))
    return out


@dataclass
class _FirstUsePrefetch:
    """A backing-only object's first compute load. Required for the simulator to
    accept any task that consumes it — unlike `_RoundTrip` this isn't
    optional, it's a feasibility precondition."""
    obj_id: str
    size: int
    first_use_task_idx: int
    first_use_ideal_start: int
    tau_pre: int
    # filled by packer:
    prefetch_boundary_idx: int | None = None
    prefetch_start_t: int | None = None


def _enumerate_first_use_prefetches(
    bare: TaskChain,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    initial_compute: set[str],
) -> list[_FirstUsePrefetch]:
    """For each backing-only object that's ever consumed as input, emit a
    first-use prefetch requirement. Excludes objects already on compute
    initially (either bare's `initial_memory` or augmented by `initial_compute`).
    """
    if bare.bandwidth_from_slow is None:
        return []
    bw_h = bare.bandwidth_from_slow
    # Only backing-resident initial objects need a first-use prefetch.
    # Outputs of tasks are produced directly on compute — no backing source exists.
    backing_initial = {
        o.id for o in bare.initial_memory if o.location == "backing"
    }
    compute_initial = {
        o.id for o in bare.initial_memory if o.location == "fast"
    } | initial_compute
    out: list[_FirstUsePrefetch] = []
    for obj_id, events in uses_by_task.items():
        if obj_id not in backing_initial:
            continue  # not on backing originally; either compute-initial or a task output
        if obj_id in compute_initial:
            continue  # already on compute; no prefetch needed
        if not events:
            continue
        first = events[0]
        size = sizes[obj_id]
        out.append(_FirstUsePrefetch(
            obj_id=obj_id, size=size,
            first_use_task_idx=first.task_idx,
            first_use_ideal_start=first.ideal_start,
            tau_pre=max(1, math.ceil(size / bw_h)),
        ))
    return out


def _rank_roundtrips(
    candidates: list[_RoundTrip], bare: TaskChain
) -> list[_RoundTrip]:
    """Rank by expected benefit. Primary: bytes saved x gap length (bigger
    objects over longer windows free more capacity for longer). Secondary
    tiebreaker: prefer candidates whose gap spans more task boundaries
    (more places where the freed bytes help)."""
    def key(rt: _RoundTrip) -> tuple[int, int, int]:
        gap_len = rt.gap_end - rt.gap_start
        span_tasks = rt.next_use_task_idx - rt.prev_use_task_idx
        return (-rt.size * gap_len, -span_tasks, rt.prev_use_task_idx)
    return sorted(candidates, key=key)


def _try_v3_pack(
    bare: TaskChain,
    initial_compute: set[str],
    sizes: dict[str, int],
    ideal_starts: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    cap: int | None,
) -> tuple[dict[int, dict[TriggerKind, list[str]]], bool]:
    """Build a complete roundtrip_planner plan with this initial placement; return
    (annotations, all_first_uses_fit_within_cap)."""
    candidates = _enumerate_roundtrips(bare, sizes, uses_by_task, ideal_starts)
    candidates = _rank_roundtrips(candidates, bare)
    first_uses = _enumerate_first_use_prefetches(
        bare, sizes, uses_by_task, initial_compute,
    )
    annotations, _committed, all_fit = _pack_roundtrips(
        bare, initial_compute, candidates, first_uses, cap, sizes, ideal_starts,
    )
    return annotations, all_fit


def _pack_roundtrips(
    bare: TaskChain,
    initial_compute: set[str],
    candidates: list[_RoundTrip],
    first_uses: list[_FirstUsePrefetch],
    cap: int | None,
    sizes: dict[str, int],
    ideal_starts: dict[str, int],
) -> tuple[dict[int, dict[TriggerKind, list[str]]], list[_RoundTrip], bool]:
    """Greedily commit candidates onto the streams. First-use prefetches are
    mandatory (a feasibility precondition) and packed first; round-trips are
    optional optimizations packed after."""
    annotations: dict[int, dict[TriggerKind, list[str]]] = defaultdict(
        lambda: {"release": [], "offload": [], "prefetch": []}
    )

    n = len(bare.tasks)
    bps = _initial_bps(bare, initial_compute, sizes)

    task_end: list[int] = []
    cum = 0
    for t in bare.tasks:
        cum += t.runtime
        task_end.append(cum)
    task_start = [task_end[i] - bare.tasks[i].runtime for i in range(n)]

    to_slow_slots: list[_StreamSlot] = []
    from_slow_slots: list[_StreamSlot] = []
    committed: list[_RoundTrip] = []

    def find_slot(
        stream: list[_StreamSlot], earliest: int, deadline: int, tau: int
    ) -> tuple[int, int] | None:
        """Find an interval (start, start+tau) such that earliest <= start,
        start+tau <= deadline, and the interval doesn't overlap any
        existing slot on `stream`."""
        if deadline - earliest < tau:
            return None
        # Walk slots sorted by start; find first gap that fits.
        cursor = earliest
        for slot in stream:
            if slot.end <= cursor:
                continue
            if slot.start >= cursor + tau:
                # Gap [cursor, slot.start) fits.
                if cursor + tau <= deadline:
                    return (cursor, cursor + tau)
                return None
            # Overlap: jump cursor past this slot.
            cursor = max(cursor, slot.end)
            if cursor + tau > deadline:
                return None
        # Past all slots.
        if cursor + tau <= deadline:
            return (cursor, cursor + tau)
        return None

    def insert_slot(stream: list[_StreamSlot], slot: _StreamSlot) -> None:
        bisect.insort(stream, slot, key=lambda s: s.start)

    last_use_idx = _last_use_task_idx(bare)

    # Compute a "headroom budget" per boundary: how much we need to reduce bps
    # by to fit all mandatory first-use prefetches without overflow.
    # This drives DEMAND-DRIVEN round-trip commitment: don't add unnecessary
    # offloads if cap pressure is already satisfied (avoids hurting makespan
    # at loose caps).
    def first_use_demand_at(k: int) -> int:
        """How many bytes of first-use prefetch will need to land at-or-before
        boundary k for objects whose live range includes k."""
        if cap is None:
            return 0
        demand = 0
        for fp in first_uses:
            last = last_use_idx.get(fp.obj_id, n - 1)
            # Prefetch must arrive by first_use_ideal_start. Latest k satisfying
            # task_end[k] + tau_pre <= first_use_ideal_start; if that's <= k
            # then the prefetch will affect bps[k].
            latest_k = -1
            for kk in range(fp.first_use_task_idx - 1, -1, -1):
                if task_end[kk] + fp.tau_pre <= fp.first_use_ideal_start:
                    latest_k = kk
                    break
            if latest_k < 0:
                latest_k = 0
            if latest_k <= k <= last:
                demand += fp.size
        return demand

    # Per-boundary "next task's output reservation" — bps[k] + outputs(k+1)
    # must fit in cap because outputs are reserved at start of task k+1
    # (before its end-of-task triggers fire).
    next_outputs: list[int] = []
    for k in range(n):
        if k + 1 < n:
            sz = sum(
                o.size for o in bare.tasks[k + 1].outputs
                if o.location == "fast"
            )
            next_outputs.append(sz)
        else:
            next_outputs.append(0)

    def overflow_at(k: int) -> int:
        """How much (bps[k] + demand + next-task outputs) exceeds cap.
        Positive means we need round-trips here."""
        if cap is None:
            return 0
        return max(
            0,
            bps[k] + first_use_demand_at(k) + next_outputs[k] - cap,
        )

    # Backing-initial objects are read-only (workload contract: outputs always
    # introduce new obj_ids), so when round-tripping them we can RELEASE
    # the compute copy instead of OFFLOADING — the backing copy is byte-identical
    # and untouched. Saves to_slow bandwidth + frees bytes instantly.
    backing_initial = {
        o.id for o in bare.initial_memory if o.location == "backing"
    }

    # --- pass 1: demand-driven round-trip commitment ---
    if cap is not None:
        for rt in candidates:
            # Only commit if it would reduce overflow at some boundary in its
            # coverage range [k_off+1, k_pre]. Without overflow, skip — keeps
            # roundtrip_planner's plan minimal at loose caps so makespan matches belady_reactive.
            k_off = rt.prev_use_task_idx
            offload_fire_t = task_end[k_off]
            k_pre = -1
            for kp in range(rt.next_use_task_idx - 1, rt.prev_use_task_idx, -1):
                if task_end[kp] + rt.tau_pre <= rt.gap_end:
                    k_pre = kp
                    break
            if k_pre < 0:
                continue
            # Check if any boundary in coverage range is over-cap with first-use demand
            covers = range(k_off + 1, k_pre + 1)
            if not any(overflow_at(k) > 0 for k in covers):
                continue  # not needed; skip to keep makespan low

            prefetch_fire_t = task_end[k_pre]
            use_release = rt.obj_id in backing_initial
            if use_release:
                # Release-instead-of-offload: no to_slow slot needed, bytes free
                # instantly at offload_fire_t, prefetch only gated by k_pre.
                off_start, off_end = offload_fire_t, offload_fire_t
            else:
                to_slow = find_slot(to_slow_slots, offload_fire_t, rt.gap_end - rt.tau_pre, rt.tau_off)
                if to_slow is None:
                    continue
                off_start, off_end = to_slow
            from_slow_earliest = max(off_end, prefetch_fire_t)
            from_slow = find_slot(from_slow_slots, from_slow_earliest, rt.gap_end, rt.tau_pre)
            if from_slow is None:
                continue
            pre_start, pre_end = from_slow
            # Subtract rt.size from bps[k] ONLY at boundaries where the
            # object is actually off-compute. The object stays in the compute
            # pool until the to_slow completes at `off_end` (pending_outbound /
            # outbound state still occupies bytes), and reappears the moment
            # the from_slow starts (`pre_start`, state=inbound). For boundaries
            # k with task_end[k] inside [off_end, pre_start), the object is
            # genuinely off-compute and bps[k] drops by rt.size.
            for k in range(k_off + 1, k_pre + 1):
                if off_end <= task_end[k] < pre_start:
                    bps[k] -= rt.size
            rt.offload_boundary_idx = k_off
            rt.prefetch_boundary_idx = k_pre
            rt.offload_end_t = off_end
            rt.prefetch_start_t = pre_start
            if not use_release:
                insert_slot(to_slow_slots, _StreamSlot(off_start, off_end, rt.obj_id, "offload"))
            insert_slot(from_slow_slots, _StreamSlot(pre_start, pre_end, rt.obj_id, "prefetch"))
            annotations[k_off]["release" if use_release else "offload"].append(rt.obj_id)
            annotations[k_pre]["prefetch"].append(rt.obj_id)
            committed.append(rt)

    # --- pass 2: first-use prefetches (mandatory) ---
    all_first_use_fit = True
    for fp in sorted(first_uses, key=lambda f: f.first_use_task_idx):
        target_t = fp.first_use_ideal_start
        last = last_use_idx.get(fp.obj_id, n - 1)
        chosen_k: int | None = None
        chosen_slot: tuple[int, int] | None = None
        for k in range(fp.first_use_task_idx - 1, -1, -1):
            if task_end[k] + fp.tau_pre > target_t:
                continue
            fire_t = task_end[k]
            slot = find_slot(from_slow_slots, fire_t, target_t, fp.tau_pre)
            if slot is None:
                continue
            if cap is not None:
                # Peak over the live range — include next-task outputs at each
                # boundary so we don't put the next task's start over cap.
                peak_after = max(
                    bps[kk] + next_outputs[kk]
                    for kk in range(k, min(last + 1, n))
                ) if k < n else 0
                if peak_after + fp.size > cap:
                    continue
            chosen_k = k
            chosen_slot = slot
            break

        if chosen_k is None:
            all_first_use_fit = False
            # Fall back: pick latest stream-feasible boundary even if it
            # overflows bps. Caller (roundtrip_planner driver) can drop initial-pool items
            # to retry.
            for k in range(fp.first_use_task_idx - 1, -1, -1):
                fire_t = task_end[k]
                slot = find_slot(from_slow_slots, fire_t, target_t, fp.tau_pre)
                if slot is not None:
                    chosen_k = k
                    chosen_slot = slot
                    break
        if chosen_k is None:
            cursor = 0
            for s in from_slow_slots:
                if s.end > cursor:
                    cursor = s.end
            chosen_k = 0
            chosen_slot = (cursor, cursor + fp.tau_pre)

        ps, pe = chosen_slot
        fp.prefetch_boundary_idx = chosen_k
        fp.prefetch_start_t = ps
        insert_slot(from_slow_slots, _StreamSlot(ps, pe, fp.obj_id, "prefetch"))
        annotations[chosen_k]["prefetch"].append(fp.obj_id)
        for k in range(chosen_k, min(last + 1, n)):
            bps[k] += fp.size

    return annotations, committed, all_first_use_fit


def _initial_bps(
    bare: TaskChain, initial_compute: set[str], sizes: dict[str, int]
) -> list[int]:
    """Compute per-boundary compute pool size assuming NO planner triggers,
    only initial placement + accumulating outputs + structural GC of
    dead-after-last-use objects (the structural release that runs at the
    last-use boundary).

    `bps[k]` represents pool size at end of task k AFTER end-of-task triggers
    fire — including the structural release for any object whose last use
    was task k itself.
    """
    n = len(bare.tasks)
    initial_dev_bytes = sum(
        o.size for o in bare.initial_memory if o.location == "fast"
    )
    augmented_bytes = sum(sizes.get(oid, 0) for oid in initial_compute)
    cur = initial_dev_bytes + augmented_bytes

    last_use_idx = _last_use_task_idx(bare)
    live: set[str] = {
        o.id for o in bare.initial_memory if o.location == "fast"
    } | set(initial_compute)

    bps: list[int] = []
    for i, task in enumerate(bare.tasks):
        # Outputs become live at end of task i.
        for out in task.outputs:
            if out.location == "fast" and out.id not in live:
                cur += out.size
                live.add(out.id)
        # GC dead objects: anything with last_use_task_idx <= i. This includes
        # current-task inputs whose last use IS i — they were consumed during
        # the task, and the structural release annotation fires AFTER the task,
        # dropping them from the pool at boundary i.
        for oid in list(live):
            if last_use_idx.get(oid, -1) <= i:
                cur -= sizes.get(oid, 0)
                live.remove(oid)
        bps.append(cur)
    return bps


def _v3_pass(
    bare: TaskChain,
    initial_compute: set[str],
    sizes: dict[str, int],
    ideal_starts: dict[str, int],
    uses: dict[str, list[int]],
) -> tuple[dict[int, dict[TriggerKind, list[str]]], set[str]]:
    """roundtrip_planner forward pass: use the (slack-aware) initial placement passed in,
    enumerate + rank + pack round-trips, then place mandatory first-use
    prefetches against the post-round-trip bps.

    Returns (annotations, final_initial_compute). The initial_compute is
    unchanged from the input (no drop-retry — we discovered that dropping
    an initial-pool item just shifts the same bytes into a first-use
    prefetch, with zero bps benefit and added stream contention).
    """
    uses_by_task = _object_uses_by_task_idx(bare, ideal_starts)
    annotations, _all_fit = _try_v3_pack(
        bare, initial_compute, sizes, ideal_starts, uses_by_task,
        bare.fast_memory_capacity,
    )

    # Add structural GC releases: for each object that will end up compute-
    # resident, release after its last-use task. Mirrors the baseline
    # `_initial_bps` assumption that anything with `last_use_task_idx == i`
    # is GC'd at boundary i.
    last_use_idx = _last_use_task_idx(bare)
    seen_release: set[str] = set()
    for oid, last_i in last_use_idx.items():
        if oid in seen_release:
            continue
        is_compute_initial = any(
            o.id == oid and o.location == "fast" for o in bare.initial_memory
        )
        is_augmented = oid in initial_compute
        is_output = any(
            out.id == oid and out.location == "fast"
            for t in bare.tasks for out in t.outputs
        )
        # First-use prefetched backing-only objects also become compute-resident
        # — include them too.
        is_prefetched = any(
            oid in ann["prefetch"] for ann in annotations.values()
        )
        if not (is_compute_initial or is_augmented or is_output or is_prefetched):
            continue
        if last_i < 0 or last_i >= len(bare.tasks):
            continue
        annotations[last_i]["release"].append(oid)
        seen_release.add(oid)
    return annotations, initial_compute


# ---------- orchestrator ----------

def apply_roundtrip_planner_policy(
    bare: TaskChain,
    *,
    fast_memory_capacity: int | None = None,
    refinement_iters: int = 20,
) -> TaskChain:
    """roundtrip_planner proactive round-trip planner entry point.

    Pipeline:
      1. Compute structural views (ideal starts, sizes, uses).
      2. Conservative initial placement (`_initial_placement` — leaves slack
         for the round-trip pass to manage activations).
      3. roundtrip_planner forward pass (`_v3_pass`): enumerate + rank + pack round-trips,
         then schedule mandatory first-use prefetches, then add structural
         GC releases.
      4. Assemble annotations into a `TaskChain`.
      5. Insert gradient writebacks (training-workload convention).
      6. Verify with the simulator; refine on common errors.
    """
    if fast_memory_capacity is not None:
        bare = replace(bare, fast_memory_capacity=fast_memory_capacity)

    ideal_starts = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses = _compute_uses(bare, ideal_starts)

    initial_compute = _initial_placement(
        bare, bare.fast_memory_capacity, uses, sizes,
    )
    annotations, final_initial = _v3_pass(
        bare, initial_compute, sizes, ideal_starts, uses,
    )
    chain = _add_gradient_writebacks(_apply_annotations(
        bare, final_initial, annotations, bare.fast_memory_capacity,
    ))
    return _verify_and_refine(chain, max_iters=refinement_iters)
