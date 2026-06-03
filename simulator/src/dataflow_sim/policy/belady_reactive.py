"""belady_reactive reactive Belady auto-policy.

Drives a `ShadowSimulator` that mirrors the simulator's state machine. As the
planner walks forward through the bare chain, the shadow tracks pool state,
stream queues, and timing. Trigger decisions (release / offload / prefetch)
are made by querying the shadow for predicted device usage, predicted input
arrival times, and stream slack, then mutating the shadow to reflect the
decision.

Workload-agnostic: operates purely on the bare chain's compute structure
and the oracle reference stream — no knowledge of `W_i / A_i / dW_i`
conventions.
"""
from __future__ import annotations

import math
from dataclasses import replace

from dataflow_sim.schema import TaskChain
from dataflow_sim.policy._common import (
    _UseEvent,
    _add_gradient_writebacks,
    _compute_ideal_starts,
    _compute_uses,
    _last_use_task_idx,
    _next_use_after,
    _object_sizes,
    _object_uses_by_task_idx,
    _verify_and_refine,
)
from dataflow_sim.policy.shadow import INF, ShadowSimulator


# ---------- Phase 1 — initial placement ----------

def _smart_initial_placement(
    bare: TaskChain,
    device_capacity: int | None,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
) -> set[str]:
    """Capacity-aware initial placement that accounts for STRUCTURAL future
    pressure (task outputs accumulating + residual streams + reserved
    next-task outputs) when deciding which host objects to pre-place.

    Algorithm (first principles):
      1. For each task boundary k, compute `min_bps[k]` — the minimum bytes
         that MUST be on device at end of task k regardless of policy:
           * device-initial objects (live until their last use)
           * task outputs of tasks 0..k (live until their last use)
           * host objects whose first use is <= k+1 and last use is >= k+1
             (they have to be on device at start of task k+1 to be consumed)
         This is a lower bound on bps[k]; cap < min_bps[k] -> infeasible.

      2. `headroom[k] = cap - min_bps[k] - reserved_outputs[k+1]`. This is
         the slack a pre-placement can occupy at boundary k.

      3. T_1 inputs and T_1 device-located outputs are forced into the
         placement (no extra cost — they're already counted in min_bps[0]).

      4. Sort remaining host candidates by first-use ascending (tiebreak by
         size DESC so big objects compete for tight headroom first). For
         each candidate, pre-placement extends its window from
         [first_use - 1, last_use] to [0, last_use], so it adds `size` bytes
         to boundaries [0, first_use - 2]. Pre-place iff that fits within
         the running headroom at every such boundary.
    """
    n = len(bare.tasks)
    if n == 0:
        return set()

    already_device = {o.id for o in bare.initial_memory if o.location == "device"}
    host_objs = {o.id: o for o in bare.initial_memory if o.location == "host"}

    if device_capacity is None:
        return {oid for oid in host_objs if uses_by_task.get(oid)}

    cap = device_capacity
    last_use_idx = _last_use_task_idx(bare)

    def first_use_task_idx(oid: str) -> int:
        events = uses_by_task.get(oid, [])
        return events[0].task_idx if events else n  # n = never used

    # Reserved bytes for the next task's device-located outputs (they're
    # allocated at start of task k+1, so must fit alongside bps[k]).
    next_outputs = [0] * n
    for k in range(n - 1):
        next_outputs[k] = sum(
            o.size for o in bare.tasks[k + 1].outputs if o.location == "device"
        )

    # --- 1. minimum bps[k] given just-in-time prefetch for everything ---
    # Each entry contributes to bps[k] for k in [appears_k, last_k].
    entries: list[tuple[int, int, int]] = []  # (appears_k, last_k, size)
    for o in bare.initial_memory:
        if o.location == "device":
            last_k = last_use_idx.get(o.id, n - 1)
            entries.append((0, last_k, o.size))
    for i, task in enumerate(bare.tasks):
        for out in task.outputs:
            if out.location == "device":
                last_k = last_use_idx.get(out.id, n - 1)
                entries.append((i, last_k, out.size))
    for oid, obj in host_objs.items():
        first = first_use_task_idx(oid)
        if first >= n:
            continue
        # Just-in-time prefetch: object must be on device at start of task
        # `first`, i.e., at bps[first - 1] onward. (Clamped at 0 for T_1 inputs.)
        appears_k = max(0, first - 1)
        last_k = last_use_idx.get(oid, n - 1)
        entries.append((appears_k, last_k, obj.size))

    # `pessimistic_bps[k]`: bps assuming NOTHING gets evicted (initial-pool
    # objects, all outputs accumulating, just-in-time prefetched hosts). It's
    # an UPPER bound on what the planner has to manage — when it overflows
    # cap, the planner will need to offload/release things at runtime.
    pessimistic_bps = [0] * n
    for appears, last, sz in entries:
        for k in range(max(0, appears), min(n, last + 1)):
            pessimistic_bps[k] += sz

    # `headroom[k]` = slack a pre-placement can occupy at boundary k. Clamp
    # negative values to 0 — if pessimistic_bps already overflows, the
    # planner has to evict at runtime regardless of our initial choices, so
    # adding more pre-placements just makes things worse (don't).
    headroom = [
        max(0, cap - pessimistic_bps[k] - next_outputs[k]) for k in range(n)
    ]

    # --- 2. force T_1 inputs. Feasibility: T_1's inputs + reserved outputs
    # must fit in cap; otherwise the chain is unrunnable regardless of any
    # policy. This is the only HARD infeasibility — everything else can be
    # managed by runtime eviction/prefetch. ---
    t1 = bare.tasks[0]
    t1_inputs_size = sum(sizes.get(i, 0) for i in t1.inputs)
    t1_outputs_size = sum(o.size for o in t1.outputs if o.location == "device")
    if t1_inputs_size + t1_outputs_size > cap:
        raise ValueError(
            f"infeasible: task 0 inputs ({t1_inputs_size} bytes) + device-located "
            f"outputs ({t1_outputs_size} bytes) exceeds capacity ({cap})"
        )
    must_place = {oid for oid in t1.inputs if oid in host_objs and oid not in already_device}
    placement = set(must_place)

    # --- 3. greedy fill by first-use ascending, size DESC tiebreak ---
    extra_per_boundary = [0] * n  # bytes consumed by pre-placement extensions
    candidates = sorted(
        (oid for oid in host_objs
         if oid not in placement and uses_by_task.get(oid)),
        key=lambda o: (first_use_task_idx(o), -sizes[o]),
    )
    for oid in candidates:
        first = first_use_task_idx(oid)
        size = sizes[oid]
        # Pre-placement adds `size` to bps[k] for k in [0, first - 2].
        # If first <= 1, no extra (already counted in min_bps from boundary 0).
        extra_range = range(max(0, first - 1))
        fits = all(extra_per_boundary[k] + size <= headroom[k] for k in extra_range)
        if fits:
            placement.add(oid)
            for k in extra_range:
                extra_per_boundary[k] += size
    return placement


# ---------- Phase 2 belady_reactive — shadow-driven forward walk ----------

def _belady_pass_v2(
    bare: TaskChain,
    initial_device: set[str],
    uses: dict[str, list[int]],
    sizes: dict[str, int],
    ideal_starts: dict[str, int],
) -> ShadowSimulator:
    """Walk the bare chain with a ShadowSimulator, issuing triggers along the
    way. Returns the shadow with all decisions recorded; caller materializes
    via `shadow.to_annotated_chain()`.

    Key belady_reactive invariant: evictions are issued for *capacity at the actual task
    start time*, which may be later than `ideal_t` if scheduled offloads need
    time to complete. We compute the earliest feasible task start as a
    function of:
      * previous compute end
      * all inputs becoming live on device
      * device capacity having room for outputs (after pending offloads)
    """
    shadow = ShadowSimulator(bare)
    for oid in initial_device:
        shadow.add_to_initial_device(oid)

    cap = bare.device_capacity
    last_use_idx = _last_use_task_idx(bare)

    for i, task in enumerate(bare.tasks):
        ideal_t = ideal_starts[task.id]
        device_outputs_size = sum(
            o.size for o in task.outputs if o.location == "device"
        )
        input_set = set(task.inputs)
        output_set = {o.id for o in task.outputs if o.location == "device"}
        pinned = input_set | output_set

        # 1. Ensure inputs scheduled to arrive
        for inp in task.inputs:
            if shadow.predicted_input_ready_t(inp) > ideal_t:
                _ensure_prefetch_v2(shadow, inp, deadline=ideal_t,
                                    current_task_idx=i, ideal_starts=ideal_starts,
                                    uses=uses, sizes=sizes,
                                    extra_pinned=pinned)

        # 2. Ensure enough evictions are scheduled that capacity will fit at SOME
        #    feasible time. The actual start may be later than ideal_t (stall).
        if cap is not None:
            guard = 0
            tried_victims: set[str] = set()
            while True:
                candidate_t = _earliest_capacity_fit_t(
                    shadow, ideal_t, device_outputs_size, cap, task, pinned,
                )
                if candidate_t is not None:
                    break
                excluded = pinned | tried_victims
                victim = _pick_belady_victim_v2(shadow, excluded, uses, ideal_t)
                if victim is None:
                    cur_usage = shadow.predicted_device_usage_at(ideal_t)
                    raise ValueError(
                        f"widest-task infeasibility at {task.id!r}: "
                        f"pinned objects already exceed capacity. "
                        f"predicted usage at ideal_t={ideal_t}: {cur_usage}, "
                        f"outputs need {device_outputs_size}, capacity {cap}"
                    )
                placed = _evict_v2(shadow, victim, deadline=ideal_t,
                                   current_task_idx=i, ideal_starts=ideal_starts,
                                   sizes=sizes, uses=uses)
                if not placed:
                    tried_victims.add(victim)
                guard += 1
                if guard > 200:
                    raise RuntimeError(
                        f"eviction loop didn't converge at task {task.id!r}"
                    )

        # 3. Compute actual task start
        input_ready_max = max(
            (shadow.predicted_input_ready_t(inp) for inp in task.inputs),
            default=0,
        )
        if math.isinf(input_ready_max):
            missing = [inp for inp in task.inputs
                       if math.isinf(shadow.predicted_input_ready_t(inp))]
            raise ValueError(
                f"task {task.id!r}: inputs {missing} can never become device-live "
                f"(no host source and no scheduled transfer)"
            )
        capacity_ready = (
            _earliest_capacity_fit_t(shadow, ideal_t, device_outputs_size, cap, task, pinned)
            if cap is not None
            else ideal_t
        )
        if capacity_ready is None:
            capacity_ready = ideal_t  # shouldn't happen after the loop above
        actual_start = max(shadow.compute_busy_until, input_ready_max, ideal_t, capacity_ready)

        # 4. Advance shadow through task
        shadow.advance_to(actual_start)
        shadow.run_task(i, task, actual_start)

        # 5. Opportunistic garbage collect: release any device-live entry that
        #    no future task in chain order consumes. Task-index-based (rather
        #    than time-based on `_next_use_after`) because actual start times
        #    drift from ideal starts due to prefetch delays — a time-based
        #    check would prematurely release objects whose use was earlier in
        #    ideal time but later in the actual schedule.
        end_t = actual_start + task.runtime
        dead = [
            oid for (oid, loc), entry in list(shadow.pool.items())
            if loc == "device"
            and entry.state == "live"
            and oid not in (set(task.inputs) | {o.id for o in task.outputs})
            and last_use_idx.get(oid, -1) <= i
        ]
        for oid in dead:
            shadow.issue_release(oid, i, end_t)

    return shadow


def _earliest_capacity_fit_t(
    shadow: ShadowSimulator,
    earliest: int,
    outputs_size: int,
    cap: int,
    task,
    pinned: set[str],
) -> int | None:
    """Earliest time t >= `earliest` at which predicted device usage + outputs
    fits in `cap`, given currently-scheduled offload completions. Returns None
    if no such t exists with the current set of scheduled evictions (caller
    should schedule more).

    Also verifies that pinned objects (inputs + outputs) fit on their own; if
    not, no amount of eviction can save us (true infeasibility)."""
    # Sanity check: do pinned items even fit?
    pinned_size = sum(shadow.sizes.get(o, 0) for o in pinned)
    if pinned_size > cap:
        return None  # infeasible regardless

    # Candidate times: `earliest` and each scheduled d2h completion >= earliest
    candidates = [earliest]
    for tx in shadow.sched_d2h:
        if tx.end_at > earliest:
            candidates.append(tx.end_at)
    candidates = sorted(set(candidates))
    for t in candidates:
        if shadow.predicted_device_usage_at(t) + outputs_size <= cap:
            return t
    return None


def _ensure_prefetch_v2(
    shadow: ShadowSimulator,
    obj_id: str,
    deadline: int,
    current_task_idx: int,
    ideal_starts: dict[str, int],
    uses: dict[str, list[int]] | None = None,
    sizes: dict[str, int] | None = None,
    cascade_budget: int = 20,
    extra_pinned: set[str] | None = None,
) -> None:
    """Schedule a prefetch for obj_id arriving by deadline. Picks the latest
    prior task boundary that satisfies (a) host source is live, (b) transfer
    completes by deadline given stream contention, (c) device has capacity
    for the prefetch reservation.

    Cascade resolution: if (c) fails at a candidate boundary, issues one or
    more evictions at the same boundary to make room. `cascade_budget` bounds
    the number of evictions per prefetch."""
    if shadow.bw_h2d is None:
        raise ValueError("bandwidth_h2d required for prefetches")
    size = shadow.sizes[obj_id]
    tau = max(1, math.ceil(size / shadow.bw_h2d))

    host_ready = shadow.predicted_object_ready_t(obj_id, "host")
    if math.isinf(host_ready):
        return  # let simulator raise

    # Walk boundaries from latest to earliest
    for k in range(current_task_idx - 1, -1, -1):
        prev_task = shadow.chain.tasks[k]
        boundary_end = ideal_starts[prev_task.id] + prev_task.runtime
        # (a) boundary after host source becomes live
        if boundary_end < host_ready:
            continue  # earlier ones are also too early
        # (b) transfer completes by deadline given stream load
        stream_busy = _h2d_busy_at(shadow, boundary_end)
        predicted_end = max(boundary_end, stream_busy) + tau
        if predicted_end > deadline:
            continue
        # (c) device capacity at boundary k AND all later boundaries through
        #     current task — prefetching at k adds `size` to every bps[k..n-1].
        #     If any of those would exceed cap, this boundary doesn't work.
        #     Cascade only at the most-recent boundary (i-1), where the current
        #     shadow.pool is accurate for victim selection.
        if shadow.device_capacity is not None:
            cap = shadow.device_capacity
            # Find the peak bps across boundaries [k, current_task_idx-1]
            n_bps = len(shadow.boundary_pool_size)
            check_range_end = min(current_task_idx, n_bps)
            peak_bps = max(
                shadow.boundary_pool_size[kk]
                for kk in range(k, check_range_end)
            ) if k < check_range_end else 0
            if peak_bps + size > cap:
                if uses is None or sizes is None:
                    continue
                cascade_used = 0
                tried_victims: set[str] = set()
                while True:
                    peak_bps = max(
                        shadow.boundary_pool_size[kk]
                        for kk in range(k, check_range_end)
                    ) if k < check_range_end else 0
                    if peak_bps + size <= cap:
                        break
                    if cascade_used >= cascade_budget:
                        break
                    excluded = {obj_id} | (extra_pinned or set()) | tried_victims
                    actual_boundary_end = (
                        shadow.actual_boundary_end[k]
                        if k <= shadow.last_snapshotted_iter
                        else boundary_end
                    )
                    victim = _pick_belady_victim_v2(
                        shadow, excluded, uses, boundary_end,
                        appeared_by=actual_boundary_end,
                        safe_after=deadline,
                    )
                    if victim is None:
                        break
                    placed = _evict_v2(shadow, victim, deadline=boundary_end,
                                       current_task_idx=k + 1, ideal_starts=ideal_starts,
                                       sizes=sizes, uses=uses)
                    if not placed:
                        # This victim can't be safely evicted at this k; try another.
                        tried_victims.add(victim)
                        continue
                    cascade_used += 1
                peak_bps = max(
                    shadow.boundary_pool_size[kk]
                    for kk in range(k, check_range_end)
                ) if k < check_range_end else 0
                if peak_bps + size > cap:
                    continue  # cascade didn't free enough; try earlier boundary
        # All checks pass: issue it
        shadow.issue_prefetch(obj_id, k, boundary_end)
        return

    # Fallback: try boundary 0
    if shadow.chain.tasks:
        t0 = shadow.chain.tasks[0]
        boundary_end = ideal_starts[t0.id] + t0.runtime
        if shadow.device_capacity is None or \
           shadow.predicted_device_usage_at(boundary_end) + size <= shadow.device_capacity:
            try:
                shadow.issue_prefetch(obj_id, 0, boundary_end)
            except ValueError:
                pass


def _h2d_busy_at(shadow: ShadowSimulator, t: int) -> int:
    """When would a new transfer enqueued at time t actually start on h2d?
    (Accounts for currently-scheduled transfers ahead in the FIFO.)"""
    busy = t
    for tx in shadow.sched_h2d:
        if tx.enqueue_at <= t:
            busy = max(busy, tx.end_at)
    return busy


def _d2h_busy_at(shadow: ShadowSimulator, t: int) -> int:
    busy = t
    for tx in shadow.sched_d2h:
        if tx.enqueue_at <= t:
            busy = max(busy, tx.end_at)
    return busy


def _pick_belady_victim_v2(
    shadow: ShadowSimulator,
    pinned: set[str],
    uses: dict[str, list[int]],
    t: int,
    appeared_by: int | None = None,
    safe_after: int | None = None,
) -> str | None:
    """Pick the device-live object (not in `pinned`) with furthest next-use.

    `appeared_by`: if set, only consider victims whose `appeared_at <=
    appeared_by`. Used when cascading at a past boundary — the victim must
    have existed in pool by that boundary's end time.

    `safe_after`: if set, reject any victim with a use in (`appeared_by`,
    `safe_after`). Used when cascading at a past boundary k: if the victim
    is consumed by an intervening task (between boundary k and current iter),
    evicting it would break that task."""
    candidates: list[tuple[float, str]] = []
    for (oid, loc), entry in shadow.pool.items():
        if loc != "device":
            continue
        if entry.state != "live":
            continue
        if oid in pinned:
            continue
        if appeared_by is not None and entry.appeared_at > appeared_by:
            continue
        if safe_after is not None:
            # Reject victims with a use in (appeared_by, safe_after). When
            # appeared_by is None, use `t` as the lower bound (we don't want
            # to reject for already-past uses).
            lo = appeared_by if appeared_by is not None else t
            nxt = _next_use_after(uses, oid, lo)
            if nxt < safe_after:
                continue  # used by intervening task / this task
        next_t = _next_use_after(uses, oid, t + 1)
        candidates.append((next_t, oid))
    if not candidates:
        return None
    # Furthest first
    candidates.sort(key=lambda x: -x[0] if not math.isinf(x[0]) else -INF)
    return candidates[0][1]


def _evict_v2(
    shadow: ShadowSimulator,
    victim: str,
    deadline: int,
    current_task_idx: int,
    ideal_starts: dict[str, int],
    sizes: dict[str, int],
    uses: dict[str, list[int]],
) -> bool:
    """Issue release (if dead, OR if host already holds a live size-matched
    copy) or offload (if no host copy and victim has a future use). Picks the
    latest boundary whose offload completion frees the bytes by deadline.
    Returns True if a feasible trigger was placed, False if no boundary
    satisfies the timing constraint."""
    next_t = _next_use_after(uses, victim, deadline + 1)

    dev_entry = shadow.pool.get((victim, "device"))
    victim_producer_idx = dev_entry.producer_task_idx if dev_entry else -1
    # Latest ideal-time use up to deadline (the next task that consumes victim
    # before we'd want it evicted); release boundary must be strictly after.
    victim_uses = uses.get(victim, [])
    prior_uses_ideal = [u for u in victim_uses if u <= deadline]
    last_prior_use_ideal = prior_uses_ideal[-1] if prior_uses_ideal else -1

    # Workload contract: existing pool entries are never mutated by tasks
    # (an output always introduces a NEW obj_id, never overwrites an input).
    # So if a `live` host copy with matching size exists, the device copy is
    # byte-identical -> a release (instant, no d2h cost) is correct even if
    # the victim has future uses (those will re-prefetch from host).
    host_entry = shadow.pool.get((victim, "host"))
    host_has_copy = (
        host_entry is not None
        and host_entry.state == "live"
        and host_entry.size == sizes[victim]
    )

    if math.isinf(next_t) or host_has_copy:
        # Release: instant; pick the latest boundary in (last_use, deadline].
        for k in range(current_task_idx - 1, -1, -1):
            prev_task = shadow.chain.tasks[k]
            boundary_end = ideal_starts[prev_task.id] + prev_task.runtime
            if k < victim_producer_idx:
                continue  # victim not produced yet at this boundary
            if boundary_end <= last_prior_use_ideal:
                continue  # release would fire before victim's final consumer
            if boundary_end <= deadline:
                shadow.issue_release(victim, k, boundary_end)
                return True
        return False  # no boundary fits between last-use and deadline

    # Offload: ideal boundary depends on whether host already has this
    # object. For weight-like objects with a host copy we'd have taken the
    # release branch above; we only reach this point for objects WITHOUT a
    # host source — typically task outputs like activations. For those
    # there's no harm in offloading EAGERLY (earliest safe boundary):
    #   * frees device bytes ASAP (helps later operations breathe)
    #   * puts the d2h stream to work during forward (otherwise idle)
    #   * host copy materializes early, so a re-prefetch later is unblocked
    # Constraints stay the same: after producer + strictly after prior use,
    # no use of victim during the pending_outbound window, and stream-time
    # plus tau must fit by deadline.
    if shadow.bw_d2h is None:
        raise ValueError("bandwidth_d2h required for offload triggers")
    tau = max(1, math.ceil(sizes[victim] / shadow.bw_d2h))
    victim_uses = uses.get(victim, [])
    prior_uses = [u for u in victim_uses if u <= deadline]
    last_prior_use = prior_uses[-1] if prior_uses else -1
    fitting_candidates = []
    for k in range(current_task_idx - 1, -1, -1):
        prev_task = shadow.chain.tasks[k]
        boundary_end = ideal_starts[prev_task.id] + prev_task.runtime
        if k < victim_producer_idx:
            continue  # victim not produced yet at this boundary
        if boundary_end <= last_prior_use:
            continue  # would set victim pending_outbound while still in use
        stream_busy = _d2h_busy_at(shadow, boundary_end)
        predicted_end = max(boundary_end, stream_busy) + tau
        # No use of victim in [boundary_end, predicted_end] (pending_outbound window)
        bad = any(boundary_end <= u <= predicted_end for u in victim_uses)
        if bad:
            continue
        fitting_candidates.append((k, boundary_end, predicted_end))
    # Prefer the EARLIEST candidate whose predicted_end fits the deadline
    # (= last in our latest-first walk, so the smallest k). Falls back to
    # the latest safe-but-late candidate if none meet the deadline.
    in_time = [c for c in fitting_candidates if c[2] <= deadline]
    chosen = in_time[-1] if in_time else (fitting_candidates[0] if fitting_candidates else None)
    if chosen is not None:
        k, boundary_end, _ = chosen
        shadow.issue_offload(victim, k, boundary_end)
        return True
    return False  # no boundary is safe for this victim; caller picks another


# ---------- orchestrator ----------

def apply_belady_reactive_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
    refinement_iters: int = 20,
) -> TaskChain:
    """belady_reactive reactive Belady auto-policy entry point.

    Pipeline:
      1. Compute structural views (ideal starts, sizes, uses).
      2. Capacity-aware initial placement (`_smart_initial_placement`).
      3. Shadow-driven forward walk (`_belady_pass_v2`) — issues
         release / offload / prefetch triggers as memory pressure binds.
      4. Materialize annotations from the shadow.
      5. Insert gradient writebacks (training-workload convention).
      6. Verify with the simulator; refine on common errors.
    """
    if device_capacity is not None:
        bare = replace(bare, device_capacity=device_capacity)

    ideal_starts = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses = _compute_uses(bare, ideal_starts)
    uses_by_task = _object_uses_by_task_idx(bare, ideal_starts)

    initial_device = _smart_initial_placement(
        bare, bare.device_capacity, sizes, uses_by_task,
    )

    shadow = _belady_pass_v2(bare, initial_device, uses, sizes, ideal_starts)
    annotated = _add_gradient_writebacks(shadow.to_annotated_chain())
    return _verify_and_refine(annotated, max_iters=refinement_iters)
