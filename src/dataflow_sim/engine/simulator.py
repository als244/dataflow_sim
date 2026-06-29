"""Event-driven simulator with compute + from-slow + to-slow streams.

Three resources, each a single-server FIFO queue. Compute tasks may stall waiting
for inputs to become fast-resident or for fast memory capacity to free up. Transfers
themselves never stall — their preconditions are validated at trigger fire time
and raise on failure (treated as policy / authoring errors).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from dataflow_sim.core.reference_stream import compute_reference_stream
from dataflow_sim.core.validate import ValidationError, validate_chain
from dataflow_sim.core.schema import (
    ActiveTask,
    Event,
    EventLog,
    Location,
    MemoryEntry,
    MemoryTracePoint,
    MemoryState,
    ObjectType,
    Snapshot,
    Task,
    TaskChain,
    TaskInterval,
    TransferDirection,
)

COMPUTE_INPUT_LOC: Location = "fast"
INF = float(sys.maxsize)


class _PoolEntry:
    __slots__ = ("size", "state", "location", "type")

    def __init__(self, size: int, state: MemoryState, location: Location, type: ObjectType):
        self.size = size
        self.state = state
        self.location = location
        self.type = type


PoolKey = tuple[str, Location]


@dataclass
class _InFlight:
    obj_id: str
    direction: TransferDirection
    src_size: int  # bytes freed on compute at completion (to_slow only)
    start_t: float
    end_t: float


@dataclass
class _Queued:
    obj_id: str
    direction: TransferDirection
    src_size: int
    runtime: float
    # Destination-object type, used to instantiate the fast-memory entry at from_slow
    # start (from_slow allocation is deferred — pending from_slow transfers consume no
    # fast bytes until they actually begin on the stream). None for to_slow
    # (the fast entry already exists in pending_outbound state).
    dst_type: ObjectType | None = None


def _precompute_task_starts(chain: TaskChain) -> dict[str, float]:
    """Ideal start times assuming no stalls. Used for the reference stream's
    next-use timestamps. Actual scheduled start may be later due to stalls;
    that's a documented limitation.
    """
    starts: dict[str, int] = {}
    t = 0
    for task in chain.tasks:
        starts[task.id] = t
        t += task.runtime
    return starts


def _fast_memory_bands(pool: dict[PoolKey, _PoolEntry]) -> dict[str, int]:
    bands = {
        "weight": 0,
        "activation": 0,
        "gradient": 0,
        "optimizer": 0,
        "other": 0,
        "inbound": 0,
        "outbound": 0,
        "pending_outbound": 0,
    }
    for (_oid, loc), entry in pool.items():
        if loc != COMPUTE_INPUT_LOC:
            continue
        if entry.state in ("pending_inbound", "inbound"):
            band = "inbound"
        elif entry.state == "outbound":
            band = "outbound"
        elif entry.state == "pending_outbound":
            band = "pending_outbound"
        else:
            band = entry.type
        bands[band] += entry.size
    return bands


def _snapshot(
    t: float,
    pool: dict[PoolKey, _PoolEntry],
    active: ActiveTask | None,
    remaining_tasks: list[Task],
    task_starts: dict[str, float],
) -> Snapshot:
    refs = compute_reference_stream(remaining_tasks, task_starts)
    next_ref_by_id = {r.obj_id: r.ref_t for r in refs}

    memory = [
        MemoryEntry(
            id=oid,
            size=entry.size,
            location=loc,
            type=entry.type,
            state=entry.state,
            next_ref_t=next_ref_by_id.get(oid),
        )
        for (oid, loc), entry in pool.items()
    ]
    memory.sort(key=lambda m: (m.location, m.id))

    return Snapshot(
        memory=memory,
        total_size=sum(e.size for e in pool.values()),
        active_task=active,
        reference_stream=refs,
    )


def run(
    chain: TaskChain,
    *,
    validate: bool = True,
    snapshots: bool = True,
    memory_trace: bool = False,
) -> EventLog:
    """Execute the task chain and return a full event log with per-event snapshots.

    By default this uses a two-pass implementation: the first pass discovers the
    *actual* scheduled start time of each compute task (accounting for stalls);
    the second pass re-runs the simulation using those actual starts as the
    basis for every snapshot's reference-stream `next_t` values, so the panels
    show realistic next-use timestamps even when transfer stalls shift the
    schedule.

    Set ``snapshots=False`` for policy scoring: the simulator still validates
    runtime behavior and returns task/transfer intervals, but skips event
    snapshots and reference streams. Set ``memory_trace=True`` to retain a
    compact fast-memory plot trace without full snapshots.

    Set ``validate=False`` to skip the static prepass — useful for tests that
    deliberately exercise runtime error paths or for benchmarking the inner
    loop without prepass overhead.
    """
    if validate:
        validate_chain(chain)
    if not snapshots:
        return _run_impl(
            chain,
            task_starts_override=None,
            snapshots=False,
            memory_trace=memory_trace,
        )

    first = _run_impl(
        chain,
        task_starts_override=None,
        snapshots=False,
        memory_trace=False,
    )
    actual_starts = {
        iv.task_id: iv.start for iv in first.task_intervals if iv.track == "compute"
    }
    return _run_impl(
        chain,
        task_starts_override=actual_starts,
        snapshots=True,
        memory_trace=memory_trace,
    )


def _run_impl(
    chain: TaskChain,
    task_starts_override: dict[str, float] | None,
    *,
    snapshots: bool,
    memory_trace: bool,
) -> EventLog:
    pool: dict[PoolKey, _PoolEntry] = {}
    events: list[Event] = []
    intervals: list[TaskInterval] = []
    memory_points: list[MemoryTracePoint] = []

    # Stream state — each may be idle (None) or have a single in-flight transfer.
    in_flight: dict[TransferDirection, _InFlight | None] = {"from_slow": None, "to_slow": None}
    queue: dict[TransferDirection, list[_Queued]] = {"from_slow": [], "to_slow": []}
    compute_busy_until = 0
    # Prefetches waiting for their backing source to become live (because a to_slow
    # is still writing it). Keyed by the source obj_id; activated when the
    # corresponding to_slow completes inside `complete("to_slow", ...)`.
    deferred_prefetches: dict[str, list[_Queued]] = {}
    # Per-(direction, obj_id) instance counter so re-prefetches/re-offloads
    # of the same object produce DISTINCT TaskInterval.task_ids — the UI
    # keys timeline bars by task_id and colliding keys cause bars to
    # render with another instance's geometry.
    transfer_seq: dict[tuple[TransferDirection, str], int] = {}

    # --- initial memory ---
    initial_alloc = {"backing": 0, COMPUTE_INPUT_LOC: 0}
    for obj in chain.initial_memory:
        key: PoolKey = (obj.id, obj.location)
        if key in pool:
            raise ValueError(f"duplicate ({obj.id!r}, {obj.location!r}) in initial_memory")
        pool[key] = _PoolEntry(size=obj.size, state="live", location=obj.location, type=obj.type)
        initial_alloc[obj.location] += obj.size
    _check_initial_capacity(chain, initial_alloc)
    peak_fast_memory_bytes = initial_alloc["fast"]
    # Running per-location byte totals, kept in sync at every pool add/del so
    # capacity checks and peak tracking are O(1) instead of summing the pool.
    loc_total = {"backing": initial_alloc["backing"], COMPUTE_INPUT_LOC: initial_alloc[COMPUTE_INPUT_LOC]}

    # Reference-stream timestamps are only needed for UI-grade snapshots.
    ref_starts = (
        task_starts_override if task_starts_override is not None
        else _precompute_task_starts(chain)
    ) if snapshots else {}
    tasks = list(chain.tasks)

    # ---------- helpers (closures over pool, in_flight, queue, events, intervals) ----------

    def emit_memory_trace(t: float) -> None:
        if not memory_trace:
            return
        point = MemoryTracePoint(
            t=t,
            fast_bytes_by_band=_fast_memory_bands(pool),
        )
        if (
            memory_points
            and memory_points[-1].t == point.t
            and memory_points[-1].fast_bytes_by_band == point.fast_bytes_by_band
        ):
            return
        memory_points.append(point)

    def loc_free(loc: Location) -> int:
        cap = chain.fast_memory_capacity if loc == "fast" else chain.backing_memory_capacity
        if cap is None:
            return INF
        return cap - loc_total[loc]

    def transfer_runtime(direction: TransferDirection, size: int, override: float | None) -> float:
        if override is not None:
            return max(float(override), 0.0)
        bw = chain.bandwidth_from_slow if direction == "from_slow" else chain.bandwidth_to_slow
        if bw is None:
            raise ValueError(
                f"transfer ({direction}) needs bandwidth_{direction} set on TaskChain "
                f"or a per-trigger `runtime` override"
            )
        return max((size + bw - 1) // bw, 1)

    def predict_schedule(direction: TransferDirection, now: float) -> list[tuple[str, float, float, int]]:
        """Return [(obj_id, start_t, end_t, src_size), ...] for in-flight + queued."""
        out: list[tuple[str, float, float, int]] = []
        cursor = now
        ifl = in_flight[direction]
        if ifl is not None:
            out.append((ifl.obj_id, ifl.start_t, ifl.end_t, ifl.src_size))
            cursor = ifl.end_t
        for tx in queue[direction]:
            start = max(cursor, now)
            end = start + tx.runtime
            out.append((tx.obj_id, start, end, tx.src_size))
            cursor = end
        return out

    def emit(
        kind: str,
        t: float,
        snap_remaining_idx: int,
        active: ActiveTask | None = None,
        **kwargs,
    ) -> None:
        nonlocal peak_fast_memory_bytes
        peak_fast_memory_bytes = max(peak_fast_memory_bytes, loc_total["fast"])
        emit_memory_trace(t)
        if not snapshots:
            return
        events.append(
            Event(
                t=t,
                kind=kind,  # type: ignore[arg-type]
                snapshot=_snapshot(t, pool, active, tasks[snap_remaining_idx:], ref_starts),
                **kwargs,
            )
        )

    def try_start(direction: TransferDirection, now: float, snap_idx: int) -> None:
        """Pop+start the queue head if the destination has room. Destination
        bytes are allocated HERE (not at trigger fire), so a queued transfer
        consumes no destination memory until its turn on the stream. If the
        destination can't fit the transfer right now, the queue head BLOCKS
        — try_start will be re-attempted whenever memory may have freed."""
        if in_flight[direction] is not None:
            return
        if not queue[direction]:
            return
        tx = queue[direction][0]  # peek (don't pop yet)

        dst_loc: Location = "fast" if direction == "from_slow" else "backing"
        dst_cap = chain.fast_memory_capacity if dst_loc == "fast" else chain.backing_memory_capacity
        existing_dst = pool.get((tx.obj_id, dst_loc))
        # Capacity check at start time. If overwriting an existing entry of
        # the same size, those bytes are already counted — don't double-count.
        if dst_cap is not None:
            already_counted = existing_dst.size if existing_dst is not None else 0
            free = dst_cap - loc_total[dst_loc] + already_counted
            if free < tx.src_size:
                return  # block the queue head

        queue[direction].pop(0)

        # Create / update destination entry. Source state flips here for to_slow
        # (was pending_outbound, becomes outbound).
        if direction == "from_slow":
            if existing_dst is None:
                pool[(tx.obj_id, COMPUTE_INPUT_LOC)] = _PoolEntry(
                    size=tx.src_size, state="inbound",
                    location="fast", type=tx.dst_type or "other",
                )
                loc_total["fast"] += tx.src_size
            else:
                existing_dst.state = "inbound"
        else:  # to_slow
            pool[(tx.obj_id, COMPUTE_INPUT_LOC)].state = "outbound"
            if existing_dst is None:
                pool[(tx.obj_id, "backing")] = _PoolEntry(
                    size=tx.src_size, state="inbound",
                    location="backing", type=tx.dst_type or "other",
                )
                loc_total["backing"] += tx.src_size
            else:
                if existing_dst.size != tx.src_size:
                    raise ValueError(
                        f"to_slow overwrite size mismatch for {tx.obj_id!r}: "
                        f"existing backing entry {existing_dst.size} bytes vs "
                        f"fast-memory source {tx.src_size} bytes"
                    )
                existing_dst.state = "inbound"

        in_flight[direction] = _InFlight(
            obj_id=tx.obj_id,
            direction=direction,
            src_size=tx.src_size,
            start_t=now,
            end_t=now + tx.runtime,
        )
        seq_key = (direction, tx.obj_id)
        seq = transfer_seq.get(seq_key, 0)
        transfer_seq[seq_key] = seq + 1
        # First instance keeps the bare "from_slow:obj" id; subsequent ones get
        # a "#N" suffix so timeline bars don't collide on React keys. The
        # UI's displayLabel strips the suffix when rendering the bar text.
        task_id = (
            f"{direction}:{tx.obj_id}" if seq == 0
            else f"{direction}:{tx.obj_id}#{seq}"
        )
        intervals.append(
            TaskInterval(
                task_id=task_id,
                start=now,
                end=now + tx.runtime,
                track=direction,
            )
        )
        emit(
            "transfer_start",
            now,
            snap_idx,
            transfer_obj=tx.obj_id,
            transfer_direction=direction,
        )

    def complete(direction: TransferDirection, t: float, snap_idx: int) -> None:
        ifl = in_flight[direction]
        assert ifl is not None
        obj_id = ifl.obj_id
        if direction == "from_slow":
            pool[(obj_id, COMPUTE_INPUT_LOC)].state = "live"
        else:  # to_slow: free fast-memory source, backing dest becomes live
            loc_total[COMPUTE_INPUT_LOC] -= pool[(obj_id, COMPUTE_INPUT_LOC)].size
            del pool[(obj_id, COMPUTE_INPUT_LOC)]
            pool[(obj_id, "backing")].state = "live"
        in_flight[direction] = None
        emit(
            "transfer_end",
            t,
            snap_idx,
            transfer_obj=obj_id,
            transfer_direction=direction,
        )
        # Activate any prefetches that were waiting on this source becoming
        # live. Compute entry was just deleted (above); we just append to the
        # from_slow queue — destination allocation + cap check happens at try_start.
        if direction == "to_slow":
            waiters = deferred_prefetches.pop(obj_id, [])
            for tx in waiters:
                queue["from_slow"].append(tx)
                emit(
                    "transfer_enqueue",
                    t,
                    snap_idx,
                    transfer_obj=tx.obj_id,
                    transfer_direction="from_slow",
                )

    def advance(target_t: float, snap_idx: int) -> None:
        """Process any transfer completions with end_t <= target_t (in time order)."""
        while True:
            from_slow_end = in_flight["from_slow"].end_t if in_flight["from_slow"] else INF
            to_slow_end = in_flight["to_slow"].end_t if in_flight["to_slow"] else INF
            next_end = min(from_slow_end, to_slow_end)
            if next_end > target_t:
                break
            # tie-breaker: process from_slow before to_slow if equal
            if from_slow_end <= to_slow_end:
                complete("from_slow", from_slow_end, snap_idx)
                try_start("from_slow", from_slow_end, snap_idx)
            else:
                complete("to_slow", to_slow_end, snap_idx)
                try_start("to_slow", to_slow_end, snap_idx)
                # A to_slow completion may have unblocked a deferred prefetch on from_slow.
                try_start("from_slow", to_slow_end, snap_idx)

    def _deferred_prefetch_ready_t(inp: str, now: float) -> float:
        """When will the deferred prefetch for `inp` complete? Computes
        `to_slow_end → from_slow_start (= max(to_slow_end, from_slow_busy_until)) → from_slow_end`."""
        waiters = deferred_prefetches.get(inp)
        if not waiters:
            raise RuntimeError(
                f"inconsistent state: no deferred prefetch for {inp!r}"
            )
        to_slow_end = None
        for obj_id, _s, end, _sz in predict_schedule("to_slow", now):
            if obj_id == inp:
                to_slow_end = end
                break
        if to_slow_end is None:
            raise RuntimeError(
                f"inconsistent state: input {inp!r} has a deferred prefetch "
                f"but no to_slow is scheduled to make its source live"
            )
        # from_slow availability at to_slow_end (use the latest end of currently
        # scheduled from_slow transfers as a lower bound for stream-busy).
        from_slow_sched = predict_schedule("from_slow", now)
        from_slow_busy_until = max((end for _, _, end, _ in from_slow_sched), default=now)
        from_slow_start = max(to_slow_end, from_slow_busy_until)
        return from_slow_start + waiters[0].runtime

    def input_ready_t(inp: str, now: float) -> float:
        entry = pool.get((inp, COMPUTE_INPUT_LOC))
        if entry is None:
            # No fast-memory entry. Either: (a) a deferred prefetch is pending,
            # (b) a normal prefetch is queued but hasn't started yet (fast-memory
            # entry creation is now deferred to try_start), or (c) input
            # isn't scheduled at all.
            if inp in deferred_prefetches:
                return _deferred_prefetch_ready_t(inp, now)
            for obj_id, _start, end, _size in predict_schedule("from_slow", now):
                if obj_id == inp:
                    return end
            if (inp, "backing") in pool:
                raise ValueError(
                    f"input {inp!r} is on backing only (no fast copy and no scheduled prefetch)"
                )
            raise ValueError(f"input {inp!r} is not present in pool")
        if entry.state == "live":
            return now
        if entry.state == "inbound":
            for obj_id, _start, end, _size in predict_schedule("from_slow", now):
                if obj_id == inp:
                    return end
            raise RuntimeError(
                f"inconsistent state: input {inp!r} state={entry.state} but no scheduled from_slow found"
            )
        if entry.state in ("pending_outbound", "outbound"):
            # An offload is in flight. If a prefetch is queued to bring it
            # back, stall the compute until that round-trip completes.
            if inp in deferred_prefetches:
                return _deferred_prefetch_ready_t(inp, now)
            raise ValueError(
                f"input {inp!r} is being offloaded (state={entry.state}) "
                f"and no re-prefetch is scheduled; compute cannot use it"
            )
        if entry.state == "reserved":
            raise ValueError(f"input {inp!r} is reserved by another task (unexpected mid-chain)")
        raise ValueError(f"input {inp!r} has unknown state {entry.state!r}")

    def compute_outputs_ready_t(needed: int, now: float, task_id: str) -> float:
        if chain.fast_memory_capacity is None:
            return now
        free = loc_free(COMPUTE_INPUT_LOC)
        if free >= needed:
            return now
        # walk through scheduled to_slow completions, accumulating freed bytes
        to_slow_schedule = sorted(predict_schedule("to_slow", now), key=lambda x: x[2])
        for _obj_id, _start, end, src_size in to_slow_schedule:
            free += src_size
            if free >= needed:
                return end
        raise ValueError(
            f"task {task_id!r} cannot satisfy fast memory need of {needed} bytes "
            f"(current free + all scheduled offloads = {free}, capacity={chain.fast_memory_capacity})"
        )

    # ---------- main loop ----------

    for i, task in enumerate(tasks):
        # 1. earliest start time
        target_start = compute_busy_until
        readiness_probe_t = target_start
        for inp in task.inputs:
            target_start = max(target_start, input_ready_t(inp, readiness_probe_t))
        compute_outputs_size = sum(out.size for out in task.outputs if out.location == "fast")
        if compute_outputs_size > 0:
            target_start = max(
                target_start,
                compute_outputs_ready_t(compute_outputs_size, readiness_probe_t, task.id),
            )

        # 2. advance time to target_start (emit any transfer events that complete in [now, target_start])
        advance(target_start, i)

        # 2b. After advance, re-verify the task can actually run: every input
        # must be live in fast memory AND fast memory must have room for outputs.
        # `predict_schedule` (used by input_ready_t / compute_outputs_ready_t)
        # assumes queued transfers run back-to-back ignoring capacity, but a
            # queued from_slow head can be BLOCKED until a to_slow completion frees fast
        # bytes. When that happens the optimistic prediction is too early —
        # drain in-flight transfers one at a time until the preconditions
        # actually hold.
        def _ready_to_run() -> bool:
            for inp in task.inputs:
                e = pool.get((inp, COMPUTE_INPUT_LOC))
                if e is None or e.state != "live":
                    return False
            if compute_outputs_size > 0 and chain.fast_memory_capacity is not None:
                if loc_total["fast"] + compute_outputs_size > chain.fast_memory_capacity:
                    return False
            return True

        while not _ready_to_run():
            # Try to pop any queued transfer whose destination has freed up
            # (advance only fires try_start on completions; standalone calls
            # here catch the case where memory was released but no in-flight
            # transfer just ended).
            try_start("from_slow", target_start, i)
            try_start("to_slow", target_start, i)
            from_slow_end = in_flight["from_slow"].end_t if in_flight["from_slow"] else INF
            to_slow_end = in_flight["to_slow"].end_t if in_flight["to_slow"] else INF
            next_end = min(from_slow_end, to_slow_end)
            if next_end == INF:
                missing_inputs = [
                    inp for inp in task.inputs
                    if not ((inp, COMPUTE_INPUT_LOC) in pool
                            and pool[(inp, COMPUTE_INPUT_LOC)].state == "live")
                ]
                msg = []
                if missing_inputs:
                    msg.append(f"inputs {missing_inputs} not live in fast memory")
                if compute_outputs_size > 0 and chain.fast_memory_capacity is not None:
                    used = loc_total["fast"]
                    if used + compute_outputs_size > chain.fast_memory_capacity:
                        msg.append(
                            f"fast memory {used}+{compute_outputs_size} bytes > cap "
                            f"{chain.fast_memory_capacity}, no offloads in flight"
                        )
                raise ValueError(
                    f"task {task.id!r} deadlocked at t={target_start}: "
                    + "; ".join(msg)
                )
            target_start = next_end
            advance(target_start, i)

        # 3. backing-output capacity check (no backing stall mechanism; raise if insufficient)
        backing_outputs_size = sum(out.size for out in task.outputs if out.location == "backing")
        if chain.backing_memory_capacity is not None and loc_total["backing"] + backing_outputs_size > chain.backing_memory_capacity:
            raise ValueError(
                f"task {task.id!r} cannot allocate {backing_outputs_size} on backing: "
                f"used={loc_total['backing']}, capacity={chain.backing_memory_capacity}"
            )

        # 4. reserve outputs
        for out in task.outputs:
            out_key: PoolKey = (out.id, out.location)
            if out_key in pool:
                raise ValueError(
                    f"task {task.id!r} output ({out.id!r}, {out.location!r}) "
                    f"collides with existing object (state={pool[out_key].state})"
                )
            pool[out_key] = _PoolEntry(
                size=out.size, state="reserved", location=out.location, type=out.type
            )
            loc_total[out.location] += out.size

        # Hard invariant: after reserving outputs, fast memory must not
        # exceed capacity. The drain loop above should have ensured this;
        # this assertion catches any subtle bug in the drain/prediction
            # logic where a plan slipped past unchecked and over-committed
            # fast memory. Without this, the simulator could silently keep running
        # with pool > cap (subsequent triggers would emit weird errors
        # later, far from the root cause).
        if chain.fast_memory_capacity is not None:
            used = loc_total["fast"]
            if used > chain.fast_memory_capacity:
                raise ValueError(
                    f"task {task.id!r} over-committed fast memory: pool={used} bytes "
                    f"exceeds capacity={chain.fast_memory_capacity}. This is a plan or "
                    f"simulator invariant violation — every task must be reachable "
                    f"under cap given the schedule. Plan needs more aggressive "
                    f"offloads or a higher cap."
                )

        # 5. record interval, emit task_start
        end_t = target_start + task.runtime
        intervals.append(TaskInterval(task_id=task.id, start=target_start, end=end_t, track="compute"))
        compute_busy_until = end_t
        active = ActiveTask(id=task.id, ends_at=end_t)
        emit("task_start", target_start, i, active=active, task_id=task.id)

        # 6. advance to end_t (transfers may complete during compute)
        advance(end_t, i + 1)

        # 7. mark outputs live, emit task_end
        for out in task.outputs:
            pool[(out.id, out.location)].state = "live"
        emit("task_end", end_t, i + 1, task_id=task.id)

        # 8. releases
        if task.releases_after:
            for obj_id in task.releases_after:
                release_key: PoolKey = (obj_id, COMPUTE_INPUT_LOC)
                entry = pool.get(release_key)
                if entry is None:
                    raise ValueError(
                        f"task {task.id!r} cannot release {obj_id!r}: no fast copy in pool"
                    )
                if entry.state != "live":
                    raise ValueError(
                        f"task {task.id!r} cannot release {obj_id!r}: state={entry.state!r} (only live)"
                    )
                loc_total[COMPUTE_INPUT_LOC] -= entry.size
                del pool[release_key]
            emit(
                "release",
                end_t,
                i + 1,
                object_ids=list(task.releases_after),
            )

        # 9. offloads: validate, mark fast memory pending_outbound, enqueue to-slow.
        # Backing destination is DEFERRED to try_start("to_slow") — a queued
        # offload consumes no backing bytes until it actually begins on the
        # stream. For an overwrite (backing entry already live, will be updated
        # by this transfer), the existing backing entry stays in place in its
        # 'live' state and try_start will flip it to 'inbound'.
        for trig in task.offload_after:
            src = pool.get((trig.obj_id, COMPUTE_INPUT_LOC))
            if src is None or src.state != "live":
                cur = src.state if src else "absent"
                raise ValueError(
                    f"task {task.id!r} cannot offload {trig.obj_id!r}: "
                    f"fast entry state={cur!r} (must be 'live')"
                )
            existing_backing = pool.get((trig.obj_id, "backing"))
            if existing_backing is not None:
                if existing_backing.state != "live":
                    raise ValueError(
                        f"task {task.id!r} cannot offload {trig.obj_id!r}: "
                        f"existing backing entry not 'live' (state={existing_backing.state!r})"
                    )
                if existing_backing.size != src.size:
                    raise ValueError(
                        f"task {task.id!r} cannot offload {trig.obj_id!r}: "
                        f"size mismatch (fast={src.size}, backing={existing_backing.size})"
                    )
            rt = transfer_runtime("to_slow", src.size, trig.runtime)
            src.state = "pending_outbound"
            queue["to_slow"].append(_Queued(
                obj_id=trig.obj_id, direction="to_slow", src_size=src.size,
                runtime=rt, dst_type=src.type,
            ))
            emit(
                "transfer_enqueue",
                end_t,
                i + 1,
                transfer_obj=trig.obj_id,
                transfer_direction="to_slow",
            )

        # 10. prefetches: validate. Fast-memory destination is DEFERRED to
        #     try_start("from_slow") — a queued prefetch consumes no fast bytes
        #     until it actually begins on the stream. If a to_slow for the same
        #     object is still in flight (fast pending_outbound/outbound OR
        #     backing pending_inbound/inbound), DEFER until the to_slow completes
        #     via complete("to_slow"). Only truly unrecoverable cases (backing
        #     absent with no scheduled to_slow, or compute already present) raise.
        for trig in task.prefetch_after:
            src = pool.get((trig.obj_id, "backing"))
            dev = pool.get((trig.obj_id, COMPUTE_INPUT_LOC))

            # Already in (or coming into) fast memory — can't re-prefetch.
            if dev is not None and dev.state in ("live", "inbound"):
                raise ValueError(
                    f"task {task.id!r} cannot prefetch {trig.obj_id!r}: "
                    f"fast copy already exists (state={dev.state!r})"
                )

            in_flight_to_slow = (
                (dev is not None and dev.state in ("pending_outbound", "outbound"))
                or (src is not None and src.state in ("pending_inbound", "inbound"))
            )

            if in_flight_to_slow:
                # to_slow in flight for this object — defer enqueue until to_slow
                # completes. dst_type carried so try_start can allocate.
                dst_type = src.type if src is not None else dev.type  # type: ignore[union-attr]
                size = src.size if src is not None else dev.size      # type: ignore[union-attr]
                rt = transfer_runtime("from_slow", size, trig.runtime)
                deferred_prefetches.setdefault(trig.obj_id, []).append(_Queued(
                    obj_id=trig.obj_id, direction="from_slow",
                    src_size=size, runtime=rt, dst_type=dst_type,
                ))
                emit(
                    "transfer_deferred",
                    end_t,
                    i + 1,
                    transfer_obj=trig.obj_id,
                    transfer_direction="from_slow",
                )
                continue

            if src is None or src.state != "live":
                state_str = src.state if src is not None else "absent"
                raise ValueError(
                    f"task {task.id!r} cannot prefetch {trig.obj_id!r}: "
                    f"no recoverable backing source (state={state_str!r})"
                )

            rt = transfer_runtime("from_slow", src.size, trig.runtime)
            queue["from_slow"].append(_Queued(
                obj_id=trig.obj_id, direction="from_slow", src_size=src.size,
                runtime=rt, dst_type=src.type,
            ))
            emit(
                "transfer_enqueue",
                end_t,
                i + 1,
                transfer_obj=trig.obj_id,
                transfer_direction="from_slow",
            )

        # 11. pop queues if streams are idle
        try_start("from_slow", end_t, i + 1)
        try_start("to_slow", end_t, i + 1)

    # ---------- final drain ----------
    while in_flight["from_slow"] is not None or in_flight["to_slow"] is not None:
        from_slow_end = in_flight["from_slow"].end_t if in_flight["from_slow"] else INF
        to_slow_end = in_flight["to_slow"].end_t if in_flight["to_slow"] else INF
        if from_slow_end <= to_slow_end:
            complete("from_slow", from_slow_end, len(tasks))
            try_start("from_slow", from_slow_end, len(tasks))
        else:
            complete("to_slow", to_slow_end, len(tasks))
            try_start("to_slow", to_slow_end, len(tasks))
            # to_slow completion may unblock deferred from_slow prefetches.
            try_start("from_slow", to_slow_end, len(tasks))

    # Deadlock check. With destination allocation deferred until transfer
    # start, a queued transfer that never fits its destination would silently
    # vanish from the schedule. Raise a clear error instead.
    if queue["from_slow"] or queue["to_slow"] or deferred_prefetches:
        stuck_from_slow = [tx.obj_id for tx in queue["from_slow"]]
        stuck_to_slow = [tx.obj_id for tx in queue["to_slow"]]
        stuck_deferred = sorted(deferred_prefetches.keys())
        raise ValueError(
            f"simulator finished with transfers still queued — destination "
            f"capacity is too tight for them to ever start. "
            f"from_slow queue: {stuck_from_slow}, to_slow queue: {stuck_to_slow}, "
            f"deferred prefetches: {stuck_deferred}"
        )

    return EventLog(
        task_intervals=intervals,
        events=events,
        peak_fast_memory_bytes=peak_fast_memory_bytes,
        memory_trace=memory_points,
    )


def _check_initial_capacity(chain: TaskChain, alloc: dict[str, int]) -> None:
    for loc, cap in ((COMPUTE_INPUT_LOC, chain.fast_memory_capacity), ("backing", chain.backing_memory_capacity)):
        if cap is None:
            continue
        if alloc[loc] > cap:
            raise ValueError(
                f"<initial_memory> cannot allocate {alloc[loc]} on {loc}: capacity={cap}"
            )
