"""Event-driven simulator with compute + H→D + D→H streams.

Three resources, each a single-server FIFO queue. Compute tasks may stall waiting
for inputs to become device-live or for device capacity to free up. Transfers
themselves never stall — their preconditions are validated at trigger fire time
and raise on failure (treated as policy / authoring errors).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from dataflow_sim.reference_stream import compute_reference_stream
from dataflow_sim.validate import ValidationError, validate_chain
from dataflow_sim.schema import (
    ActiveTask,
    Event,
    EventLog,
    Location,
    MemoryEntry,
    MemoryState,
    ObjectType,
    Snapshot,
    Task,
    TaskChain,
    TaskInterval,
    TransferDirection,
)

COMPUTE_INPUT_LOC: Location = "device"
INF = sys.maxsize


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
    src_size: int  # bytes freed on device at completion (d2h only)
    start_t: int
    end_t: int


@dataclass
class _Queued:
    obj_id: str
    direction: TransferDirection
    src_size: int
    runtime: int
    # Destination-object type, used to instantiate the device entry at h2d
    # start (h2d allocation is deferred — pending h2d transfers consume no
    # device bytes until they actually begin on the stream). None for d2h
    # (the device entry already exists in pending_outbound state).
    dst_type: ObjectType | None = None


def _precompute_task_starts(chain: TaskChain) -> dict[str, int]:
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


def _location_total(pool: dict[PoolKey, _PoolEntry], loc: Location) -> int:
    return sum(e.size for (_, l), e in pool.items() if l == loc)


def _snapshot(
    t: int,
    pool: dict[PoolKey, _PoolEntry],
    active: ActiveTask | None,
    remaining_tasks: list[Task],
    task_starts: dict[str, int],
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


def run(chain: TaskChain, *, validate: bool = True) -> EventLog:
    """Execute the task chain and return a full event log with per-event snapshots.

    Two-pass implementation: the first pass discovers the *actual* scheduled
    start time of each compute task (accounting for stalls); the second pass
    re-runs the simulation using those actual starts as the basis for every
    snapshot's reference-stream `next_t` values, so the panels show realistic
    next-use timestamps even when transfer stalls shift the schedule.

    Set ``validate=False`` to skip the static prepass — useful for tests that
    deliberately exercise runtime error paths or for benchmarking the inner
    loop without prepass overhead.
    """
    if validate:
        validate_chain(chain)
    first = _run_impl(chain, task_starts_override=None)
    actual_starts = {
        iv.task_id: iv.start for iv in first.task_intervals if iv.track == "compute"
    }
    return _run_impl(chain, task_starts_override=actual_starts)


def _run_impl(
    chain: TaskChain,
    task_starts_override: dict[str, int] | None,
) -> EventLog:
    pool: dict[PoolKey, _PoolEntry] = {}
    events: list[Event] = []
    intervals: list[TaskInterval] = []

    # Stream state — each may be idle (None) or have a single in-flight transfer.
    in_flight: dict[TransferDirection, _InFlight | None] = {"h2d": None, "d2h": None}
    queue: dict[TransferDirection, list[_Queued]] = {"h2d": [], "d2h": []}
    compute_busy_until = 0
    # Prefetches waiting for their host source to become live (because a d2h
    # is still writing it). Keyed by the source obj_id; activated when the
    # corresponding d2h completes inside `complete("d2h", ...)`.
    deferred_prefetches: dict[str, list[_Queued]] = {}
    # Per-(direction, obj_id) instance counter so re-prefetches/re-offloads
    # of the same object produce DISTINCT TaskInterval.task_ids — the UI
    # keys timeline bars by task_id and colliding keys cause bars to
    # render with another instance's geometry.
    transfer_seq: dict[tuple[TransferDirection, str], int] = {}

    # --- initial memory ---
    initial_alloc = {"host": 0, "device": 0}
    for obj in chain.initial_memory:
        key: PoolKey = (obj.id, obj.location)
        if key in pool:
            raise ValueError(f"duplicate ({obj.id!r}, {obj.location!r}) in initial_memory")
        pool[key] = _PoolEntry(size=obj.size, state="live", location=obj.location, type=obj.type)
        initial_alloc[obj.location] += obj.size
    _check_initial_capacity(chain, initial_alloc)

    # Reference-stream timestamps: use actual starts if provided, else cumsum.
    ref_starts = (
        task_starts_override if task_starts_override is not None
        else _precompute_task_starts(chain)
    )
    tasks = list(chain.tasks)

    # ---------- helpers (closures over pool, in_flight, queue, events, intervals) ----------

    def loc_free(loc: Location) -> int:
        cap = chain.device_capacity if loc == "device" else chain.host_capacity
        if cap is None:
            return INF
        return cap - _location_total(pool, loc)

    def transfer_runtime(direction: TransferDirection, size: int, override: int | None) -> int:
        if override is not None:
            return max(int(override), 0)
        bw = chain.bandwidth_h2d if direction == "h2d" else chain.bandwidth_d2h
        if bw is None:
            raise ValueError(
                f"transfer ({direction}) needs bandwidth_{direction} set on TaskChain "
                f"or a per-trigger `runtime` override"
            )
        return max((size + bw - 1) // bw, 1)

    def predict_schedule(direction: TransferDirection, now: int) -> list[tuple[str, int, int, int]]:
        """Return [(obj_id, start_t, end_t, src_size), ...] for in-flight + queued."""
        out: list[tuple[str, int, int, int]] = []
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
        t: int,
        snap_remaining_idx: int,
        active: ActiveTask | None = None,
        **kwargs,
    ) -> None:
        events.append(
            Event(
                t=t,
                kind=kind,  # type: ignore[arg-type]
                snapshot=_snapshot(t, pool, active, tasks[snap_remaining_idx:], ref_starts),
                **kwargs,
            )
        )

    def try_start(direction: TransferDirection, now: int, snap_idx: int) -> None:
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

        dst_loc: Location = "device" if direction == "h2d" else "host"
        dst_cap = chain.device_capacity if dst_loc == "device" else chain.host_capacity
        existing_dst = pool.get((tx.obj_id, dst_loc))
        # Capacity check at start time. If overwriting an existing entry of
        # the same size, those bytes are already counted — don't double-count.
        if dst_cap is not None:
            already_counted = existing_dst.size if existing_dst is not None else 0
            free = dst_cap - _location_total(pool, dst_loc) + already_counted
            if free < tx.src_size:
                return  # block the queue head

        queue[direction].pop(0)

        # Create / update destination entry. Source state flips here for d2h
        # (was pending_outbound, becomes outbound).
        if direction == "h2d":
            if existing_dst is None:
                pool[(tx.obj_id, "device")] = _PoolEntry(
                    size=tx.src_size, state="inbound",
                    location="device", type=tx.dst_type or "other",
                )
            else:
                existing_dst.state = "inbound"
        else:  # d2h
            pool[(tx.obj_id, "device")].state = "outbound"
            if existing_dst is None:
                pool[(tx.obj_id, "host")] = _PoolEntry(
                    size=tx.src_size, state="inbound",
                    location="host", type=tx.dst_type or "other",
                )
            else:
                if existing_dst.size != tx.src_size:
                    raise ValueError(
                        f"d2h overwrite size mismatch for {tx.obj_id!r}: "
                        f"existing host entry {existing_dst.size} bytes vs "
                        f"device source {tx.src_size} bytes"
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
        # First instance keeps the bare "h2d:obj" id; subsequent ones get
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

    def complete(direction: TransferDirection, t: int, snap_idx: int) -> None:
        ifl = in_flight[direction]
        assert ifl is not None
        obj_id = ifl.obj_id
        if direction == "h2d":
            pool[(obj_id, "device")].state = "live"
        else:  # d2h: free device source, host dest becomes live
            del pool[(obj_id, "device")]
            pool[(obj_id, "host")].state = "live"
        in_flight[direction] = None
        emit(
            "transfer_end",
            t,
            snap_idx,
            transfer_obj=obj_id,
            transfer_direction=direction,
        )
        # Activate any prefetches that were waiting on this source becoming
        # live. Device entry was just deleted (above); we just append to the
        # h2d queue — destination allocation + cap check happens at try_start.
        if direction == "d2h":
            waiters = deferred_prefetches.pop(obj_id, [])
            for tx in waiters:
                queue["h2d"].append(tx)
                emit(
                    "transfer_enqueue",
                    t,
                    snap_idx,
                    transfer_obj=tx.obj_id,
                    transfer_direction="h2d",
                )

    def advance(target_t: int, snap_idx: int) -> None:
        """Process any transfer completions with end_t <= target_t (in time order)."""
        while True:
            h2d_end = in_flight["h2d"].end_t if in_flight["h2d"] else INF
            d2h_end = in_flight["d2h"].end_t if in_flight["d2h"] else INF
            next_end = min(h2d_end, d2h_end)
            if next_end > target_t:
                break
            # tie-breaker: process h2d before d2h if equal
            if h2d_end <= d2h_end:
                complete("h2d", h2d_end, snap_idx)
                try_start("h2d", h2d_end, snap_idx)
            else:
                complete("d2h", d2h_end, snap_idx)
                try_start("d2h", d2h_end, snap_idx)
                # A d2h completion may have unblocked a deferred prefetch on h2d.
                try_start("h2d", d2h_end, snap_idx)

    def _deferred_prefetch_ready_t(inp: str, now: int) -> int:
        """When will the deferred prefetch for `inp` complete? Computes
        `d2h_end → h2d_start (= max(d2h_end, h2d_busy_until)) → h2d_end`."""
        waiters = deferred_prefetches.get(inp)
        if not waiters:
            raise RuntimeError(
                f"inconsistent state: no deferred prefetch for {inp!r}"
            )
        d2h_end = None
        for obj_id, _s, end, _sz in predict_schedule("d2h", now):
            if obj_id == inp:
                d2h_end = end
                break
        if d2h_end is None:
            raise RuntimeError(
                f"inconsistent state: input {inp!r} has a deferred prefetch "
                f"but no d2h is scheduled to make its source live"
            )
        # h2d availability at d2h_end (use the latest end of currently
        # scheduled h2d transfers as a lower bound for stream-busy).
        h2d_sched = predict_schedule("h2d", now)
        h2d_busy_until = max((end for _, _, end, _ in h2d_sched), default=now)
        h2d_start = max(d2h_end, h2d_busy_until)
        return h2d_start + waiters[0].runtime

    def input_ready_t(inp: str, now: int) -> int:
        entry = pool.get((inp, "device"))
        if entry is None:
            # No device entry. Either: (a) a deferred prefetch is pending,
            # (b) a normal prefetch is queued but hasn't started yet (device
            # entry creation is now deferred to try_start), or (c) input
            # isn't scheduled at all.
            if inp in deferred_prefetches:
                return _deferred_prefetch_ready_t(inp, now)
            for obj_id, _start, end, _size in predict_schedule("h2d", now):
                if obj_id == inp:
                    return end
            if (inp, "host") in pool:
                raise ValueError(
                    f"input {inp!r} is on host only (no device copy and no scheduled prefetch)"
                )
            raise ValueError(f"input {inp!r} is not present in pool")
        if entry.state == "live":
            return now
        if entry.state == "inbound":
            for obj_id, _start, end, _size in predict_schedule("h2d", now):
                if obj_id == inp:
                    return end
            raise RuntimeError(
                f"inconsistent state: input {inp!r} state={entry.state} but no scheduled h2d found"
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

    def device_outputs_ready_t(needed: int, now: int, task_id: str) -> int:
        if chain.device_capacity is None:
            return now
        free = loc_free("device")
        if free >= needed:
            return now
        # walk through scheduled d2h completions, accumulating freed bytes
        d2h_schedule = sorted(predict_schedule("d2h", now), key=lambda x: x[2])
        for _obj_id, _start, end, src_size in d2h_schedule:
            free += src_size
            if free >= needed:
                return end
        raise ValueError(
            f"task {task_id!r} cannot satisfy device memory need of {needed} bytes "
            f"(current free + all scheduled offloads = {free}, capacity={chain.device_capacity})"
        )

    # ---------- main loop ----------

    for i, task in enumerate(tasks):
        # 1. earliest start time
        target_start = compute_busy_until
        for inp in task.inputs:
            target_start = max(target_start, input_ready_t(inp, target_start))
        device_outputs_size = sum(out.size for out in task.outputs if out.location == "device")
        if device_outputs_size > 0:
            target_start = max(
                target_start,
                device_outputs_ready_t(device_outputs_size, target_start, task.id),
            )

        # 2. advance time to target_start (emit any transfer events that complete in [now, target_start])
        advance(target_start, i)

        # 2b. After advance, re-verify the task can actually run: every input
        # must be live on device AND the device must have room for outputs.
        # `predict_schedule` (used by input_ready_t / device_outputs_ready_t)
        # assumes queued transfers run back-to-back ignoring capacity, but a
        # queued h2d head can be BLOCKED until a d2h completion frees device
        # bytes. When that happens the optimistic prediction is too early —
        # drain in-flight transfers one at a time until the preconditions
        # actually hold.
        def _ready_to_run() -> bool:
            for inp in task.inputs:
                e = pool.get((inp, COMPUTE_INPUT_LOC))
                if e is None or e.state != "live":
                    return False
            if device_outputs_size > 0 and chain.device_capacity is not None:
                if _location_total(pool, "device") + device_outputs_size > chain.device_capacity:
                    return False
            return True

        while not _ready_to_run():
            # Try to pop any queued transfer whose destination has freed up
            # (advance only fires try_start on completions; standalone calls
            # here catch the case where memory was released but no in-flight
            # transfer just ended).
            try_start("h2d", target_start, i)
            try_start("d2h", target_start, i)
            h2d_end = in_flight["h2d"].end_t if in_flight["h2d"] else INF
            d2h_end = in_flight["d2h"].end_t if in_flight["d2h"] else INF
            next_end = min(h2d_end, d2h_end)
            if next_end == INF:
                missing_inputs = [
                    inp for inp in task.inputs
                    if not ((inp, COMPUTE_INPUT_LOC) in pool
                            and pool[(inp, COMPUTE_INPUT_LOC)].state == "live")
                ]
                msg = []
                if missing_inputs:
                    msg.append(f"inputs {missing_inputs} not live on device")
                if device_outputs_size > 0 and chain.device_capacity is not None:
                    used = _location_total(pool, "device")
                    if used + device_outputs_size > chain.device_capacity:
                        msg.append(
                            f"device {used}+{device_outputs_size} bytes > cap "
                            f"{chain.device_capacity}, no offloads in flight"
                        )
                raise ValueError(
                    f"task {task.id!r} deadlocked at t={target_start}: "
                    + "; ".join(msg)
                )
            target_start = next_end
            advance(target_start, i)

        # 3. host-output capacity check (no host stall mechanism; raise if insufficient)
        host_outputs_size = sum(out.size for out in task.outputs if out.location == "host")
        if chain.host_capacity is not None and _location_total(pool, "host") + host_outputs_size > chain.host_capacity:
            raise ValueError(
                f"task {task.id!r} cannot allocate {host_outputs_size} on host: "
                f"used={_location_total(pool, 'host')}, capacity={chain.host_capacity}"
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

        # Hard invariant: after reserving outputs, device pool must not
        # exceed capacity. The drain loop above should have ensured this;
        # this assertion catches any subtle bug in the drain/prediction
        # logic where a plan slipped past unchecked and over-committed the
        # device. Without this, the simulator could silently keep running
        # with pool > cap (subsequent triggers would emit weird errors
        # later, far from the root cause).
        if chain.device_capacity is not None:
            used = _location_total(pool, "device")
            if used > chain.device_capacity:
                raise ValueError(
                    f"task {task.id!r} over-committed device: pool={used} bytes "
                    f"exceeds capacity={chain.device_capacity}. This is a plan or "
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
                        f"task {task.id!r} cannot release {obj_id!r}: no device copy in pool"
                    )
                if entry.state != "live":
                    raise ValueError(
                        f"task {task.id!r} cannot release {obj_id!r}: state={entry.state!r} (only live)"
                    )
                del pool[release_key]
            emit(
                "release",
                end_t,
                i + 1,
                object_ids=list(task.releases_after),
            )

        # 9. offloads: validate, mark device pending_outbound, enqueue D→H.
        # Host destination is DEFERRED to try_start("d2h") — a queued
        # offload consumes no host bytes until it actually begins on the
        # stream. For an overwrite (host entry already live, will be updated
        # by this transfer), the existing host entry stays in place in its
        # 'live' state and try_start will flip it to 'inbound'.
        for trig in task.offload_after:
            src = pool.get((trig.obj_id, "device"))
            if src is None or src.state != "live":
                cur = src.state if src else "absent"
                raise ValueError(
                    f"task {task.id!r} cannot offload {trig.obj_id!r}: "
                    f"device entry state={cur!r} (must be 'live')"
                )
            existing_host = pool.get((trig.obj_id, "host"))
            if existing_host is not None:
                if existing_host.state != "live":
                    raise ValueError(
                        f"task {task.id!r} cannot offload {trig.obj_id!r}: "
                        f"existing host entry not 'live' (state={existing_host.state!r})"
                    )
                if existing_host.size != src.size:
                    raise ValueError(
                        f"task {task.id!r} cannot offload {trig.obj_id!r}: "
                        f"size mismatch (device={src.size}, host={existing_host.size})"
                    )
            rt = transfer_runtime("d2h", src.size, trig.runtime)
            src.state = "pending_outbound"
            queue["d2h"].append(_Queued(
                obj_id=trig.obj_id, direction="d2h", src_size=src.size,
                runtime=rt, dst_type=src.type,
            ))
            emit(
                "transfer_enqueue",
                end_t,
                i + 1,
                transfer_obj=trig.obj_id,
                transfer_direction="d2h",
            )

        # 10. prefetches: validate. Device destination is DEFERRED to
        #     try_start("h2d") — a queued prefetch consumes no device bytes
        #     until it actually begins on the stream. If a d2h for the same
        #     object is still in flight (device pending_outbound/outbound OR
        #     host pending_inbound/inbound), DEFER until the d2h completes
        #     via complete("d2h"). Only truly unrecoverable cases (host
        #     absent with no scheduled d2h, or device already present) raise.
        for trig in task.prefetch_after:
            src = pool.get((trig.obj_id, "host"))
            dev = pool.get((trig.obj_id, "device"))

            # Already on (or coming onto) the device — can't re-prefetch.
            if dev is not None and dev.state in ("live", "inbound"):
                raise ValueError(
                    f"task {task.id!r} cannot prefetch {trig.obj_id!r}: "
                    f"device copy already exists (state={dev.state!r})"
                )

            in_flight_d2h = (
                (dev is not None and dev.state in ("pending_outbound", "outbound"))
                or (src is not None and src.state in ("pending_inbound", "inbound"))
            )

            if in_flight_d2h:
                # d2h in flight for this object — defer enqueue until d2h
                # completes. dst_type carried so try_start can allocate.
                dst_type = src.type if src is not None else dev.type  # type: ignore[union-attr]
                size = src.size if src is not None else dev.size      # type: ignore[union-attr]
                rt = transfer_runtime("h2d", size, trig.runtime)
                deferred_prefetches.setdefault(trig.obj_id, []).append(_Queued(
                    obj_id=trig.obj_id, direction="h2d",
                    src_size=size, runtime=rt, dst_type=dst_type,
                ))
                emit(
                    "transfer_deferred",
                    end_t,
                    i + 1,
                    transfer_obj=trig.obj_id,
                    transfer_direction="h2d",
                )
                continue

            if src is None or src.state != "live":
                state_str = src.state if src is not None else "absent"
                raise ValueError(
                    f"task {task.id!r} cannot prefetch {trig.obj_id!r}: "
                    f"no recoverable host source (state={state_str!r})"
                )

            rt = transfer_runtime("h2d", src.size, trig.runtime)
            queue["h2d"].append(_Queued(
                obj_id=trig.obj_id, direction="h2d", src_size=src.size,
                runtime=rt, dst_type=src.type,
            ))
            emit(
                "transfer_enqueue",
                end_t,
                i + 1,
                transfer_obj=trig.obj_id,
                transfer_direction="h2d",
            )

        # 11. pop queues if streams are idle
        try_start("h2d", end_t, i + 1)
        try_start("d2h", end_t, i + 1)

    # ---------- final drain ----------
    while in_flight["h2d"] is not None or in_flight["d2h"] is not None:
        h2d_end = in_flight["h2d"].end_t if in_flight["h2d"] else INF
        d2h_end = in_flight["d2h"].end_t if in_flight["d2h"] else INF
        if h2d_end <= d2h_end:
            complete("h2d", h2d_end, len(tasks))
            try_start("h2d", h2d_end, len(tasks))
        else:
            complete("d2h", d2h_end, len(tasks))
            try_start("d2h", d2h_end, len(tasks))
            # d2h completion may unblock deferred h2d prefetches.
            try_start("h2d", d2h_end, len(tasks))

    # Deadlock check. With destination allocation deferred until transfer
    # start, a queued transfer that never fits its destination would silently
    # vanish from the schedule. Raise a clear error instead.
    if queue["h2d"] or queue["d2h"] or deferred_prefetches:
        stuck_h2d = [tx.obj_id for tx in queue["h2d"]]
        stuck_d2h = [tx.obj_id for tx in queue["d2h"]]
        stuck_deferred = sorted(deferred_prefetches.keys())
        raise ValueError(
            f"simulator finished with transfers still queued — destination "
            f"capacity is too tight for them to ever start. "
            f"h2d queue: {stuck_h2d}, d2h queue: {stuck_d2h}, "
            f"deferred prefetches: {stuck_deferred}"
        )

    return EventLog(task_intervals=intervals, events=events)


def _check_initial_capacity(chain: TaskChain, alloc: dict[str, int]) -> None:
    for loc, cap in (("device", chain.device_capacity), ("host", chain.host_capacity)):
        if cap is None:
            continue
        if alloc[loc] > cap:
            raise ValueError(
                f"<initial_memory> cannot allocate {alloc[loc]} on {loc}: capacity={cap}"
            )
