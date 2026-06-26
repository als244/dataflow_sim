import pytest

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain, TransferTrigger
from dataflow_sim.engine.simulator import run


def _trig(x):
    if isinstance(x, str):
        return TransferTrigger(obj_id=x)
    return TransferTrigger(obj_id=x["id"], runtime=x.get("runtime"))


def _task(id_, inputs, runtime, outputs=None, releases=None, offloads=None, prefetches=None):
    return Task(
        id=id_,
        inputs=list(inputs),
        outputs=[OutputAlloc(**o) for o in (outputs or [])],
        runtime=runtime,
        releases_after=list(releases or []),
        offload_after=[_trig(x) for x in (offloads or [])],
        prefetch_after=[_trig(x) for x in (prefetches or [])],
    )


def _chain(initial, tasks):
    return TaskChain(
        initial_memory=[Object(**o) for o in initial],
        tasks=tasks,
    )


def test_single_task_emits_start_then_end():
    chain = _chain(
        [{"id": "w", "size": 10}, {"id": "x", "size": 5}],
        [_task("t0", ["w", "x"], runtime=4, outputs=[{"id": "y", "size": 7}])],
    )
    log = run(chain)

    assert [e.kind for e in log.events] == ["task_start", "task_end"]
    assert [e.t for e in log.events] == [0, 4]
    assert log.task_intervals[0].start == 0
    assert log.task_intervals[0].end == 4


def test_snapshot_free_run_keeps_intervals_without_events():
    chain = TaskChain(
        initial_memory=[
            Object(id="x", size=1, location="fast"),
            Object(id="w", size=10, location="backing"),
        ],
        tasks=[
            _task("t0", ["x"], runtime=4, prefetches=["w"]),
            _task("t1", ["w"], runtime=3),
        ],
        bandwidth_from_slow=5,
    )

    full = run(chain)
    simple = run(chain, snapshots=False)

    assert simple.events == []
    assert simple.peak_fast_memory_bytes == full.peak_fast_memory_bytes
    assert simple.memory_trace == []
    assert [(iv.task_id, iv.start, iv.end, iv.track) for iv in simple.task_intervals] == [
        (iv.task_id, iv.start, iv.end, iv.track) for iv in full.task_intervals
    ]


def test_snapshot_free_run_can_emit_compact_memory_trace():
    chain = TaskChain(
        initial_memory=[
            Object(id="x", size=1, location="fast", type="activation"),
            Object(id="w", size=10, location="backing", type="weight"),
        ],
        tasks=[
            _task("t0", ["x"], runtime=4, prefetches=["w"]),
            _task("t1", ["w"], runtime=3),
        ],
        bandwidth_from_slow=5,
    )

    log = run(chain, snapshots=False, memory_trace=True)

    assert log.events == []
    assert log.memory_trace
    assert any(
        point.fast_bytes_by_band["weight"] > 0
        or point.fast_bytes_by_band["inbound"] > 0
        for point in log.memory_trace
    )


def test_output_reserved_at_start_visible_at_end():
    chain = _chain(
        [{"id": "in", "size": 1}],
        [
            _task("t0", ["in"], runtime=3, outputs=[{"id": "out", "size": 100}]),
            _task("t1", ["out"], runtime=2),
        ],
    )
    log = run(chain)

    start_snap = log.events[0].snapshot
    assert start_snap.total_size == 101
    out_entry = next(m for m in start_snap.memory if m.id == "out")
    assert out_entry.state == "reserved"

    end_snap = log.events[1].snapshot
    out_entry_end = next(m for m in end_snap.memory if m.id == "out")
    assert out_entry_end.state == "live"


def test_releases_after_emits_release_event_and_frees_memory():
    chain = _chain(
        [{"id": "a", "size": 10}],
        [
            _task("t0", ["a"], runtime=2, outputs=[{"id": "b", "size": 3}], releases=["a"]),
        ],
    )
    log = run(chain)

    kinds = [e.kind for e in log.events]
    assert kinds == ["task_start", "task_end", "release"]

    release_evt = log.events[-1]
    assert release_evt.t == 2
    assert release_evt.object_ids == ["a"]
    assert {m.id for m in release_evt.snapshot.memory} == {"b"}
    assert release_evt.snapshot.total_size == 3


def test_missing_input_raises():
    # validate=False to exercise the simulator's runtime-side rejection;
    # the static prepass would catch this first with a different message.
    chain = _chain([], [_task("t0", ["gbacking"], runtime=1)])
    with pytest.raises(ValueError, match="not present"):
        run(chain, validate=False)


def test_output_visible_only_for_downstream_not_self():
    """An output declared by task t cannot be re-input to t itself."""
    chain = _chain(
        [],
        [_task("t0", ["self_out"], runtime=1, outputs=[{"id": "self_out", "size": 1}])],
    )
    with pytest.raises(ValueError, match="not present"):
        run(chain, validate=False)


def test_output_collision_raises():
    chain = _chain(
        [{"id": "x", "size": 1}],
        [_task("t0", [], runtime=1, outputs=[{"id": "x", "size": 1}])],
    )
    with pytest.raises(ValueError, match="collides"):
        run(chain, validate=False)


def test_release_nonexistent_raises():
    chain = _chain([], [_task("t0", [], runtime=1, releases=["nope"])])
    with pytest.raises(ValueError, match="cannot release"):
        run(chain, validate=False)


def test_reference_stream_in_snapshot_tracks_next_use():
    chain = _chain(
        [{"id": "w", "size": 1}],
        [
            _task("t0", ["w"], runtime=2, outputs=[{"id": "a", "size": 1}]),
            _task("t1", ["a"], runtime=3, outputs=[{"id": "b", "size": 1}]),
            _task("t2", ["w", "b"], runtime=1),
        ],
    )
    log = run(chain)

    # At TASK_START of t0 (t=0), w's next ref is by t0 itself at t=0.
    start_t0 = log.events[0].snapshot
    w_ref_at_start = next(r for r in start_t0.reference_stream if r.obj_id == "w")
    assert w_ref_at_start.ref_t == 0
    assert w_ref_at_start.ref_task == "t0"

    # At TASK_END of t0 (t=2), t0 has run; w's next ref is by t2 at t=5.
    end_t0 = log.events[1].snapshot
    w_ref_at_end = next(r for r in end_t0.reference_stream if r.obj_id == "w")
    assert w_ref_at_end.ref_t == 5
    assert w_ref_at_end.ref_task == "t2"


def test_reference_stream_in_snapshot_includes_future_outputs():
    """An object's first reference is the producer task allocating it, even
    when the consumer is further out."""
    chain = _chain(
        [],
        [
            _task("t0", [], runtime=2, outputs=[{"id": "future", "size": 1}]),
            _task("t1", ["future"], runtime=1),
        ],
    )
    log = run(chain)
    start_t0 = log.events[0].snapshot
    # future's first reference is t=0 by t0 (the producer), not t=2 by t1.
    r = next(r for r in start_t0.reference_stream if r.obj_id == "future")
    assert r.ref_t == 0
    assert r.ref_task == "t0"


def test_location_propagates_to_memory_entries():
    chain = TaskChain(
        initial_memory=[
            Object(id="w", size=10, location="backing"),
            Object(id="w", size=10, location="fast"),
            Object(id="x", size=5, location="fast"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["w", "x"],
                outputs=[OutputAlloc(id="y", size=7, location="fast")],
                runtime=1,
                releases_after=[],
            )
        ],
    )
    log = run(chain)
    end_snap = log.events[1].snapshot
    by_loc = {(m.id, m.location) for m in end_snap.memory}
    assert ("w", "backing") in by_loc
    assert ("w", "fast") in by_loc
    assert ("x", "fast") in by_loc
    assert ("y", "fast") in by_loc


def test_compute_rejects_backing_only_input():
    chain = TaskChain(
        initial_memory=[Object(id="w", size=10, location="backing")],
        tasks=[
            Task(id="t0", inputs=["w"], outputs=[], runtime=1, releases_after=[]),
        ],
    )
    # validate=False: the prepass catches backing-only inputs as
    # `released-then-referenced`; this test pins the runtime message.
    with pytest.raises(ValueError, match="backing only"):
        run(chain, validate=False)


def test_same_id_on_backing_and_compute_coexist():
    chain = TaskChain(
        initial_memory=[
            Object(id="W", size=10, location="backing"),
            Object(id="W", size=10, location="fast"),
        ],
        tasks=[Task(id="t0", inputs=["W"], outputs=[], runtime=1, releases_after=[])],
    )
    log = run(chain)
    assert len(log.events) == 2  # start + end (no releases)
    snap_ids = [(m.id, m.location) for m in log.events[0].snapshot.memory]
    assert ("W", "backing") in snap_ids
    assert ("W", "fast") in snap_ids


def test_fast_memory_capacity_enforced():
    """When a compute task can't fit its outputs and there's no in-flight offload
    to free space, the simulator raises (compute would stall forever)."""
    chain = TaskChain(
        initial_memory=[Object(id="x", size=8, location="fast")],
        tasks=[
            Task(
                id="t0",
                inputs=["x"],
                outputs=[OutputAlloc(id="y", size=5, location="fast")],
                runtime=1,
                releases_after=[],
            )
        ],
        fast_memory_capacity=10,
    )
    with pytest.raises(ValueError, match="cannot satisfy fast memory"):
        run(chain)


def test_backing_memory_capacity_enforced_on_initial_memory():
    chain = TaskChain(
        initial_memory=[
            Object(id="a", size=6, location="backing"),
            Object(id="b", size=6, location="backing"),
        ],
        tasks=[],
        backing_memory_capacity=10,
    )
    with pytest.raises(ValueError, match="cannot allocate"):
        run(chain)


def test_active_task_inputs_show_in_reference_stream_at_task_start():
    chain = TaskChain(
        initial_memory=[Object(id="w", size=1, location="fast")],
        tasks=[Task(id="t0", inputs=["w"], outputs=[], runtime=5, releases_after=[])],
    )
    log = run(chain)
    start_snap = log.events[0].snapshot  # TASK_START t0 at t=0
    w_ref = next(r for r in start_snap.reference_stream if r.obj_id == "w")
    assert w_ref.ref_t == 0
    assert w_ref.ref_task == "t0"
    # w_entry's next_ref_t in the memory panel should also reflect t=0
    w_entry = next(m for m in start_snap.memory if m.id == "w")
    assert w_entry.next_ref_t == 0


def test_task_intervals_match_runtime_sum():
    chain = _chain(
        [],
        [
            _task("t0", [], runtime=5),
            _task("t1", [], runtime=3),
            _task("t2", [], runtime=7),
        ],
    )
    log = run(chain)
    assert [(i.task_id, i.start, i.end) for i in log.task_intervals] == [
        ("t0", 0, 5),
        ("t1", 5, 8),
        ("t2", 8, 15),
    ]


# ============================================================
# Transfer + stall + capacity tests (Phase 2)
# ============================================================


def _xfer_chain(initial, tasks, *, bandwidth=4, compute_cap=None, backing_cap=None):
    return TaskChain(
        initial_memory=[Object(**o) for o in initial],
        tasks=tasks,
        fast_memory_capacity=compute_cap,
        backing_memory_capacity=backing_cap,
        bandwidth_from_slow=bandwidth,
        bandwidth_to_slow=bandwidth,
    )


def test_offload_emits_transfer_events_and_frees_compute():
    chain = _xfer_chain(
        [{"id": "w", "size": 8, "location": "fast"}],
        [
            _task("t0", ["w"], runtime=5, offloads=["w"]),
            _task("t1", [], runtime=1),
        ],
    )
    log = run(chain)
    kinds = [(e.t, e.kind, e.transfer_obj, e.transfer_direction) for e in log.events
             if e.kind in ("transfer_enqueue", "transfer_start", "transfer_end")]
    assert ("to_slow", 5, "transfer_enqueue") in [(d, t, k) for (t, k, _, d) in kinds]
    # transfer should start immediately (stream idle) and complete 8 bytes / 4 bw = 2 units later
    assert any(t == 5 and k == "transfer_start" and o == "w" and d == "to_slow" for (t, k, o, d) in kinds)
    assert any(t == 7 and k == "transfer_end" and o == "w" and d == "to_slow" for (t, k, o, d) in kinds)
    # at the end, w should be live on backing, gone from fast memory
    final = log.events[-1].snapshot
    by_loc = {(m.id, m.location): m.state for m in final.memory}
    assert by_loc == {("w", "backing"): "live"}


def test_prefetch_emits_transfer_events_and_lands_live_on_compute():
    chain = _xfer_chain(
        [{"id": "w", "size": 8, "location": "backing"}],
        [
            _task("t0", [], runtime=5, prefetches=["w"]),
            _task("t1", [], runtime=1),
        ],
    )
    log = run(chain)
    # from_slow should fire after t0
    transfer_events = [(e.t, e.kind, e.transfer_obj, e.transfer_direction) for e in log.events
                       if e.kind.startswith("transfer_")]
    assert ("to_slow" not in [d for (_, _, _, d) in transfer_events])
    assert any(k == "transfer_end" and o == "w" and d == "from_slow" for (_, k, o, d) in transfer_events)
    final = log.events[-1].snapshot
    by_loc = {(m.id, m.location): m.state for m in final.memory}
    assert by_loc.get(("w", "backing")) == "live"
    assert by_loc.get(("w", "fast")) == "live"


def test_compute_stalls_waiting_on_inbound_prefetch():
    """t0 prefetches w (8 bytes / bw=2 = 4 unit transfer), t1 takes 1 unit but
    needs w on compute. t1 should stall until prefetch completes."""
    chain = _xfer_chain(
        [{"id": "w", "size": 8, "location": "backing"}],
        [
            _task("t0", [], runtime=3, prefetches=["w"]),
            _task("t1", ["w"], runtime=1),
        ],
        bandwidth=2,  # 8/2 = 4 unit transfer
    )
    log = run(chain)
    intervals = {iv.task_id: (iv.start, iv.end) for iv in log.task_intervals}
    assert intervals["t0"] == (0, 3)
    # transfer enqueued at 3, starts at 3, ends at 3+4=7. t1 must wait until 7.
    assert intervals["t1"] == (7, 8)


def test_compute_waits_for_needed_queued_inputs_not_unneeded_tail():
    """A task with multiple queued inputs should start once those inputs are
    live, not after later from-slow queue entries that the task does not need."""
    chain = _xfer_chain(
        [
            {"id": "X", "size": 10, "location": "backing"},
            {"id": "A", "size": 10, "location": "backing"},
            {"id": "B", "size": 10, "location": "backing"},
            {"id": "C", "size": 10, "location": "backing"},
        ],
        [
            _task("t0", [], runtime=1, prefetches=["X", "A", "B", "C"]),
            _task("t1", ["A", "B"], runtime=1),
        ],
        bandwidth=1,
    )
    log = run(chain)
    intervals = {iv.task_id: (iv.start, iv.end) for iv in log.task_intervals}

    assert intervals["from_slow:X"] == (1, 11)
    assert intervals["from_slow:A"] == (11, 21)
    assert intervals["from_slow:B"] == (21, 31)
    assert intervals["from_slow:C"] == (31, 41)
    assert intervals["t1"] == (31, 32)


def test_compute_stalls_waiting_on_fast_memory_capacity_to_free():
    """compute capacity is tight; an offload must complete before the next task
    can allocate its output."""
    chain = _xfer_chain(
        [{"id": "a", "size": 6, "location": "fast"}],
        [
            _task("t0", ["a"], runtime=1, offloads=["a"]),
            _task("t1", [], runtime=1, outputs=[{"id": "b", "size": 8, "location": "fast"}]),
        ],
        bandwidth=2,
        compute_cap=8,
        backing_cap=16,
    )
    log = run(chain)
    intervals = {iv.task_id: (iv.start, iv.end) for iv in log.task_intervals}
    # t0 ends at 1, offload of `a` (6 bytes / bw 2 = 3 units) ends at 4 — at which
    # point compute is freed and t1 can allocate its 8-byte output.
    assert intervals["t0"] == (0, 1)
    assert intervals["t1"] == (4, 5)


def test_offload_deadlocks_when_backing_memory_capacity_too_tight():
    """With deferred destination allocation, an offload that can never fit
    on the backing stays queued forever; the simulator raises at end-of-run."""
    chain = _xfer_chain(
        [{"id": "w", "size": 10, "location": "fast"}],
        [_task("t0", ["w"], runtime=1, offloads=["w"])],
        bandwidth=2,
        backing_cap=5,
    )
    with pytest.raises(ValueError, match="to_slow queue.*'w'"):
        run(chain)


def test_prefetch_deadlocks_when_fast_memory_capacity_too_tight():
    """With deferred destination allocation, a prefetch that can never fit
    on the compute stays queued forever; the simulator raises at end-of-run."""
    chain = _xfer_chain(
        [
            {"id": "blocker", "size": 8, "location": "fast"},
            {"id": "w", "size": 10, "location": "backing"},
        ],
        [_task("t0", ["blocker"], runtime=1, prefetches=["w"])],
        bandwidth=2,
        compute_cap=8,
    )
    with pytest.raises(ValueError, match="from_slow queue.*'w'"):
        run(chain)


def test_transfer_queue_serializes_multiple_offloads():
    """Two offloads from the same task queue on to-slow; the second starts only
    when the first completes."""
    chain = _xfer_chain(
        [
            {"id": "a", "size": 4, "location": "fast"},
            {"id": "b", "size": 4, "location": "fast"},
        ],
        [_task("t0", ["a", "b"], runtime=1, offloads=["a", "b"])],
        bandwidth=2,  # 4/2 = 2 units per transfer
    )
    log = run(chain)
    from_slow_intervals = [iv for iv in log.task_intervals if iv.track == "to_slow"]
    assert len(from_slow_intervals) == 2
    # First transfer: 1..3; second: 3..5 (queued)
    starts = sorted(iv.start for iv in from_slow_intervals)
    ends = sorted(iv.end for iv in from_slow_intervals)
    assert starts == [1, 3]
    assert ends == [3, 5]


def test_per_trigger_runtime_overrides_bandwidth():
    chain = _xfer_chain(
        [{"id": "w", "size": 100, "location": "fast"}],
        [_task("t0", ["w"], runtime=1,
               offloads=[{"id": "w", "runtime": 5}])],
        bandwidth=2,  # would otherwise take 50 units
    )
    log = run(chain)
    end = next(e.t for e in log.events
               if e.kind == "transfer_end" and e.transfer_obj == "w")
    assert end == 1 + 5  # task end (1) + override runtime (5)


def test_missing_bandwidth_raises_when_used():
    chain = TaskChain(
        initial_memory=[Object(id="w", size=4, location="fast")],
        tasks=[_task("t0", ["w"], runtime=1, offloads=["w"])],
        # bandwidth_to_slow not set, no per-trigger override
    )
    with pytest.raises(ValueError, match="bandwidth"):
        run(chain)


def test_offload_overwrites_existing_backing_copy():
    """Offload with an existing live backing copy = write-back. Backing entry
    stays 'live' until the to_slow actually begins (destination is deferred —
    no backing bytes consumed by a merely-queued overwrite either), then
    flips to 'inbound' for the duration, and lands back on 'live' at
    completion. No new backing capacity consumed."""
    chain = _xfer_chain(
        [
            {"id": "w", "size": 4, "location": "fast"},
            {"id": "w", "size": 4, "location": "backing"},
        ],
        [_task("t0", ["w"], runtime=1, offloads=["w"])],
        bandwidth=2,
    )
    log = run(chain)
    # At trigger fire (enqueue), the backing entry is still 'live' — destination
    # allocation is deferred to transfer-start.
    enqueue = next(
        e for e in log.events
        if e.kind == "transfer_enqueue" and e.transfer_obj == "w"
    )
    backing_w_at_enqueue = next(
        m for m in enqueue.snapshot.memory if m.id == "w" and m.location == "backing"
    )
    assert backing_w_at_enqueue.state == "live"
    # When the to_slow actually starts, the backing entry flips to 'inbound'.
    start = next(
        e for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "w"
    )
    backing_w_at_start = next(
        m for m in start.snapshot.memory if m.id == "w" and m.location == "backing"
    )
    assert backing_w_at_start.state == "inbound"
    # On completion, backing w is live (and compute gone).
    final = log.events[-1].snapshot
    by_loc = {(m.id, m.location): m.state for m in final.memory}
    assert by_loc == {("w", "backing"): "live"}


def test_offload_overwrite_size_mismatch_raises():
    chain = _xfer_chain(
        [
            {"id": "w", "size": 4, "location": "fast"},
            {"id": "w", "size": 8, "location": "backing"},
        ],
        [_task("t0", ["w"], runtime=1, offloads=["w"])],
        bandwidth=2,
    )
    with pytest.raises(ValueError, match="size mismatch"):
        run(chain)


def test_prefetch_already_on_compute_raises():
    chain = _xfer_chain(
        [
            {"id": "w", "size": 4, "location": "fast"},
            {"id": "w", "size": 4, "location": "backing"},
        ],
        [_task("t0", ["w"], runtime=1, prefetches=["w"])],
        bandwidth=2,
    )
    with pytest.raises(ValueError, match="fast copy already exists"):
        run(chain, validate=False)


# ---------- deferred prefetch: source is still mid-to_slow ----------

def test_prefetch_defers_when_source_still_offloading():
    """Triggering a prefetch while the source's to_slow is still in flight
    must DEFER the from_slow enqueue (not raise). The from_slow auto-enqueues when
    the to_slow completes and any downstream compute correctly stalls."""
    chain = _xfer_chain(
        [{"id": "X", "size": 10, "location": "fast"}],
        [
            # t0 (1µs): fires offload of X. to_slow enqueues at 1, runtime=5,
            # so to_slow is in_flight 1→6.
            _task("t0", inputs=[], runtime=1, offloads=["X"]),
            # t1 (1µs): fires prefetch of X. At t1.end=2, to_slow is mid-flight
            # (X compute state=outbound). Must DEFER, not raise.
            _task("t1", inputs=[], runtime=1, prefetches=["X"]),
            # t2 needs X. to_slow ends at 6 → from_slow starts at 6, ends at 11.
            # t2 must stall until 11.
            _task("t2", inputs=["X"], runtime=1),
        ],
        bandwidth=2,  # size 10 / bw 2 = 5µs per transfer
    )
    log = run(chain)
    n_def = sum(1 for e in log.events if e.kind == "transfer_deferred")
    assert n_def == 1
    t2_iv = next(iv for iv in log.task_intervals if iv.task_id == "t2")
    assert t2_iv.start == 11, f"expected t2 to stall until 11, got {t2_iv.start}"
    # Both ends are paired with starts.
    starts = sum(1 for e in log.events if e.kind == "transfer_start")
    ends = sum(1 for e in log.events if e.kind == "transfer_end")
    assert starts == 2 and ends == 2  # one to_slow + one from_slow


def test_prefetch_defers_when_backing_source_pending_inbound():
    """If the to_slow is queued (not yet in flight) so the backing source is
    still `pending_inbound`, prefetch also defers."""
    chain = _xfer_chain(
        [
            # Two objects: A and X. Offload A first to keep to_slow busy.
            {"id": "A", "size": 10, "location": "fast"},
            {"id": "X", "size": 10, "location": "fast"},
        ],
        [
            # t0: offload BOTH A and X. to_slow FIFO: A starts at 1 ends at 6,
            # X queued, starts at 6 ends at 11. At t1.end (=2), X is still
            # pending_outbound (waiting in queue) — defer.
            _task("t0", inputs=[], runtime=1, offloads=["A", "X"]),
            _task("t1", inputs=[], runtime=1, prefetches=["X"]),
            _task("t2", inputs=["X"], runtime=1),
        ],
        bandwidth=2,
    )
    log = run(chain)
    assert sum(1 for e in log.events if e.kind == "transfer_deferred") == 1
    t2_iv = next(iv for iv in log.task_intervals if iv.task_id == "t2")
    # X's to_slow ends at 11; from_slow enqueues at 11, starts at 11 (stream idle by
    # then since A's from_slow wasn't requested), ends at 16. t2 starts at 16.
    assert t2_iv.start == 16


def test_truly_absent_backing_source_still_raises():
    """Prefetch where the backing source neither exists nor is in flight
    is still a hard error (matches the user's invariant)."""
    chain = _xfer_chain(
        [{"id": "A", "size": 4, "location": "fast"}],
        [_task("t0", inputs=[], runtime=1, prefetches=["X"])],  # X doesn't exist
        bandwidth=2,
    )
    with pytest.raises(ValueError, match="no recoverable backing source"):
        run(chain, validate=False)


# ---------- deferred destination allocation (symmetric from_slow + to_slow) ----------

def test_queued_prefetch_consumes_no_compute_bytes_until_transfer_starts():
    """A prefetch enqueued at time T but blocked behind another from_slow should
    not occupy any compute bytes while it waits in the queue."""
    chain = _xfer_chain(
        [
            {"id": "A", "size": 10, "location": "backing"},
            {"id": "B", "size": 10, "location": "backing"},
        ],
        [
            # t0 fires both prefetches; A goes in-flight, B queues behind it.
            _task("t0", inputs=[], runtime=1, prefetches=["A", "B"]),
            # padding so the schedule extends past the first from_slow
            _task("t1", inputs=[], runtime=1),
        ],
        bandwidth=2,  # 10/2 = 5µs per transfer
        compute_cap=20,
    )
    log = run(chain)
    # Take the snapshot at A's transfer_start: A is now in_flight (compute
    # entry created), B is still queued (no compute entry).
    start_A = next(
        e for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "A"
    )
    compute_objs = {
        m.id: m for m in start_A.snapshot.memory if m.location == "fast"
    }
    assert "A" in compute_objs and compute_objs["A"].state == "inbound"
    assert "B" not in compute_objs  # deferred — no compute entry yet
    compute_total = sum(m.size for m in start_A.snapshot.memory if m.location == "fast")
    assert compute_total == 10  # only A counted, not B


def test_queued_offload_consumes_no_backing_bytes_until_transfer_starts():
    """An offload enqueued at time T but blocked behind another to_slow should
    not occupy any backing bytes while it waits in the queue."""
    chain = _xfer_chain(
        [
            {"id": "A", "size": 10, "location": "fast"},
            {"id": "B", "size": 10, "location": "fast"},
        ],
        [
            _task("t0", inputs=[], runtime=1, offloads=["A", "B"]),
            _task("t1", inputs=[], runtime=1),
        ],
        bandwidth=2,  # 10/2 = 5µs per transfer
        backing_cap=20,
    )
    log = run(chain)
    # Snapshot at A's to_slow start: A's backing dest is created, B's is still
    # deferred (queued).
    start_A = next(
        e for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "A"
    )
    backing_objs = {m.id: m for m in start_A.snapshot.memory if m.location == "backing"}
    assert "A" in backing_objs and backing_objs["A"].state == "inbound"
    assert "B" not in backing_objs  # B's backing dest deferred until B starts
    backing_total = sum(m.size for m in start_A.snapshot.memory if m.location == "backing")
    assert backing_total == 10


def test_prefetch_blocks_at_start_until_compute_release_frees_bytes():
    """Compute is full; a prefetch is queued. When a downstream compute
    task releases its input, the freed bytes should unblock the queued
    prefetch and let it begin."""
    chain = _xfer_chain(
        [
            {"id": "blocker", "size": 8, "location": "fast"},
            {"id": "X", "size": 8, "location": "backing"},
        ],
        [
            # t0 uses blocker, fires prefetch of X. Compute is full (8/8),
            # so X's from_slow cannot start yet — it must block on cap.
            _task("t0", ["blocker"], runtime=1, prefetches=["X"]),
            # t1 releases blocker, freeing 8 bytes. X's queued from_slow should
            # then start. (blocker is listed as input so the static prepass
            # accepts the release.)
            _task("t1", inputs=["blocker"], runtime=1, releases=["blocker"]),
            # t2 actually uses X — must stall until X arrives.
            _task("t2", ["X"], runtime=1),
        ],
        bandwidth=2,  # 8/2 = 4µs transfer
        compute_cap=8,
    )
    log = run(chain)
    # X's from_slow cannot start at t=1 (compute still has blocker). It must
    # start at t=2 (after t1's release at t=2). from_slow ends at 2+4=6.
    from_slow_start = next(
        e.t for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "X"
    )
    assert from_slow_start == 2
    t2_iv = next(iv for iv in log.task_intervals if iv.task_id == "t2")
    assert t2_iv.start == 6  # waited for from_slow to finish at 6


# ---------- task_id uniqueness for re-prefetch / re-offload ----------

def test_repeated_transfers_of_same_object_get_unique_task_ids():
    """The UI keys timeline bars by task_id, so re-prefetching/re-offloading
    the same object must produce DISTINCT task_ids. First instance keeps the
    bare `<dir>:<obj>` id; subsequent ones get a `#N` suffix."""
    chain = _xfer_chain(
        [
            {"id": "W", "size": 4, "location": "backing"},
            {"id": "filler", "size": 4, "location": "fast"},
        ],
        [
            # t0 prefetches W (1st from_slow)
            _task("t0", inputs=[], runtime=1, prefetches=["W"]),
            # t1 uses W then offloads it
            _task("t1", ["W"], runtime=1, offloads=["W"]),
            # t2 prefetches W again (2nd from_slow — must NOT collide)
            _task("t2", inputs=[], runtime=1, prefetches=["W"]),
            # t3 uses W then offloads again
            _task("t3", ["W"], runtime=1, offloads=["W"]),
        ],
        bandwidth=2,
    )
    log = run(chain)
    from_slow_ids = [iv.task_id for iv in log.task_intervals if iv.track == "from_slow"]
    to_slow_ids = [iv.task_id for iv in log.task_intervals if iv.track == "to_slow"]
    assert from_slow_ids == ["from_slow:W", "from_slow:W#1"]
    assert to_slow_ids == ["to_slow:W", "to_slow:W#1"]
    # All transfer task_ids globally unique.
    all_xfer = from_slow_ids + to_slow_ids
    assert len(set(all_xfer)) == len(all_xfer)
