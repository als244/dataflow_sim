import pytest

from dataflow_sim.schema import Object, OutputAlloc, Task, TaskChain, TransferTrigger
from dataflow_sim.simulator import run


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
    chain = _chain([], [_task("t0", ["ghost"], runtime=1)])
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
            Object(id="w", size=10, location="host"),
            Object(id="w", size=10, location="device"),
            Object(id="x", size=5, location="device"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["w", "x"],
                outputs=[OutputAlloc(id="y", size=7, location="device")],
                runtime=1,
                releases_after=[],
            )
        ],
    )
    log = run(chain)
    end_snap = log.events[1].snapshot
    by_loc = {(m.id, m.location) for m in end_snap.memory}
    assert ("w", "host") in by_loc
    assert ("w", "device") in by_loc
    assert ("x", "device") in by_loc
    assert ("y", "device") in by_loc


def test_compute_rejects_host_only_input():
    chain = TaskChain(
        initial_memory=[Object(id="w", size=10, location="host")],
        tasks=[
            Task(id="t0", inputs=["w"], outputs=[], runtime=1, releases_after=[]),
        ],
    )
    # validate=False: the prepass catches host-only inputs as
    # `released-then-referenced`; this test pins the runtime message.
    with pytest.raises(ValueError, match="host only"):
        run(chain, validate=False)


def test_same_id_on_host_and_device_coexist():
    chain = TaskChain(
        initial_memory=[
            Object(id="W", size=10, location="host"),
            Object(id="W", size=10, location="device"),
        ],
        tasks=[Task(id="t0", inputs=["W"], outputs=[], runtime=1, releases_after=[])],
    )
    log = run(chain)
    assert len(log.events) == 2  # start + end (no releases)
    snap_ids = [(m.id, m.location) for m in log.events[0].snapshot.memory]
    assert ("W", "host") in snap_ids
    assert ("W", "device") in snap_ids


def test_device_capacity_enforced():
    """When a compute task can't fit its outputs and there's no in-flight offload
    to free space, the simulator raises (compute would stall forever)."""
    chain = TaskChain(
        initial_memory=[Object(id="x", size=8, location="device")],
        tasks=[
            Task(
                id="t0",
                inputs=["x"],
                outputs=[OutputAlloc(id="y", size=5, location="device")],
                runtime=1,
                releases_after=[],
            )
        ],
        device_capacity=10,
    )
    with pytest.raises(ValueError, match="cannot satisfy device memory"):
        run(chain)


def test_host_capacity_enforced_on_initial_memory():
    chain = TaskChain(
        initial_memory=[
            Object(id="a", size=6, location="host"),
            Object(id="b", size=6, location="host"),
        ],
        tasks=[],
        host_capacity=10,
    )
    with pytest.raises(ValueError, match="cannot allocate"):
        run(chain)


def test_active_task_inputs_show_in_reference_stream_at_task_start():
    chain = TaskChain(
        initial_memory=[Object(id="w", size=1, location="device")],
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


def _xfer_chain(initial, tasks, *, bandwidth=4, device_cap=None, host_cap=None):
    return TaskChain(
        initial_memory=[Object(**o) for o in initial],
        tasks=tasks,
        device_capacity=device_cap,
        host_capacity=host_cap,
        bandwidth_h2d=bandwidth,
        bandwidth_d2h=bandwidth,
    )


def test_offload_emits_transfer_events_and_frees_device():
    chain = _xfer_chain(
        [{"id": "w", "size": 8, "location": "device"}],
        [
            _task("t0", ["w"], runtime=5, offloads=["w"]),
            _task("t1", [], runtime=1),
        ],
    )
    log = run(chain)
    kinds = [(e.t, e.kind, e.transfer_obj, e.transfer_direction) for e in log.events
             if e.kind in ("transfer_enqueue", "transfer_start", "transfer_end")]
    assert ("d2h", 5, "transfer_enqueue") in [(d, t, k) for (t, k, _, d) in kinds]
    # transfer should start immediately (stream idle) and complete 8 bytes / 4 bw = 2 units later
    assert any(t == 5 and k == "transfer_start" and o == "w" and d == "d2h" for (t, k, o, d) in kinds)
    assert any(t == 7 and k == "transfer_end" and o == "w" and d == "d2h" for (t, k, o, d) in kinds)
    # at the end, w should be live on host, gone from device
    final = log.events[-1].snapshot
    by_loc = {(m.id, m.location): m.state for m in final.memory}
    assert by_loc == {("w", "host"): "live"}


def test_prefetch_emits_transfer_events_and_lands_live_on_device():
    chain = _xfer_chain(
        [{"id": "w", "size": 8, "location": "host"}],
        [
            _task("t0", [], runtime=5, prefetches=["w"]),
            _task("t1", [], runtime=1),
        ],
    )
    log = run(chain)
    # h2d should fire after t0
    transfer_events = [(e.t, e.kind, e.transfer_obj, e.transfer_direction) for e in log.events
                       if e.kind.startswith("transfer_")]
    assert ("d2h" not in [d for (_, _, _, d) in transfer_events])
    assert any(k == "transfer_end" and o == "w" and d == "h2d" for (_, k, o, d) in transfer_events)
    final = log.events[-1].snapshot
    by_loc = {(m.id, m.location): m.state for m in final.memory}
    assert by_loc.get(("w", "host")) == "live"
    assert by_loc.get(("w", "device")) == "live"


def test_compute_stalls_waiting_on_inbound_prefetch():
    """t0 prefetches w (8 bytes / bw=2 = 4 unit transfer), t1 takes 1 unit but
    needs w on device. t1 should stall until prefetch completes."""
    chain = _xfer_chain(
        [{"id": "w", "size": 8, "location": "host"}],
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


def test_compute_stalls_waiting_on_device_capacity_to_free():
    """device capacity is tight; an offload must complete before the next task
    can allocate its output."""
    chain = _xfer_chain(
        [{"id": "a", "size": 6, "location": "device"}],
        [
            _task("t0", ["a"], runtime=1, offloads=["a"]),
            _task("t1", [], runtime=1, outputs=[{"id": "b", "size": 8, "location": "device"}]),
        ],
        bandwidth=2,
        device_cap=8,
        host_cap=16,
    )
    log = run(chain)
    intervals = {iv.task_id: (iv.start, iv.end) for iv in log.task_intervals}
    # t0 ends at 1, offload of `a` (6 bytes / bw 2 = 3 units) ends at 4 — at which
    # point device is freed and t1 can allocate its 8-byte output.
    assert intervals["t0"] == (0, 1)
    assert intervals["t1"] == (4, 5)


def test_offload_deadlocks_when_host_capacity_too_tight():
    """With deferred destination allocation, an offload that can never fit
    on the host stays queued forever; the simulator raises at end-of-run."""
    chain = _xfer_chain(
        [{"id": "w", "size": 10, "location": "device"}],
        [_task("t0", ["w"], runtime=1, offloads=["w"])],
        bandwidth=2,
        host_cap=5,
    )
    with pytest.raises(ValueError, match="d2h queue.*'w'"):
        run(chain)


def test_prefetch_deadlocks_when_device_capacity_too_tight():
    """With deferred destination allocation, a prefetch that can never fit
    on the device stays queued forever; the simulator raises at end-of-run."""
    chain = _xfer_chain(
        [
            {"id": "blocker", "size": 8, "location": "device"},
            {"id": "w", "size": 10, "location": "host"},
        ],
        [_task("t0", ["blocker"], runtime=1, prefetches=["w"])],
        bandwidth=2,
        device_cap=8,
    )
    with pytest.raises(ValueError, match="h2d queue.*'w'"):
        run(chain)


def test_transfer_queue_serializes_multiple_offloads():
    """Two offloads from the same task queue on D→H; the second starts only
    when the first completes."""
    chain = _xfer_chain(
        [
            {"id": "a", "size": 4, "location": "device"},
            {"id": "b", "size": 4, "location": "device"},
        ],
        [_task("t0", ["a", "b"], runtime=1, offloads=["a", "b"])],
        bandwidth=2,  # 4/2 = 2 units per transfer
    )
    log = run(chain)
    h2d_intervals = [iv for iv in log.task_intervals if iv.track == "d2h"]
    assert len(h2d_intervals) == 2
    # First transfer: 1..3; second: 3..5 (queued)
    starts = sorted(iv.start for iv in h2d_intervals)
    ends = sorted(iv.end for iv in h2d_intervals)
    assert starts == [1, 3]
    assert ends == [3, 5]


def test_per_trigger_runtime_overrides_bandwidth():
    chain = _xfer_chain(
        [{"id": "w", "size": 100, "location": "device"}],
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
        initial_memory=[Object(id="w", size=4, location="device")],
        tasks=[_task("t0", ["w"], runtime=1, offloads=["w"])],
        # bandwidth_d2h not set, no per-trigger override
    )
    with pytest.raises(ValueError, match="bandwidth"):
        run(chain)


def test_offload_overwrites_existing_host_copy():
    """Offload with an existing live host copy = write-back. Host entry
    stays 'live' until the d2h actually begins (destination is deferred —
    no host bytes consumed by a merely-queued overwrite either), then
    flips to 'inbound' for the duration, and lands back on 'live' at
    completion. No new host capacity consumed."""
    chain = _xfer_chain(
        [
            {"id": "w", "size": 4, "location": "device"},
            {"id": "w", "size": 4, "location": "host"},
        ],
        [_task("t0", ["w"], runtime=1, offloads=["w"])],
        bandwidth=2,
    )
    log = run(chain)
    # At trigger fire (enqueue), the host entry is still 'live' — destination
    # allocation is deferred to transfer-start.
    enqueue = next(
        e for e in log.events
        if e.kind == "transfer_enqueue" and e.transfer_obj == "w"
    )
    host_w_at_enqueue = next(
        m for m in enqueue.snapshot.memory if m.id == "w" and m.location == "host"
    )
    assert host_w_at_enqueue.state == "live"
    # When the d2h actually starts, the host entry flips to 'inbound'.
    start = next(
        e for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "w"
    )
    host_w_at_start = next(
        m for m in start.snapshot.memory if m.id == "w" and m.location == "host"
    )
    assert host_w_at_start.state == "inbound"
    # On completion, host w is live (and device gone).
    final = log.events[-1].snapshot
    by_loc = {(m.id, m.location): m.state for m in final.memory}
    assert by_loc == {("w", "host"): "live"}


def test_offload_overwrite_size_mismatch_raises():
    chain = _xfer_chain(
        [
            {"id": "w", "size": 4, "location": "device"},
            {"id": "w", "size": 8, "location": "host"},
        ],
        [_task("t0", ["w"], runtime=1, offloads=["w"])],
        bandwidth=2,
    )
    with pytest.raises(ValueError, match="size mismatch"):
        run(chain)


def test_prefetch_already_on_device_raises():
    chain = _xfer_chain(
        [
            {"id": "w", "size": 4, "location": "device"},
            {"id": "w", "size": 4, "location": "host"},
        ],
        [_task("t0", ["w"], runtime=1, prefetches=["w"])],
        bandwidth=2,
    )
    with pytest.raises(ValueError, match="device copy already exists"):
        run(chain, validate=False)


# ---------- deferred prefetch: source is still mid-d2h ----------

def test_prefetch_defers_when_source_still_offloading():
    """Triggering a prefetch while the source's d2h is still in flight
    must DEFER the h2d enqueue (not raise). The h2d auto-enqueues when
    the d2h completes and any downstream compute correctly stalls."""
    chain = _xfer_chain(
        [{"id": "X", "size": 10, "location": "device"}],
        [
            # t0 (1µs): fires offload of X. d2h enqueues at 1, runtime=5,
            # so d2h is in_flight 1→6.
            _task("t0", inputs=[], runtime=1, offloads=["X"]),
            # t1 (1µs): fires prefetch of X. At t1.end=2, d2h is mid-flight
            # (X device state=outbound). Must DEFER, not raise.
            _task("t1", inputs=[], runtime=1, prefetches=["X"]),
            # t2 needs X. d2h ends at 6 → h2d starts at 6, ends at 11.
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
    assert starts == 2 and ends == 2  # one d2h + one h2d


def test_prefetch_defers_when_host_source_pending_inbound():
    """If the d2h is queued (not yet in flight) so the host source is
    still `pending_inbound`, prefetch also defers."""
    chain = _xfer_chain(
        [
            # Two objects: A and X. Offload A first to keep d2h busy.
            {"id": "A", "size": 10, "location": "device"},
            {"id": "X", "size": 10, "location": "device"},
        ],
        [
            # t0: offload BOTH A and X. d2h FIFO: A starts at 1 ends at 6,
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
    # X's d2h ends at 11; h2d enqueues at 11, starts at 11 (stream idle by
    # then since A's h2d wasn't requested), ends at 16. t2 starts at 16.
    assert t2_iv.start == 16


def test_truly_absent_host_source_still_raises():
    """Prefetch where the host source neither exists nor is in flight
    is still a hard error (matches the user's invariant)."""
    chain = _xfer_chain(
        [{"id": "A", "size": 4, "location": "device"}],
        [_task("t0", inputs=[], runtime=1, prefetches=["X"])],  # X doesn't exist
        bandwidth=2,
    )
    with pytest.raises(ValueError, match="no recoverable host source"):
        run(chain, validate=False)


# ---------- deferred destination allocation (symmetric h2d + d2h) ----------

def test_queued_prefetch_consumes_no_device_bytes_until_transfer_starts():
    """A prefetch enqueued at time T but blocked behind another h2d should
    not occupy any device bytes while it waits in the queue."""
    chain = _xfer_chain(
        [
            {"id": "A", "size": 10, "location": "host"},
            {"id": "B", "size": 10, "location": "host"},
        ],
        [
            # t0 fires both prefetches; A goes in-flight, B queues behind it.
            _task("t0", inputs=[], runtime=1, prefetches=["A", "B"]),
            # padding so the schedule extends past the first h2d
            _task("t1", inputs=[], runtime=1),
        ],
        bandwidth=2,  # 10/2 = 5µs per transfer
        device_cap=20,
    )
    log = run(chain)
    # Take the snapshot at A's transfer_start: A is now in_flight (device
    # entry created), B is still queued (no device entry).
    start_A = next(
        e for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "A"
    )
    device_objs = {
        m.id: m for m in start_A.snapshot.memory if m.location == "device"
    }
    assert "A" in device_objs and device_objs["A"].state == "inbound"
    assert "B" not in device_objs  # deferred — no device entry yet
    device_total = sum(m.size for m in start_A.snapshot.memory if m.location == "device")
    assert device_total == 10  # only A counted, not B


def test_queued_offload_consumes_no_host_bytes_until_transfer_starts():
    """An offload enqueued at time T but blocked behind another d2h should
    not occupy any host bytes while it waits in the queue."""
    chain = _xfer_chain(
        [
            {"id": "A", "size": 10, "location": "device"},
            {"id": "B", "size": 10, "location": "device"},
        ],
        [
            _task("t0", inputs=[], runtime=1, offloads=["A", "B"]),
            _task("t1", inputs=[], runtime=1),
        ],
        bandwidth=2,  # 10/2 = 5µs per transfer
        host_cap=20,
    )
    log = run(chain)
    # Snapshot at A's d2h start: A's host dest is created, B's is still
    # deferred (queued).
    start_A = next(
        e for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "A"
    )
    host_objs = {m.id: m for m in start_A.snapshot.memory if m.location == "host"}
    assert "A" in host_objs and host_objs["A"].state == "inbound"
    assert "B" not in host_objs  # B's host dest deferred until B starts
    host_total = sum(m.size for m in start_A.snapshot.memory if m.location == "host")
    assert host_total == 10


def test_prefetch_blocks_at_start_until_device_release_frees_bytes():
    """Device is full; a prefetch is queued. When a downstream compute
    task releases its input, the freed bytes should unblock the queued
    prefetch and let it begin."""
    chain = _xfer_chain(
        [
            {"id": "blocker", "size": 8, "location": "device"},
            {"id": "X", "size": 8, "location": "host"},
        ],
        [
            # t0 uses blocker, fires prefetch of X. Device is full (8/8),
            # so X's h2d cannot start yet — it must block on cap.
            _task("t0", ["blocker"], runtime=1, prefetches=["X"]),
            # t1 releases blocker, freeing 8 bytes. X's queued h2d should
            # then start. (blocker is listed as input so the static prepass
            # accepts the release.)
            _task("t1", inputs=["blocker"], runtime=1, releases=["blocker"]),
            # t2 actually uses X — must stall until X arrives.
            _task("t2", ["X"], runtime=1),
        ],
        bandwidth=2,  # 8/2 = 4µs transfer
        device_cap=8,
    )
    log = run(chain)
    # X's h2d cannot start at t=1 (device still has blocker). It must
    # start at t=2 (after t1's release at t=2). h2d ends at 2+4=6.
    h2d_start = next(
        e.t for e in log.events
        if e.kind == "transfer_start" and e.transfer_obj == "X"
    )
    assert h2d_start == 2
    t2_iv = next(iv for iv in log.task_intervals if iv.task_id == "t2")
    assert t2_iv.start == 6  # waited for h2d to finish at 6


# ---------- task_id uniqueness for re-prefetch / re-offload ----------

def test_repeated_transfers_of_same_object_get_unique_task_ids():
    """The UI keys timeline bars by task_id, so re-prefetching/re-offloading
    the same object must produce DISTINCT task_ids. First instance keeps the
    bare `<dir>:<obj>` id; subsequent ones get a `#N` suffix."""
    chain = _xfer_chain(
        [
            {"id": "W", "size": 4, "location": "host"},
            {"id": "filler", "size": 4, "location": "device"},
        ],
        [
            # t0 prefetches W (1st h2d)
            _task("t0", inputs=[], runtime=1, prefetches=["W"]),
            # t1 uses W then offloads it
            _task("t1", ["W"], runtime=1, offloads=["W"]),
            # t2 prefetches W again (2nd h2d — must NOT collide)
            _task("t2", inputs=[], runtime=1, prefetches=["W"]),
            # t3 uses W then offloads again
            _task("t3", ["W"], runtime=1, offloads=["W"]),
        ],
        bandwidth=2,
    )
    log = run(chain)
    h2d_ids = [iv.task_id for iv in log.task_intervals if iv.track == "h2d"]
    d2h_ids = [iv.task_id for iv in log.task_intervals if iv.track == "d2h"]
    assert h2d_ids == ["h2d:W", "h2d:W#1"]
    assert d2h_ids == ["d2h:W", "d2h:W#1"]
    # All transfer task_ids globally unique.
    all_xfer = h2d_ids + d2h_ids
    assert len(set(all_xfer)) == len(all_xfer)
