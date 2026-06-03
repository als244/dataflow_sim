import pytest

from dataflow_sim.simulator import run
from dataflow_app.workloads.training import build_training_chain


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_training_chain_runs_clean(L):
    chain = build_training_chain(L, fwd_runtime=10)
    log = run(chain)
    # Compute intervals: L fwd + head + L (r_i, b_i) pairs = 3L + 1
    compute_intervals = [iv for iv in log.task_intervals if iv.track == "compute"]
    assert len(compute_intervals) == 3 * L + 1


def test_training_chain_dy_head_lives_only_until_b_last():
    chain = build_training_chain(L=3, fwd_runtime=10)
    log = run(chain)
    snap = next(
        e.snapshot for e in log.events
        if e.kind == "task_start" and e.task_id == "b_2"
    )
    assert any(m.id == "dy_head" and m.state == "live" for m in snap.memory)
    final_snap = log.events[-1].snapshot
    assert all(m.id != "dy_head" for m in final_snap.memory)


def test_activations_released_from_device_after_their_backward():
    """Device copies of A_i are gone at end of run."""
    chain = build_training_chain(L=3)
    log = run(chain)
    final_device_ids = {m.id for m in log.events[-1].snapshot.memory if m.location == "device"}
    assert final_device_ids.isdisjoint({"A_0", "A_1", "A_2"})


def test_backward_runtime_is_2x_forward_and_r_is_zero():
    chain = build_training_chain(L=2, fwd_runtime=10)
    log = run(chain)
    durations = {
        iv.task_id: iv.end - iv.start
        for iv in log.task_intervals
        if iv.track == "compute"
    }
    for i in range(2):
        assert durations[f"f_{i}"] == 10
        assert durations[f"b_{i}"] == 20
        assert durations[f"r_{i}"] == 0


def test_r_i_immediately_precedes_b_i():
    chain = build_training_chain(L=3)
    log = run(chain)
    order = [iv.task_id for iv in log.task_intervals if iv.track == "compute"]
    for i in range(3):
        ri = order.index(f"r_{i}")
        bi = order.index(f"b_{i}")
        assert bi == ri + 1, f"r_{i} should immediately precede b_{i} in {order}"


def test_r_i_starts_and_ends_at_same_timestamp():
    chain = build_training_chain(L=3)
    log = run(chain)
    for iv in log.task_intervals:
        if iv.task_id.startswith("r_"):
            assert iv.start == iv.end


def test_initial_pool_window_size_2():
    """With window_size=2, only W_0..W_1 plus W_head/dW_head are on device
    initially. All weights/dWs exist on host."""
    chain = build_training_chain(L=3, window_size=2)
    log = run(chain)
    snap = log.events[0].snapshot
    on_device = {m.id for m in snap.memory if m.location == "device"}
    on_host = {m.id for m in snap.memory if m.location == "host"}

    # Device: input, W_0, W_1, W_head, dW_head (initial). A_0 + y_0 reserved by f_0.
    assert {"input", "W_0", "W_1", "W_head", "dW_head"}.issubset(on_device)
    assert "W_2" not in on_device  # outside window
    assert "dW_0" not in on_device
    assert "dW_1" not in on_device
    assert "dW_2" not in on_device

    # Host: all weights, all dWs, head weights.
    for i in range(3):
        assert f"W_{i}" in on_host
        assert f"dW_{i}" in on_host
    assert "W_head" in on_host
    assert "dW_head" in on_host
    assert "input" not in on_host  # device-only


def test_initial_pool_window_size_3():
    """window_size=3 puts W_0..W_2 on device initially."""
    chain = build_training_chain(L=5, window_size=3)
    log = run(chain)
    snap = log.events[0].snapshot
    on_device = {m.id for m in snap.memory if m.location == "device"}
    assert {"W_0", "W_1", "W_2"}.issubset(on_device)
    assert "W_3" not in on_device


def test_forward_window_slide_releases_old_prefetches_next():
    """After f_0 with window=2, W_0 is released and W_{0+2}=W_2 is prefetched."""
    chain = build_training_chain(L=3, window_size=2)
    f0 = next(t for t in chain.tasks if t.id == "f_0")
    assert "W_0" in f0.releases_after
    assert any(p.obj_id == "W_2" for p in f0.prefetch_after)


def test_forward_dw_preamble():
    """f_{L-2} prefetches dW_{L-1}, f_{L-1} prefetches dW_{L-2}."""
    chain = build_training_chain(L=3)
    f1 = next(t for t in chain.tasks if t.id == "f_1")  # L-2 = 1
    f2 = next(t for t in chain.tasks if t.id == "f_2")  # L-1 = 2
    assert any(p.obj_id == "dW_2" for p in f1.prefetch_after)
    assert any(p.obj_id == "dW_1" for p in f2.prefetch_after)


def test_backward_offloads_dW_and_cascades_prefetch():
    """b_i offloads dW_i; if i-2 >= 0 it also prefetches dW_{i-2}."""
    chain = build_training_chain(L=5)
    b4 = next(t for t in chain.tasks if t.id == "b_4")
    b3 = next(t for t in chain.tasks if t.id == "b_3")
    b0 = next(t for t in chain.tasks if t.id == "b_0")
    # b_4: offload dW_4, cascade-prefetch dW_2
    assert any(o.obj_id == "dW_4" for o in b4.offload_after)
    assert any(p.obj_id == "dW_2" for p in b4.prefetch_after)
    # b_3: offload dW_3, cascade-prefetch dW_1
    assert any(o.obj_id == "dW_3" for o in b3.offload_after)
    assert any(p.obj_id == "dW_1" for p in b3.prefetch_after)
    # b_0: offload dW_0, no cascade (i-2 = -2)
    assert any(o.obj_id == "dW_0" for o in b0.offload_after)
    assert not any(p.obj_id.startswith("dW_") for p in b0.prefetch_after)


def test_backward_prefetches_offloaded_activations():
    """b_i prefetches A_{i-2} if A_{i-2} was offloaded."""
    chain = build_training_chain(L=5)
    # A_i offloaded for i in [0, L-3] = [0, 1, 2]
    b4 = next(t for t in chain.tasks if t.id == "b_4")
    b3 = next(t for t in chain.tasks if t.id == "b_3")
    b2 = next(t for t in chain.tasks if t.id == "b_2")
    # b_4 prefetches A_2 (in offloaded set)
    assert any(p.obj_id == "A_2" for p in b4.prefetch_after)
    # b_3 prefetches A_1
    assert any(p.obj_id == "A_1" for p in b3.prefetch_after)
    # b_2 prefetches A_0
    assert any(p.obj_id == "A_0" for p in b2.prefetch_after)


def test_head_depends_on_dW_head():
    chain = build_training_chain(L=2)
    head_task = next(t for t in chain.tasks if t.id == "head")
    assert "dW_head" in head_task.inputs


def test_b_i_sees_A_i_live_at_start():
    """A_i must be live when b_i begins — possibly via prefetch if it was offloaded."""
    chain = build_training_chain(L=3)
    log = run(chain)
    for i in range(3):
        snap = next(
            e.snapshot for e in log.events
            if e.kind == "task_start" and e.task_id == f"b_{i}"
        )
        assert any(m.id == f"A_{i}" and m.state == "live" for m in snap.memory), (
            f"A_{i} should be live at start of b_{i}"
        )
