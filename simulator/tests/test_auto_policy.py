"""Tests for the V2 auto-policy.

V2 working envelope (per AUTOPOLICY.md):
  L=3: device_capacity >= 500 or None
  L=5: device_capacity >= 500 or None
  L=10: device_capacity >= 500 or None
"""
import pytest

from dataflow_sim.policy._common import (
    _compute_ideal_starts,
    _compute_uses,
    _next_use_after,
    _object_sizes,
)
from dataflow_sim.policy.roundtrip_planner import _initial_placement
from dataflow_sim.policy.belady_reactive import (
    apply_belady_reactive_policy as apply_auto_policy,
)
from dataflow_sim.policy.race_best import apply_race_best_policy
from dataflow_sim.policy.sliding_window import apply_sliding_window_policy
from dataflow_sim.simulator import run
from conftest import build_bare_training_chain


# ---------- Phase 0a / reference stream ----------

def test_next_use_after_returns_first_use_at_or_after():
    uses = {"a": [0, 10, 20], "b": [5, 15]}
    assert _next_use_after(uses, "a", 0) == 0
    assert _next_use_after(uses, "a", 5) == 10
    assert _next_use_after(uses, "a", 21) == float("inf")
    assert _next_use_after(uses, "b", 5) == 5
    assert _next_use_after(uses, "b", 16) == float("inf")
    assert _next_use_after(uses, "missing", 0) == float("inf")


def test_compute_uses_collects_input_timestamps():
    bare = build_bare_training_chain(L=2)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # input is used by f_0 at t=0
    assert uses["input"] == [0]
    # W_0 used by f_0 (t=0) and by r_0/b_0 (later)
    assert 0 in uses["W_0"]


# ---------- Phase 1 / initial placement ----------

def test_initial_placement_must_place_T1_inputs():
    bare = build_bare_training_chain(L=3)
    sizes = _object_sizes(bare)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # T_1 = f_0; inputs are input, W_0
    placement = _initial_placement(bare, device_capacity=2000, uses=uses, sizes=sizes)
    assert "W_0" in placement  # input is already device-resident, so excluded


def test_initial_placement_raises_when_widest_T1_too_big():
    bare = build_bare_training_chain(L=3, weight_size=64, input_size=16)
    sizes = _object_sizes(bare)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # T_1 needs input(16) + W_0(64) on device + outputs A_0(32) + y_0(32) reserved = 144
    # Initial pool also has input already on device (16 bytes)
    # If capacity is 30, can't fit even input
    with pytest.raises(ValueError, match="widest-task infeasibility"):
        _initial_placement(bare, device_capacity=30, uses=uses, sizes=sizes)


# ---------- End-to-end / working envelope ----------

@pytest.mark.parametrize("cap", [None, 1200, 1000, 800, 600, 500])
def test_auto_policy_L3_works_at_loose_caps(cap):
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, device_capacity=cap)
    log = run(annotated)  # must not raise
    # All compute tasks should appear in intervals
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "f_0" in compute_ids
    assert "b_0" in compute_ids


@pytest.mark.parametrize("cap", [None, 1200, 1000, 800, 600, 500])
def test_auto_policy_L5_works_at_v2_envelope(cap):
    """V2 extends L=5 down to cap=500."""
    bare = build_bare_training_chain(L=5)
    annotated = apply_auto_policy(bare, device_capacity=cap)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "b_0" in compute_ids


@pytest.mark.parametrize("cap", [None, 1500, 1000, 800, 600, 500])
def test_auto_policy_L10_works_at_v2_envelope(cap):
    """V2 extends L=10 down to cap=500 (was unlimited-only in V1)."""
    bare = build_bare_training_chain(L=10)
    annotated = apply_auto_policy(bare, device_capacity=cap)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "b_0" in compute_ids


def test_phase5_recovers_l10_cap600():
    """Phase 5 iterative refinement: at L=10 cap=600 the initial planning
    overshoots capacity at some prefetches; Phase 5 shifts the prefetch
    earlier until the simulator accepts."""
    bare = build_bare_training_chain(L=10)
    annotated = apply_auto_policy(bare, device_capacity=600)
    log = run(annotated)
    makespan = max(iv.end for iv in log.task_intervals)
    # Should be a valid run (no exceptions); makespan within reasonable bound.
    assert makespan < 400


# ---------- V3 (proactive) ----------

@pytest.mark.parametrize("cap", [None, 1000, 800])
def test_v3_proactive_matches_v2_at_loose_caps(cap):
    """V3 should produce a feasible plan for the loose-cap envelope where it
    has working coverage (None / 1000 / 800 for L=3)."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_race_best_policy(bare, device_capacity=cap)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "b_0" in compute_ids


@pytest.mark.parametrize("L,cap", [
    (3, None), (3, 800), (3, 500),
    (5, None), (5, 800), (5, 500),
    (10, None), (10, 800), (10, 500),
])
def test_v3_envelope_at_least_v2(L, cap):
    """V3 must always be at least as good as V2 — fallback path guarantees
    that V3 succeeds iff V2 would, and produces a makespan ≤ V2's."""
    bare = build_bare_training_chain(L=L)
    v2 = apply_auto_policy(bare, device_capacity=cap)
    v3 = apply_race_best_policy(bare, device_capacity=cap)
    v2_ms = max(iv.end for iv in run(v2).task_intervals)
    v3_ms = max(iv.end for iv in run(v3).task_intervals)
    assert v3_ms <= v2_ms, f"V3 ms={v3_ms} worse than V2 ms={v2_ms} at L={L} cap={cap}"


def test_v3_enumerates_roundtrips_for_l3():
    """V3's gap enumeration should find candidate round-trips for forward
    weights (W_0..W_2) since they have large gaps between f_i and r_i/b_i."""
    from dataflow_sim.policy._common import (
        _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx,
    )
    from dataflow_sim.policy.roundtrip_planner import _enumerate_roundtrips
    bare = build_bare_training_chain(L=3)
    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)
    candidates = _enumerate_roundtrips(bare, sizes, uses_by_task, ideal)
    obj_ids = {c.obj_id for c in candidates}
    # W_0 and W_1 have wide gaps between forward and backward. W_2's gap
    # (f_2 end → r_2 start = 2 ticks) is too narrow for an 8+8 round-trip.
    assert "W_0" in obj_ids
    assert "W_1" in obj_ids


def test_v3_object_uses_by_task_idx_collapses_duplicates():
    """A task's input list may reference the same object once; we should
    record a single use event per (task, obj) pair."""
    from dataflow_sim.policy._common import _compute_ideal_starts, _object_uses_by_task_idx
    bare = build_bare_training_chain(L=2)
    ideal = _compute_ideal_starts(bare)
    by_task = _object_uses_by_task_idx(bare, ideal)
    # W_0 used by f_0 (task 0), r_0, b_0 — exactly three events.
    w0_events = by_task["W_0"]
    assert len(w0_events) == 3
    assert [e.task_idx for e in w0_events] == sorted(e.task_idx for e in w0_events)


def test_initial_placement_leaves_slack_for_widest_task():
    """The slack-aware initial placement should leave room equal to the widest
    single-task footprint so future prefetches/cascade have headroom."""
    from dataflow_sim.policy.roundtrip_planner import _initial_placement
    from dataflow_sim.policy._common import _compute_ideal_starts, _compute_uses, _object_sizes
    bare = build_bare_training_chain(L=3)
    sizes = _object_sizes(bare)
    ideal = _compute_ideal_starts(bare)
    uses = _compute_uses(bare, ideal)
    # At cap=400 (which is < 2 × widest 224), slack should keep initial small.
    placement = _initial_placement(bare, device_capacity=400, uses=uses, sizes=sizes)
    initial_bytes = 16 + sum(sizes[oid] for oid in placement)  # input + placements
    # widest = 224, so cap - widest = 176. Initial pool (after T_1 outputs reserved
    # = 64) should fit in 400 - 224 = 176 bytes free. With must_place W_0 (64)
    # + input (16) = 80, headroom for greedy is small.
    assert initial_bytes <= 400 - 224 + 80  # widest-task slack + T_1 pinned


def test_auto_policy_L3_zero_stalls_at_unlimited():
    """At unlimited capacity, both policies should be stall-free."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, device_capacity=None)
    log = run(annotated)
    compute = sorted(
        [iv for iv in log.task_intervals if iv.track == "compute"],
        key=lambda iv: iv.start,
    )
    # No gaps between consecutive compute tasks
    for prev, cur in zip(compute, compute[1:]):
        assert cur.start <= prev.end, f"stall: {prev.task_id}→{cur.task_id}"


def test_auto_policy_L3_unlimited_emits_only_gradient_writebacks():
    """At unlimited capacity, the only required offloads are the gradient
    write-backs at each gradient's last-use task (workload convention:
    dW_i / dW_head must end on host). No other offloads, no prefetches."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, device_capacity=None)
    n_pre = sum(len(t.prefetch_after) for t in annotated.tasks)
    assert n_pre == 0
    # Every host-initial gradient must be offloaded exactly once.
    grads = {o.id for o in bare.initial_memory
             if o.location == "host" and o.type == "gradient"}
    offloaded = [tr.obj_id for t in annotated.tasks for tr in t.offload_after]
    from collections import Counter
    c = Counter(offloaded)
    for g in grads:
        assert c[g] == 1, f"gradient {g} offloaded {c[g]} times (want 1)"
    # Nothing else is offloaded.
    extra = set(offloaded) - grads
    assert not extra, f"unexpected offloads at unlimited cap: {sorted(extra)}"


def test_auto_policy_emits_releases_at_tight_cap():
    """At a tight-but-feasible cap, the policy should emit at least some triggers."""
    bare = build_bare_training_chain(L=3)
    annotated = apply_auto_policy(bare, device_capacity=800)
    n_rel = sum(len(t.releases_after) for t in annotated.tasks)
    assert n_rel >= 1


# ---------- equivalence at generous capacity ----------

def test_v2_releases_w_instead_of_offloading_when_host_copy_exists():
    """W_i is host-initial and never mutated (workload contract). When the
    planner needs to free its device bytes between fwd-use and bwd-use, it
    should RELEASE (instant, no d2h cost) — NOT offload — because the host
    copy is byte-identical. Pre-fix V2 always offloaded weights with a
    future use, wasting d2h bandwidth on a write-back to identical data."""
    bare = build_bare_training_chain(L=5)
    annotated = apply_auto_policy(bare, device_capacity=600)
    # Collect every per-task d2h trigger by object.
    offloaded_objs = set()
    for task in annotated.tasks:
        for trig in task.offload_after:
            offloaded_objs.add(trig.obj_id)
    # No W_i should be offloaded — releases handle them.
    w_offloads = {o for o in offloaded_objs if o.startswith("W_")}
    assert not w_offloads, f"V2 wastefully offloaded weights with host copies: {sorted(w_offloads)}"


def test_smart_initial_placement_defers_to_leave_room_for_outputs():
    """The smart initial placement should DEFER host objects whose
    pre-placement would push pessimistic-bps over cap at some boundary,
    leaving room for task outputs (activations) that accumulate over
    forward. Regression for: the old greedy fill would pre-place every
    dW_i + W_head + dW_head as long as the SUM fit under cap, ignoring
    that activations would arrive later and need that room."""
    from dataflow_sim.policy.belady_reactive import _smart_initial_placement
    from dataflow_sim.policy._common import (
        _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx,
    )
    bare = build_bare_training_chain(
        L=5, input_size=50, weight_size=100, activation_size=200,
        grad_size=100, head_weight_size=100,
        fwd_runtime=10, head_runtime=10, bwd_runtime=20,
        bandwidth_h2d=50, bandwidth_d2h=50,
    )
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, _compute_ideal_starts(bare))

    # At loose cap, smart init places everything host with a use.
    loose = _smart_initial_placement(bare, 100_000, sizes, uses_by_task)
    all_host_with_use = {
        o.id for o in bare.initial_memory
        if o.location == "host" and uses_by_task.get(o.id)
    }
    assert loose == all_host_with_use

    # At tight cap where SUM of weights+grads alone fits but adding
    # activations would overflow, smart init defers the host objects whose
    # first-use is LATEST (backward grads, head, dW_head). At least one of
    # {W_head, dW_head, dW_4, dW_3, dW_2, dW_1, dW_0} must be deferred.
    tight = _smart_initial_placement(bare, 1200, sizes, uses_by_task)
    deferred = all_host_with_use - tight
    late_use_objs = {"W_head", "dW_head"} | {f"dW_{i}" for i in range(5)}
    assert deferred & late_use_objs, (
        f"smart init didn't defer any backward-only object at tight cap; "
        f"placed={sorted(tight)}, deferred={sorted(deferred)}"
    )


def test_smart_initial_placement_at_loose_cap_eliminates_forward_stalls():
    """Concrete user-reported scenario: with cap big enough that smart init
    has room to fit everything live during forward (without pre-placing
    things that aren't needed until backward), forward tasks should run
    back-to-back with no stall."""
    bare = build_bare_training_chain(
        L=5, input_size=50, weight_size=100, activation_size=200,
        grad_size=100, head_weight_size=100,
        fwd_runtime=10, head_runtime=10, bwd_runtime=20,
        bandwidth_h2d=50, bandwidth_d2h=50,
    )
    # Cap that comfortably holds: 5 W + 5 A + y + input + reserved next-task
    # outputs + a couple of dW prefetches during forward.
    annotated = apply_auto_policy(bare, device_capacity=2500)
    log = run(annotated)
    f_compute = sorted(
        [iv for iv in log.task_intervals
         if iv.task_id.startswith("f_") and iv.track == "compute"],
        key=lambda iv: iv.start,
    )
    assert len(f_compute) == 5
    for i in range(1, 5):
        gap = f_compute[i].start - f_compute[i - 1].end
        assert gap == 0, (
            f"forward stall between f_{i-1} and f_{i}: gap={gap}; "
            f"smart initial placement should have reserved room for "
            f"accumulating activations"
        )


def test_auto_writes_back_all_gradients_to_host():
    """Workload convention: every host-resident gradient (dW_*, dW_head)
    must end up on host with state='live' after the chain runs — the
    auto policy must emit a write-back offload for each one at its
    last-use task (mirroring the sliding-window policy's behavior)."""
    bare = build_bare_training_chain(L=5)
    for label, policy_fn in (("V2", apply_auto_policy), ("V3", apply_race_best_policy)):
        annotated = policy_fn(bare, device_capacity=None)
        log = run(annotated)
        final = log.events[-1].snapshot
        grads = {o.id for o in bare.initial_memory
                 if o.location == "host" and o.type == "gradient"}
        for g in grads:
            host_entry = next(
                (m for m in final.memory
                 if m.id == g and m.location == "host"),
                None,
            )
            assert host_entry is not None, (
                f"{label}: gradient {g!r} not on host at end"
            )
            assert host_entry.state == "live", (
                f"{label}: {g!r} state={host_entry.state}, want 'live'"
            )


def test_activation_offload_fires_eagerly_at_production():
    """When V2 decides an activation (no host source) must be offloaded,
    it should pick the EARLIEST safe boundary (right after production) so
    the d2h fires while the stream would otherwise be idle. Previously V2
    used 'latest boundary that still meets deadline', which delayed A_0's
    offload by tens of ms even though d2h was sitting idle the whole time."""
    # Tight cap that forces activation offloads.
    bare = build_bare_training_chain(
        L=8, input_size=50, weight_size=100, activation_size=500,
        grad_size=100, head_weight_size=100,
        fwd_runtime=10, head_runtime=10, bwd_runtime=20,
        bandwidth_h2d=50, bandwidth_d2h=50,
    )
    # Cap chosen so A_0 MUST be offloaded — the assertion below otherwise
    # no-ops (was a defensive skip pre-restructure). At cap=2000 the L=8
    # backward-needed activations cycle off-device during forward.
    annotated = apply_auto_policy(bare, device_capacity=2000)
    log = run(annotated)
    f_0 = next(iv for iv in log.task_intervals if iv.task_id == "f_0")
    # Look for A_0's d2h start.
    a0_d2h = next(
        (iv for iv in log.task_intervals
         if iv.track == "d2h" and iv.task_id.split(":", 1)[1].startswith("A_0")),
        None,
    )
    assert a0_d2h is not None, (
        "A_0 wasn't offloaded at cap=1600; tighten further or check policy "
        "behaviour — the eagerness assertion below needs a real d2h to inspect"
    )
    # A_0 must start within ONE compute task of production (not "as late as
    # the deadline allows", which would be many tasks later).
    fwd_runtime = f_0.end - f_0.start
    delay = a0_d2h.start - f_0.end
    assert delay < fwd_runtime, (
        f"A_0 d2h delayed by {delay} units after production (f_0 ends at "
        f"{f_0.end}); expected < {fwd_runtime} (one task) since d2h is "
        f"idle and earliest-safe boundary should be picked"
    )


def test_v3_round_trip_releases_host_initial_objects():
    """V3's round-trip planner should pick RELEASE (not offload) for
    host-initial weights — the host copy is byte-identical and untouched."""
    bare = build_bare_training_chain(L=5)
    annotated = apply_race_best_policy(bare, device_capacity=600)
    offloaded_objs = set()
    for task in annotated.tasks:
        for trig in task.offload_after:
            offloaded_objs.add(trig.obj_id)
    w_offloads = {o for o in offloaded_objs if o.startswith("W_")}
    assert not w_offloads, f"V3 wastefully offloaded weights with host copies: {sorted(w_offloads)}"


def test_auto_at_least_as_fast_as_sliding_window_at_unlimited_cap():
    """At unlimited capacity, the auto-policy should be at least as fast as
    sliding-window because it doesn't emit unnecessary triggers (the
    sliding-window's tail-end `dW_*` offloads extend the makespan past the
    last compute task end)."""
    bare = build_bare_training_chain(L=3)
    sliding = apply_sliding_window_policy(bare, window_size=2, device_capacity=None)
    auto = apply_auto_policy(bare, device_capacity=None)
    sliding_dur = max(iv.end for iv in run(sliding).task_intervals)
    auto_dur = max(iv.end for iv in run(auto).task_intervals)
    assert auto_dur <= sliding_dur, f"auto={auto_dur} > sliding={sliding_dur}"
