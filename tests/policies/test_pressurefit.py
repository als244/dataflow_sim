from __future__ import annotations

from dataflow_sim.policies._common import _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx
from dataflow_sim.policies.pressurefit import (
    _build_facts,
    _extend_inbound_lead_time,
    _initial_residency,
    _pressure_initial_placement,
    _reduce_to_fit,
    apply_pressurefit_policy,
    plan_pressurefit_policy,
)
from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.engine.simulator import run
from conftest import build_bare_training_chain


def _initial_choice_chain() -> TaskChain:
    return TaskChain(
        initial_memory=[
            Object(id="early", size=10, location="backing", type="weight"),
            Object(id="late", size=10, location="backing", type="weight"),
        ],
        tasks=[
            Task(id="t0", inputs=["early"], outputs=[], runtime=50),
            Task(id="t1", inputs=[], outputs=[], runtime=100),
            Task(id="t2", inputs=["late"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=15,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )


def test_pressure_initial_placement_skips_hidden_future_use():
    bare = _initial_choice_chain()
    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)

    placement = _pressure_initial_placement(
        bare, bare.fast_memory_capacity, sizes, uses_by_task,
    )

    assert "early" in placement  # task-0 input, no prior trigger slot
    assert "late" not in placement  # its first inbound hides behind t1


def test_pressurefit_prefetches_late_object_instead_of_preplacing():
    bare = _initial_choice_chain()
    annotated = apply_pressurefit_policy(bare)
    run(annotated)

    initial_compute = {o.id for o in annotated.initial_memory if o.location == "fast"}
    assert "early" in initial_compute
    assert "late" not in initial_compute
    assert any(
        trig.obj_id == "late"
        for task in annotated.tasks
        for trig in task.prefetch_after
    )


def test_pressurefit_runs_training_chain_at_moderate_cap():
    bare = build_bare_training_chain(L=5)
    annotated = apply_pressurefit_policy(bare, fast_memory_capacity=800)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "f_0" in compute_ids
    assert "b_0" in compute_ids


def test_pressurefit_diagnostics_describe_selected_candidate():
    bare = build_bare_training_chain(L=5)
    annotated, diagnostics = plan_pressurefit_policy(bare, fast_memory_capacity=800)
    log = run(annotated)
    makespan = max(iv.end for iv in log.task_intervals)

    assert diagnostics.selected_makespan_us == makespan
    assert diagnostics.valid_candidate_count > 0
    selected = [c for c in diagnostics.candidates if c.selected]
    assert len(selected) == 1
    assert selected[0].name == diagnostics.selected_candidate
    assert selected[0].status == "valid"


def test_pressurefit_evaluates_exactly_four_inbound_schedules():
    bare = build_bare_training_chain(L=5)
    _annotated, diagnostics = plan_pressurefit_policy(bare, fast_memory_capacity=800)

    assert [c.name for c in diagnostics.candidates] == [
        "packed-fifo",
        "packed-fit",
        "interval-entry",
        "latest-safe",
    ]
    assert diagnostics.candidate_count == 4
    # The selected plan is the fastest valid one.
    valid = [c for c in diagnostics.candidates if c.status == "valid"]
    assert diagnostics.selected_makespan_us == min(c.makespan_us for c in valid)


def test_pressurefit_can_release_disposable_mutation_after_final_use():
    bare = TaskChain(
        initial_memory=[
            Object(id="buf", size=10, location="backing", type="other"),
        ],
        tasks=[
            Task(
                id="mut",
                inputs=["buf"],
                outputs=[OutputAlloc(id="out", size=1, location="fast")],
                runtime=1,
                mutates_inputs=["buf"],
            ),
        ],
        fast_memory_capacity=32,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert annotated.tasks[0].mutates_inputs == ["buf"]
    assert "buf" in annotated.tasks[0].releases_after
    assert not any(trig.obj_id == "buf" for trig in annotated.tasks[0].offload_after)
    run(annotated)


def test_pressurefit_preserves_final_backing_mutation_writeback():
    bare = TaskChain(
        initial_memory=[
            Object(id="buf", size=10, location="backing", type="other"),
        ],
        tasks=[
            Task(
                id="mut",
                inputs=["buf"],
                outputs=[OutputAlloc(id="out", size=1, location="fast")],
                runtime=1,
                mutates_inputs=["buf"],
            ),
        ],
        final_locations={"buf": "backing"},
        fast_memory_capacity=32,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert annotated.tasks[0].mutates_inputs == ["buf"]
    assert "buf" not in annotated.tasks[0].releases_after
    assert any(trig.obj_id == "buf" for trig in annotated.tasks[0].offload_after)
    run(annotated)


def test_pressurefit_uses_timing_relief_when_static_boundary_is_impossible():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=1, location="fast", type="activation"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["x"],
                outputs=[
                    OutputAlloc(id="A", size=60, location="fast"),
                    OutputAlloc(id="y", size=1, location="fast"),
                ],
                runtime=10,
            ),
            Task(
                id="t1",
                inputs=["y"],
                outputs=[OutputAlloc(id="B", size=60, location="fast")],
                runtime=10,
            ),
            Task(id="t2", inputs=[], outputs=[], runtime=10),
            Task(id="t3", inputs=["A"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=100,
        bandwidth_from_slow=10,
        bandwidth_to_slow=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert any(trig.obj_id == "A" for trig in annotated.tasks[0].offload_after)
    assert any(
        trig.obj_id == "A"
        for task in annotated.tasks[1:3]
        for trig in task.prefetch_after
    )
    run(annotated)


def test_packed_fifo_clamps_prefetch_fire_to_pressure():
    """Deadline packing must not fire a prefetch into boundaries whose
    modeled bytes leave no room for the transfer's destination."""
    from dataflow_sim.policies.pressurefit import _emit_chain

    bare = TaskChain(
        initial_memory=[
            # `hog` pins boundaries -1..1 (anchors at every gap), leaving no
            # room for x's 30 bytes before boundary 2.
            Object(id="hog", size=90, location="fast", type="other"),
            Object(id="x", size=30, location="backing", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=["hog"], outputs=[], runtime=10),
            Task(id="t1", inputs=["hog"], outputs=[], runtime=10),
            Task(id="t2", inputs=["hog"], outputs=[], runtime=10),
            Task(id="t3", inputs=[], outputs=[], runtime=10),
            Task(id="t4", inputs=["x"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )
    facts = _build_facts(bare)
    intervals = _initial_residency(facts, initial_compute=set())
    _reduce_to_fit(facts, intervals, bare.fast_memory_capacity)
    assert intervals["x"] == [(3, 3)]

    unclamped = _emit_chain(bare, facts, intervals, pack_inbound=True)
    assert "x" in [t.obj_id for t in unclamped.tasks[0].prefetch_after]

    annotated = _emit_chain(
        bare, facts, intervals, pack_inbound=True, clamp_inbound=True,
    )
    prefetch_by_task = {
        task.id: [trig.obj_id for trig in task.prefetch_after]
        for task in annotated.tasks
    }
    # Deadline packing alone fires x on t0 (tau=30, deadline=40), but
    # boundaries 0..1 hold 90/100 bytes; the clamp slides the trigger to t2.
    assert prefetch_by_task["t0"] == []
    assert prefetch_by_task["t1"] == []
    assert "x" in prefetch_by_task["t2"]
    run(annotated)


def test_pressurefit_extends_prefetch_intervals_under_strict_cap():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=25, location="backing", type="other"),
            Object(id="y", size=25, location="backing", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=[], outputs=[], runtime=10),
            Task(id="t1", inputs=[], outputs=[], runtime=10),
            Task(id="t2", inputs=[], outputs=[], runtime=10),
            Task(id="t3", inputs=["x"], outputs=[], runtime=10),
            Task(id="t4", inputs=["y"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=50,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )
    facts = _build_facts(bare)
    intervals = _initial_residency(facts, initial_compute=set())
    _reduce_to_fit(facts, intervals, bare.fast_memory_capacity)

    assert intervals["x"] == [(2, 2)]
    assert intervals["y"] == [(3, 3)]

    _extend_inbound_lead_time(
        facts, intervals, bare.fast_memory_capacity, bare.bandwidth_from_slow,
    )

    assert intervals["x"] == [(1, 2)]
    assert intervals["y"] == [(1, 3)]


