from __future__ import annotations

from dataflow_sim.policy._common import _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx
from dataflow_sim.policy.pressurefit import (
    _build_facts,
    _extend_h2d_lead_time,
    _initial_residency,
    _late_first_use_candidates,
    _pressure_initial_placement,
    _protected_warm_start_candidates_from_probe,
    _reduce_to_fit,
    apply_pressurefit_policy,
)
from dataflow_sim.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.simulator import run
from conftest import build_bare_training_chain


def _warm_choice_chain() -> TaskChain:
    return TaskChain(
        initial_memory=[
            Object(id="early", size=10, location="host", type="weight"),
            Object(id="late", size=10, location="host", type="weight"),
        ],
        tasks=[
            Task(id="t0", inputs=["early"], outputs=[], runtime=50),
            Task(id="t1", inputs=[], outputs=[], runtime=100),
            Task(id="t2", inputs=["late"], outputs=[], runtime=1),
        ],
        device_capacity=15,
        bandwidth_h2d=10,
        bandwidth_d2h=10,
    )


def test_pressure_initial_placement_skips_hidden_late_first_use():
    bare = _warm_choice_chain()
    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)

    placement = _pressure_initial_placement(
        bare, bare.device_capacity, sizes, uses_by_task,
    )

    assert "early" in placement  # task-0 input, no prior trigger slot
    assert "late" not in placement  # its first H2D hides behind t1


def test_pressurefit_prefetches_late_object_instead_of_preplacing():
    bare = _warm_choice_chain()
    annotated = apply_pressurefit_policy(bare)
    run(annotated)

    initial_device = {o.id for o in annotated.initial_memory if o.location == "device"}
    assert "early" in initial_device
    assert "late" not in initial_device
    assert any(
        trig.obj_id == "late"
        for task in annotated.tasks
        for trig in task.prefetch_after
    )


def test_pressurefit_runs_training_chain_at_moderate_cap():
    bare = build_bare_training_chain(L=5)
    annotated = apply_pressurefit_policy(bare, device_capacity=800)
    log = run(annotated)
    compute_ids = {iv.task_id for iv in log.task_intervals if iv.track == "compute"}
    assert "f_0" in compute_ids
    assert "b_0" in compute_ids


def test_pressurefit_preserves_generic_mutation_writeback():
    bare = TaskChain(
        initial_memory=[
            Object(id="buf", size=10, location="host", type="other"),
        ],
        tasks=[
            Task(
                id="mut",
                inputs=["buf"],
                outputs=[OutputAlloc(id="out", size=1, location="device")],
                runtime=1,
                mutates_inputs=["buf"],
            ),
        ],
        device_capacity=32,
        bandwidth_h2d=10,
        bandwidth_d2h=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert annotated.tasks[0].mutates_inputs == ["buf"]
    assert "buf" not in annotated.tasks[0].releases_after
    assert any(trig.obj_id == "buf" for trig in annotated.tasks[0].offload_after)
    run(annotated)


def test_pressurefit_uses_timing_relief_when_static_boundary_is_impossible():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=1, location="device", type="activation"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["x"],
                outputs=[
                    OutputAlloc(id="A", size=60, location="device"),
                    OutputAlloc(id="y", size=1, location="device"),
                ],
                runtime=10,
            ),
            Task(
                id="t1",
                inputs=["y"],
                outputs=[OutputAlloc(id="B", size=60, location="device")],
                runtime=10,
            ),
            Task(id="t2", inputs=[], outputs=[], runtime=10),
            Task(id="t3", inputs=["A"], outputs=[], runtime=10),
        ],
        device_capacity=100,
        bandwidth_h2d=10,
        bandwidth_d2h=10,
    )

    annotated = apply_pressurefit_policy(bare)

    assert any(trig.obj_id == "A" for trig in annotated.tasks[0].offload_after)
    assert any(
        trig.obj_id == "A"
        for task in annotated.tasks[1:3]
        for trig in task.prefetch_after
    )
    run(annotated)


def test_pressurefit_extends_prefetch_intervals_under_strict_cap():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=25, location="host", type="other"),
            Object(id="y", size=25, location="host", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=[], outputs=[], runtime=10),
            Task(id="t1", inputs=[], outputs=[], runtime=10),
            Task(id="t2", inputs=[], outputs=[], runtime=10),
            Task(id="t3", inputs=["x"], outputs=[], runtime=10),
            Task(id="t4", inputs=["y"], outputs=[], runtime=10),
        ],
        device_capacity=50,
        bandwidth_h2d=1,
        bandwidth_d2h=1,
    )
    facts = _build_facts(bare)
    intervals = _initial_residency(facts, initial_device=set())
    _reduce_to_fit(facts, intervals, bare.device_capacity)

    assert intervals["x"] == [(2, 2)]
    assert intervals["y"] == [(3, 3)]

    _extend_h2d_lead_time(
        facts, intervals, bare.device_capacity, bare.bandwidth_h2d,
    )

    assert intervals["x"] == [(1, 2)]
    assert intervals["y"] == [(1, 3)]


def test_pressurefit_warm_probe_and_protection_are_generic():
    bare = TaskChain(
        initial_memory=[
            Object(id="x", size=20, location="host", type="other"),
            Object(id="y", size=20, location="host", type="other"),
            Object(id="z", size=20, location="host", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=[], outputs=[], runtime=10),
            Task(id="t1", inputs=["x"], outputs=[], runtime=10),
            Task(id="t2", inputs=["y"], outputs=[], runtime=10),
            Task(id="t3", inputs=["z"], outputs=[], runtime=10),
        ],
        device_capacity=40,
        bandwidth_h2d=1,
        bandwidth_d2h=1,
    )
    facts = _build_facts(bare)

    assert _protected_warm_start_candidates_from_probe(
        facts, bare.device_capacity,
    ) == ["z"]

    intervals = _initial_residency(facts, initial_device={"x", "y", "z"})
    _reduce_to_fit(
        facts, intervals, bare.device_capacity, protected_initial={"z"},
    )

    assert intervals["z"] == [(-1, 2)]
    assert intervals["y"] == [(1, 1)]


def test_pressurefit_late_first_use_candidates_are_schema_driven():
    bare = TaskChain(
        initial_memory=[
            Object(id="early_clean", size=10, location="host", type="other"),
            Object(id="late_clean", size=20, location="host", type="other"),
            Object(id="late_mutated", size=30, location="host", type="other"),
        ],
        tasks=[
            Task(id="t0", inputs=["early_clean"], outputs=[], runtime=1),
            Task(id="t1", inputs=[], outputs=[], runtime=1),
            Task(
                id="t2",
                inputs=["late_mutated"],
                outputs=[],
                runtime=1,
                mutates_inputs=["late_mutated"],
            ),
            Task(id="t3", inputs=["late_clean"], outputs=[], runtime=1),
        ],
        device_capacity=64,
        bandwidth_h2d=10,
        bandwidth_d2h=10,
    )
    facts = _build_facts(bare)

    assert _late_first_use_candidates(facts, read_only=True) == [
        "late_clean",
        "early_clean",
    ]
    assert _late_first_use_candidates(facts, read_only=False) == [
        "late_clean",
        "late_mutated",
        "early_clean",
    ]
