from __future__ import annotations

from dataflow_sim.policy._common import _compute_ideal_starts, _object_sizes, _object_uses_by_task_idx
from dataflow_sim.policy.pressurefit import (
    _InitialProtectionJob,
    _build_facts,
    _extend_inbound_lead_time,
    _initial_protection_headroom,
    _initial_protection_jobs_from_probe,
    _initial_protection_sets,
    _initial_residency,
    _pressure_initial_placement,
    _reduce_to_fit,
    _select_initial_protection_set,
    apply_pressurefit_policy,
    plan_pressurefit_policy,
)
from dataflow_sim.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.simulator import run
from conftest import build_bare_training_chain


def _initial_choice_chain() -> TaskChain:
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


def test_pressure_initial_placement_skips_hidden_future_use():
    bare = _initial_choice_chain()
    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)

    placement = _pressure_initial_placement(
        bare, bare.device_capacity, sizes, uses_by_task,
    )

    assert "early" in placement  # task-0 input, no prior trigger slot
    assert "late" not in placement  # its first inbound hides behind t1


def test_pressurefit_prefetches_late_object_instead_of_preplacing():
    bare = _initial_choice_chain()
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


def test_pressurefit_diagnostics_describe_selected_candidate():
    bare = build_bare_training_chain(L=5)
    annotated, diagnostics = plan_pressurefit_policy(
        bare,
        device_capacity=800,
        portfolio_mode="fast",
    )
    log = run(annotated)
    makespan = max(iv.end for iv in log.task_intervals)

    assert diagnostics.portfolio_mode == "fast"
    assert diagnostics.effective_portfolio_mode == "fast"
    assert diagnostics.selected_makespan_us == makespan
    assert diagnostics.valid_candidate_count > 0
    selected = [c for c in diagnostics.candidates if c.selected]
    assert len(selected) == 1
    assert selected[0].name == diagnostics.selected_candidate
    assert selected[0].status == "valid"


def test_pressurefit_can_release_disposable_mutation_after_final_use():
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
    assert "buf" in annotated.tasks[0].releases_after
    assert not any(trig.obj_id == "buf" for trig in annotated.tasks[0].offload_after)
    run(annotated)


def test_pressurefit_preserves_final_host_mutation_writeback():
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
        final_locations={"buf": "host"},
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


def test_pressurefit_can_delay_prefetch_to_preserve_next_task_capacity():
    bare = TaskChain(
        initial_memory=[
            Object(id="input_0", size=10, location="device", type="activation"),
            Object(id="input_1", size=30, location="host", type="activation"),
            Object(id="W", size=10, location="host", type="weight"),
            Object(id="dW", size=10, location="host", type="gradient"),
        ],
        tasks=[
            Task(
                id="f_0_0",
                inputs=["input_0", "W"],
                outputs=[OutputAlloc(id="A_0_0", size=70, location="device")],
                runtime=1,
            ),
            Task(id="r_0_0", inputs=["A_0_0", "W"], outputs=[], runtime=0),
            Task(
                id="b_0_0",
                inputs=["A_0_0", "W", "dW"],
                outputs=[OutputAlloc(id="dy_0_0", size=10, location="device")],
                runtime=100,
                mutates_inputs=["dW"],
            ),
            Task(
                id="f_0_1",
                inputs=["input_1", "W"],
                outputs=[OutputAlloc(id="A_0_1", size=1, location="device")],
                runtime=1,
            ),
        ],
        device_capacity=120,
        bandwidth_h2d=1,
        bandwidth_d2h=10,
    )

    annotated = apply_pressurefit_policy(bare)
    run(annotated)

    prefetch_by_task = {
        task.id: [trig.obj_id for trig in task.prefetch_after]
        for task in annotated.tasks
    }
    assert "input_1" not in prefetch_by_task["r_0_0"]
    assert "input_1" in prefetch_by_task["b_0_0"]


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

    _extend_inbound_lead_time(
        facts, intervals, bare.device_capacity, bare.bandwidth_h2d,
    )

    assert intervals["x"] == [(1, 2)]
    assert intervals["y"] == [(1, 3)]


def test_pressurefit_initial_protection_uses_deadline_demand():
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

    assert _initial_protection_headroom(facts, bare.device_capacity) == 40
    jobs = _initial_protection_jobs_from_probe(
        facts, bare.device_capacity, bare.bandwidth_h2d,
    )
    assert [(job.oid, job.release_t, job.deadline, job.tau) for job in jobs] == [
        ("z", 20, 30, 20),
    ]
    protection_sets = _initial_protection_sets(
        facts, bare.device_capacity, bare.bandwidth_h2d,
    )
    assert {"z"} in protection_sets
    headroom = _initial_protection_headroom(facts, bare.device_capacity)
    assert all(
        sum(facts.sizes[oid] for oid in protected) <= headroom
        for protected in protection_sets
    )

    intervals = _initial_residency(facts, initial_device={"x", "y", "z"})
    _reduce_to_fit(
        facts, intervals, bare.device_capacity, protected_initial={"z"},
    )

    assert intervals["z"] == [(-1, 2)]
    assert intervals["y"] == [(1, 1)]


def test_pressurefit_initial_protection_selection_uses_capacity_and_inbound_work():
    jobs = [
        _InitialProtectionJob(
            oid="large",
            release_t=10,
            deadline=40,
            tau=30,
            first_use=4,
            size=30,
            residency_cost=1200,
        ),
        _InitialProtectionJob(
            oid="small",
            release_t=10,
            deadline=40,
            tau=5,
            first_use=4,
            size=5,
            residency_cost=200,
        ),
    ]

    assert _select_initial_protection_set(jobs, headroom=5) == {"small"}
    assert _select_initial_protection_set(jobs, headroom=30) == {"large"}
