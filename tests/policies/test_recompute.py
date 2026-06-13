from __future__ import annotations

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.engine.simulator import run
from dataflow_sim.engine.stall_report import build_stall_report
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.common.recompute import RecomputeOption, RecomputeRewrite
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_layerwise_training_chain,
    build_transformer_training_workload,
)


# ---------------------------------------------------------------- stall report

def test_stall_report_attributes_input_wait_to_object():
    # Hand-annotated plan: x prefetched at t0's end, t1 stalls on its arrival.
    from dataflow_sim.core.schema import TransferTrigger

    annotated = TaskChain(
        initial_memory=[Object(id="x", size=100, location="host", type="other")],
        tasks=[
            Task(
                id="t0", inputs=[], outputs=[], runtime=10,
                prefetch_after=[TransferTrigger(obj_id="x")],
            ),
            Task(id="t1", inputs=["x"], outputs=[], runtime=10),
        ],
        device_capacity=200,
        bandwidth_h2d=1,   # x takes 100us; t1 must stall
        bandwidth_d2h=1,
    )
    log = run(annotated, snapshots=False)
    report = build_stall_report(annotated, log)

    assert report.stall_us > 0
    assert report.input_wait_us == report.stall_us
    assert report.stall_by_object.get("x", 0) == report.input_wait_us
    assert report.capacity_wait_us == 0


def test_stall_report_attributes_capacity_wait():
    # t1's output cannot be reserved until big's offload completes.
    bare = TaskChain(
        initial_memory=[Object(id="big", size=90, location="device", type="other")],
        tasks=[
            Task(id="t0", inputs=["big"], outputs=[], runtime=1),
            Task(
                id="t1",
                inputs=[],
                outputs=[OutputAlloc(id="out", size=50, location="device")],
                runtime=1,
            ),
        ],
        final_locations={"big": "host"},
        device_capacity=100,
        bandwidth_h2d=1,
        bandwidth_d2h=1,   # offload of big takes 90us
    )
    annotated = apply_pressurefit_policy(bare)
    log = run(annotated, snapshots=False)
    report = build_stall_report(annotated, log)

    assert report.capacity_wait_us > 0
    assert report.input_wait_us == 0


def test_stall_report_backlog_windows_cover_queued_transfers():
    # Two prefetches enqueued by t0; the second waits behind the first.
    from dataflow_sim.core.schema import TransferTrigger

    annotated = TaskChain(
        initial_memory=[
            Object(id="x", size=50, location="host", type="other"),
            Object(id="y", size=50, location="host", type="other"),
        ],
        tasks=[
            Task(
                id="t0", inputs=[], outputs=[], runtime=1,
                prefetch_after=[
                    TransferTrigger(obj_id="x"),
                    TransferTrigger(obj_id="y"),
                ],
            ),
            Task(id="t1", inputs=["x", "y"], outputs=[], runtime=1),
        ],
        device_capacity=200,
        bandwidth_h2d=1,
        bandwidth_d2h=1,
    )
    log = run(annotated, snapshots=False)
    report = build_stall_report(annotated, log)

    assert report.backlog_us["h2d"] >= 50  # second transfer waits ~50us
    assert sum(report.transfer_backlog_overlap.values()) > 0


# ------------------------------------------------------- workload chain variant

def _small():
    spec = load_model_presets()["nanogpt_124M"]
    hw = HARDWARE_PRESETS["H100"]
    cfg = TrainingConfig(seqlen=1024, num_seqs=1)
    return spec, hw, cfg


def test_recompute_variant_rewires_activation_producer():
    spec, hw, cfg = _small()
    base = build_transformer_training_workload(spec, hw, cfg)
    var = build_transformer_training_workload(
        spec, hw, cfg, recompute={"A_0_0_3": 1},
    )

    base_tasks = {t.id: t for t in base.chain.tasks}
    var_tasks = {t.id: t for t in var.chain.tasks}

    assert any(o.id == "A_0_0_3" for o in base_tasks["f_0_0_3"].outputs)
    assert not any(o.id == "A_0_0_3" for o in var_tasks["f_0_0_3"].outputs)
    assert [o.id for o in var_tasks["r_0_0_3"].outputs] == ["A_0_0_3"]
    assert var_tasks["r_0_0_3"].runtime > 0
    assert "y_0_0_2" in var_tasks["r_0_0_3"].inputs
    # Backward runtimes are identical across variants.
    assert var_tasks["b_0_0_3"].runtime == base_tasks["b_0_0_3"].runtime
    # Rewrite table declares binary options for every activation.
    rewrites = var.metadata["recompute_rewrites"]
    assert len(rewrites) == spec.n_layers
    assert [opt.level for opt in rewrites[0].options] == [0, 1]

    annotated = apply_pressurefit_policy(var.chain)
    run(annotated, snapshots=False)


def test_recompute_variant_rejects_unknown_object():
    spec, hw, cfg = _small()
    try:
        build_transformer_training_workload(spec, hw, cfg, recompute={"nope": 1})
    except ValueError as e:
        assert "unknown recompute object" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ------------------------------------------------------------ recompute planner

def _structural_family(L, cap, *, activation_size, recompute_us, bandwidth):
    """Variant builder + rewrites over the structural training chain."""
    def build(levels):
        instances = frozenset(
            (0, 0, int(obj.rsplit("_", 1)[1]))
            for obj, lvl in levels.items() if lvl >= 1
        )
        chain = build_layerwise_training_chain(
            L,
            input_size=8,
            weight_size=8,
            activation_size=activation_size,
            layer_output_size=8,
            grad_size=8,
            head_weight_size=8,
            fwd_runtime=10,
            head_runtime=2,
            bwd_runtime=20,
            bandwidth_h2d=bandwidth,
            bandwidth_d2h=bandwidth,
            recompute=instances,
            recompute_runtime=recompute_us,
        )
        return TaskChain(
            initial_memory=chain.initial_memory,
            tasks=chain.tasks,
            final_locations=chain.final_locations,
            device_capacity=cap,
            host_capacity=None,
            bandwidth_h2d=chain.bandwidth_h2d,
            bandwidth_d2h=chain.bandwidth_d2h,
        )

    options = (
        RecomputeOption(level=0, saved_bytes=activation_size, recompute_us=0),
        RecomputeOption(level=1, saved_bytes=0, recompute_us=recompute_us),
    )
    rewrites = [
        RecomputeRewrite(
            object_id=f"A_0_0_{i}",
            f_task_id=f"f_0_0_{i}",
            r_task_id=f"r_0_0_{i}",
            options=options,
        )
        for i in range(L)
    ]
    return build, rewrites


def test_recompute_loop_converts_under_pressure_and_improves():
    # Activations dominate memory and the link is slow: round-tripping them
    # stalls backward, while recompute costs almost nothing.
    build, rewrites = _structural_family(
        L=6, cap=600, activation_size=400, recompute_us=1, bandwidth=1,
    )
    result = plan_with_recompute(
        build, rewrites, lambda b: apply_pressurefit_policy(b),
    )
    assert result.makespan_us < result.baseline_makespan_us
    assert any(lvl >= 1 for lvl in result.levels.values())


def test_recompute_loop_keeps_everything_saved_when_memory_is_loose():
    build, rewrites = _structural_family(
        L=6, cap=None, activation_size=400, recompute_us=1, bandwidth=1,
    )
    result = plan_with_recompute(
        build, rewrites, lambda b: apply_pressurefit_policy(b),
    )
    assert result.makespan_us == result.baseline_makespan_us
    assert all(lvl == 0 for lvl in result.levels.values())
