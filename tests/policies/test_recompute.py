from __future__ import annotations

import re
from dataclasses import replace

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.engine.simulator import run
from dataflow_sim.engine.stall_report import build_stall_report
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.common.recompute import RecomputeOption, RecomputeRewrite
from dataflow_sim.workloads.dataflow_builder import TrainingConfig
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining


_BWD_TASK_ID = re.compile(r"^b_(\d+)_(\d+)_(\d+)$")
_RECOMPUTE_TASK_ID = re.compile(r"^r_(\d+)_(\d+)_(\d+)$")


def _legacy_placeholder_recompute_chain(chain: TaskChain) -> TaskChain:
    real_recompute = {
        tuple(map(int, _RECOMPUTE_TASK_ID.match(task.id).groups()))
        for task in chain.tasks
        if _RECOMPUTE_TASK_ID.match(task.id)
    }
    tasks: list[Task] = []
    for task in chain.tasks:
        match = _BWD_TASK_ID.match(task.id)
        if match is not None:
            k, j, i = map(int, match.groups())
            if (k, j, i) not in real_recompute:
                in_act = f"input_{k}_{j}" if i == 0 else f"y_{k}_{j}_{i - 1}"
                tasks.append(
                    Task(
                        id=f"r_{k}_{j}_{i}",
                        inputs=[in_act, f"W_{i}"],
                        outputs=[],
                        runtime=0.0,
                    )
                )
        tasks.append(task)
    return replace(chain, tasks=tasks)


def _makespan_us(chain: TaskChain, *, fast_memory_capacity: int | None = None) -> float:
    annotated = apply_pressurefit_policy(
        chain,
        fast_memory_capacity=fast_memory_capacity,
    )
    log = run(annotated, snapshots=False)
    return max(iv.end for iv in log.task_intervals)


# ---------------------------------------------------------------- stall report

def test_stall_report_attributes_input_wait_to_object():
    # Hand-annotated plan: x prefetched at t0's end, t1 stalls on its arrival.
    from dataflow_sim.core.schema import TransferTrigger

    annotated = TaskChain(
        initial_memory=[Object(id="x", size=100, location="backing", type="other")],
        tasks=[
            Task(
                id="t0", inputs=[], outputs=[], runtime=10,
                prefetch_after=[TransferTrigger(obj_id="x")],
            ),
            Task(id="t1", inputs=["x"], outputs=[], runtime=10),
        ],
        fast_memory_capacity=200,
        bandwidth_from_slow=1,   # x takes 100us; t1 must stall
        bandwidth_to_slow=1,
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
        initial_memory=[Object(id="big", size=90, location="fast", type="other")],
        tasks=[
            Task(id="t0", inputs=["big"], outputs=[], runtime=1),
            Task(
                id="t1",
                inputs=[],
                outputs=[OutputAlloc(id="out", size=50, location="fast")],
                runtime=1,
            ),
        ],
        final_locations={"big": "backing"},
        fast_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,   # offload of big takes 90us
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
            Object(id="x", size=50, location="backing", type="other"),
            Object(id="y", size=50, location="backing", type="other"),
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
        fast_memory_capacity=200,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )
    log = run(annotated, snapshots=False)
    report = build_stall_report(annotated, log)

    assert report.backlog_us["from_slow"] >= 50  # second transfer waits ~50us
    assert sum(report.transfer_backlog_overlap.values()) > 0


# ------------------------------------------------------- workload chain variant

def _small():
    config = Llama3Config.preset(
        "8B",
        n_layers=4,
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        expert_dim=2048,
        vocab_size=32_000,
        qk_norm=True,
    )
    model = Llama3ForTraining(config)
    hw = HARDWARE_PRESETS["H100"]
    cfg = TrainingConfig(seqlen=1024, num_seqs=1)
    return model, hw, cfg


def test_recompute_variant_rewires_activation_producer():
    model, hw, cfg = _small()
    base = model.build_training_workload(cfg, hw)
    var = model.build_training_workload(cfg, hw, recompute={"A_0_0_3": 1})

    base_tasks = {t.id: t for t in base.chain.tasks}
    var_tasks = {t.id: t for t in var.chain.tasks}

    assert any(o.id == "A_0_0_3" for o in base_tasks["f_0_0_3"].outputs)
    assert not any(t.id.startswith("r_") for t in base.chain.tasks)
    assert not any(o.id == "A_0_0_3" for o in var_tasks["f_0_0_3"].outputs)
    assert [t.id for t in var.chain.tasks if t.id.startswith("r_")] == [
        "r_0_0_3"
    ]
    assert [o.id for o in var_tasks["r_0_0_3"].outputs] == ["A_0_0_3"]
    assert var_tasks["r_0_0_3"].runtime > 0
    assert "y_0_0_2" in var_tasks["r_0_0_3"].inputs
    recompute_blocks = [
        block for block in var.metadata["compute_blocks"]
        if block["category"] == "recompute" and block["total_runtime_us"] > 0
    ]
    assert len(recompute_blocks) == 1
    assert recompute_blocks[0]["name"] == "Transformer Block Recompute"
    assert recompute_blocks[0]["subops"]
    assert recompute_blocks[0]["total_flops"] > 0
    assert recompute_blocks[0]["total_effective_flops"] == 0
    # Backward runtimes are identical across variants.
    assert var_tasks["b_0_0_3"].runtime == base_tasks["b_0_0_3"].runtime
    # Rewrite table declares binary options for every activation.
    rewrites = var.metadata["recompute_rewrites"]
    assert len(rewrites) == model.n_layers
    assert [opt.level for opt in rewrites[0].options] == [0, 1]
    assert rewrites[0].f_compute_block_key == "transformer_block.forward"
    assert rewrites[0].r_compute_block_key == "transformer_block.recompute"

    annotated = apply_pressurefit_policy(var.chain)
    run(annotated, snapshots=False)
    ann_tasks = {t.id: t for t in annotated.tasks}
    assert "y_0_0_0" in ann_tasks["f_0_0_1"].releases_after
    assert "y_0_0_1" in ann_tasks["f_0_0_2"].releases_after
    assert "y_0_0_2" in ann_tasks["r_0_0_3"].releases_after


def test_removing_noop_recompute_placeholders_does_not_reduce_throughput():
    model, hw, cfg = _small()
    chain = model.build_training_workload(cfg, hw).chain
    legacy = _legacy_placeholder_recompute_chain(chain)

    current_makespan = _makespan_us(chain, fast_memory_capacity=96 * 1024 * 1024)
    legacy_makespan = _makespan_us(legacy, fast_memory_capacity=96 * 1024 * 1024)

    assert current_makespan <= legacy_makespan


def test_recompute_variant_rejects_unknown_object():
    model, hw, cfg = _small()
    try:
        model.build_training_workload(cfg, hw, recompute={"nope": 1})
    except ValueError as e:
        assert "unknown recompute object" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ------------------------------------------------------------ recompute planner

def _structural_family(L, cap, *, activation_size, recompute_us, bandwidth):
    """Variant builder + rewrites over the structural training chain."""
    def build(levels):
        recomputed = {
            int(obj.rsplit("_", 1)[1])
            for obj, lvl in levels.items()
            if lvl >= 1
        }
        initial = [Object(id="input_0_0", size=8, location="fast", type="activation")]
        initial.extend(
            Object(id=f"W_{i}", size=8, location="backing", type="parameter")
            for i in range(L)
        )
        initial.append(Object(id="W_head", size=8, location="backing", type="parameter"))
        tasks = []
        for i in range(L):
            in_act = "input_0_0" if i == 0 else f"y_0_0_{i - 1}"
            outputs = [OutputAlloc(id=f"y_0_0_{i}", size=8, type="activation")]
            if i not in recomputed:
                outputs.insert(
                    0,
                    OutputAlloc(id=f"A_0_0_{i}", size=activation_size, type="activation"),
                )
            tasks.append(
                Task(
                    id=f"f_0_0_{i}",
                    inputs=[in_act, f"W_{i}"],
                    outputs=outputs,
                    runtime=10,
                )
            )
        tasks.append(
            Task(
                id="head_0_0",
                inputs=[f"y_0_0_{L - 1}", "W_head"],
                outputs=[OutputAlloc(id="dy_head_0_0", size=8, type="activation")],
                runtime=2,
            )
        )
        for i in range(L - 1, -1, -1):
            if i in recomputed:
                in_act = "input_0_0" if i == 0 else f"y_0_0_{i - 1}"
                tasks.append(
                    Task(
                        id=f"r_0_0_{i}",
                        inputs=[in_act, f"W_{i}"],
                        outputs=[
                            OutputAlloc(
                                id=f"A_0_0_{i}",
                                size=activation_size,
                                type="activation",
                            )
                        ],
                        runtime=recompute_us,
                    )
                )
            upstream = "dy_head_0_0" if i == L - 1 else f"dy_0_0_{i + 1}"
            tasks.append(
                Task(
                    id=f"b_0_0_{i}",
                    inputs=[upstream, f"A_0_0_{i}", f"W_{i}"],
                    outputs=[OutputAlloc(id=f"dy_0_0_{i}", size=8, type="activation")],
                    runtime=20,
                )
            )
        return TaskChain(
            initial_memory=initial,
            tasks=tasks,
            final_locations={},
            fast_memory_capacity=cap,
            backing_memory_capacity=None,
            bandwidth_from_slow=bandwidth,
            bandwidth_to_slow=bandwidth,
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
            f_compute_block_key="structural.forward",
            r_compute_block_key="structural.recompute",
            group_key="structural",
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
    assert all(t.outputs for t in result.chain.tasks if t.id.startswith("r_"))


def test_recompute_loop_keeps_everything_saved_when_memory_is_loose():
    build, rewrites = _structural_family(
        L=6, cap=None, activation_size=400, recompute_us=1, bandwidth=1,
    )
    result = plan_with_recompute(
        build, rewrites, lambda b: apply_pressurefit_policy(b),
    )
    assert result.makespan_us == result.baseline_makespan_us
    assert all(lvl == 0 for lvl in result.levels.values())
    assert not any(t.id.startswith("r_") for t in result.chain.tasks)
