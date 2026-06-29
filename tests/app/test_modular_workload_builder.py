from __future__ import annotations

from dataflow_sim.engine.simulator import run as simulator_run
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.dataflow_builder import (
    DTypePolicy,
    TensorRef,
    TrainingConfig,
    dtype_nbytes,
)
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining
from dataflow_sim.workloads.models.olmoe import OLMoEConfig, OLMoEForTraining
from dataflow_sim.workloads.models.qwen3 import Qwen3Config, Qwen3ForTraining
from dataflow_sim.workloads.models.qwen3_moe import Qwen3MoEConfig, Qwen3MoEForTraining
from dataflow_sim.workloads.modules import DenseAttention, SwiGLUMLP, TransformerDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.summary import compute_workload_summary


def _tasks_by_id(program):
    return {task.id: task for task in program.tasks}


def _block_keys(program) -> set[str]:
    return {block.key for block in program.compute_blocks}


def test_dtype_policy_defaults_and_low_precision_sizes():
    policy = DTypePolicy()

    assert policy.param == "bf16"
    assert policy.activation == "bf16"
    assert policy.gradient == "bf16"
    assert policy.optimizer_state == "bf16"
    assert dtype_nbytes("fp8") == 1
    assert dtype_nbytes("fp4") == 0.5
    assert TensorRef("packed", (3, 1), dtype="fp4").size_bytes == 2


def test_matmul_accumulate_epilogue_bytes():
    base = fwd.matmul("proj", tokens=4, input_dim=3, output_dim=5)
    fused = fwd.matmul(
        "proj",
        tokens=4,
        input_dim=3,
        output_dim=5,
        accumulate=True,
    )
    dgrad = bwd.matmul_input_grad(
        "proj_dgrad",
        tokens=4,
        input_dim=3,
        output_dim=5,
    )
    wgrad_accum = bwd.matmul_weight_grad(
        "proj_wgrad",
        tokens=4,
        input_dim=3,
        output_dim=5,
    )
    wgrad_no_accum = bwd.matmul_weight_grad(
        "proj_wgrad",
        tokens=4,
        input_dim=3,
        output_dim=5,
        accumulate=False,
    )

    assert base.memory_bytes == (4 * 3 + 3 * 5 + 4 * 5) * 2
    assert fused.memory_bytes == base.memory_bytes + 4 * 5 * 2
    assert dgrad.memory_bytes == base.memory_bytes
    assert wgrad_no_accum.memory_bytes == base.memory_bytes
    assert wgrad_accum.memory_bytes == base.memory_bytes + 3 * 5 * 2

    dims = TransformerDimensions(
        vocab_size=128,
        n_layers=1,
        d_model=16,
        head_dim=8,
        n_heads=2,
        n_kv_heads=1,
        expert_dim=32,
        num_shared_experts=1,
        num_routed_experts=0,
        top_k=0,
        qk_norm=False,
    )
    attn_proj = next(
        op for op in DenseAttention(dims).forward_ops(tokens=4, seqlen=4)
        if op.name == "attn_proj"
    )
    mlp_down = next(
        op for op in SwiGLUMLP(dims).forward_ops(tokens=4)
        if op.name == "shared_mlp_down"
    )

    assert attn_proj.memory_bytes == (4 * 16 + 16 * 16 + 4 * 16 + 4 * 16) * 2
    assert mlp_down.memory_bytes == (4 * 32 + 32 * 16 + 4 * 16 + 4 * 16) * 2


def test_family_presets_are_easy_to_override():
    llama = Llama3Config.preset("8B", n_layers=80, d_model=8192)
    llama_405b = Llama3Config.preset("405B", n_layers=4)
    qwen_4b = Qwen3Config.preset("4B", n_layers=12)
    qwen_32b = Qwen3Config.preset("32B", n_layers=36)
    moe = Qwen3MoEConfig.preset("30B-3B", top_k=4)
    olmoe = OLMoEConfig.preset("7B-1B", top_k=4)

    assert llama.preset_name == "llama3_8B"
    assert llama.n_layers == 80
    assert llama.d_model == 8192
    assert llama_405b.preset_name == "llama3_405B"
    assert llama_405b.d_model == 16384
    assert qwen_4b.preset_name == "qwen3_4B"
    assert qwen_4b.n_layers == 12
    assert qwen_32b.n_layers == 36
    assert moe.top_k == 4
    assert moe.preset_name == "qwen3_moe_30B-3B"
    assert olmoe.preset_name == "olmoe_7B-1B"
    assert olmoe.top_k == 4


def test_training_program_uses_model_order_reverse_backward_and_optimizer_tail():
    config = Llama3Config.preset(
        "8B",
        n_layers=2,
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        expert_dim=2048,
        vocab_size=32_000,
        qk_norm=True,
    )
    training = TrainingConfig(
        seqlen=128,
        num_seqs=1,
        grad_accum_rounds=2,
        optimizer="adamw",
        final_model_state_on_backing=True,
    )
    program = Llama3ForTraining(config).build_training_program(training)
    tasks = _tasks_by_id(program)

    assert [task.id for task in program.tasks[:7]] == [
        "f_0_0_0",
        "f_0_0_1",
        "head_0_0",
        "r_0_0_1",
        "b_0_0_1",
        "r_0_0_0",
        "b_0_0_0",
    ]
    assert [task.id for task in program.tasks[-2:]] == ["step_0_0", "step_0_1"]
    assert tasks["r_0_0_1"].inputs == ["y_0_0_0", "W_1"]
    assert tasks["r_0_0_0"].inputs == ["input_0_0", "W_0"]
    assert tasks["b_0_1_1"].inputs == ["dy_head_0_1", "A_0_1_1", "W_1", "dW_0_1"]
    assert tasks["b_0_1_1"].mutates == ["dW_0_1"]
    assert tasks["step_0_1"].inputs == ["dW_0_1", "W_1", "O_1"]
    assert tasks["step_0_1"].mutates == ["W_1", "O_1"]
    assert program.final_locations == {
        "W_0": "backing",
        "O_0": "backing",
        "W_1": "backing",
        "O_1": "backing",
    }
    assert _block_keys(program) >= {
        "transformer_block.forward",
        "transformer_block.backward",
        "transformer_block.recompute_slot",
        "transformer_head.training",
        "optimizer_step.adamw",
    }


def test_moe_recompute_variant_rewires_activation_producer_and_block_metadata():
    config = Qwen3MoEConfig.preset(
        "30B-3B",
        n_layers=2,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=32_000,
    )
    training = TrainingConfig(seqlen=128, num_seqs=1, optimizer="muon")
    model = Qwen3MoEForTraining(config)
    base = model.build_training_program(training)
    variant = model.build_training_program(training, recompute={"A_0_0_1": 1})
    base_tasks = _tasks_by_id(base)
    variant_tasks = _tasks_by_id(variant)

    assert any(out.id == "A_0_0_1" for out in base_tasks["f_0_0_1"].outputs)
    assert not any(out.id == "A_0_0_1" for out in variant_tasks["f_0_0_1"].outputs)
    assert [out.id for out in variant_tasks["r_0_0_1"].outputs] == ["A_0_0_1"]
    assert variant_tasks["r_0_0_1"].compute_block_key == "transformer_block.recompute"

    workload = model.build_training_workload(
        training,
        HARDWARE_PRESETS["H100"],
        recompute={"A_0_0_1": 1},
    )
    rewrites = workload.metadata["recompute_rewrites"]
    assert len(rewrites) == config.n_layers
    assert [opt.level for opt in rewrites[0].options] == [0, 1]
    assert rewrites[0].f_compute_block_key == "transformer_block.forward"
    assert rewrites[0].r_compute_block_key == "transformer_block.recompute"
    recompute_blocks = [
        block for block in workload.metadata["compute_blocks"]
        if block["category"] == "recompute" and block["total_runtime_us"] > 0
    ]
    assert len(recompute_blocks) == 1
    assert recompute_blocks[0]["key"] == "transformer_block.recompute"
    assert recompute_blocks[0]["total_flops"] > 0
    assert recompute_blocks[0]["total_effective_flops"] == 0


def test_varied_family_workloads_run_and_report_kpis():
    hw = HARDWARE_PRESETS["H100"]
    cases = [
        (
            Llama3ForTraining(
                Llama3Config.preset(
                    "8B",
                    n_layers=2,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=8,
                    expert_dim=2048,
                    vocab_size=32_000,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="adamw"),
        ),
        (
            Qwen3ForTraining(
                Qwen3Config.preset(
                    "32B",
                    n_layers=3,
                    d_model=768,
                    head_dim=64,
                    n_heads=12,
                    n_kv_heads=4,
                    expert_dim=3072,
                    vocab_size=48_000,
                )
            ),
            TrainingConfig(seqlen=192, num_seqs=1, optimizer="none"),
        ),
        (
            Qwen3MoEForTraining(
                Qwen3MoEConfig.preset(
                    "30B-3B",
                    n_layers=4,
                    d_model=1024,
                    n_heads=16,
                    n_kv_heads=4,
                    expert_dim=512,
                    num_routed_experts=16,
                    top_k=4,
                    vocab_size=64_000,
                )
            ),
            TrainingConfig(seqlen=256, num_seqs=1, optimizer="muon"),
        ),
        (
            OLMoEForTraining(
                OLMoEConfig.preset(
                    "7B-1B",
                    n_layers=3,
                    d_model=512,
                    head_dim=64,
                    n_heads=8,
                    n_kv_heads=8,
                    expert_dim=256,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                )
            ),
            TrainingConfig(seqlen=192, num_seqs=1, optimizer="adamw"),
        ),
    ]

    for model, training in cases:
        workload = model.build_training_workload(training, hw)
        chain = apply_pressurefit_policy(workload.chain)
        log = simulator_run(chain, snapshots=False)
        summary = compute_workload_summary(workload, log)

        assert summary["makespan_us"] > 0
        assert summary["tokens_per_second"] > 0
        assert summary["effective_tflops"] > 0
        assert summary["hardware_tflops"] >= summary["effective_tflops"]
        assert {block["key"] for block in workload.metadata["compute_blocks"]} >= {
            "transformer_block.forward",
            "transformer_block.backward",
            "transformer_head.training",
        }


def test_constrained_memory_recompute_planning_selects_useful_variants():
    hw = HARDWARE_PRESETS["H100"]
    cases = [
        (
            Llama3ForTraining(
                Llama3Config.preset(
                    "8B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=4,
                    expert_dim=2048,
                    vocab_size=32_000,
                    qk_norm=True,
                )
            ),
            TrainingConfig(seqlen=512, num_seqs=1, optimizer="none"),
            96 * 1024 * 1024,
        ),
        (
            Qwen3MoEForTraining(
                Qwen3MoEConfig.preset(
                    "30B-3B",
                    n_layers=6,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                )
            ),
            TrainingConfig(seqlen=1024, num_seqs=1, optimizer="none"),
            128 * 1024 * 1024,
        ),
    ]

    for model, training, cap in cases:
        base = model.build_training_workload(training, hw)
        result = plan_with_recompute(
            lambda levels, model=model, training=training: model.build_training_workload(
                training,
                hw,
                recompute=levels,
            ).chain,
            base.metadata["recompute_rewrites"],
            lambda chain, cap=cap: apply_pressurefit_policy(
                chain,
                fast_memory_capacity=cap,
            ),
            max_iters=2,
            max_wall_s=10,
        )

        assert sum(1 for level in result.levels.values() if level >= 1) > 0
        assert result.makespan_us < result.baseline_makespan_us
        selected = model.build_training_workload(training, hw, recompute=result.levels)
        summary = compute_workload_summary(
            selected,
            simulator_run(result.chain, snapshots=False),
        )
        assert summary["recompute_pct"] > 0
