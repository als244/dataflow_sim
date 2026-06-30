from __future__ import annotations

from dataflow_sim.engine.simulator import run as simulator_run
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS, HardwareSpec
from dataflow_sim.workloads.dataflow_builder import (
    DTypePolicy,
    OpDTypePolicy,
    ParallelismConfig,
    TensorRef,
    TrainingConfig,
    dtype_nbytes,
)
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining
from dataflow_sim.workloads.models.olmoe import OLMoEConfig, OLMoEForTraining
from dataflow_sim.workloads.models.qwen3 import Qwen3Config, Qwen3ForTraining
from dataflow_sim.workloads.models.qwen3_hybrid_dense import (
    QwenHybridDenseConfig,
    QwenHybridDenseForTraining,
)
from dataflow_sim.workloads.models.qwen3_hybrid_moe import (
    QwenHybridMoEConfig,
    QwenHybridMoEForTraining,
)
from dataflow_sim.workloads.models.qwen3_moe import Qwen3MoEConfig, Qwen3MoEForTraining
from dataflow_sim.workloads.models.deepseek_v3 import DeepSeekV3Config, DeepSeekV3ForTraining
from dataflow_sim.workloads.models.deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForTraining
from dataflow_sim.workloads.models.glm5 import GLM5Config, GLM5ForTraining
from dataflow_sim.workloads.models.glm5_2 import GLM52Config, GLM52ForTraining
from dataflow_sim.workloads.models.gpt_oss import GPTOSSConfig, GPTOSSForTraining
from dataflow_sim.workloads.models.kimi_k2 import KimiK2Config, KimiK2ForTraining
from dataflow_sim.workloads.models.nemotron_h import NemotronHConfig, NemotronHForTraining
from dataflow_sim.workloads.modules import (
    DenseAttention,
    DeepSeekV32Block,
    DSASparseAttention,
    GPTOSSBlock,
    MoE,
    NemotronDimensions,
    NemotronBlock,
    ReLU2MoE,
    SwiGLUMLP,
    TransformerDimensions,
    layer_activation_elements_per_token,
    optimizer_ops_for_matrices,
)
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.ops import optimizer as opt_ops
from dataflow_sim.workloads.summary import compute_workload_summary


def _tasks_by_id(program):
    return {task.id: task for task in program.tasks}


def _block_keys(program) -> set[str]:
    return {block.key for block in program.compute_blocks}


def _blocks_by_key(program):
    return {block.key: block for block in program.compute_blocks}


def test_dtype_policy_defaults_and_low_precision_sizes():
    policy = DTypePolicy()

    assert policy.param == "bf16"
    assert policy.activation == "bf16"
    assert policy.expert_dispatch == "bf16"
    assert policy.gradient == "bf16"
    assert policy.optimizer_state == "bf16"
    assert policy.compute == "bf16"
    assert policy.expert_param == "bf16"
    assert policy.expert_compute == "bf16"
    assert policy.indexer_activation == "fp8"
    assert policy.indexer_compute == "fp8"
    assert dtype_nbytes("fp8") == 1
    assert dtype_nbytes("fp4") == 0.5
    assert TensorRef("packed", (3, 1), dtype="fp4").size_bytes == 2


def test_muon_uses_split_real_matrix_shapes_and_expert_counts():
    fused = [
        opt_ops.OptimizerMatrix("fused_gate_up", rows=16, cols=8),
    ]
    split = [
        opt_ops.OptimizerMatrix("gate_proj", rows=16, cols=4),
        opt_ops.OptimizerMatrix("up_proj", rows=16, cols=4),
    ]
    fused_ops = optimizer_ops_for_matrices(
        "test",
        matrices=fused,
        optimizer="muon",
    )
    split_ops = optimizer_ops_for_matrices(
        "test",
        matrices=split,
        optimizer="muon",
    )

    assert [op.name for op in split_ops] == [
        "gate_proj_muon_step",
        "up_proj_muon_step",
    ]
    assert sum(op.flops * op.count for op in split_ops) < fused_ops[0].flops

    experts = [
        opt_ops.OptimizerMatrix("routed_mlp_up", rows=16, cols=4, count=8, expert=True)
    ]
    expert_ops = optimizer_ops_for_matrices(
        "test",
        matrices=experts,
        optimizer="muon",
    )
    base_flops, _ = opt_ops.muon_matrix_flops_bytes(16, 4)
    assert expert_ops[0].name == "routed_mlp_up_muon_step"
    assert expert_ops[0].flops == base_flops
    assert expert_ops[0].count == 8


def test_module_optimizer_matrices_use_real_parameter_shapes():
    dims = TransformerDimensions(
        vocab_size=128,
        n_layers=1,
        d_model=16,
        head_dim=4,
        n_heads=4,
        n_kv_heads=2,
        expert_dim=8,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2,
    )

    assert [m.name for m in DenseAttention(dims).optimizer_matrices()] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_proj",
    ]
    assert [m.name for m in SwiGLUMLP(dims).optimizer_matrices()] == [
        "shared_mlp_gate",
        "shared_mlp_up",
        "shared_mlp_down",
    ]
    moe_matrices = MoE(dims).optimizer_matrices()
    assert [m.name for m in moe_matrices] == [
        "shared_mlp_gate",
        "shared_mlp_up",
        "shared_mlp_down",
        "routed_mlp_gate",
        "routed_mlp_up",
        "routed_mlp_down",
    ]
    assert [m.count for m in moe_matrices[-3:]] == [4, 4, 4]
    assert all(m.expert for m in moe_matrices)
    assert sum(m.rows * m.cols * m.count for m in DenseAttention(dims).optimizer_matrices() + moe_matrices) == (
        16 * 16
        + 16 * 8
        + 16 * 8
        + 16 * 16
        + 3 * 16 * 8
        + 3 * 16 * 8 * 4
    )

    nemotron_dims = NemotronDimensions(
        vocab_size=128,
        n_layers=1,
        d_model=16,
        head_dim=4,
        n_heads=4,
        n_kv_heads=2,
        expert_dim=8,
        shared_expert_dim=12,
        num_shared_experts=1,
        num_routed_experts=4,
        top_k=2,
        intermediate_size=24,
        mamba_num_heads=4,
        mamba_head_dim=4,
        ssm_state_size=4,
        conv_kernel=4,
        mamba_chunk_size=8,
        n_groups=2,
        layer_types=("moe",),
        hybrid_override_pattern="E",
    )
    relu2_matrices = ReLU2MoE(nemotron_dims).optimizer_matrices()
    assert [m.name for m in relu2_matrices] == [
        "shared_mlp_up",
        "shared_mlp_down",
        "routed_mlp_up",
        "routed_mlp_down",
    ]
    assert [m.count for m in relu2_matrices[-2:]] == [4, 4]


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


def test_backward_matmul_policy_separates_activation_and_parameter_gradients():
    policy = OpDTypePolicy.from_dtype_policy(
        DTypePolicy(param="fp8", activation="fp8", gradient="bf16")
    )
    dgrad = bwd.matmul_input_grad(
        "proj_dgrad",
        tokens=4,
        input_dim=3,
        output_dim=5,
        bytes_per_element=policy,
    )
    wgrad = bwd.matmul_weight_grad(
        "proj_wgrad",
        tokens=4,
        input_dim=3,
        output_dim=5,
        bytes_per_element=policy,
    )

    assert dgrad.memory_bytes == 4 * 5 + 3 * 5 + 4 * 3
    assert wgrad.memory_bytes == 4 * 3 + 4 * 5 + 3 * 5 * 2 + 3 * 5 * 2


def test_datatype_policy_changes_program_bytes_and_compute_precision():
    training = TrainingConfig(seqlen=64, num_seqs=1, optimizer="adamw")
    model = Qwen3MoEForTraining(
        Qwen3MoEConfig.preset(
            "30B-3B",
            n_layers=2,
            d_model=256,
            n_heads=4,
            n_kv_heads=2,
            expert_dim=64,
            num_routed_experts=8,
            top_k=2,
            vocab_size=4096,
        )
    )
    bf16_program = model.build_training_program(training, dtype_policy=DTypePolicy())
    lowp_program = model.build_training_program(
        training,
        dtype_policy=DTypePolicy(
            param="fp8",
            activation="fp8",
            gradient="fp8",
            optimizer_state="fp8",
            expert_param="fp4",
            expert_compute="fp8",
        ),
    )
    split_grad_program = model.build_training_program(
        training,
        dtype_policy=DTypePolicy(activation="bf16", gradient="fp8"),
    )

    bf16_objects = {obj.id: obj.size_bytes for obj in bf16_program.objects}
    lowp_objects = {obj.id: obj.size_bytes for obj in lowp_program.objects}
    assert lowp_objects["W_0"] < bf16_objects["W_0"]
    assert lowp_objects["O_0"] < bf16_objects["O_0"]
    assert bf16_objects["input_0_0"] == training.tokens * model.dims.d_model * 2
    assert lowp_objects["input_0_0"] == training.tokens * model.dims.d_model

    bf16_outputs = {
        output.id: output.size_bytes
        for task in bf16_program.tasks
        for output in task.outputs
    }
    lowp_outputs = {
        output.id: output.size_bytes
        for task in lowp_program.tasks
        for output in task.outputs
    }
    saved_width = layer_activation_elements_per_token(model.dims)
    assert bf16_outputs["y_0_0_0"] == training.tokens * model.dims.d_model * 2
    assert lowp_outputs["y_0_0_0"] == training.tokens * model.dims.d_model
    assert bf16_outputs["A_0_0_0"] == training.tokens * saved_width * 2
    assert lowp_outputs["A_0_0_0"] == training.tokens * saved_width
    assert lowp_outputs["dy_head_0_0"] == training.tokens * model.dims.d_model
    assert lowp_outputs["dW_0_0"] == model.layers[0].param_count
    split_grad_outputs = {
        output.id: output.size_bytes
        for task in split_grad_program.tasks
        for output in task.outputs
    }
    assert split_grad_outputs["dy_head_0_0"] == training.tokens * model.dims.d_model * 2
    assert split_grad_outputs["dW_0_0"] == model.layers[0].param_count

    moe_forward = next(
        block
        for block in lowp_program.compute_blocks
        if block.key == "transformer_block.forward"
    )
    routed_up = next(
        op for op in moe_forward.subops if op.name == "routed_mlp_up_one_expert"
    )
    qkv = next(op for op in moe_forward.subops if op.name == "qkv_proj")
    assert routed_up.efficiency == "matmul_fp8"
    assert qkv.efficiency == "matmul_bf16"


def test_expert_dispatch_dtype_changes_forward_and_backward_dispatch_lanes():
    training = TrainingConfig(seqlen=64, num_seqs=1, optimizer="none")
    model = QwenHybridMoEForTraining(
        QwenHybridMoEConfig.preset("35B-A3B", n_layers=1)
    )

    bf16_program = model.build_training_program(training, dtype_policy=DTypePolicy())
    dispatch_program = model.build_training_program(
        training,
        dtype_policy=DTypePolicy(expert_dispatch="fp8"),
    )

    bf16_outputs = {
        output.id: output.size_bytes
        for task in bf16_program.tasks
        for output in task.outputs
    }
    dispatch_outputs = {
        output.id: output.size_bytes
        for task in dispatch_program.tasks
        for output in task.outputs
    }
    assert dispatch_outputs["A_0_0_0"] == bf16_outputs["A_0_0_0"]
    assert dispatch_outputs["dy_head_0_0"] == bf16_outputs["dy_head_0_0"]
    assert dispatch_outputs["dW_0_0"] == bf16_outputs["dW_0_0"]

    bf16_forward = next(
        block for block in bf16_program.compute_blocks
        if any(op.name == "shared_mlp_up" for op in block.subops)
    )
    dispatch_forward = next(
        block for block in dispatch_program.compute_blocks
        if any(op.name == "shared_mlp_up" for op in block.subops)
    )
    bf16_ops = {op.name: op for op in bf16_forward.subops}
    dispatch_ops = {op.name: op for op in dispatch_forward.subops}

    for name in ("x_scatter", "shared_mlp_up", "routed_mlp_up_one_expert"):
        assert dispatch_ops[name].memory_bytes < bf16_ops[name].memory_bytes
    for name in (
        "ffn_norm",
        "swiglu",
        "shared_mlp_down",
        "routed_mlp_down_one_expert",
        "x_gather",
    ):
        assert dispatch_ops[name].memory_bytes == bf16_ops[name].memory_bytes

    bf16_backward = next(
        block for block in bf16_program.compute_blocks
        if any(op.name == "dy_scatter" for op in block.subops)
    )
    dispatch_backward = next(
        block for block in dispatch_program.compute_blocks
        if any(op.name == "dy_scatter" for op in block.subops)
    )
    bf16_bwd_ops = {op.name: op for op in bf16_backward.subops}
    dispatch_bwd_ops = {op.name: op for op in dispatch_backward.subops}
    for name in (
        "dy_scatter",
        "routed_mlp_down_one_expert_dgrad",
        "routed_mlp_up_one_expert_dgrad",
        "dy_gather",
        "shared_mlp_up_dgrad",
        "routed_mlp_down_one_expert_wgrad",
        "routed_mlp_up_one_expert_wgrad",
        "shared_mlp_up_wgrad",
    ):
        assert dispatch_bwd_ops[name].memory_bytes < bf16_bwd_ops[name].memory_bytes
    for name in (
        "shared_mlp_down_dgrad",
        "swiglu_bwd",
        "shared_mlp_down_wgrad",
    ):
        assert dispatch_bwd_ops[name].memory_bytes == bf16_bwd_ops[name].memory_bytes


def test_ep_group_size_shards_routed_experts_and_uses_scale_up_movement():
    training = TrainingConfig(seqlen=64, num_seqs=1, optimizer="muon")
    config = Qwen3MoEConfig.preset(
        "30B-3B",
        n_layers=1,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,
        expert_dim=64,
        num_shared_experts=1,
        num_routed_experts=8,
        top_k=2,
        vocab_size=4096,
    )
    model = Qwen3MoEForTraining(config)

    ep1 = model.build_training_program(training)
    ep2 = model.build_training_program(
        training,
        parallelism=ParallelismConfig(ep_group_size=2),
    )

    ep1_objects = {obj.id: obj.size_bytes for obj in ep1.objects}
    ep2_objects = {obj.id: obj.size_bytes for obj in ep2.objects}
    ep1_outputs = {
        output.id: output.size_bytes
        for task in ep1.tasks
        for output in task.outputs
    }
    ep2_outputs = {
        output.id: output.size_bytes
        for task in ep2.tasks
        for output in task.outputs
    }
    assert ep2_objects["W_0"] < ep1_objects["W_0"]
    assert ep2_objects["O_0"] < ep1_objects["O_0"]
    assert ep2_outputs["dW_0_0"] < ep1_outputs["dW_0_0"]

    ep1_forward = _blocks_by_key(ep1)["transformer_block.forward"]
    ep2_forward = _blocks_by_key(ep2)["transformer_block.forward"]
    ep1_ops = {op.name: op for op in ep1_forward.subops}
    ep2_ops = {op.name: op for op in ep2_forward.subops}

    assert ep1_ops["x_scatter"].efficiency == "memory"
    assert ep2_ops["x_scatter"].efficiency == "scale_up"
    assert ep1_ops["x_gather"].efficiency == "memory"
    assert ep2_ops["x_gather"].efficiency == "scale_up"
    assert ep1_ops["routed_mlp_up_one_expert"].count == config.num_routed_experts
    assert ep2_ops["routed_mlp_up_one_expert"].count == config.num_routed_experts // 2
    assert ep2_ops["routed_mlp_up_one_expert"].flops == (
        2 * ep1_ops["routed_mlp_up_one_expert"].flops
    )
    assert (
        ep2_ops["routed_mlp_up_one_expert"].flops
        * ep2_ops["routed_mlp_up_one_expert"].count
    ) == (
        ep1_ops["routed_mlp_up_one_expert"].flops
        * ep1_ops["routed_mlp_up_one_expert"].count
    )

    ep2_optimizer = _blocks_by_key(ep2)["optimizer_step.muon"]
    opt_rows = {op.name: op for op in ep2_optimizer.subops}
    assert opt_rows["routed_mlp_gate_muon_step"].count == config.num_routed_experts // 2
    assert opt_rows["shared_mlp_gate_muon_step"].count == config.num_shared_experts
    assert ep2.metadata["parallelism"] == {"ep_group_size": 2}


def test_invalid_ep_group_size_for_moe_raises_clear_error():
    training = TrainingConfig(seqlen=64, num_seqs=1, optimizer="none")
    model = Qwen3MoEForTraining(
        Qwen3MoEConfig.preset(
            "30B-3B",
            n_layers=1,
            num_routed_experts=8,
            top_k=2,
        )
    )

    try:
        model.build_training_program(
            training,
            parallelism=ParallelismConfig(ep_group_size=3),
        )
    except ValueError as exc:
        assert "ep_group_size=3 must divide routed expert count 8" in str(exc)
    else:
        raise AssertionError("expected invalid EP group size to raise")


def test_compute_precision_changes_realized_runtime_under_unlimited_memory():
    hw = HardwareSpec(
        peak_tflops_bf16=100,
        peak_tflops_fp8=200,
        peak_tflops_fp4=400,
        fast_memory_bw_gbs=1_000_000,
        from_slow_bw_gbs=1_000_000,
        to_slow_bw_gbs=1_000_000,
        matmul_eff_bf16=1.0,
        matmul_eff_fp8=1.0,
        matmul_eff_fp4=1.0,
        attn_fwd_eff=1.0,
        attn_bwd_eff=1.0,
        mem_eff=1.0,
    )
    training = TrainingConfig(seqlen=64, num_seqs=1, optimizer="none")
    model = Llama3ForTraining(
        Llama3Config.preset(
            "8B",
            n_layers=2,
            d_model=512,
            n_heads=8,
            n_kv_heads=4,
            expert_dim=1024,
            vocab_size=4096,
        )
    )

    bf16 = model.build_training_workload(
        training,
        hw,
        dtype_policy=DTypePolicy(compute="bf16"),
    )
    fp8 = model.build_training_workload(
        training,
        hw,
        dtype_policy=DTypePolicy(compute="fp8"),
    )
    bf16_forward = next(
        block
        for block in bf16.metadata["compute_blocks"]
        if block["key"] == "transformer_block.forward"
    )
    fp8_forward = next(
        block
        for block in fp8.metadata["compute_blocks"]
        if block["key"] == "transformer_block.forward"
    )

    assert fp8_forward["total_runtime_us"] < bf16_forward["total_runtime_us"]

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
    qwen_235b_moe = Qwen3MoEConfig.preset("235B-A22B", n_layers=8)
    olmoe = OLMoEConfig.preset("7B-1B", top_k=4)
    qwen_hybrid = QwenHybridDenseConfig.preset("9B", n_layers=12)
    qwen_hybrid_moe = QwenHybridMoEConfig.preset("397B-A17B", top_k=8)
    deepseek = DeepSeekV3Config.preset("671B-37B", n_layers=4)
    deepseek_v32 = DeepSeekV32Config.preset("671B-37B", n_layers=4)
    glm5 = GLM5Config.preset("5", n_layers=4)
    glm51 = DeepSeekV32Config.preset("glm-5.1", n_layers=4)
    glm52 = GLM52Config.preset("5.2", n_layers=4)
    kimi = KimiK2Config.preset("1T-32B", first_k_dense_replace=2)
    nemotron = NemotronHConfig.preset("nano", n_layers=4, hybrid_override_pattern="M*E-")
    gpt_oss = GPTOSSConfig.preset("20B", n_layers=4)

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
    assert qwen_235b_moe.preset_name == "qwen3_moe_235B-A22B"
    assert qwen_235b_moe.n_layers == 8
    assert qwen_235b_moe.d_model == 4096
    assert olmoe.preset_name == "olmoe_7B-1B"
    assert olmoe.top_k == 4
    assert qwen_hybrid.preset_name == "qwen3_5_9B"
    assert qwen_hybrid.n_layers == 12
    assert qwen_hybrid.layer_types[:4] == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    assert qwen_hybrid_moe.preset_name == "qwen3_5_397B-A17B"
    assert qwen_hybrid_moe.top_k == 8
    assert deepseek.preset_name == "deepseek_v3_671B-37B"
    assert deepseek.n_layers == 4
    assert deepseek_v32.preset_name == "deepseek_v3_2_671B-37B"
    assert deepseek_v32.n_layers == 4
    assert DeepSeekV32ForTraining(deepseek_v32).family_name == "deepseek_v3_2"
    assert glm5.preset_name == "glm_5_744B-40B"
    assert glm5.n_layers == 4
    assert GLM5ForTraining(glm5).family_name == "deepseek_v3_2"
    assert glm51.preset_name == "glm_5_744B-40B"
    assert glm51.n_layers == 4
    assert glm52.preset_name == "glm_5_2_744B-40B"
    assert glm52.n_layers == 4
    assert glm52.indexer_modes() == ("full", "full", "full", "shared")
    assert kimi.preset_name == "kimi_k2_1T-32B"
    assert kimi.first_k_dense_replace == 2
    assert KimiK2ForTraining(kimi).family_name == "deepseek_v3"
    assert nemotron.preset_name == "nemotron3_nano_30B-A3B"
    assert nemotron.n_layers == 4
    assert nemotron.dimensions().layer_types == ("mamba", "attention", "moe", "mlp")
    assert gpt_oss.preset_name == "gpt_oss_20B"
    assert gpt_oss.n_layers == 4
    assert gpt_oss.dimensions().layer_types == (
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    )


def test_new_family_public_preset_values_match_source_configs():
    qwen_dense = QwenHybridDenseConfig.preset("qwen3_5_27B")
    qwen3_moe = Qwen3MoEConfig.preset("qwen3_moe_235B-A22B")
    qwen_moe = QwenHybridMoEConfig.preset("qwen3_5_35B-A3B")
    qwen_large_moe = QwenHybridMoEConfig.preset("qwen3_5_397B-A17B")
    deepseek = DeepSeekV3Config.preset("671B-37B")
    deepseek_v32 = DeepSeekV32Config.preset("671B-37B")
    glm5 = GLM5Config.preset("5")
    glm51 = GLM5Config.preset("5.1")
    glm52 = GLM52Config.preset("5.2")
    kimi = KimiK2Config.preset("1T-32B")
    nemotron_nano = NemotronHConfig.preset("nano")
    nemotron_super = NemotronHConfig.preset("super")
    nemotron_ultra = NemotronHConfig.preset("ultra")
    gpt_oss_20b = GPTOSSConfig.preset("20B")
    gpt_oss_120b = GPTOSSConfig.preset("120B")

    assert qwen_dense.vocab_size == 248_320
    assert qwen_dense.n_layers == 64
    assert qwen_dense.d_model == 5120
    assert qwen_dense.linear_num_value_heads == 48
    assert QwenHybridDenseConfig.preset("qwen3_6_27B").preset_name == "qwen3_5_27B"
    assert qwen3_moe.vocab_size == 151_936
    assert qwen3_moe.n_layers == 94
    assert qwen3_moe.d_model == 4096
    assert qwen3_moe.n_heads == 64
    assert qwen3_moe.n_kv_heads == 4
    assert qwen3_moe.expert_dim == 1536
    assert qwen3_moe.num_routed_experts == 128
    assert qwen3_moe.top_k == 8
    assert qwen_moe.n_layers == 40
    assert qwen_moe.expert_dim == 512
    assert qwen_moe.num_routed_experts == 256
    assert qwen_moe.top_k == 8
    assert QwenHybridMoEConfig.preset("qwen3_6_35B-A3B").preset_name == "qwen3_5_35B-A3B"
    assert qwen_large_moe.num_routed_experts == 512
    assert qwen_large_moe.top_k == 10
    assert deepseek.vocab_size == 129_280
    assert deepseek.n_heads == 128
    assert deepseek.first_k_dense_replace == 3
    assert deepseek.q_lora_rank == 1536
    assert deepseek.kv_lora_rank == 512
    assert deepseek_v32.vocab_size == 129_280
    assert deepseek_v32.n_layers == 61
    assert deepseek_v32.first_k_dense_replace == 3
    assert deepseek_v32.d_model == 7168
    assert deepseek_v32.n_heads == 128
    assert deepseek_v32.q_lora_rank == 1536
    assert deepseek_v32.kv_lora_rank == 512
    assert deepseek_v32.qk_nope_head_dim == 128
    assert deepseek_v32.qk_rope_head_dim == 64
    assert deepseek_v32.v_head_dim == 128
    assert deepseek_v32.expert_dim == 2048
    assert deepseek_v32.num_routed_experts == 256
    assert deepseek_v32.num_shared_experts == 1
    assert deepseek_v32.top_k == 8
    assert deepseek_v32.index_n_heads == 64
    assert deepseek_v32.index_head_dim == 128
    assert deepseek_v32.index_topk == 2048
    assert glm5.vocab_size == 154_880
    assert glm5.n_layers == 78
    assert glm5.first_k_dense_replace == 3
    assert glm5.d_model == 6144
    assert glm5.head_dim == 64
    assert glm5.n_heads == 64
    assert glm5.n_kv_heads == 64
    assert glm5.intermediate_size == 12_288
    assert glm5.expert_dim == 2048
    assert glm5.num_routed_experts == 256
    assert glm5.num_shared_experts == 1
    assert glm5.top_k == 8
    assert glm5.q_lora_rank == 2048
    assert glm5.kv_lora_rank == 512
    assert glm5.qk_nope_head_dim == 192
    assert glm5.qk_rope_head_dim == 64
    assert glm5.v_head_dim == 256
    assert glm5.index_n_heads == 32
    assert glm5.index_head_dim == 128
    assert glm5.index_topk == 2048
    assert glm51 == glm5
    assert glm52.vocab_size == 154_880
    assert glm52.n_layers == 78
    assert glm52.first_k_dense_replace == 3
    assert glm52.d_model == 6144
    assert glm52.head_dim == 192
    assert glm52.n_heads == 64
    assert glm52.n_kv_heads == 64
    assert glm52.intermediate_size == 12_288
    assert glm52.expert_dim == 2048
    assert glm52.num_routed_experts == 256
    assert glm52.num_shared_experts == 1
    assert glm52.top_k == 8
    assert glm52.q_lora_rank == 2048
    assert glm52.kv_lora_rank == 512
    assert glm52.qk_nope_head_dim == 192
    assert glm52.qk_rope_head_dim == 64
    assert glm52.v_head_dim == 256
    assert glm52.index_n_heads == 32
    assert glm52.index_head_dim == 128
    assert glm52.index_topk == 2048
    assert glm52.index_topk_freq == 4
    assert glm52.index_skip_topk_offset == 3
    glm52_modes = glm52.indexer_modes()
    assert len(glm52_modes) == 78
    assert glm52_modes.count("full") == 21
    assert glm52_modes.count("shared") == 57
    assert [i for i, mode in enumerate(glm52_modes) if mode == "full"] == [
        0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74
    ]
    assert kimi.vocab_size == 163_840
    assert kimi.n_heads == 64
    assert kimi.num_routed_experts == 384
    assert kimi.routed_scaling_factor == 2.827
    assert nemotron_nano.vocab_size == 131_072
    assert nemotron_nano.n_layers == 52
    assert nemotron_nano.d_model == 2688
    assert nemotron_nano.expert_dim == 1856
    assert nemotron_nano.shared_expert_dim == 3712
    assert nemotron_nano.mamba_num_heads == 64
    assert nemotron_nano.mamba_chunk_size == 128
    assert nemotron_super.n_layers == 88
    assert nemotron_super.d_model == 4096
    assert nemotron_super.num_routed_experts == 512
    assert nemotron_ultra.n_layers == 108
    assert nemotron_ultra.d_model == 8192
    assert nemotron_ultra.n_heads == 64
    assert gpt_oss_20b.vocab_size == 201_088
    assert gpt_oss_20b.n_layers == 24
    assert gpt_oss_20b.d_model == 2880
    assert gpt_oss_20b.head_dim == 64
    assert gpt_oss_20b.n_heads == 64
    assert gpt_oss_20b.n_kv_heads == 8
    assert gpt_oss_20b.expert_dim == 2880
    assert gpt_oss_20b.num_routed_experts == 32
    assert gpt_oss_20b.top_k == 4
    assert gpt_oss_20b.sliding_window == 128
    assert gpt_oss_120b.n_layers == 36
    assert gpt_oss_120b.d_model == 2880
    assert gpt_oss_120b.num_routed_experts == 128
    assert gpt_oss_120b.top_k == 4
    assert gpt_oss_120b.sliding_window == 128
    assert gpt_oss_20b.dimensions().layer_types.count("sliding_attention") == 12
    assert gpt_oss_20b.dimensions().layer_types.count("full_attention") == 12
    assert gpt_oss_120b.dimensions().layer_types.count("sliding_attention") == 18
    assert gpt_oss_120b.dimensions().layer_types.count("full_attention") == 18
    for config, counts in (
        (nemotron_nano, {"mamba": 23, "attention": 6, "moe": 23}),
        (nemotron_super, {"mamba": 40, "attention": 8, "moe": 40}),
        (nemotron_ultra, {"mamba": 48, "attention": 12, "moe": 48}),
    ):
        layer_types = config.dimensions().layer_types
        assert len(layer_types) == config.n_layers
        assert {name: layer_types.count(name) for name in counts} == counts


def test_new_op_helper_formulas_are_hand_checkable():
    conv = fwd.depthwise_causal_conv1d(
        "conv",
        tokens=5,
        dim=7,
        kernel_size=3,
    )
    delta = fwd.gated_delta_rule(
        "delta",
        tokens=5,
        num_key_heads=2,
        key_head_dim=4,
        num_value_heads=3,
        value_head_dim=6,
        chunk_size=2,
    )
    mla = fwd.mla_attention(
        "mla",
        tokens=8,
        n_heads=2,
        qk_head_dim=5,
        value_head_dim=3,
        seqlen=4,
    )
    mla_bwd = bwd.mla_attention_grad(
        "mla_bwd",
        tokens=8,
        n_heads=2,
        qk_head_dim=5,
        value_head_dim=3,
        seqlen=4,
    )
    relu2 = fwd.relu2("relu2", tokens=5, dim=7)
    relu2_bwd = bwd.relu2_grad("relu2_bwd", tokens=5, dim=7)
    mamba = fwd.mamba_chunk_scan(
        "mamba_scan",
        tokens=8,
        seqlen=4,
        num_heads=3,
        head_dim=5,
        state_dim=7,
        n_groups=2,
        chunk_size=2,
    )
    mamba_bwd = bwd.mamba_chunk_scan_grad(
        "mamba_scan_bwd",
        tokens=8,
        seqlen=4,
        num_heads=3,
        head_dim=5,
        state_dim=7,
        n_groups=2,
        chunk_size=2,
    )
    sliding = fwd.sliding_attention(
        "sliding_attn",
        tokens=8,
        n_heads=2,
        n_kv_heads=1,
        head_dim=5,
        window_size=2,
        seqlen=4,
    )
    sliding_full = fwd.sliding_attention(
        "sliding_attn_full",
        tokens=8,
        n_heads=2,
        n_kv_heads=1,
        head_dim=5,
        window_size=8,
        seqlen=4,
    )
    sliding_varlen = fwd.sliding_attention(
        "sliding_attn_varlen",
        tokens=8,
        n_heads=2,
        n_kv_heads=1,
        head_dim=5,
        window_size=2,
        sequence_lengths=[3, 5],
    )
    sliding_bwd = bwd.sliding_attention_grad(
        "sliding_attn_bwd",
        tokens=8,
        n_heads=2,
        n_kv_heads=1,
        head_dim=5,
        window_size=2,
        seqlen=4,
    )
    index_policy = OpDTypePolicy.from_dtype_policy(
        DTypePolicy(indexer_activation="fp8", indexer_compute="fp8")
    )
    index_score = fwd.lightning_index_score(
        "index_score",
        tokens=8,
        index_n_heads=2,
        index_head_dim=3,
        index_topk=2,
        seqlen=4,
        bytes_per_element=index_policy,
    )
    index_score_bwd = bwd.lightning_index_score_grad(
        "index_score_bwd",
        tokens=8,
        index_n_heads=2,
        index_head_dim=3,
        index_topk=2,
        seqlen=4,
        bytes_per_element=index_policy,
    )
    dsa = fwd.dsa_sparse_attention(
        "dsa_sparse_attn",
        tokens=8,
        n_heads=2,
        kv_lora_rank=5,
        rope_head_dim=3,
        value_head_dim=7,
        index_topk=2,
        seqlen=4,
    )
    dsa_bwd = bwd.dsa_sparse_attention_grad(
        "dsa_sparse_attn_bwd",
        tokens=8,
        n_heads=2,
        kv_lora_rank=5,
        rope_head_dim=3,
        value_head_dim=7,
        index_topk=2,
        seqlen=4,
    )

    assert conv.flops == 2 * 5 * 7 * 3
    assert conv.memory_bytes == (5 * 7 + 7 * 3 + 5 * 7) * 2
    assert delta.flops == 5 * 3 * (6 * 4 * 6 + 2 * 2 * (4 + 6))
    assert delta.memory_bytes == (2 * 5 * 8 + 4 * 5 * 18 + 2 * 5 * 3 + 5 * 3 * 2) * 2
    assert mla.flops == 2 * (5 + 3) * (2 * 4 * 4)
    assert mla_bwd.flops == 2 * (2 * 5 + 3 * 3) * (2 * 4 * 4)
    assert mla_bwd.effective_flops == 2 * (2 * 5 + 2 * 3) * (2 * 4 * 4)
    assert relu2.memory_bytes == 3 * 5 * 7 * 2
    assert relu2_bwd.memory_bytes == 5 * 5 * 7 * 2
    assert mamba.flops == (
        2 * 2 * 2 * 2 * 2 * 3 * (7 + 5)
        + 4 * 8 * 3 * 5 * 7
        + 2 * 2 * 2 * 3 * 5 * 7
        + 2 * 8 * 15
    )
    assert mamba.memory_bytes == (
        8 * (15 + 2 * 2 * 7 + 3 + 15)
        + 2 * 2 * 2 * 2 * 3
        + 2 * 2 * 3 * 5 * 7
    ) * 2
    assert mamba_bwd.flops == 2 * mamba.flops
    assert mamba_bwd.memory_bytes == 2 * mamba.memory_bytes
    assert sliding.flops == 2 * 2 * 5 * (2 * 4 * 2)
    assert sliding.memory_bytes == (8 * (2 + 2 * 1) * 5 + 8 * 2 * 5) * 2
    assert sliding_full.flops == 2 * 2 * 5 * (2 * 4 * 4)
    assert sliding_varlen.flops == 2 * 2 * 5 * (3 * 2 + 5 * 2)
    assert sliding_bwd.flops == 5 * 2 * 5 * (2 * 4 * 2)
    assert sliding_bwd.effective_flops == 4 * 2 * 5 * (2 * 4 * 2)
    assert sliding_bwd.memory_bytes == (
        (8 * (2 + 2 * 1) * 5 + 8 * 2 * 5) * 2
        + 8 * (2 + 2 * 1) * 5 * 2
    )
    assert index_score.flops == 2 * (2 * 4 * 4) * 2 * 3
    assert index_score.memory_bytes == 8 * 2 * 3 + 8 * 3 + 8 * 2 + (2 * 4 * 2) * (4 + 1)
    assert index_score.efficiency == "matmul_fp8"
    assert index_score_bwd.flops == 6 * (2 * 4 * 2) * 2 * 3
    assert index_score_bwd.effective_flops == 4 * (2 * 4 * 2) * 2 * 3
    assert index_score_bwd.memory_bytes == (2 * 4 * 2) + 8 * 2 * 3 + 8 * 3 + 8 * 2
    assert dsa.flops == 2 * (2 * 5 + 3) * (2 * 4 * 2)
    assert dsa.memory_bytes == 8 * 2 * (2 * (5 + 3) + 2 * 7) * 2
    assert dsa_bwd.flops == 2 * (5 * 5 + 2 * 3) * (2 * 4 * 2)
    assert dsa_bwd.effective_flops == 2 * (4 * 5 + 2 * 3) * (2 * 4 * 2)
    assert dsa_bwd.memory_bytes == 8 * 2 * (3 * (5 + 3) + 3 * 7) * 2


def test_qwen_deepseek_and_gpt_oss_reuse_existing_moe_subops_without_router_ops():
    qwen = QwenHybridMoEConfig.preset(
        "35B-A3B",
        n_layers=1,
        d_model=256,
        n_heads=4,
        n_kv_heads=1,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=1024,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    ).dimensions()
    deepseek = DeepSeekV3Config.preset(
        "671B-37B",
        n_layers=1,
        first_k_dense_replace=0,
        d_model=256,
        n_heads=4,
        n_kv_heads=4,
        intermediate_size=512,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=1024,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        head_dim=48,
    ).dimensions()
    deepseek_v32 = DeepSeekV32Config.preset(
        "671B-37B",
        n_layers=1,
        first_k_dense_replace=0,
        d_model=256,
        n_heads=4,
        n_kv_heads=4,
        intermediate_size=512,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=1024,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        head_dim=48,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
    ).dimensions()
    gpt_oss = GPTOSSConfig.preset(
        "20B",
        n_layers=2,
        d_model=256,
        n_heads=4,
        n_kv_heads=1,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=1024,
    ).dimensions()

    qwen_expected = [op.name for op in MoE(qwen.ffn_dimensions()).forward_ops(tokens=16)]
    gpt_oss_expected = [
        op.name for op in MoE(gpt_oss.ffn_dimensions()).forward_ops(tokens=16)
    ]
    qwen_ops = [
        op.name
        for op in QwenHybridMoEForTraining(
            QwenHybridMoEConfig.preset(
                "35B-A3B",
                n_layers=1,
                d_model=256,
                n_heads=4,
                n_kv_heads=1,
                expert_dim=128,
                num_routed_experts=8,
                top_k=2,
                vocab_size=1024,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
            )
        ).layers[0].forward_ops(16, 16, 2)
    ]
    deepseek_ops = [
        op.name
        for op in DeepSeekV3ForTraining(
            DeepSeekV3Config.preset(
                "671B-37B",
                n_layers=4,
                d_model=256,
                n_heads=4,
                n_kv_heads=4,
                intermediate_size=512,
                expert_dim=128,
                num_routed_experts=8,
                top_k=2,
                vocab_size=1024,
                q_lora_rank=64,
                kv_lora_rank=32,
                qk_nope_head_dim=32,
                qk_rope_head_dim=16,
                v_head_dim=32,
                head_dim=48,
            )
        ).layers[-1].forward_ops(16, 16, 2)
    ]
    deepseek_v32_ops = [
        op.name
        for op in DeepSeekV32ForTraining(
            DeepSeekV32Config.preset(
                "671B-37B",
                n_layers=4,
                d_model=256,
                n_heads=4,
                n_kv_heads=4,
                intermediate_size=512,
                expert_dim=128,
                num_routed_experts=8,
                top_k=2,
                vocab_size=1024,
                q_lora_rank=64,
                kv_lora_rank=32,
                qk_nope_head_dim=32,
                qk_rope_head_dim=16,
                v_head_dim=32,
                head_dim=48,
                index_n_heads=2,
                index_head_dim=16,
                index_topk=4,
            )
        ).layers[-1].forward_ops(16, 16, 2)
    ]
    gpt_oss_ops = [
        op.name
        for op in GPTOSSForTraining(
            GPTOSSConfig.preset(
                "20B",
                n_layers=2,
                d_model=256,
                n_heads=4,
                n_kv_heads=1,
                expert_dim=128,
                num_routed_experts=8,
                top_k=2,
                vocab_size=1024,
            )
        ).layers[0].forward_ops(16, 16, 2)
    ]

    for name in qwen_expected:
        assert name in qwen_ops
        assert name in deepseek_ops
        assert name in deepseek_v32_ops
    for name in gpt_oss_expected:
        assert name in gpt_oss_ops
    assert not any("router" in name for name in qwen_ops)
    assert not any("router" in name for name in deepseek_ops)
    assert not any("router" in name for name in deepseek_v32_ops)
    assert not any("router" in name for name in gpt_oss_ops)
    assert deepseek.ffn_dimensions(dense=False).num_routed_experts == 8
    assert deepseek_v32.ffn_dimensions(dense=False).num_routed_experts == 8
    assert gpt_oss.ffn_dimensions().num_routed_experts == 8


def test_deepseek_v32_modules_emit_expected_subop_chains_blocks_and_indexer_dtypes():
    config = DeepSeekV32Config.preset(
        "671B-37B",
        n_layers=4,
        first_k_dense_replace=1,
        d_model=256,
        n_heads=4,
        n_kv_heads=4,
        intermediate_size=512,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=4096,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        head_dim=48,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
    )
    frozen_indexer_config = DeepSeekV32Config.preset(
        "671B-37B",
        n_layers=4,
        first_k_dense_replace=1,
        d_model=256,
        n_heads=4,
        n_kv_heads=4,
        intermediate_size=512,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=4096,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        head_dim=48,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        train_indexer=False,
    )
    dims = config.dimensions()
    dense_ops = [
        op.name
        for op in DeepSeekV32Block(dims, dense_ffn=True).forward_ops(tokens=16, seqlen=16)
    ]
    moe_ops = [
        op.name
        for op in DeepSeekV32Block(dims, dense_ffn=False).forward_ops(tokens=16, seqlen=16)
    ]

    assert dense_ops == [
        "attn_norm",
        "q_a_proj",
        "q_a_norm",
        "q_b_proj",
        "index_q_b_proj",
        "index_k_proj",
        "index_weight_proj",
        "lightning_index_score",
        "kv_a_proj_with_mqa",
        "kv_a_norm",
        "kv_b_proj",
        "dsa_rope",
        "dsa_sparse_attn",
        "o_proj",
        "ffn_norm",
        "shared_mlp_up",
        "swiglu",
        "shared_mlp_down",
    ]
    assert moe_ops[-8:] == [
        "ffn_norm",
        "shared_mlp_up",
        "x_scatter",
        "routed_mlp_up_one_expert",
        "swiglu",
        "shared_mlp_down",
        "routed_mlp_down_one_expert",
        "x_gather",
    ]
    assert not any("router" in name for name in moe_ops)

    program = DeepSeekV32ForTraining(config).build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw")
    )
    blocks = _blocks_by_key(program)
    assert _block_keys(program) >= {
        "deepseek_v3_2.dense_prefix_block.forward",
        "deepseek_v3_2.moe_suffix_block.forward",
        "deepseek_v3_2.dense_prefix_block.backward",
        "deepseek_v3_2.moe_suffix_block.backward",
        "deepseek_v3_2.dense_prefix_block.optimizer_step.adamw",
        "deepseek_v3_2.moe_suffix_block.optimizer_step.adamw",
    }
    assert blocks["deepseek_v3_2.dense_prefix_block.forward"].name == (
        "DeepSeek-V3.2 Dense Prefix Block Forward"
    )
    assert blocks["deepseek_v3_2.moe_suffix_block.forward"].name == (
        "DeepSeek-V3.2 MoE Block Forward"
    )

    glm_program = GLM5ForTraining(
        GLM5Config.preset(
            "5",
            n_layers=4,
            first_k_dense_replace=1,
            d_model=256,
            n_heads=4,
            n_kv_heads=4,
            intermediate_size=512,
            expert_dim=128,
            num_routed_experts=8,
            top_k=2,
            vocab_size=4096,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            head_dim=48,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=4,
        )
    ).build_training_program(TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw"))
    assert _block_keys(glm_program) >= {
        "deepseek_v3_2.dense_prefix_block.forward",
        "deepseek_v3_2.moe_suffix_block.forward",
        "deepseek_v3_2.dense_prefix_block.backward",
        "deepseek_v3_2.moe_suffix_block.backward",
    }

    default_forward = blocks["deepseek_v3_2.dense_prefix_block.forward"]
    index_score = next(op for op in default_forward.subops if op.name == "lightning_index_score")
    assert index_score.efficiency == "matmul_fp8"

    bf16_index_program = DeepSeekV32ForTraining(config).build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="none"),
        dtype_policy=DTypePolicy(indexer_activation="bf16", indexer_compute="bf16"),
    )
    default_outputs = {
        output.id: output.size_bytes
        for task in program.tasks
        for output in task.outputs
    }
    bf16_index_outputs = {
        output.id: output.size_bytes
        for task in bf16_index_program.tasks
        for output in task.outputs
    }
    assert default_outputs["A_0_0_0"] < bf16_index_outputs["A_0_0_0"]
    bf16_forward = _blocks_by_key(bf16_index_program)["deepseek_v3_2.dense_prefix_block.forward"]
    bf16_index_score = next(op for op in bf16_forward.subops if op.name == "lightning_index_score")
    assert bf16_index_score.efficiency == "matmul_bf16"
    assert index_score.memory_bytes < bf16_index_score.memory_bytes

    frozen_program = DeepSeekV32ForTraining(frozen_indexer_config).build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw")
    )
    frozen_blocks = _blocks_by_key(frozen_program)
    frozen_forward_names = [
        op.name
        for op in frozen_blocks["deepseek_v3_2.dense_prefix_block.forward"].subops
    ]
    frozen_backward_names = [
        op.name
        for op in frozen_blocks["deepseek_v3_2.dense_prefix_block.backward"].subops
    ]
    assert "lightning_index_score" in frozen_forward_names
    assert "lightning_index_score_bwd" not in frozen_backward_names
    assert "index_q_b_proj_wgrad" not in frozen_backward_names
    assert "index_k_proj_wgrad" not in frozen_backward_names
    assert "index_weight_proj_wgrad" not in frozen_backward_names

    frozen_outputs = {
        output.id: output.size_bytes
        for task in frozen_program.tasks
        for output in task.outputs
    }
    assert frozen_outputs["A_0_0_0"] < default_outputs["A_0_0_0"]
    assert frozen_outputs["dW_0_0"] < default_outputs["dW_0_0"]
    default_objects = {obj.id: obj.size_bytes for obj in program.objects}
    frozen_objects = {obj.id: obj.size_bytes for obj in frozen_program.objects}
    assert frozen_objects["W_0"] == default_objects["W_0"]
    assert frozen_objects["O_0"] < default_objects["O_0"]


def test_glm52_indexshare_blocks_skip_shared_indexer_work():
    config = GLM52Config.preset(
        "5.2",
        n_layers=8,
        first_k_dense_replace=3,
        d_model=256,
        n_heads=4,
        n_kv_heads=4,
        intermediate_size=512,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=4096,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        head_dim=48,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    assert config.indexer_modes() == (
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
    )

    model = GLM52ForTraining(config)
    program = model.build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw")
    )
    blocks = _blocks_by_key(program)
    task_counts = {
        key: sum(1 for task in program.tasks if task.compute_block_key == key)
        for key in _block_keys(program)
    }
    assert task_counts["glm_5_2.dense_full_index_block.forward"] == 3
    assert task_counts["glm_5_2.moe_full_index_block.forward"] == 1
    assert task_counts["glm_5_2.moe_shared_index_block.forward"] == 4
    assert task_counts["glm_5_2.moe_shared_index_block.backward"] == 4

    assert blocks["glm_5_2.dense_full_index_block.forward"].name == (
        "GLM-5.2 Dense Full-Index Block Forward"
    )
    assert blocks["glm_5_2.moe_full_index_block.forward"].name == (
        "GLM-5.2 MoE Full-Index Block Forward"
    )
    assert blocks["glm_5_2.moe_shared_index_block.forward"].name == (
        "GLM-5.2 MoE Shared-Index Block Forward"
    )

    full_forward_names = [
        op.name for op in blocks["glm_5_2.moe_full_index_block.forward"].subops
    ]
    shared_forward_names = [
        op.name for op in blocks["glm_5_2.moe_shared_index_block.forward"].subops
    ]
    shared_backward_names = [
        op.name for op in blocks["glm_5_2.moe_shared_index_block.backward"].subops
    ]

    assert "lightning_index_score" in full_forward_names
    assert "index_q_b_proj" in full_forward_names
    assert "dsa_sparse_attn" in shared_forward_names
    assert "lightning_index_score" not in shared_forward_names
    assert "index_q_b_proj" not in shared_forward_names
    assert "index_k_proj" not in shared_forward_names
    assert "index_weight_proj" not in shared_forward_names
    assert "lightning_index_score_bwd" not in shared_backward_names
    assert not any(name.startswith("index_") for name in shared_backward_names)

    indexer_matrix_params = (
        config.q_lora_rank * config.index_n_heads * config.index_head_dim
        + config.d_model * config.index_head_dim
        + config.d_model * config.index_n_heads
    )
    assert model.layers[6].param_count - model.layers[7].param_count == indexer_matrix_params
    assert (
        model.layers[6].gradient_param_count - model.layers[7].gradient_param_count
        == indexer_matrix_params
    )


def test_gpt_oss_modules_emit_expected_subop_chains_and_block_keys():
    config = GPTOSSConfig.preset(
        "20B",
        n_layers=2,
        d_model=256,
        n_heads=4,
        n_kv_heads=1,
        expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=4096,
        sliding_window=8,
    )
    dims = config.dimensions()
    sliding_ops = [
        op.name
        for op in GPTOSSBlock(dims, "sliding_attention").forward_ops(tokens=16, seqlen=16)
    ]
    full_ops = [
        op.name
        for op in GPTOSSBlock(dims, "full_attention").forward_ops(tokens=16, seqlen=16)
    ]

    assert sliding_ops == [
        "attn_norm",
        "qkv_proj",
        "rope",
        "sliding_attn",
        "attn_proj",
        "ffn_norm",
        "x_scatter",
        "routed_mlp_up_one_expert",
        "swiglu",
        "routed_mlp_down_one_expert",
        "x_gather",
    ]
    assert full_ops == [
        "attn_norm",
        "qkv_proj",
        "rope",
        "attn",
        "attn_proj",
        "ffn_norm",
        "x_scatter",
        "routed_mlp_up_one_expert",
        "swiglu",
        "routed_mlp_down_one_expert",
        "x_gather",
    ]
    program = GPTOSSForTraining(config).build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw")
    )
    assert _block_keys(program) >= {
        "gpt_oss.sliding_attention_moe_block.forward",
        "gpt_oss.full_attention_moe_block.forward",
        "gpt_oss.sliding_attention_moe_block.backward",
        "gpt_oss.full_attention_moe_block.backward",
        "gpt_oss.sliding_attention_moe_block.optimizer_step.adamw",
        "gpt_oss.full_attention_moe_block.optimizer_step.adamw",
    }
    blocks = _blocks_by_key(program)
    assert blocks["gpt_oss.sliding_attention_moe_block.forward"].name == (
        "GPT-OSS Sliding Attention MoE Block Forward"
    )
    assert blocks["gpt_oss.full_attention_moe_block.forward"].name == (
        "GPT-OSS Full Attention MoE Block Forward"
    )


def test_nemotron_modules_emit_expected_subop_chains_and_dtype_lanes():
    config = NemotronHConfig.preset(
        "nano",
        n_layers=3,
        d_model=256,
        n_heads=4,
        n_kv_heads=1,
        expert_dim=64,
        shared_expert_dim=128,
        num_routed_experts=8,
        top_k=2,
        vocab_size=4096,
        mamba_num_heads=4,
        mamba_head_dim=16,
        ssm_state_size=8,
        n_groups=2,
        intermediate_size=128,
        hybrid_override_pattern="M*E",
    )
    dims = config.dimensions()
    mamba_ops = [
        op.name
        for op in NemotronBlock(dims, "mamba").forward_ops(tokens=16, seqlen=16)
    ]
    attention_ops = [
        op.name
        for op in NemotronBlock(dims, "attention").forward_ops(tokens=16, seqlen=16)
    ]
    moe_ops = [
        op.name
        for op in NemotronBlock(dims, "moe").forward_ops(tokens=16, seqlen=16)
    ]

    assert mamba_ops == [
        "block_norm",
        "mamba_in_proj",
        "mamba_depthwise_conv1d",
        "mamba_silu",
        "mamba_chunk_scan",
        "mamba_gated_rms_norm",
        "mamba_out_proj",
    ]
    assert attention_ops == [
        "block_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "attn",
        "o_proj",
    ]
    assert moe_ops == [
        "block_norm",
        "shared_mlp_up",
        "x_scatter",
        "routed_mlp_up_one_expert",
        "shared_relu2",
        "routed_relu2",
        "shared_mlp_down",
        "routed_mlp_down_one_expert",
        "x_gather",
        "moe_combine_residual",
    ]
    assert not any("router" in name for name in moe_ops)
    program = NemotronHForTraining(config).build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw")
    )
    assert _block_keys(program) >= {
        "nemotron_h.mamba_block.forward",
        "nemotron_h.attention_block.forward",
        "nemotron_h.moe_block.forward",
        "nemotron_h.mamba_block.backward",
        "nemotron_h.attention_block.backward",
        "nemotron_h.moe_block.backward",
        "nemotron_h.mamba_block.optimizer_step.adamw",
        "nemotron_h.attention_block.optimizer_step.adamw",
        "nemotron_h.moe_block.optimizer_step.adamw",
    }

    bf16_ops = {
        op.name: op
        for op in ReLU2MoE(dims).forward_ops(tokens=16, bytes_per_element=2)
    }
    dispatch_ops = {
        op.name: op
        for op in ReLU2MoE(dims).forward_ops(
            tokens=16,
            bytes_per_element=OpDTypePolicy.from_dtype_policy(
                DTypePolicy(expert_dispatch="fp8")
            ),
        )
    }
    for name in ("x_scatter", "shared_mlp_up", "routed_mlp_up_one_expert"):
        assert dispatch_ops[name].memory_bytes < bf16_ops[name].memory_bytes
    for name in ("shared_relu2", "routed_relu2", "shared_mlp_down", "x_gather"):
        assert dispatch_ops[name].memory_bytes == bf16_ops[name].memory_bytes


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

    assert [task.id for task in program.tasks[:8]] == [
        "f_0_0_0",
        "f_0_0_1",
        "head_fwd_0_0",
        "head_bwd_0_0",
        "b_0_0_1",
        "b_0_0_0",
        "f_0_1_0",
        "f_0_1_1",
    ]
    assert [task.id for task in program.tasks[-3:]] == [
        "step_0_0",
        "step_0_1",
        "step_0_head",
    ]
    assert not any(task.id.startswith("r_") for task in program.tasks)
    assert tasks["b_0_1_1"].inputs == ["dy_head_0_1", "A_0_1_1", "W_1", "dW_0_1"]
    assert tasks["b_0_1_1"].mutates == ["dW_0_1"]
    assert tasks["step_0_1"].inputs == ["dW_0_1", "W_1", "O_1"]
    assert tasks["step_0_1"].mutates == ["W_1", "O_1"]
    assert tasks["step_0_head"].inputs == ["dW_head_0", "W_head", "O_head"]
    assert tasks["step_0_head"].mutates == ["W_head", "O_head"]
    blocks = _blocks_by_key(program)
    assert blocks["lm_head.forward"].name == "LM Head Forward"
    assert blocks["lm_head.backward"].name == "LM Head Bwd"
    assert [op.name for op in blocks["lm_head.forward"].subops] == [
        "final_norm",
        "head_proj",
        "cross_entropy",
    ]
    assert [op.name for op in blocks["lm_head.backward"].subops] == [
        "head_proj_dgrad",
        "head_proj_wgrad",
        "final_norm_bwd",
    ]
    assert program.final_locations == {
        "W_0": "backing",
        "O_0": "backing",
        "W_1": "backing",
        "O_1": "backing",
        "W_head": "backing",
        "O_head": "backing",
    }
    assert _block_keys(program) >= {
        "transformer_block.forward",
        "transformer_block.backward",
        "lm_head.forward",
        "lm_head.backward",
        "lm_head.optimizer_step.adamw",
        "optimizer_step.adamw",
    }
    assert blocks["optimizer_step.adamw"].name == (
        "AdamW Optimizer Step: Transformer Block"
    )
    assert blocks["lm_head.optimizer_step.adamw"].name == (
        "AdamW Optimizer Step: LM Head"
    )
    assert "transformer_block.recompute_slot" not in _block_keys(program)


def test_lm_head_optimizer_defaults_to_adamw_for_non_none_layer_optimizer():
    config = Llama3Config.preset(
        "8B",
        n_layers=1,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        expert_dim=256,
        vocab_size=1_024,
    )
    model = Llama3ForTraining(config)

    none_program = model.build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="none")
    )
    assert "step_0_head" not in _tasks_by_id(none_program)
    assert "lm_head.optimizer_step.adamw" not in _block_keys(none_program)
    assert not any(obj.id == "O_head" for obj in none_program.objects)

    muon_program = model.build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="muon")
    )
    muon_blocks = _blocks_by_key(muon_program)
    assert "optimizer_step.muon" in muon_blocks
    assert "lm_head.optimizer_step.adamw" in muon_blocks
    assert muon_blocks["optimizer_step.muon"].name == (
        "Muon Optimizer Step: Transformer Block"
    )
    assert muon_blocks["lm_head.optimizer_step.adamw"].name == (
        "AdamW Optimizer Step: LM Head"
    )
    assert [op.name for op in muon_blocks["lm_head.optimizer_step.adamw"].subops] == [
        "adamw_step"
    ]
    assert _tasks_by_id(muon_program)["step_0_head"].compute_block_key == (
        "lm_head.optimizer_step.adamw"
    )

    adamw_program = model.build_training_program(
        TrainingConfig(seqlen=16, num_seqs=1, optimizer="adamw")
    )
    assert "optimizer_step.adamw" in _block_keys(adamw_program)
    assert "lm_head.optimizer_step.adamw" in _block_keys(adamw_program)


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
    assert not any(task.id.startswith("r_") for task in base.tasks)
    assert not any(out.id == "A_0_0_1" for out in variant_tasks["f_0_0_1"].outputs)
    assert [task.id for task in variant.tasks if task.id.startswith("r_")] == [
        "r_0_0_1"
    ]
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
    optimizer_blocks = [
        block for block in workload.metadata["compute_blocks"]
        if block["key"] == "optimizer_step.muon"
    ]
    assert len(optimizer_blocks) == 1
    opt_rows = {row["name"]: row for row in optimizer_blocks[0]["subops"]}
    assert opt_rows["routed_mlp_gate_muon_step"]["count"] == config.num_routed_experts
    assert opt_rows["routed_mlp_up_muon_step"]["count"] == config.num_routed_experts
    old_fused_gate_up_flops, _ = opt_ops.muon_matrix_flops_bytes(
        config.d_model,
        2 * config.expert_dim,
    )
    split_gate_up_flops = (
        opt_rows["routed_mlp_gate_muon_step"]["flops"]
        * opt_rows["routed_mlp_gate_muon_step"]["count"]
        + opt_rows["routed_mlp_up_muon_step"]["flops"]
        * opt_rows["routed_mlp_up_muon_step"]["count"]
    )
    assert split_gate_up_flops < old_fused_gate_up_flops * config.num_routed_experts


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
        (
            QwenHybridDenseForTraining(
                QwenHybridDenseConfig.preset(
                    "9B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=1024,
                    intermediate_size=1024,
                    vocab_size=32_000,
                    linear_num_key_heads=4,
                    linear_num_value_heads=8,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="adamw"),
        ),
        (
            QwenHybridMoEForTraining(
                QwenHybridMoEConfig.preset(
                    "35B-A3B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    linear_num_key_heads=4,
                    linear_num_value_heads=8,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="muon"),
        ),
        (
            DeepSeekV3ForTraining(
                DeepSeekV3Config.preset(
                    "671B-37B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=8,
                    intermediate_size=1024,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    q_lora_rank=128,
                    kv_lora_rank=64,
                    qk_nope_head_dim=32,
                    qk_rope_head_dim=16,
                    v_head_dim=32,
                    head_dim=48,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="adamw"),
        ),
        (
            DeepSeekV32ForTraining(
                DeepSeekV32Config.preset(
                    "671B-37B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=8,
                    intermediate_size=1024,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    q_lora_rank=128,
                    kv_lora_rank=64,
                    qk_nope_head_dim=32,
                    qk_rope_head_dim=16,
                    v_head_dim=32,
                    head_dim=48,
                    index_n_heads=4,
                    index_head_dim=32,
                    index_topk=32,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="adamw"),
        ),
        (
            KimiK2ForTraining(
                KimiK2Config.preset(
                    "1T-32B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=8,
                    intermediate_size=1024,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    q_lora_rank=128,
                    kv_lora_rank=64,
                    qk_nope_head_dim=32,
                    qk_rope_head_dim=16,
                    v_head_dim=32,
                    head_dim=48,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="none"),
        ),
        (
            NemotronHForTraining(
                NemotronHConfig.preset(
                    "nano",
                    n_layers=3,
                    d_model=512,
                    head_dim=64,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    shared_expert_dim=256,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    mamba_num_heads=8,
                    mamba_head_dim=32,
                    ssm_state_size=16,
                    n_groups=2,
                    intermediate_size=256,
                    hybrid_override_pattern="M*E",
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="adamw"),
        ),
        (
            GPTOSSForTraining(
                GPTOSSConfig.preset(
                    "20B",
                    n_layers=4,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    sliding_window=64,
                )
            ),
            TrainingConfig(seqlen=128, num_seqs=1, optimizer="adamw"),
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
        block_keys = {block["key"] for block in workload.metadata["compute_blocks"]}
        assert {"lm_head.forward", "lm_head.backward"} <= block_keys
        assert any(key.endswith(".forward") for key in block_keys)
        assert any(key.endswith(".backward") for key in block_keys)


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
        (
            QwenHybridMoEForTraining(
                QwenHybridMoEConfig.preset(
                    "35B-A3B",
                    n_layers=6,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    linear_num_key_heads=4,
                    linear_num_value_heads=8,
                )
            ),
            TrainingConfig(seqlen=512, num_seqs=1, optimizer="none"),
            140 * 1024 * 1024,
        ),
        (
            DeepSeekV3ForTraining(
                DeepSeekV3Config.preset(
                    "671B-37B",
                    n_layers=6,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=8,
                    intermediate_size=1024,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    q_lora_rank=128,
                    kv_lora_rank=64,
                    qk_nope_head_dim=32,
                    qk_rope_head_dim=16,
                    v_head_dim=32,
                    head_dim=48,
                )
            ),
            TrainingConfig(seqlen=512, num_seqs=1, optimizer="none"),
            80 * 1024 * 1024,
        ),
        (
            DeepSeekV32ForTraining(
                DeepSeekV32Config.preset(
                    "671B-37B",
                    n_layers=6,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=8,
                    intermediate_size=1024,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    q_lora_rank=128,
                    kv_lora_rank=64,
                    qk_nope_head_dim=32,
                    qk_rope_head_dim=16,
                    v_head_dim=32,
                    head_dim=48,
                    index_n_heads=4,
                    index_head_dim=32,
                    index_topk=64,
                )
            ),
            TrainingConfig(seqlen=512, num_seqs=1, optimizer="none"),
            80 * 1024 * 1024,
        ),
        (
            NemotronHForTraining(
                NemotronHConfig.preset(
                    "nano",
                    n_layers=6,
                    d_model=512,
                    head_dim=64,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    shared_expert_dim=256,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    mamba_num_heads=8,
                    mamba_head_dim=32,
                    ssm_state_size=16,
                    n_groups=2,
                    intermediate_size=256,
                    hybrid_override_pattern="M*EM*E",
                )
            ),
            TrainingConfig(seqlen=512, num_seqs=1, optimizer="none"),
            80 * 1024 * 1024,
        ),
        (
            GPTOSSForTraining(
                GPTOSSConfig.preset(
                    "20B",
                    n_layers=6,
                    d_model=512,
                    n_heads=8,
                    n_kv_heads=2,
                    expert_dim=128,
                    num_routed_experts=8,
                    top_k=2,
                    vocab_size=32_000,
                    sliding_window=128,
                )
            ),
            TrainingConfig(seqlen=512, num_seqs=1, optimizer="none"),
            96 * 1024 * 1024,
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
