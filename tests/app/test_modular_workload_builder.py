from __future__ import annotations

from dataflow_sim.engine.simulator import run as simulator_run
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS, HardwareSpec
from dataflow_sim.workloads.dataflow_builder import (
    DTypePolicy,
    OpDTypePolicy,
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
from dataflow_sim.workloads.models.kimi_k2 import KimiK2Config, KimiK2ForTraining
from dataflow_sim.workloads.modules import (
    DenseAttention,
    MoE,
    SwiGLUMLP,
    TransformerDimensions,
    layer_activation_elements_per_token,
)
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
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
    kimi = KimiK2Config.preset("1T-32B", first_k_dense_replace=2)

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
    assert kimi.preset_name == "kimi_k2_1T-32B"
    assert kimi.first_k_dense_replace == 2
    assert KimiK2ForTraining(kimi).family_name == "deepseek_v3"


def test_new_family_public_preset_values_match_source_configs():
    qwen_dense = QwenHybridDenseConfig.preset("qwen3_5_27B")
    qwen3_moe = Qwen3MoEConfig.preset("qwen3_moe_235B-A22B")
    qwen_moe = QwenHybridMoEConfig.preset("qwen3_5_35B-A3B")
    qwen_large_moe = QwenHybridMoEConfig.preset("qwen3_5_397B-A17B")
    deepseek = DeepSeekV3Config.preset("671B-37B")
    kimi = KimiK2Config.preset("1T-32B")

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
    assert kimi.vocab_size == 163_840
    assert kimi.n_heads == 64
    assert kimi.num_routed_experts == 384
    assert kimi.routed_scaling_factor == 2.827


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

    assert conv.flops == 2 * 5 * 7 * 3
    assert conv.memory_bytes == (5 * 7 + 7 * 3 + 5 * 7) * 2
    assert delta.flops == 5 * 3 * (6 * 4 * 6 + 2 * 2 * (4 + 6))
    assert delta.memory_bytes == (2 * 5 * 8 + 4 * 5 * 18 + 2 * 5 * 3 + 5 * 3 * 2) * 2
    assert mla.flops == 2 * (5 + 3) * (2 * 4 * 4)
    assert mla_bwd.flops == 2 * (2 * 5 + 3 * 3) * (2 * 4 * 4)
    assert mla_bwd.effective_flops == 2 * (2 * 5 + 2 * 3) * (2 * 4 * 4)


def test_qwen_and_deepseek_reuse_existing_moe_subops_without_router_ops():
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

    expected = [op.name for op in MoE(qwen.ffn_dimensions()).forward_ops(tokens=16)]
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

    for name in expected:
        assert name in qwen_ops
        assert name in deepseek_ops
    assert not any("router" in name for name in qwen_ops)
    assert not any("router" in name for name in deepseek_ops)
    assert deepseek.ffn_dimensions(dense=False).num_routed_experts == 8


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
    }
    assert _block_keys(program) >= {
        "transformer_block.forward",
        "transformer_block.backward",
        "transformer_block.recompute_slot",
        "lm_head.forward",
        "lm_head.backward",
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
