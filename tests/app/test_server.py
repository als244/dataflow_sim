from dataflow_sim.app.server.main import (
    SimulationParams,
    WorkloadPreviewParams,
    preview_workload,
    presets,
    simulate,
)


_MODEL_KEYS = {
    "family", "vocab_size", "n_layers", "d_model", "head_dim", "n_heads",
    "n_kv_heads", "expert_dim", "num_shared_experts", "num_routed_experts",
    "top_k", "qk_norm", "intermediate_size", "full_attention_interval",
    "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
    "linear_value_head_dim", "linear_conv_kernel_dim", "gdn_chunk_size",
    "router_aux_loss_coef", "mtp_num_hidden_layers", "first_k_dense_replace", "q_lora_rank",
    "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
    "index_n_heads", "index_head_dim", "index_topk",
    "routed_scaling_factor", "scoring_func", "shared_expert_dim",
    "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "conv_kernel",
    "mamba_chunk_size", "n_groups", "hybrid_override_pattern", "sliding_window",
}
_TRAINING_KEYS = {
    "seqlen", "num_seqs", "grad_accum_rounds", "num_steps", "optimizer",
    "final_model_state_on_backing",
}
_PLANNER_KEYS = {
    "policy", "window_size", "fast_memory_capacity_gb", "recompute",
}


def _payload(**overrides):
    payload = {
        "workload": {
            "source": "model_training",
            "preset": "custom",
            "model": {
                "preset": "custom",
                "family": "llama3",
                "vocab_size": 64,
                "n_layers": 2,
                "d_model": 16,
                "head_dim": 8,
                "n_heads": 2,
                "n_kv_heads": 1,
                "expert_dim": 32,
                "num_shared_experts": 1,
                "num_routed_experts": 0,
                "top_k": 0,
                "qk_norm": False,
            },
            "training": {
                "seqlen": 8,
                "num_seqs": 1,
                "grad_accum_rounds": 1,
                "num_steps": 1,
                "optimizer": "none",
                "final_model_state_on_backing": False,
            },
        },
        "hardware": {
            "preset": "custom",
            "peak_tflops_bf16": 100,
            "peak_tflops_fp8": 200,
            "peak_tflops_fp4": 400,
            "fast_memory_bw_gbs": 1000,
            "from_slow_bw_gbs": 100,
            "to_slow_bw_gbs": 100,
            "matmul_eff_bf16": 0.8,
            "matmul_eff_fp8": 0.8,
            "matmul_eff_fp4": 0.8,
            "attn_fwd_eff": 0.8,
            "attn_bwd_eff": 0.8,
            "mem_eff": 0.9,
        },
        "planner": {
            "policy": "pressurefit",
            "window_size": 2,
            "fast_memory_capacity_gb": 1,
            "recompute": False,
        },
    }
    for key, value in overrides.items():
        if key == "hardware":
            payload["hardware"] = value
        elif key == "workload":
            payload["workload"] = value
        elif key == "planner":
            payload["planner"] = value
        elif key in _MODEL_KEYS:
            payload["workload"]["model"][key] = value
        elif key in _TRAINING_KEYS:
            payload["workload"]["training"][key] = value
        elif key in _PLANNER_KEYS:
            payload["planner"][key] = value
        else:
            payload[key] = value
    return payload


def _schema_payload():
    payload = _payload(policy="pressurefit", fast_memory_capacity_gb=1)
    payload["workload"] = {
        "source": "schema",
        "schema": {
            "schema_version": "dataflow/v1",
            "name": "tiny-generic",
            "description": "",
            "metadata": {},
            "objects": [
                {
                    "id": "x",
                    "size_bytes": 16,
                    "initial_location": "fast",
                    "role": "activation",
                }
            ],
            "tasks": [
                {
                    "id": "op0",
                    "label": "op0",
                    "group": "generic",
                    "inputs": ["x"],
                    "outputs": [
                        {"id": "y", "size_bytes": 8, "role": "activation"}
                    ],
                    "cost": {"kind": "fixed", "runtime_us": 7},
                }
            ],
            "final_locations": {},
        },
    }
    return payload


def test_simulate_keeps_exact_training_step_count():
    body = simulate(SimulationParams.model_validate(_payload(num_steps=4)))

    summary = body["summary"]
    assert summary["makespan_us"] > 0
    assert body["policy_diagnostics"]["valid_candidate_count"] > 0
    task_ids = {iv["task_id"] for iv in body["log"]["task_intervals"]}
    assert "f_3_0_0" in task_ids


def test_simulate_exposes_pressurefit_diagnostics():
    body = simulate(SimulationParams.model_validate(_payload()))
    diagnostics = body["policy_diagnostics"]

    assert diagnostics["candidate_count"] == 4
    assert diagnostics["selected_candidate"]
    assert any(c["selected"] for c in diagnostics["candidates"])


def test_simulate_accepts_asymmetric_transfer_bandwidths():
    body = simulate(
        SimulationParams.model_validate(
            _payload(
                hardware={
                    **_payload()["hardware"],
                    "from_slow_bw_gbs": 123,
                    "to_slow_bw_gbs": 45,
                }
            )
        )
    )

    assert body["chain"]["bandwidth_from_slow"] == 123_000
    assert body["chain"]["bandwidth_to_slow"] == 45_000


def test_simulate_accepts_fractional_fast_memory_budget():
    body = simulate(
        SimulationParams.model_validate(_payload(fast_memory_capacity_gb=0.125))
    )

    assert body["chain"]["fast_memory_capacity"] == int(round(0.125 * (1024 ** 3)))


def test_simulate_recompute_toggle_off_matches_default_and_can_change_makespan():
    # A tight cap where recompute relieves transfer pressure.
    off = simulate(SimulationParams.model_validate(_payload(
        seqlen=4096,
        num_seqs=2,
        fast_memory_capacity_gb=0.005,
        recompute=False,
    )))
    on = simulate(SimulationParams.model_validate(_payload(
        seqlen=4096,
        num_seqs=2,
        fast_memory_capacity_gb=0.005,
        recompute=True,
    )))
    omitted_payload = _payload(
        seqlen=4096,
        num_seqs=2,
        fast_memory_capacity_gb=0.005,
    )
    del omitted_payload["planner"]["recompute"]
    omitted = simulate(SimulationParams.model_validate(omitted_payload))

    # Default is off: omitting the field reproduces recompute=False exactly.
    assert omitted["summary"]["makespan_us"] == off["summary"]["makespan_us"]
    # Recompute never loses (it seeds the no-recompute plan) and here helps.
    assert on["summary"]["makespan_us"] <= off["summary"]["makespan_us"]
    # PressureFit diagnostics still surface under recompute.
    assert on["policy_diagnostics"]["candidate_count"] == 4
    recompute_blocks = [
        block for block in on["breakdown"]["compute_blocks"]
        if block["category"] == "recompute" and block["total_runtime_us"] > 0
    ]
    assert recompute_blocks
    assert recompute_blocks[0]["name"] == "Transformer Block Recompute"
    assert recompute_blocks[0]["subops"]
    assert recompute_blocks[0]["total_flops"] > 0
    assert recompute_blocks[0]["total_effective_flops"] == 0
    assert on["summary"]["total_effective_flops"] == off["summary"]["total_effective_flops"]
    assert on["summary"]["total_flops"] > off["summary"]["total_flops"]


def test_simulate_omits_policy_diagnostics_for_other_policies():
    body = simulate(
        SimulationParams.model_validate(_payload(policy="max_reduce"))
    )

    assert body["policy_diagnostics"] is None


def test_simulate_large_chain_uses_snapshot_free_response():
    body = simulate(
        SimulationParams.model_validate(_payload(policy="max_reduce", num_steps=600))
    )

    assert body["log"]["events"] == []
    assert body["log"]["memory_trace"]
    assert len(body["log"]["task_intervals"]) > 3_000
    assert body["summary"]["peak_fast_memory_gb"] > 0


def test_simulate_final_model_state_on_backing_is_opt_in():
    default_body = simulate(
        SimulationParams.model_validate(_payload(optimizer="adamw"))
    )
    finalized_body = simulate(
        SimulationParams.model_validate(
            _payload(optimizer="adamw", final_model_state_on_backing=True)
        )
    )

    assert default_body["chain"]["final_locations"] == {}
    assert finalized_body["chain"]["final_locations"] == {
        "W_0": "backing", "O_0": "backing",
        "W_1": "backing", "O_1": "backing",
    }


def test_preview_model_training_workload_returns_dataflow_schema():
    body = preview_workload(
        WorkloadPreviewParams.model_validate(
            {
                "workload": _payload()["workload"],
                "hardware": _payload()["hardware"],
            }
        )
    )

    assert body["schema"]["schema_version"] == "dataflow/v1"
    assert body["preview"]["task_count"] > 0
    assert body["preview"]["aggregate_task_runtime_us"] > 0
    assert body["schema"]["metadata"]["kind"] == "training.transformer.llama3.modular"
    assert body["chain"]["tasks"]
    assert body["compute_blocks"]
    assert body["breakdown"]["compute_blocks"]


def test_preview_accepts_uploaded_schema_and_returns_bare_chain():
    payload = _schema_payload()
    body = preview_workload(
        WorkloadPreviewParams.model_validate(
            {
                "workload": payload["workload"],
                "hardware": payload["hardware"],
            }
        )
    )

    assert body["schema"]["name"] == "tiny-generic"
    assert body["chain"]["tasks"][0]["id"] == "op0"
    assert body["chain"]["tasks"][0]["releases_after"] == []
    assert body["compute_blocks"][0]["instance_count"] == 1


def test_presets_include_only_public_model_workloads():
    body = presets()
    expected_models = {
        "llama3_8B": "llama3",
        "llama3_70B": "llama3",
        "llama3_405B": "llama3",
        "qwen3_4B": "qwen3",
        "qwen3_8B": "qwen3",
        "qwen3_32B": "qwen3",
        "qwen3_moe_30B-3B": "qwen3_moe",
        "qwen3_moe_235B-A22B": "qwen3_moe",
        "olmoe_7B-1B": "olmoe",
        "qwen3_5_9B": "qwen3_hybrid_dense",
        "qwen3_5_27B": "qwen3_hybrid_dense",
        "qwen3_5_35B-A3B": "qwen3_hybrid_moe",
        "qwen3_5_122B-A10B": "qwen3_hybrid_moe",
        "qwen3_5_397B-A17B": "qwen3_hybrid_moe",
        "deepseek_v3_671B-37B": "deepseek_v3",
        "deepseek_v3_2_671B-37B": "deepseek_v3_2",
        "glm_5_744B-40B": "deepseek_v3_2",
        "kimi_k2_1T-32B": "deepseek_v3",
        "gpt_oss_20B": "gpt_oss",
        "gpt_oss_120B": "gpt_oss",
        "nemotron3_nano_30B-A3B": "nemotron_h",
        "nemotron3_super_120B-A12B": "nemotron_h",
        "nemotron3_ultra_550B-A55B": "nemotron_h",
    }

    assert set(body["workloads"]) == set(expected_models)
    assert list(body["workloads"]) == sorted(body["workloads"], key=str.lower)
    assert body["workloads"]["llama3_8B"]["datatypes"] == {
        "weight_dtype": "bf16",
        "activation_dtype": "bf16",
        "expert_dispatch_dtype": "bf16",
        "gradient_dtype": "bf16",
        "optimizer_dtype": "bf16",
        "compute_precision": "bf16",
        "expert_weight_dtype": "bf16",
        "expert_compute_precision": "bf16",
        "indexer_activation_dtype": "fp8",
        "indexer_compute_precision": "fp8",
    }
    for name, family in expected_models.items():
        assert body["workloads"][name]["model"]["family"] == family
        assert body["workloads"][name]["source"] == "model_training"

    assert set(body["model_families"]) == {
        "llama3",
        "qwen3",
        "qwen3_moe",
        "olmoe",
        "qwen3_hybrid_dense",
        "qwen3_hybrid_moe",
        "deepseek_v3",
        "deepseek_v3_2",
        "gpt_oss",
        "nemotron_h",
    }
    qwen_fields = {
        field["key"]
        for field in body["model_families"]["qwen3_hybrid_moe"]["fields"]
    }
    deepseek_fields = {
        field["key"]
        for field in body["model_families"]["deepseek_v3"]["fields"]
    }
    deepseek_v32_fields = {
        field["key"]
        for field in body["model_families"]["deepseek_v3_2"]["fields"]
    }
    nemotron_fields = {
        field["key"]
        for field in body["model_families"]["nemotron_h"]["fields"]
    }
    gpt_oss_fields = {
        field["key"]
        for field in body["model_families"]["gpt_oss"]["fields"]
    }
    assert "linear_num_value_heads" in qwen_fields
    assert "gdn_chunk_size" in qwen_fields
    assert "router_aux_loss_coef" not in qwen_fields
    assert "mtp_num_hidden_layers" not in qwen_fields
    assert "q_lora_rank" in deepseek_fields
    assert "routed_scaling_factor" not in deepseek_fields
    assert "scoring_func" not in deepseek_fields
    assert "index_n_heads" in deepseek_v32_fields
    assert "index_head_dim" in deepseek_v32_fields
    assert "index_topk" in deepseek_v32_fields
    assert "train_indexer" in deepseek_v32_fields
    assert body["workloads"]["deepseek_v3_2_671B-37B"]["model"]["train_indexer"] is True
    assert body["model_families"]["llama3"]["capabilities"] == {
        "has_moe": False,
        "has_indexer": False,
    }
    assert body["model_families"]["deepseek_v3"]["capabilities"] == {
        "has_moe": True,
        "has_indexer": False,
    }
    assert body["model_families"]["deepseek_v3_2"]["capabilities"] == {
        "has_moe": True,
        "has_indexer": True,
    }
    assert "mamba_num_heads" in nemotron_fields
    assert "shared_expert_dim" in nemotron_fields
    assert "hybrid_override_pattern" in nemotron_fields
    assert "sliding_window" in gpt_oss_fields
    assert "layer_types" not in gpt_oss_fields
    assert "layer_types" not in qwen_fields


def test_preview_accepts_datatype_policy_and_exports_metadata():
    payload = _payload()
    payload["workload"]["datatypes"] = {
        "weight_dtype": "fp8",
        "activation_dtype": "fp8",
        "expert_dispatch_dtype": "fp8",
        "gradient_dtype": "fp8",
        "optimizer_dtype": "fp8",
        "compute_precision": "fp8",
        "expert_weight_dtype": "fp4",
        "expert_compute_precision": "fp4",
        "indexer_activation_dtype": "fp8",
        "indexer_compute_precision": "fp8",
    }

    body = preview_workload(WorkloadPreviewParams.model_validate(payload))
    policy = body["schema"]["metadata"]["dtype_policy"]

    assert policy["param"] == "fp8"
    assert policy["activation"] == "fp8"
    assert policy["expert_dispatch"] == "fp8"
    assert policy["gradient"] == "fp8"
    assert policy["optimizer_state"] == "fp8"
    assert policy["compute"] == "fp8"
    assert policy["expert_param"] == "fp4"
    assert policy["expert_compute"] == "fp4"
    assert policy["indexer_activation"] == "fp8"
    assert policy["indexer_compute"] == "fp8"


def test_presets_include_sram_accelerator_hardware():
    body = presets()
    hw = body["hardware"]["SRAM Accelerator"]

    assert hw == {
        "peak_tflops_bf16": 3200.0,
        "peak_tflops_fp8": 6400.0,
        "peak_tflops_fp4": 12800.0,
        "fast_memory_bw_gbs": 40000.0,
        "from_slow_bw_gbs": 3000.0,
        "to_slow_bw_gbs": 3000.0,
        "matmul_eff_bf16": 1.0,
        "matmul_eff_fp8": 1.0,
        "matmul_eff_fp4": 1.0,
        "attn_fwd_eff": 1.0,
        "attn_bwd_eff": 1.0,
        "mem_eff": 1.0,
    }


def test_h100_marks_fp4_matmul_as_unsupported():
    body = presets()
    hw = body["hardware"]["H100"]

    assert hw["peak_tflops_fp4"] is None
    assert hw["matmul_eff_fp4"] is None


def test_simulate_uploaded_schema_workload():
    body = simulate(SimulationParams.model_validate(_schema_payload()))

    assert body["workload_preview"]["name"] == "tiny-generic"
    assert body["chain"]["tasks"][0]["id"] == "op0"
    assert body["summary"]["makespan_us"] >= 7
    assert body["summary"]["tokens_per_second"] == 0.0
    assert body["summary"]["primary_unit"] is None
