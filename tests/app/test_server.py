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
    "top_k", "qk_norm",
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
            "peak_tflops": 100,
            "fast_memory_bw_gbs": 1000,
            "from_slow_bw_gbs": 100,
            "to_slow_bw_gbs": 100,
            "matmul_eff": 0.8,
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
        "olmoe_7B-1B": "olmoe",
    }

    assert set(body["models"]) == set(expected_models)
    assert set(body["workloads"]) == set(expected_models)
    for name, family in expected_models.items():
        assert body["models"][name]["family"] == family
        assert body["workloads"][name]["source"] == "model_training"


def test_simulate_uploaded_schema_workload():
    body = simulate(SimulationParams.model_validate(_schema_payload()))

    assert body["workload_preview"]["name"] == "tiny-generic"
    assert body["chain"]["tasks"][0]["id"] == "op0"
    assert body["summary"]["makespan_us"] >= 7
    assert body["summary"]["tokens_per_second"] == 0.0
    assert body["summary"]["primary_unit"] is None
