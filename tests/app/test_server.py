from dataflow_sim.app.server.main import SimulationParams, simulate


def _payload(**overrides):
    payload = {
        "hardware": {
            "preset": "custom",
            "peak_tflops": 100,
            "gpu_membw_gbs": 1000,
            "interconnect_bw_gbs": 100,
            "matmul_eff": 0.8,
            "attn_fwd_eff": 0.8,
            "attn_bwd_eff": 0.8,
            "mem_eff": 0.9,
        },
        "model": {
            "preset": "custom",
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
        "seqlen": 8,
        "num_seqs": 1,
        "grad_accum_rounds": 1,
        "num_steps": 1,
        "optimizer": "none",
        "final_model_state_on_host": False,
        "policy": "pressurefit",
        "window_size": 2,
        "device_capacity_gb": 1,
    }
    payload.update(overrides)
    return payload


def test_simulate_keeps_exact_step_count():
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
    assert body["summary"]["peak_memory_gb"] > 0


def test_simulate_final_model_state_on_host_is_opt_in():
    default_body = simulate(
        SimulationParams.model_validate(_payload(optimizer="adamw"))
    )
    finalized_body = simulate(
        SimulationParams.model_validate(
            _payload(optimizer="adamw", final_model_state_on_host=True)
        )
    )

    assert default_body["chain"]["final_locations"] == {}
    assert finalized_body["chain"]["final_locations"] == {
        "W_0": "host", "O_0": "host",
        "W_1": "host", "O_1": "host",
    }
