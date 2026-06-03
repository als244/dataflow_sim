"""Tests for `dataflow_app.workloads.transformer` and `dataflow_app.workloads.presets`."""
from __future__ import annotations

import math

import pytest

from dataflow_sim.simulator import run as sim_run
from dataflow_app.workloads.presets import HARDWARE_PRESETS, load_model_presets
from dataflow_app.workloads.training import build_transformer_bare_chain
from dataflow_app.workloads.transformer import (
    BYTES_PER_ELEMENT,
    HardwareEnv,
    SubOp,
    TrainingConfig,
    TransformerSpec,
    active_params_per_layer,
    activation_bytes,
    backward_subops,
    forward_subops,
    head_breakdown,
    head_microseconds,
    head_params,
    head_subops,
    head_weight_bytes,
    input_bytes,
    layer_bwd_microseconds,
    layer_fwd_breakdown,
    layer_fwd_microseconds,
    layer_output_bytes,
    layer_weight_bytes,
    params_per_layer,
    time_subop,
)


@pytest.fixture
def models() -> dict[str, TransformerSpec]:
    return load_model_presets()


@pytest.fixture
def h100() -> HardwareEnv:
    return HARDWARE_PRESETS["H100"]


@pytest.fixture
def cfg() -> TrainingConfig:
    return TrainingConfig(seqlen=4096, num_seqs=4)


# ---------- param counts ----------

def test_params_per_layer_llama3_8B(models):
    """llama3_8B: dense, no routed experts. Hand-computed."""
    s = models["llama3_8B"]
    # attn = head_dim * (2*n_heads + 2*n_kv_heads) = 128 * (64 + 16) = 10240
    # mlp = 3 * 14336 * (1 + 0) = 43008
    # total per layer = 4096 * (10240 + 43008) = 4096 * 53248 = 218103808
    assert params_per_layer(s) == 218_103_808
    # active = same (no routed)
    assert active_params_per_layer(s) == 218_103_808


def test_active_params_for_moe(models):
    """olmoe: num_routed=64, top_k=8. Active < total."""
    s = models["olmoe_7Bx1B"]
    assert active_params_per_layer(s) < params_per_layer(s)
    # ratio should be (shared + top_k) / (shared + num_routed) for the MLP term
    attn_params = s.d_model * s.head_dim * (2 * s.n_heads + 2 * s.n_kv_heads)
    full_mlp = s.d_model * 3 * s.expert_dim * (s.num_shared_experts + s.num_routed_experts)
    active_mlp = s.d_model * 3 * s.expert_dim * (s.num_shared_experts + s.top_k)
    assert params_per_layer(s) == attn_params + full_mlp
    assert active_params_per_layer(s) == attn_params + active_mlp


def test_head_params(models):
    s = models["llama3_8B"]
    assert head_params(s) == 4096 * 128256


# ---------- byte helpers ----------

def test_byte_helpers_use_bf16(models, cfg):
    s = models["nanogpt_124M"]
    assert input_bytes(s, cfg) == cfg.num_seqs * cfg.seqlen * s.d_model * BYTES_PER_ELEMENT
    assert layer_output_bytes(s, cfg) == input_bytes(s, cfg)
    assert layer_weight_bytes(s) == params_per_layer(s) * BYTES_PER_ELEMENT
    assert head_weight_bytes(s) == head_params(s) * BYTES_PER_ELEMENT


def test_activation_bytes_uses_2_d_model_factor(models, cfg):
    """A_i uses `2 * d_model` per the user's correction (not 3)."""
    s = models["llama3_8B"]
    expected_elements = cfg.num_seqs * cfg.seqlen * (
        s.head_dim * (2 * s.n_heads + 2 * s.n_kv_heads)
        + 2 * s.d_model
        + 2 * (s.num_shared_experts + s.top_k) * s.expert_dim
    )
    assert activation_bytes(s, cfg) == expected_elements * BYTES_PER_ELEMENT


# ---------- sub-op enumeration ----------

def test_qk_norm_skipped_for_llama3(models, cfg):
    """llama3 has qk_norm=False → qk_norm/qk_norm_bwd sub-ops are skipped."""
    llama = models["llama3_8B"]
    other = models["nanogpt_124M"]
    assert llama.qk_norm is False
    assert other.qk_norm is True

    fwd_names_llama = {s.name for s in forward_subops(llama, cfg)}
    fwd_names_other = {s.name for s in forward_subops(other, cfg)}
    assert "qk_norm" not in fwd_names_llama
    assert "qk_norm" in fwd_names_other

    bwd_names_llama = {s.name for s in backward_subops(llama, cfg)}
    bwd_names_other = {s.name for s in backward_subops(other, cfg)}
    assert "qk_norm_bwd" not in bwd_names_llama
    assert "qk_norm_bwd" in bwd_names_other


def test_routed_mlp_dispatch_count(models, cfg):
    """For MoE, routed up/down sub-ops have count == num_routed_experts."""
    s = models["olmoe_7Bx1B"]  # num_routed=64, top_k=8
    fwd = {sop.name: sop for sop in forward_subops(s, cfg)}
    assert "routed_mlp_up_one_expert" in fwd
    assert "routed_mlp_down_one_expert" in fwd
    assert fwd["routed_mlp_up_one_expert"].count == s.num_routed_experts
    assert fwd["routed_mlp_down_one_expert"].count == s.num_routed_experts


def test_no_routed_subops_for_dense(models, cfg):
    s = models["llama3_8B"]  # num_routed=0
    fwd_names = {sop.name for sop in forward_subops(s, cfg)}
    assert "routed_mlp_up_one_expert" not in fwd_names
    assert "routed_mlp_down_one_expert" not in fwd_names


def test_no_shared_subop_when_zero(cfg):
    """num_shared_experts == 0 → shared mlp sub-ops skipped."""
    s = TransformerSpec(
        vocab_size=32000, n_layers=4, d_model=512, head_dim=64,
        n_heads=8, n_kv_heads=8, expert_dim=2048,
        num_shared_experts=0, num_routed_experts=0, top_k=0,
        qk_norm=True,
    )
    names = {sop.name for sop in forward_subops(s, cfg)}
    assert "shared_mlp_up" not in names
    assert "shared_mlp_down" not in names


def test_mlp_split_up_down_flops_sum_to_old_formula(models, cfg):
    """The split up+down per-expert flops should sum to the pre-split
    `6 * tokens * d_model * expert_dim` total work per expert."""
    s = models["llama3_8B"]  # dense, num_shared=1
    fwd = {sop.name: sop for sop in forward_subops(s, cfg)}
    up = fwd["shared_mlp_up"]
    down = fwd["shared_mlp_down"]
    # up: 2 * tt * d * 2*edim; down: 2 * tt * edim * d. Sum: 6 * tt * d * edim.
    tt = cfg.num_seqs * cfg.seqlen
    expected = 6 * tt * s.d_model * s.expert_dim
    assert up.flops + down.flops == expected
    # Both have count == num_shared_experts.
    assert up.count == s.num_shared_experts == down.count


def test_backward_order_dgrad_before_wgrad(models, cfg):
    """All wgrad sub-ops come after the last dgrad/memory/attn_bwd sub-op."""
    s = models["llama3_8B"]
    names = [sop.name for sop in backward_subops(s, cfg)]
    wgrad_indices = [i for i, n in enumerate(names) if n.endswith("_wgrad")]
    non_wgrad_indices = [i for i, n in enumerate(names) if not n.endswith("_wgrad")]
    assert wgrad_indices, "expected at least one wgrad sub-op"
    assert max(non_wgrad_indices) < min(wgrad_indices), (
        f"wgrad ops must follow all non-wgrad ops; got order: {names}"
    )


def test_backward_mlp_dgrad_order_is_down_swiglu_up(models, cfg):
    """Within the dgrad block, MLP order is down → swiglu_bwd → up
    (reverse of fwd, matching autograd order)."""
    s = models["llama3_8B"]
    names = [sop.name for sop in backward_subops(s, cfg)]
    up_idx = names.index("shared_mlp_up_dgrad")
    swiglu_idx = names.index("swiglu_bwd")
    down_idx = names.index("shared_mlp_down_dgrad")
    assert down_idx < swiglu_idx < up_idx, (
        f"expected shared_mlp_down_dgrad < swiglu_bwd < shared_mlp_up_dgrad; "
        f"got order: {names}"
    )


def test_backward_mlp_wgrad_order_down_before_up(models, cfg):
    """Within the wgrad block, MLP down_wgrad comes before up_wgrad
    (reverse of fwd)."""
    s = models["llama3_8B"]
    names = [sop.name for sop in backward_subops(s, cfg)]
    assert names.index("shared_mlp_down_wgrad") < names.index("shared_mlp_up_wgrad")


def test_backward_mlp_dgrad_order_moe(models, cfg):
    """For MoE: any down_dgrad (routed and/or shared) comes before
    swiglu_bwd, which comes before any up_dgrad."""
    s = models["olmoe_7Bx1B"]
    names = [sop.name for sop in backward_subops(s, cfg)]
    swiglu_idx = names.index("swiglu_bwd")
    up_names = [n for n in ("shared_mlp_up_dgrad", "routed_mlp_up_one_expert_dgrad") if n in names]
    down_names = [n for n in ("shared_mlp_down_dgrad", "routed_mlp_down_one_expert_dgrad") if n in names]
    assert up_names, "expected at least one up_dgrad op"
    assert down_names, "expected at least one down_dgrad op"
    for down in down_names:
        assert names.index(down) < swiglu_idx, f"{down} must precede swiglu_bwd"
    for up in up_names:
        assert names.index(up) > swiglu_idx, f"{up} must follow swiglu_bwd"


def test_forward_moe_has_x_scatter_around_routed_mlp(models, cfg):
    """For MoE models: x_scatter precedes routed_mlp_up; x_gather follows
    routed_mlp_down."""
    s = models["olmoe_7Bx1B"]
    names = [sop.name for sop in forward_subops(s, cfg)]
    assert names.index("x_scatter") < names.index("routed_mlp_up_one_expert")
    assert names.index("x_gather") > names.index("routed_mlp_down_one_expert")


def test_backward_moe_has_dy_scatter_and_dy_gather(models, cfg):
    """For MoE models: dy_scatter precedes routed_mlp_down_dgrad;
    dy_gather follows routed_mlp_up_dgrad."""
    s = models["olmoe_7Bx1B"]
    names = [sop.name for sop in backward_subops(s, cfg)]
    assert names.index("dy_scatter") < names.index("routed_mlp_down_one_expert_dgrad")
    assert names.index("dy_gather") > names.index("routed_mlp_up_one_expert_dgrad")


def test_dense_model_has_no_scatter_or_gather(models, cfg):
    """Non-MoE models (no routed experts) should not emit scatter/gather."""
    s = models["llama3_8B"]
    fwd_names = {sop.name for sop in forward_subops(s, cfg)}
    bwd_names = {sop.name for sop in backward_subops(s, cfg)}
    for nm in ("x_scatter", "x_gather", "dy_scatter", "dy_gather"):
        assert nm not in fwd_names
        assert nm not in bwd_names


def test_scatter_gather_bytes_match_formula(models, cfg):
    """Scatter/gather bytes = total_tokens * (1 + top_k) * d_model * BPE."""
    s = models["olmoe_7Bx1B"]
    tt = cfg.num_seqs * cfg.seqlen
    expected = tt * (1 + s.top_k) * s.d_model * BYTES_PER_ELEMENT
    fwd_by_name = {sop.name: sop for sop in forward_subops(s, cfg)}
    bwd_by_name = {sop.name: sop for sop in backward_subops(s, cfg)}
    assert fwd_by_name["x_scatter"].bytes == expected
    assert fwd_by_name["x_gather"].bytes == expected
    assert bwd_by_name["dy_scatter"].bytes == expected
    assert bwd_by_name["dy_gather"].bytes == expected


# ---------- timing ----------

def test_compute_subop_timing_compute_bound(h100):
    """Big matmul, low bandwidth pressure → math-bound. effective_tflops is
    computed from un-rounded binding seconds, so it equals peak × matmul_eff
    EXACTLY (no ceil-induced quantization)."""
    s = SubOp(name="big_matmul", kind="compute", flops=10**14, bytes=1000,
              eff_name="matmul", count=1)
    t = time_subop(s, h100)
    assert t.bound_by == "compute"
    assert t.math_us is not None and t.math_us > t.mem_us
    assert t.effective_tflops is not None
    assert t.effective_tflops == pytest.approx(
        h100.peak_tflops * h100.matmul_eff, rel=1e-12,
    )


def test_compute_bound_tflops_is_size_independent(h100):
    """Several compute-bound matmuls of different sizes all report the SAME
    effective_tflops = peak × matmul_eff. Regression for the old rounding
    bug where ceil(math_us) made small ops look slower per-unit."""
    expected = h100.peak_tflops * h100.matmul_eff
    for flops in (10**11, 10**12, 10**13, 10**14, 10**15):
        s = SubOp(name=f"m_{flops}", kind="compute", flops=flops, bytes=1000,
                  eff_name="matmul", count=1)
        t = time_subop(s, h100)
        assert t.bound_by == "compute"
        assert t.effective_tflops == pytest.approx(expected, rel=1e-12), (
            f"flops={flops}: got {t.effective_tflops}, expected {expected}"
        )


def test_attn_bwd_tflops_is_4_5ths_of_peak(h100):
    """attn_bwd has effective_flops = 4× while flops = 5× (the extra 1× is
    fwd recompute). Compute-bound rate should be exactly 4/5 × peak × eff."""
    s = SubOp(name="attn_bwd", kind="compute", flops=5 * 10**13,
              bytes=1000, eff_name="attn_bwd", count=1,
              effective_flops=4 * 10**13)
    t = time_subop(s, h100)
    assert t.bound_by == "compute"
    assert t.effective_tflops == pytest.approx(
        (4 / 5) * h100.peak_tflops * h100.attn_bwd_eff, rel=1e-12,
    )


def test_per_op_compute_eff_override(h100):
    """A per-op compute_eff override beats the HW eff value."""
    s_default = SubOp(name="m_def", kind="compute", flops=10**14, bytes=1000,
                      eff_name="matmul", count=1)
    s_bad = SubOp(name="m_bad", kind="compute", flops=10**14, bytes=1000,
                  eff_name="matmul", count=1, compute_eff=0.2)
    t_def = time_subop(s_default, h100)
    t_bad = time_subop(s_bad, h100)
    assert t_def.effective_tflops == pytest.approx(
        h100.peak_tflops * h100.matmul_eff, rel=1e-12,
    )
    assert t_bad.effective_tflops == pytest.approx(
        h100.peak_tflops * 0.2, rel=1e-12,
    )
    # Override slows the op down (smaller eff → more seconds → more µs).
    assert t_bad.per_call_us > t_def.per_call_us


def test_per_op_mem_eff_override(h100):
    """A per-op mem_eff override beats hw.mem_eff."""
    s_default = SubOp(name="norm_def", kind="memory", flops=0, bytes=10**9,
                      eff_name="none", count=1)
    s_bad = SubOp(name="norm_bad", kind="memory", flops=0, bytes=10**9,
                  eff_name="none", count=1, mem_eff=0.3)
    t_def = time_subop(s_default, h100)
    t_bad = time_subop(s_bad, h100)
    # Lower mem_eff → more seconds → larger total_us.
    assert t_bad.total_us > t_def.total_us
    # Ratio matches the eff ratio (within rounding).
    ratio = t_bad.total_us / t_def.total_us
    expected_ratio = h100.mem_eff / 0.3
    assert ratio == pytest.approx(expected_ratio, rel=0.01)


def test_per_call_us_exact_unrounded(h100):
    """`per_call_us_exact` is the un-rounded float µs of the binding term;
    `per_call_us` is its ceil. Equality only when the exact value happens to
    be integer."""
    s = SubOp(name="big_matmul", kind="compute", flops=10**14, bytes=1000,
              eff_name="matmul", count=1)
    t = time_subop(s, h100)
    expected_exact = 10**14 / (h100.peak_tflops * 1e12 * h100.matmul_eff) * 1e6
    assert t.per_call_us_exact == pytest.approx(expected_exact, rel=1e-12)
    assert t.per_call_us == math.ceil(t.per_call_us_exact)


def test_compute_subop_timing_memory_bound(h100):
    """Small flops, huge bytes → memory dominates."""
    s = SubOp(name="streaming", kind="compute", flops=1000, bytes=10**10,
              eff_name="matmul", count=1)
    t = time_subop(s, h100)
    assert t.bound_by == "memory"
    assert t.mem_us > (t.math_us or 0)
    assert t.effective_tflops is not None
    # Effective TFLOPS far below peak.
    assert t.effective_tflops < h100.peak_tflops * 0.001


def test_memory_subop_timing(h100):
    """Memory-bound sub-op: total_us = ceil(bytes / (M * mem_eff) * 1e6),
    math_us is None, effective_tflops is None, bound_by == 'memory'."""
    s = SubOp(name="rms_norm", kind="memory", flops=0, bytes=10**9,
              eff_name="none", count=1)
    t = time_subop(s, h100)
    assert t.math_us is None
    assert t.effective_tflops is None
    assert t.bound_by == "memory"
    expected_seconds = 10**9 / (h100.gpu_membw_gbs * 1e9 * h100.mem_eff)
    expected_us = max(1, math.ceil(expected_seconds * 1e6))
    assert t.total_us == expected_us


def test_routed_expert_total_is_count_times_per_call(models, h100, cfg):
    s = models["olmoe_7Bx1B"]
    fwd_timings = layer_fwd_breakdown(s, h100, cfg)
    rt = next(t for t in fwd_timings if t.name == "routed_mlp_up_one_expert")
    assert rt.total_us == rt.per_call_us * rt.count
    assert rt.count == s.num_routed_experts


# ---------- end-to-end: build_transformer_bare_chain ----------

def test_build_transformer_bare_chain_llama3_h100(models, h100, cfg):
    bare, breakdown = build_transformer_bare_chain(models["llama3_8B"], h100, cfg)

    # Task count: 2*L + 1 + 2*L (= 3L+1)? Let me count: L fwd + 1 head + L r_i + L b_i = 3L+1
    expected_tasks = 3 * models["llama3_8B"].n_layers + 1
    assert len(bare.tasks) == expected_tasks

    # Runtimes are in plausible µs range
    f_runtimes = [t.runtime for t in bare.tasks if t.id.startswith("f_")]
    assert all(rt > 0 for rt in f_runtimes)
    assert all(rt < 1_000_000 for rt in f_runtimes)  # under 1 second per layer
    assert all(rt == f_runtimes[0] for rt in f_runtimes)  # identical across layers

    # Breakdown structure
    assert set(breakdown.keys()) == {"fwd", "bwd", "head", "totals_us"}
    assert set(breakdown["totals_us"].keys()) == {"layer_fwd", "layer_bwd", "head"}
    assert breakdown["totals_us"]["layer_fwd"] == layer_fwd_microseconds(
        models["llama3_8B"], h100, cfg
    )

    # All fwd sub-ops appear in breakdown
    fwd_names = {sop["name"] for sop in breakdown["fwd"]}
    assert "qkv_proj" in fwd_names
    assert "attn" in fwd_names
    assert "attn_proj" in fwd_names
    assert "shared_mlp_up" in fwd_names
    assert "shared_mlp_down" in fwd_names
    # llama3 has no qk_norm
    assert "qk_norm" not in fwd_names


def test_head_subops_order_and_kinds(models, cfg):
    """head_subops returns the 6-op block in exec order: final_norm,
    head_proj, cross_entropy, head_proj_dgrad, head_proj_wgrad,
    final_norm_bwd. The three memory ops have flops=0; the three head_proj
    matmuls are compute."""
    s = models["nanogpt_124M"]
    ops = head_subops(s, cfg)
    assert [o.name for o in ops] == [
        "final_norm", "head_proj", "cross_entropy",
        "head_proj_dgrad", "head_proj_wgrad", "final_norm_bwd",
    ]
    assert [o.kind for o in ops] == [
        "memory", "compute", "memory", "compute", "compute", "memory",
    ]
    assert ops[0].flops == 0 and ops[2].flops == 0 and ops[5].flops == 0


def test_head_memory_subop_bytes_match_formula(models, cfg):
    """Memory-bound head ops:
      final_norm     bytes = 2 * total_tokens * d_model    * BYTES_PER_ELEMENT
      cross_entropy  bytes = 2 * total_tokens * vocab_size * BYTES_PER_ELEMENT
      final_norm_bwd bytes = 7 * total_tokens * d_model    * BYTES_PER_ELEMENT
    cross_entropy works on logits (size = vocab_size), not the residual stream.
    """
    s = models["nanogpt_124M"]
    tt = cfg.num_seqs * cfg.seqlen
    d = s.d_model
    v = s.vocab_size
    by_name = {o.name: o for o in head_subops(s, cfg)}
    assert by_name["final_norm"].bytes == 2 * tt * d * BYTES_PER_ELEMENT
    assert by_name["cross_entropy"].bytes == 2 * tt * v * BYTES_PER_ELEMENT
    assert by_name["final_norm_bwd"].bytes == 7 * tt * d * BYTES_PER_ELEMENT


def test_head_breakdown_returns_six_timings(models, h100, cfg):
    """head_breakdown is a list of SubOpTimings, one per head sub-op
    (6 ops after head_proj was split into fwd/dgrad/wgrad)."""
    s = models["nanogpt_124M"]
    timings = head_breakdown(s, h100, cfg)
    assert len(timings) == 6
    assert [t.name for t in timings] == [
        "final_norm", "head_proj", "cross_entropy",
        "head_proj_dgrad", "head_proj_wgrad", "final_norm_bwd",
    ]


def test_head_microseconds_sums_over_all_subops(models, h100, cfg):
    """head_microseconds is the sum of per-sub-op total_us (capped at 1)."""
    s = models["nanogpt_124M"]
    timings = head_breakdown(s, h100, cfg)
    expected = max(1, sum(t.total_us for t in timings))
    assert head_microseconds(s, h100, cfg) == expected


def test_breakdown_head_is_six_entry_list(models, h100, cfg):
    """The breakdown payload exposes head as a 6-entry list (used by the UI
    panel's head section). Includes memory-bound entries with effective_flops=0."""
    _, breakdown = build_transformer_bare_chain(models["nanogpt_124M"], h100, cfg)
    head_rows = breakdown["head"]
    assert len(head_rows) == 6
    assert [r["name"] for r in head_rows] == [
        "final_norm", "head_proj", "cross_entropy",
        "head_proj_dgrad", "head_proj_wgrad", "final_norm_bwd",
    ]
    by_name = {r["name"]: r for r in head_rows}
    for compute_name in ("head_proj", "head_proj_dgrad", "head_proj_wgrad"):
        assert by_name[compute_name]["effective_flops"] > 0
    for mem_name in ("final_norm", "cross_entropy", "final_norm_bwd"):
        assert by_name[mem_name]["effective_flops"] == 0


def test_transformer_chain_is_runnable_with_policy(models, h100, cfg):
    """Smoke test: build chain + apply auto-policy at unlimited cap + run.
    (Sliding-window at realistic transfer/compute ratios can hit timing
    races on activation round-trips; auto-policy is more robust.)"""
    from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
    spec = models["nanogpt_124M"]
    bare, _ = build_transformer_bare_chain(spec, h100, cfg)
    chain = apply_belady_reactive_policy(bare, device_capacity=None)
    log = sim_run(chain)
    # Every bare task produces a compute interval; the auto policy adds
    # gradient-writeback d2h intervals on top (one per host-initial dW_*).
    compute_intervals = [iv for iv in log.task_intervals if iv.track == "compute"]
    assert len(compute_intervals) == len(bare.tasks)
    assert max(iv.end for iv in log.task_intervals) > 0


# ---------- preset registry ----------

def test_all_presets_load():
    m = load_model_presets()
    assert set(m.keys()) >= {"nanogpt_124M", "llama3_8B", "olmoe_7Bx1B"}
    for spec in m.values():
        assert isinstance(spec, TransformerSpec)
        assert spec.n_layers >= 1


def test_hardware_preset_defaults():
    h = HARDWARE_PRESETS["H100"]
    assert h.peak_tflops == 989
    assert h.gpu_membw_gbs == 3000
    assert h.mem_eff == 0.9
    r = HARDWARE_PRESETS["RTX_5090"]
    assert r.peak_tflops == 210
    assert r.mem_eff == 0.9
