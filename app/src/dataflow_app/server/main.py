"""FastAPI bridge: UI POSTs transformer-spec + hardware + training params,
server returns the event log + sub-op breakdown.

Run with:
    uvicorn server.main:app --reload --port 8000
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dataflow_sim.policy.sliding_window import apply_sliding_window_policy
from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policy.roundtrip_planner import apply_roundtrip_planner_policy
from dataflow_sim.policy.race_best import apply_race_best_policy
from dataflow_sim.policy.max_reduce import apply_max_reduce_policy
from dataflow_sim.policy.min_grow import apply_min_grow_policy
from dataflow_sim.policy.pressurefit import apply_pressurefit_policy
from dataflow_sim.simulator import run as sim_run
from dataflow_app.workloads.presets import HARDWARE_PRESETS, load_model_presets
from dataflow_app.workloads.training import build_transformer_bare_chain
from dataflow_app.workloads.transformer import (
    HardwareEnv,
    TrainingConfig,
    TransformerSpec,
)

app = FastAPI(title="dataflow_sim")

Policy = Literal[
    "sliding_window",
    "belady_reactive",
    "roundtrip_planner",
    "race_best",
    "max_reduce",
    "min_grow",
    "pressurefit",
]


class HardwareParams(BaseModel):
    preset: str = "custom"
    peak_tflops: float = Field(..., gt=0)
    gpu_membw_gbs: float = Field(..., gt=0)
    interconnect_bw_gbs: float = Field(..., gt=0)
    matmul_eff: float = Field(..., gt=0, le=1)
    attn_fwd_eff: float = Field(..., gt=0, le=1)
    attn_bwd_eff: float = Field(..., gt=0, le=1)
    mem_eff: float = Field(0.9, gt=0, le=1)


class ModelParams(BaseModel):
    preset: str = "custom"
    vocab_size: int = Field(..., ge=1)
    n_layers: int = Field(..., ge=1, le=256)
    d_model: int = Field(..., ge=1)
    head_dim: int = Field(..., ge=1)
    n_heads: int = Field(..., ge=1)
    n_kv_heads: int = Field(..., ge=1)
    expert_dim: int = Field(0, ge=0)
    num_shared_experts: int = Field(0, ge=0)
    num_routed_experts: int = Field(0, ge=0)
    top_k: int = Field(0, ge=0)
    qk_norm: bool = Field(True)


class SimulationParams(BaseModel):
    hardware: HardwareParams
    model: ModelParams
    seqlen: int = Field(4096, ge=1)
    num_seqs: int = Field(4, ge=1)
    policy: Policy = "pressurefit"
    window_size: int = Field(2, ge=1, le=32)
    device_capacity_gb: float | None = Field(None, gt=0)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/presets")
def presets() -> dict:
    """Models + hardware preset registries for UI dropdowns."""
    return {
        "models": {name: asdict(spec) for name, spec in load_model_presets().items()},
        "hardware": {name: asdict(hw) for name, hw in HARDWARE_PRESETS.items()},
    }


def _compute_summary(log, breakdown: dict, n_layers: int, total_tokens: int) -> dict:
    """Aggregate top-level KPIs from the simulator log + per-layer breakdown.

    All times in µs.
    `effective_tflops` uses `effective_flops` per sub-op (excludes recompute
    overhead — e.g. attn_bwd's 1x recompute portion).
    `hardware_tflops` uses raw `flops` per sub-op (includes ALL flops the
    hardware actually executes — useful + recompute). When `r_i` recompute
    tasks later carry non-zero flops, those should be added to `total_flops`
    too so this metric stays accurate.
    `peak_memory_gb` is the high-water mark of device-pool bytes across
    every event snapshot. Utilization = `<stream-busy-time> / makespan × 100`.
    """
    makespan = max((iv.end for iv in log.task_intervals), default=0)
    # Peak device memory across all event snapshots.
    peak_bytes = 0
    for ev in log.events:
        b = sum(m.size for m in ev.snapshot.memory if m.location == "device")
        if b > peak_bytes:
            peak_bytes = b
    peak_gb = peak_bytes / (1024 ** 3)

    if makespan <= 0:
        return {
            "makespan_us": 0,
            "tokens_per_second": 0.0,
            "effective_tflops": 0.0,
            "hardware_tflops": 0.0,
            "peak_memory_gb": peak_gb,
            "idle_pct": 0.0,
            "recompute_pct": 0.0,
            "ingress_util_pct": 0.0,
            "egress_util_pct": 0.0,
            "total_flops": 0,
            "total_effective_flops": 0,
        }
    compute_busy = sum(iv.end - iv.start for iv in log.task_intervals if iv.track == "compute")
    h2d_busy = sum(iv.end - iv.start for iv in log.task_intervals if iv.track == "h2d")
    d2h_busy = sum(iv.end - iv.start for iv in log.task_intervals if iv.track == "d2h")
    # Recompute time = (a) all `r_i` task intervals (explicit recompute tasks)
    # PLUS (b) the discounted portion of any sub-op whose `effective_flops`
    # is strictly less than its `flops` — that gap is recompute overhead.
    # For attn_bwd today: discount = 1/5 of total_us; generalizes to any
    # future sub-op with the same pattern.
    recompute_busy = sum(
        iv.end - iv.start for iv in log.task_intervals
        if iv.track == "compute" and iv.task_id.startswith("r_")
    )
    def _subop_recompute_us(subop: dict) -> int:
        if subop["kind"] != "compute" or subop["flops"] <= 0:
            return 0
        if subop["effective_flops"] >= subop["flops"]:
            return 0
        discount = (subop["flops"] - subop["effective_flops"]) / subop["flops"]
        return int(round(subop["total_us"] * discount))
    per_layer_subop_recompute_us = sum(
        _subop_recompute_us(s) for s in breakdown["fwd"] + breakdown["bwd"]
    )
    head_subop_recompute_us = sum(
        _subop_recompute_us(s) for s in breakdown["head"]
    )
    recompute_busy += per_layer_subop_recompute_us * n_layers + head_subop_recompute_us
    per_layer_fwd_flops = sum(s["flops"] * s["count"] for s in breakdown["fwd"])
    per_layer_bwd_flops = sum(s["flops"] * s["count"] for s in breakdown["bwd"])
    head_flops_total = sum(s["flops"] * s["count"] for s in breakdown["head"])
    per_layer_fwd_eff = sum(s["effective_flops"] * s["count"] for s in breakdown["fwd"])
    per_layer_bwd_eff = sum(s["effective_flops"] * s["count"] for s in breakdown["bwd"])
    head_eff_total = sum(s["effective_flops"] * s["count"] for s in breakdown["head"])

    total_flops = (per_layer_fwd_flops + per_layer_bwd_flops) * n_layers + head_flops_total
    total_eff_flops = (per_layer_fwd_eff + per_layer_bwd_eff) * n_layers + head_eff_total
    return {
        "makespan_us": makespan,
        "total_flops": total_flops,
        "total_effective_flops": total_eff_flops,
        "tokens_per_second": total_tokens / (makespan * 1e-6),
        "effective_tflops": total_eff_flops / (makespan * 1e-6) / 1e12,
        "hardware_tflops": total_flops / (makespan * 1e-6) / 1e12,
        "peak_memory_gb": peak_gb,
        "idle_pct": (makespan - compute_busy) / makespan * 100.0,
        "recompute_pct": recompute_busy / makespan * 100.0,
        "ingress_util_pct": h2d_busy / makespan * 100.0,
        "egress_util_pct": d2h_busy / makespan * 100.0,
    }


@app.post("/api/simulate")
def simulate(params: SimulationParams) -> dict:
    try:
        spec = TransformerSpec(
            vocab_size=params.model.vocab_size,
            n_layers=params.model.n_layers,
            d_model=params.model.d_model,
            head_dim=params.model.head_dim,
            n_heads=params.model.n_heads,
            n_kv_heads=params.model.n_kv_heads,
            expert_dim=params.model.expert_dim,
            num_shared_experts=params.model.num_shared_experts,
            num_routed_experts=params.model.num_routed_experts,
            top_k=params.model.top_k,
            qk_norm=params.model.qk_norm,
        )
        hw = HardwareEnv(
            peak_tflops=params.hardware.peak_tflops,
            gpu_membw_gbs=params.hardware.gpu_membw_gbs,
            interconnect_bw_gbs=params.hardware.interconnect_bw_gbs,
            matmul_eff=params.hardware.matmul_eff,
            attn_fwd_eff=params.hardware.attn_fwd_eff,
            attn_bwd_eff=params.hardware.attn_bwd_eff,
            mem_eff=params.hardware.mem_eff,
        )
        cfg = TrainingConfig(seqlen=params.seqlen, num_seqs=params.num_seqs)

        bare, breakdown = build_transformer_bare_chain(spec, hw, cfg)
        cap_bytes = (
            int(round(params.device_capacity_gb * (1024 ** 3)))
            if params.device_capacity_gb is not None
            else None
        )
        if params.policy == "sliding_window":
            chain = apply_sliding_window_policy(
                bare, window_size=params.window_size, device_capacity=cap_bytes,
            )
        elif params.policy == "belady_reactive":
            chain = apply_belady_reactive_policy(bare, device_capacity=cap_bytes)
        elif params.policy == "roundtrip_planner":
            chain = apply_roundtrip_planner_policy(bare, device_capacity=cap_bytes)
        elif params.policy == "race_best":
            chain = apply_race_best_policy(bare, device_capacity=cap_bytes)
        elif params.policy == "max_reduce":
            bare_capped = replace(bare, device_capacity=cap_bytes) if cap_bytes is not None else bare
            chain = apply_max_reduce_policy(bare_capped)
        elif params.policy == "min_grow":
            bare_capped = replace(bare, device_capacity=cap_bytes) if cap_bytes is not None else bare
            chain = apply_min_grow_policy(bare_capped)
        elif params.policy == "pressurefit":
            chain = apply_pressurefit_policy(bare, device_capacity=cap_bytes)
        else:  # pragma: no cover (validated by pydantic)
            raise ValueError(f"unknown policy: {params.policy!r}")

        log = sim_run(chain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    summary = _compute_summary(log, breakdown, spec.n_layers, params.seqlen * params.num_seqs)
    return {"log": asdict(log), "breakdown": breakdown, "summary": summary}
