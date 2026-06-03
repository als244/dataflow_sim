"""FastAPI bridge: UI POSTs transformer-spec + hardware + training params,
server returns the event log + sub-op breakdown.

Run with:
    uvicorn dataflow_app.server.main:app --reload --port 8000
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dataflow_sim.policy.sliding_window import apply_sliding_window_policy
from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policy.roundtrip_planner import apply_roundtrip_planner_policy
from dataflow_sim.policy.max_reduce import apply_max_reduce_policy
from dataflow_sim.policy.min_grow import apply_min_grow_policy
from dataflow_sim.policy.pressurefit import apply_pressurefit_policy
from dataflow_sim.simulator import run as sim_run
from dataflow_sim.schema import EventLog, TaskChain
from dataflow_app.workloads.presets import HARDWARE_PRESETS, load_model_presets
from dataflow_app.workloads.training import build_transformer_bare_chain
from dataflow_app.workloads.transformer import (
    HardwareEnv,
    TrainingConfig,
    TransformerSpec,
)

app = FastAPI(title="dataflow_sim")
MAX_RESPONSE_EVENTS = 300

Policy = Literal[
    "sliding_window",
    "belady_reactive",
    "roundtrip_planner",
    "max_reduce",
    "min_grow",
    "pressurefit",
]
Optimizer = Literal["none", "adamw", "muon"]


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
    grad_accum_rounds: int = Field(1, ge=1, le=128)
    num_steps: int = Field(1, ge=1, le=1_000_000)
    optimizer: Optimizer = "none"
    final_model_state_on_host: bool = False
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


def _log_makespan(log) -> int:
    return max((iv.end for iv in log.task_intervals), default=0)


def _peak_device_gb(log) -> float:
    peak_bytes = 0
    for ev in log.events:
        b = sum(m.size for m in ev.snapshot.memory if m.location == "device")
        if b > peak_bytes:
            peak_bytes = b
    return peak_bytes / (1024 ** 3)


def _interval_busy(log, *, track: str | None = None,
                   task_prefix: str | None = None) -> int:
    total = 0
    for iv in log.task_intervals:
        if track is not None and iv.track != track:
            continue
        if task_prefix is not None and not iv.task_id.startswith(task_prefix):
            continue
        total += iv.end - iv.start
    return total


def _apply_policy(params: SimulationParams, bare, cap_bytes: int | None) -> TaskChain:
    if params.policy == "sliding_window":
        return apply_sliding_window_policy(
            bare, window_size=params.window_size, device_capacity=cap_bytes,
        )
    if params.policy == "belady_reactive":
        return apply_belady_reactive_policy(bare, device_capacity=cap_bytes)
    if params.policy == "roundtrip_planner":
        return apply_roundtrip_planner_policy(bare, device_capacity=cap_bytes)
    if params.policy == "max_reduce":
        bare_capped = replace(bare, device_capacity=cap_bytes) if cap_bytes is not None else bare
        return apply_max_reduce_policy(bare_capped)
    if params.policy == "min_grow":
        bare_capped = replace(bare, device_capacity=cap_bytes) if cap_bytes is not None else bare
        return apply_min_grow_policy(bare_capped)
    if params.policy == "pressurefit":
        return apply_pressurefit_policy(bare, device_capacity=cap_bytes)
    raise ValueError(f"unknown policy: {params.policy!r}")


def _run_config(
    params: SimulationParams,
    spec: TransformerSpec,
    hw: HardwareEnv,
    cfg: TrainingConfig,
    cap_bytes: int | None,
) -> tuple[TaskChain, dict, EventLog]:
    bare, breakdown = build_transformer_bare_chain(spec, hw, cfg)
    chain = _apply_policy(params, bare, cap_bytes)
    return chain, breakdown, sim_run(chain)


def _compute_summary(
    log,
    breakdown: dict,
    n_layers: int,
    total_tokens: int,
    grad_accum_rounds: int,
    num_steps: int,
) -> dict:
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
    makespan = _log_makespan(log)
    peak_gb = _peak_device_gb(log)

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
    compute_busy = _interval_busy(log, track="compute")
    h2d_busy = _interval_busy(log, track="h2d")
    d2h_busy = _interval_busy(log, track="d2h")
    # Recompute time = (a) all `r_i` task intervals (explicit recompute tasks)
    # PLUS (b) the discounted portion of any sub-op whose `effective_flops`
    # is strictly less than its `flops` — that gap is recompute overhead.
    # For attn_bwd today: discount = 1/5 of total_us; generalizes to any
    # future sub-op with the same pattern.
    recompute_busy = _interval_busy(log, track="compute", task_prefix="r_")
    def _subop_recompute_us(subop: dict) -> int:
        if subop["kind"] != "compute" or subop["flops"] <= 0:
            return 0
        if subop["effective_flops"] >= subop["flops"]:
            return 0
        discount = (subop["flops"] - subop["effective_flops"]) / subop["flops"]
        return int(round(subop["total_us"] * discount))
    fwd_rows = breakdown.get("fwd", [])
    bwd_rows = breakdown.get("bwd", [])
    head_rows = breakdown.get("head", [])
    optimizer_rows = breakdown.get("optimizer", [])

    per_layer_subop_recompute_us = sum(_subop_recompute_us(s) for s in fwd_rows + bwd_rows)
    head_subop_recompute_us = sum(_subop_recompute_us(s) for s in head_rows)
    recompute_busy += (
        (per_layer_subop_recompute_us * n_layers + head_subop_recompute_us)
        * grad_accum_rounds
        * num_steps
    )
    per_layer_fwd_flops = sum(s["flops"] * s["count"] for s in fwd_rows)
    per_layer_bwd_flops = sum(s["flops"] * s["count"] for s in bwd_rows)
    head_flops_total = sum(s["flops"] * s["count"] for s in head_rows)
    optimizer_flops_per_step = sum(s["flops"] * s["count"] for s in optimizer_rows) * n_layers
    per_layer_fwd_eff = sum(s["effective_flops"] * s["count"] for s in fwd_rows)
    per_layer_bwd_eff = sum(s["effective_flops"] * s["count"] for s in bwd_rows)
    head_eff_total = sum(s["effective_flops"] * s["count"] for s in head_rows)
    optimizer_eff_per_step = sum(s["effective_flops"] * s["count"] for s in optimizer_rows) * n_layers

    per_round_flops = (per_layer_fwd_flops + per_layer_bwd_flops) * n_layers + head_flops_total
    per_round_eff_flops = (per_layer_fwd_eff + per_layer_bwd_eff) * n_layers + head_eff_total
    total_flops = (
        per_round_flops * grad_accum_rounds * num_steps
        + optimizer_flops_per_step * num_steps
    )
    total_eff_flops = (
        per_round_eff_flops * grad_accum_rounds * num_steps
        + optimizer_eff_per_step * num_steps
    )
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


def _compact_log_for_response(log, max_events: int = MAX_RESPONSE_EVENTS):
    """Keep exact intervals but downsample heavy UI snapshots.

    Large repeated-step chains can produce thousands of events, and every event
    snapshot contains the full memory pool. Returning all snapshots can create a
    hundred-megabyte JSON response. Summary statistics are computed from the
    full log before this compaction; the compacted log is only for interactive
    timeline/memory browsing.
    """
    if len(log.events) <= max_events:
        return log
    last = len(log.events) - 1
    keep = {
        round(i * last / (max_events - 1))
        for i in range(max_events)
    }
    keep.add(0)
    keep.add(last)
    return replace(log, events=[log.events[i] for i in sorted(keep)])


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
        cap_bytes = (
            int(round(params.device_capacity_gb * (1024 ** 3)))
            if params.device_capacity_gb is not None
            else None
        )
        total_tokens = (
            params.seqlen
            * params.num_seqs
            * params.grad_accum_rounds
            * params.num_steps
        )

        cfg = TrainingConfig(
            seqlen=params.seqlen,
            num_seqs=params.num_seqs,
            grad_accum_rounds=params.grad_accum_rounds,
            num_steps=params.num_steps,
            optimizer=params.optimizer,
            final_model_state_on_host=params.final_model_state_on_host,
        )
        chain, breakdown, log = _run_config(params, spec, hw, cfg, cap_bytes)
        summary = _compute_summary(
            log,
            breakdown,
            spec.n_layers,
            total_tokens,
            params.grad_accum_rounds,
            params.num_steps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_log = _compact_log_for_response(log)
    return {
        "log": asdict(response_log),
        "breakdown": breakdown,
        "summary": summary,
        "chain": asdict(chain),
    }
