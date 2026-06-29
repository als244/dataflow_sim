"""FastAPI bridge for workload preview and simulation.

The UI sends split `{workload, hardware}` preview requests and
`{workload, hardware, planner}` simulation requests. The server owns
DataflowProgram validation, hardware realization, memory planning, simulation,
and summary aggregation.

Run with:
    uvicorn dataflow_sim.app.server.main:app --reload --port 8000
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from dataflow_sim.policies.sliding_window import apply_sliding_window_policy
from dataflow_sim.policies.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policies.roundtrip_planner import apply_roundtrip_planner_policy
from dataflow_sim.policies.max_reduce import apply_max_reduce_policy
from dataflow_sim.policies.min_grow import apply_min_grow_policy
from dataflow_sim.policies.pressurefit import (
    PressureFitDiagnostics,
    apply_pressurefit_policy,
    plan_pressurefit_policy,
)
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.engine.simulator import run as sim_run
from dataflow_sim.core.schema import EventLog, TaskChain
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS, HardwareSpec
from dataflow_sim.workloads.common.workload import Workload
from dataflow_sim.workloads.dataflow import DataflowProgram, realize_dataflow_program
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_example_heterogeneous_transformer_program,
    build_transformer_training_program,
    build_transformer_training_workload,
)
from dataflow_sim.workloads.models.transformer import (
    TransformerSpec,
)

app = FastAPI(title="dataflow_sim")
MAX_RESPONSE_EVENTS = 300
MAX_MEMORY_TRACE_POINTS = 6_000
MAX_SNAPSHOT_TASKS = 3_000

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
    fast_memory_bw_gbs: float = Field(..., gt=0)
    from_slow_bw_gbs: float = Field(..., gt=0)
    to_slow_bw_gbs: float = Field(..., gt=0)
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


class TrainingParams(BaseModel):
    seqlen: int = Field(4096, ge=1)
    num_seqs: int = Field(4, ge=1)
    grad_accum_rounds: int = Field(1, ge=1, le=128)
    num_steps: int = Field(1, ge=1, le=1_000_000)
    optimizer: Optimizer = "none"
    final_model_state_on_backing: bool = False


class TransformerWorkloadParams(BaseModel):
    source: Literal["training_transformer"] = "training_transformer"
    preset: str = "custom"
    model: ModelParams
    training: TrainingParams = Field(default_factory=TrainingParams)


class SchemaWorkloadParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: Literal["schema"] = "schema"
    program: DataflowProgram = Field(alias="schema")


WorkloadParams = Annotated[
    TransformerWorkloadParams | SchemaWorkloadParams,
    Field(discriminator="source"),
]


class PlannerParams(BaseModel):
    policy: Policy = "pressurefit"
    window_size: int = Field(2, ge=1, le=32)
    fast_memory_capacity_gb: float | None = Field(None, gt=0)
    # When true for transformer workloads, an evidence-directed loop picks
    # which activations to recompute. Generic schema workloads currently ignore it.
    recompute: bool = False


class SimulationParams(BaseModel):
    workload: WorkloadParams
    hardware: HardwareParams
    planner: PlannerParams = Field(default_factory=PlannerParams)


class WorkloadPreviewParams(BaseModel):
    workload: WorkloadParams
    hardware: HardwareParams


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/presets")
def presets() -> dict:
    """Workload + hardware preset registries for UI dropdowns."""
    model_presets = load_model_presets()
    workloads = {
        name: {
            "source": "training_transformer",
            "preset": name,
            "model": asdict(spec),
            "training": TrainingParams().model_dump(mode="json"),
            "description": f"Transformer training workload for {name}",
        }
        for name, spec in model_presets.items()
    }
    hetero = build_example_heterogeneous_transformer_program()
    workloads["heterogeneous_dense_moe_demo"] = {
        "source": "schema",
        "preset": "heterogeneous_dense_moe_demo",
        "schema": hetero.model_dump(mode="json"),
        "description": "Example heterogeneous transformer with repeated dense layers and one MoE layer",
    }
    return {
        "workloads": workloads,
        # Kept for older local scripts/UI snapshots that still expect a model
        # registry; the webapp uses `workloads`.
        "models": {name: asdict(spec) for name, spec in model_presets.items()},
        "hardware": {name: asdict(hw) for name, hw in HARDWARE_PRESETS.items()},
    }


def _hardware_from_params(params: HardwareParams) -> HardwareSpec:
    return HardwareSpec(
        peak_tflops=params.peak_tflops,
        fast_memory_bw_gbs=params.fast_memory_bw_gbs,
        from_slow_bw_gbs=params.from_slow_bw_gbs,
        to_slow_bw_gbs=params.to_slow_bw_gbs,
        matmul_eff=params.matmul_eff,
        attn_fwd_eff=params.attn_fwd_eff,
        attn_bwd_eff=params.attn_bwd_eff,
        mem_eff=params.mem_eff,
    )


def _transformer_spec_from_params(params: ModelParams) -> TransformerSpec:
    return TransformerSpec(
        vocab_size=params.vocab_size,
        n_layers=params.n_layers,
        d_model=params.d_model,
        head_dim=params.head_dim,
        n_heads=params.n_heads,
        n_kv_heads=params.n_kv_heads,
        expert_dim=params.expert_dim,
        num_shared_experts=params.num_shared_experts,
        num_routed_experts=params.num_routed_experts,
        top_k=params.top_k,
        qk_norm=params.qk_norm,
    )


def _training_config_from_params(params: TrainingParams) -> TrainingConfig:
    return TrainingConfig(
        seqlen=params.seqlen,
        num_seqs=params.num_seqs,
        grad_accum_rounds=params.grad_accum_rounds,
        num_steps=params.num_steps,
        optimizer=params.optimizer,
        final_model_state_on_backing=params.final_model_state_on_backing,
    )


def _program_from_workload_params(params: WorkloadParams) -> DataflowProgram:
    if params.source == "schema":
        return params.program
    spec = _transformer_spec_from_params(params.model)
    cfg = _training_config_from_params(params.training)
    return build_transformer_training_program(spec, cfg)


def _workload_from_params(
    params: WorkloadParams,
    hw: HardwareSpec,
) -> tuple[Workload, tuple[TransformerSpec, TrainingConfig] | None]:
    if params.source == "schema":
        return realize_dataflow_program(params.program, hw), None
    spec = _transformer_spec_from_params(params.model)
    cfg = _training_config_from_params(params.training)
    return build_transformer_training_workload(spec, hw, cfg), (spec, cfg)


@app.post("/api/workloads/preview")
def preview_workload(params: WorkloadPreviewParams) -> dict:
    try:
        hw = _hardware_from_params(params.hardware)
        workload, _ = _workload_from_params(params.workload, hw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "schema": workload.metadata["program"],
        "preview": workload.metadata["preview"],
        "chain": asdict(workload.chain),
        "breakdown": workload.metadata["breakdown"],
        "compute_blocks": workload.metadata.get("compute_blocks", []),
        "task_summaries": workload.metadata.get("task_summaries", []),
    }


def _log_makespan(log) -> float:
    return max((iv.end for iv in log.task_intervals), default=0)


def _peak_fast_memory_gb(log) -> float:
    if getattr(log, "peak_fast_memory_bytes", 0):
        return log.peak_fast_memory_bytes / (1024 ** 3)
    peak_bytes = 0
    for ev in log.events:
        b = sum(m.size for m in ev.snapshot.memory if m.location == "fast")
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


def _apply_policy(planner: PlannerParams, bare, cap_bytes: int | None) -> TaskChain:
    if planner.policy == "sliding_window":
        return apply_sliding_window_policy(
            bare, window_size=planner.window_size, fast_memory_capacity=cap_bytes,
        )
    if planner.policy == "belady_reactive":
        return apply_belady_reactive_policy(bare, fast_memory_capacity=cap_bytes)
    if planner.policy == "roundtrip_planner":
        return apply_roundtrip_planner_policy(bare, fast_memory_capacity=cap_bytes)
    if planner.policy == "max_reduce":
        bare_capped = replace(bare, fast_memory_capacity=cap_bytes) if cap_bytes is not None else bare
        return apply_max_reduce_policy(bare_capped)
    if planner.policy == "min_grow":
        bare_capped = replace(bare, fast_memory_capacity=cap_bytes) if cap_bytes is not None else bare
        return apply_min_grow_policy(bare_capped)
    if planner.policy == "pressurefit":
        return apply_pressurefit_policy(bare, fast_memory_capacity=cap_bytes)
    raise ValueError(f"unknown policy: {planner.policy!r}")


# Interactive bound on the recompute-selection refinement loop. The loop's
# baseline + seed evaluations are unbudgeted, so a very large chain can still
# exceed this; it caps the exploratory portion for UI responsiveness.
RECOMPUTE_MAX_WALL_S = 30.0


def _run_config(
    params: SimulationParams,
    hw: HardwareSpec,
    cap_bytes: int | None,
) -> tuple[TaskChain, Workload, EventLog, PressureFitDiagnostics | None]:
    workload, transformer_context = _workload_from_params(params.workload, hw)
    bare = workload.chain
    diagnostics = None
    if params.planner.recompute and transformer_context is not None:
        spec, cfg = transformer_context
        chain, workload, diagnostics = _plan_with_recompute(
            params.planner, spec, hw, cfg, cap_bytes,
        )
    elif params.planner.policy == "pressurefit":
        chain, diagnostics = plan_pressurefit_policy(
            bare, fast_memory_capacity=cap_bytes,
        )
    else:
        chain = _apply_policy(params.planner, bare, cap_bytes)
    snapshots = len(chain.tasks) <= MAX_SNAPSHOT_TASKS
    return (
        chain,
        workload,
        sim_run(chain, snapshots=snapshots, memory_trace=not snapshots),
        diagnostics,
    )


def _plan_with_recompute(
    planner: PlannerParams,
    spec: TransformerSpec,
    hw: HardwareSpec,
    cfg: TrainingConfig,
    cap_bytes: int | None,
) -> tuple[TaskChain, Workload, PressureFitDiagnostics | None]:
    """Select per-layer recompute levels (evidence loop), then return the
    annotated chain for the chosen levels under the requested policy."""
    base = build_transformer_training_workload(spec, hw, cfg)
    rewrites = base.metadata["recompute_rewrites"]

    def build_variant(levels) -> TaskChain:
        chain = build_transformer_training_workload(
            spec, hw, cfg, recompute=dict(levels),
        ).chain
        return replace(chain, fast_memory_capacity=cap_bytes) if cap_bytes is not None else chain

    result = plan_with_recompute(
        build_variant,
        rewrites,
        lambda b: _apply_policy(planner, b, cap_bytes),
        max_wall_s=RECOMPUTE_MAX_WALL_S,
    )
    selected_workload = build_transformer_training_workload(
        spec, hw, cfg, recompute=dict(result.levels),
    )
    # Re-derive PressureFit candidate diagnostics on the chosen variant so the
    # UI panel still works. Deterministic — yields the same chain the loop
    # selected. Other policies don't produce diagnostics (None, as without
    # recompute).
    if planner.policy == "pressurefit":
        chain, diagnostics = plan_pressurefit_policy(
            build_variant(result.levels), fast_memory_capacity=cap_bytes,
        )
        return chain, selected_workload, diagnostics
    return result.chain, selected_workload, None


def _compute_summary(
    log,
    breakdown: dict,
    summary_meta: dict | None,
) -> dict:
    """Aggregate top-level KPIs from the simulator log + per-layer breakdown.

    All times in µs.
    `effective_tflops` uses `effective_flops` per sub-op (excludes recompute
    overhead — e.g. attn_bwd's 1x recompute portion).
    `hardware_tflops` uses raw `flops` per sub-op (includes ALL flops the
    hardware actually executes — useful + recompute). When `r_i` recompute
    tasks later carry non-zero flops, those should be added to `total_flops`
    too so this metric stays accurate.
    `peak_fast_memory_gb` is the high-water mark of compute-pool bytes across
    every event snapshot. Utilization = `<stream-busy-time> / makespan × 100`.
    """
    makespan = _log_makespan(log)
    peak_gb = _peak_fast_memory_gb(log)
    summary_meta = summary_meta or {}
    n_layers = int(summary_meta.get("n_layers", 1))
    metrics = summary_meta.get("metrics") or {}
    primary_unit = metrics.get("primary_unit")
    primary_count = float(metrics.get("primary_count", 0) or 0)
    total_tokens = int(
        primary_count
        if primary_unit == "tokens"
        else summary_meta.get("total_tokens", 0)
    )
    grad_accum_rounds = int(summary_meta.get("grad_accum_rounds", 1))
    num_steps = int(summary_meta.get("num_steps", 1))

    if makespan <= 0:
        return {
            "makespan_us": 0,
            "tokens_per_second": 0.0,
            "primary_unit": primary_unit,
            "primary_count": primary_count,
            "primary_rate_per_second": 0.0,
            "effective_tflops": 0.0,
            "hardware_tflops": 0.0,
            "peak_fast_memory_gb": peak_gb,
            "idle_pct": 0.0,
            "recompute_pct": 0.0,
            "from_slow_util_pct": 0.0,
            "to_slow_util_pct": 0.0,
            "total_flops": 0,
            "total_effective_flops": 0,
        }
    compute_busy = _interval_busy(log, track="compute")
    from_slow_busy = _interval_busy(log, track="from_slow")
    to_slow_busy = _interval_busy(log, track="to_slow")
    # Recompute time = (a) all `r_i` task intervals (explicit recompute tasks)
    # PLUS (b) the discounted portion of any sub-op whose `effective_flops`
    # is strictly less than its `flops` — that gap is recompute overhead.
    # For attn_bwd today: discount = 1/5 of total_us; generalizes to any
    # future sub-op with the same pattern.
    recompute_busy = _interval_busy(log, track="compute", task_prefix="r_")
    def _subop_recompute_us(subop: dict) -> float:
        if subop["kind"] != "compute" or subop["flops"] <= 0:
            return 0.0
        if subop["effective_flops"] >= subop["flops"]:
            return 0.0
        discount = (subop["flops"] - subop["effective_flops"]) / subop["flops"]
        return float(subop["total_us"]) * discount
    block_rows = breakdown.get("compute_blocks") or []
    if block_rows:
        total_flops = sum(int(block.get("total_flops", 0)) for block in block_rows)
        total_eff_flops = sum(
            int(block.get("total_effective_flops", 0)) for block in block_rows
        )
        recompute_busy += sum(
            _subop_recompute_us(subop) * int(block.get("instance_count", 1))
            for block in block_rows
            if block.get("category") != "recompute"
            for subop in block.get("subops", [])
        )
        primary_rate = (
            primary_count / (makespan * 1e-6) if primary_count > 0 else 0.0
        )
        return {
            "makespan_us": makespan,
            "total_flops": total_flops,
            "total_effective_flops": total_eff_flops,
            "tokens_per_second": (
                primary_rate if primary_unit == "tokens" else 0.0
            ),
            "primary_unit": primary_unit,
            "primary_count": primary_count,
            "primary_rate_per_second": primary_rate,
            "effective_tflops": total_eff_flops / (makespan * 1e-6) / 1e12,
            "hardware_tflops": total_flops / (makespan * 1e-6) / 1e12,
            "peak_fast_memory_gb": peak_gb,
            "idle_pct": (makespan - compute_busy) / makespan * 100.0,
            "recompute_pct": recompute_busy / makespan * 100.0,
            "from_slow_util_pct": from_slow_busy / makespan * 100.0,
            "to_slow_util_pct": to_slow_busy / makespan * 100.0,
        }

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
    primary_rate = (
        primary_count / (makespan * 1e-6) if primary_count > 0 else 0.0
    )
    return {
        "makespan_us": makespan,
        "total_flops": total_flops,
        "total_effective_flops": total_eff_flops,
        "tokens_per_second": (
            primary_rate
            if primary_unit == "tokens"
            else (total_tokens / (makespan * 1e-6) if total_tokens > 0 else 0.0)
        ),
        "primary_unit": primary_unit,
        "primary_count": primary_count,
        "primary_rate_per_second": primary_rate,
        "effective_tflops": total_eff_flops / (makespan * 1e-6) / 1e12,
        "hardware_tflops": total_flops / (makespan * 1e-6) / 1e12,
        "peak_fast_memory_gb": peak_gb,
        "idle_pct": (makespan - compute_busy) / makespan * 100.0,
        "recompute_pct": recompute_busy / makespan * 100.0,
        "from_slow_util_pct": from_slow_busy / makespan * 100.0,
        "to_slow_util_pct": to_slow_busy / makespan * 100.0,
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


def _compact_memory_trace_for_response(
    log,
    max_points: int = MAX_MEMORY_TRACE_POINTS,
):
    """Downsample compact memory traces while preserving endpoints and peaks."""
    points = getattr(log, "memory_trace", [])
    if len(points) <= max_points:
        return log

    bucket_count = max(1, max_points // 3)
    last = len(points) - 1
    keep: set[int] = {0, last}
    for bucket_idx in range(bucket_count):
        lo = round(bucket_idx * last / bucket_count)
        hi = round((bucket_idx + 1) * last / bucket_count)
        if hi < lo:
            lo, hi = hi, lo
        hi = min(last, max(lo, hi))
        keep.add(lo)
        keep.add(hi)
        peak_idx = max(
            range(lo, hi + 1),
            key=lambda i: sum(points[i].fast_bytes_by_band.values()),
        )
        keep.add(peak_idx)

    return replace(log, memory_trace=[points[i] for i in sorted(keep)])


@app.post("/api/simulate")
def simulate(params: SimulationParams) -> dict:
    try:
        hw = _hardware_from_params(params.hardware)
        cap_bytes = (
            int(round(params.planner.fast_memory_capacity_gb * (1024 ** 3)))
            if params.planner.fast_memory_capacity_gb is not None
            else None
        )
        chain, workload, log, policy_diagnostics = _run_config(
            params, hw, cap_bytes,
        )
        breakdown = workload.metadata["breakdown"]
        summary_meta = workload.metadata.get("summary", {})
        summary = _compute_summary(
            log,
            breakdown,
            summary_meta,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_log = _compact_memory_trace_for_response(_compact_log_for_response(log))
    return {
        "log": asdict(response_log),
        "breakdown": breakdown,
        "summary": summary,
        "chain": asdict(chain),
        "workload_preview": workload.metadata["preview"],
        "compute_blocks": workload.metadata.get("compute_blocks", []),
        "policy_diagnostics": (
            asdict(policy_diagnostics)
            if policy_diagnostics is not None
            else None
        ),
    }
