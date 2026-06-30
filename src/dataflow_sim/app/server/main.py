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
from dataflow_sim.workloads.dataflow_builder import DTypePolicy, TrainingConfig
from dataflow_sim.workloads.models.registry import (
    MODEL_FAMILIES,
    iter_model_presets,
    model_families_payload,
)
from dataflow_sim.workloads.summary import compute_workload_summary
from dataflow_sim.workloads.training_builder import TrainingWorkloadBuilder

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
DTypeName = Literal["bf16", "fp8", "fp4"]
ModelFamily = Literal[
    "llama3",
    "qwen3",
    "qwen3_moe",
    "olmoe",
    "qwen3_hybrid_dense",
    "qwen3_hybrid_moe",
    "deepseek_v3",
    "gpt_oss",
    "nemotron_h",
]


class HardwareParams(BaseModel):
    preset: str = "custom"
    peak_tflops_bf16: float = Field(..., gt=0)
    peak_tflops_fp8: float = Field(..., gt=0)
    peak_tflops_fp4: float | None = Field(None, gt=0)
    fast_memory_bw_gbs: float = Field(..., gt=0)
    from_slow_bw_gbs: float = Field(..., gt=0)
    to_slow_bw_gbs: float = Field(..., gt=0)
    matmul_eff_bf16: float = Field(..., gt=0, le=1)
    matmul_eff_fp8: float = Field(..., gt=0, le=1)
    matmul_eff_fp4: float | None = Field(None, gt=0, le=1)
    attn_fwd_eff: float = Field(..., gt=0, le=1)
    attn_bwd_eff: float = Field(..., gt=0, le=1)
    mem_eff: float = Field(0.9, gt=0, le=1)


class ModelParams(BaseModel):
    preset: str = "custom"
    family: ModelFamily
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
    intermediate_size: int = Field(0, ge=0)
    full_attention_interval: int = Field(4, ge=1)
    linear_num_key_heads: int = Field(1, ge=1)
    linear_key_head_dim: int = Field(1, ge=1)
    linear_num_value_heads: int = Field(1, ge=1)
    linear_value_head_dim: int = Field(1, ge=1)
    linear_conv_kernel_dim: int = Field(1, ge=1)
    gdn_chunk_size: int = Field(64, ge=1)
    router_aux_loss_coef: float = Field(0.0, ge=0)
    mtp_num_hidden_layers: int = Field(0, ge=0)
    first_k_dense_replace: int = Field(0, ge=0)
    q_lora_rank: int = Field(0, ge=0)
    kv_lora_rank: int = Field(0, ge=0)
    qk_nope_head_dim: int = Field(0, ge=0)
    qk_rope_head_dim: int = Field(0, ge=0)
    v_head_dim: int = Field(0, ge=0)
    routed_scaling_factor: float = Field(1.0, ge=0)
    scoring_func: str = "sigmoid"
    shared_expert_dim: int = Field(0, ge=0)
    mamba_num_heads: int = Field(1, ge=1)
    mamba_head_dim: int = Field(1, ge=1)
    ssm_state_size: int = Field(1, ge=1)
    conv_kernel: int = Field(1, ge=1)
    mamba_chunk_size: int = Field(128, ge=1)
    n_groups: int = Field(1, ge=1)
    hybrid_override_pattern: str = ""
    sliding_window: int = Field(128, ge=1)


class TrainingParams(BaseModel):
    seqlen: int = Field(4096, ge=1)
    num_seqs: int = Field(4, ge=1)
    grad_accum_rounds: int = Field(1, ge=1, le=128)
    num_steps: int = Field(1, ge=1, le=1_000_000)
    optimizer: Optimizer = "none"
    final_model_state_on_backing: bool = False


class DatatypeParams(BaseModel):
    weight_dtype: DTypeName = "bf16"
    activation_dtype: DTypeName = "bf16"
    expert_dispatch_dtype: DTypeName = "bf16"
    gradient_dtype: DTypeName = "bf16"
    optimizer_dtype: DTypeName = "bf16"
    compute_precision: DTypeName = "bf16"
    expert_weight_dtype: DTypeName = "bf16"
    expert_compute_precision: DTypeName = "bf16"


class ModelTrainingWorkloadParams(BaseModel):
    source: Literal["model_training"] = "model_training"
    preset: str = "custom"
    model: ModelParams
    training: TrainingParams = Field(default_factory=TrainingParams)
    datatypes: DatatypeParams = Field(default_factory=DatatypeParams)


class SchemaWorkloadParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: Literal["schema"] = "schema"
    program: DataflowProgram = Field(alias="schema")


WorkloadParams = Annotated[
    ModelTrainingWorkloadParams | SchemaWorkloadParams,
    Field(discriminator="source"),
]


class PlannerParams(BaseModel):
    policy: Policy = "pressurefit"
    window_size: int = Field(2, ge=1, le=32)
    fast_memory_capacity_gb: float | None = Field(None, gt=0)
    # When true for modular training workloads, an evidence-directed loop picks
    # which saved activation instances to recompute. Generic schema workloads
    # currently ignore it because they do not publish recompute rewrite tables.
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
    workloads = {}
    for name, family, config in iter_model_presets():
        model = _model_params_from_config(config, family)
        workloads[name] = {
            "source": "model_training",
            "preset": name,
            "model": model,
            "training": TrainingParams().model_dump(mode="json"),
            "datatypes": DatatypeParams().model_dump(mode="json"),
            "description": f"{family} training workload for {name}",
        }

    return {
        "workloads": workloads,
        "model_families": model_families_payload(),
        "hardware": {name: asdict(hw) for name, hw in HARDWARE_PRESETS.items()},
    }


def _model_params_from_config(config, family: str) -> dict:
    """Return the preset payload consumed by the webapp model editor.

    The workload preset already carries its top-level preset name, so this
    nested model payload intentionally omits `preset`. The UI adds the selected
    preset back when it sends a simulation request.
    """

    data = asdict(config)
    data.pop("preset_name", None)
    data.pop("layer_types", None)
    return {"family": family, **data}


def _config_kwargs_from_params(params: ModelParams, family: str) -> dict:
    entry = MODEL_FAMILIES[family]
    kwargs = {
        field: getattr(params, field)
        for field in entry.config_field_names
        if hasattr(params, field)
    }
    kwargs["preset_name"] = params.preset
    return kwargs


def _training_model_from_params(params: ModelParams) -> TrainingWorkloadBuilder:
    family = params.family
    entry = MODEL_FAMILIES[family]
    return entry.builder_cls(entry.config_cls(**_config_kwargs_from_params(params, family)))


def _hardware_from_params(params: HardwareParams) -> HardwareSpec:
    return HardwareSpec(
        peak_tflops_bf16=params.peak_tflops_bf16,
        peak_tflops_fp8=params.peak_tflops_fp8,
        peak_tflops_fp4=params.peak_tflops_fp4,
        fast_memory_bw_gbs=params.fast_memory_bw_gbs,
        from_slow_bw_gbs=params.from_slow_bw_gbs,
        to_slow_bw_gbs=params.to_slow_bw_gbs,
        matmul_eff_bf16=params.matmul_eff_bf16,
        matmul_eff_fp8=params.matmul_eff_fp8,
        matmul_eff_fp4=params.matmul_eff_fp4,
        attn_fwd_eff=params.attn_fwd_eff,
        attn_bwd_eff=params.attn_bwd_eff,
        mem_eff=params.mem_eff,
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


def _dtype_policy_from_params(params: DatatypeParams) -> DTypePolicy:
    return DTypePolicy(
        param=params.weight_dtype,
        activation=params.activation_dtype,
        expert_dispatch=params.expert_dispatch_dtype,
        gradient=params.gradient_dtype,
        optimizer_state=params.optimizer_dtype,
        compute=params.compute_precision,
        expert_param=params.expert_weight_dtype,
        expert_compute=params.expert_compute_precision,
    )


def _workload_from_params(
    params: WorkloadParams,
    hw: HardwareSpec,
) -> tuple[Workload, tuple[TrainingWorkloadBuilder, TrainingConfig, DTypePolicy] | None]:
    if params.source == "schema":
        return realize_dataflow_program(params.program, hw), None
    cfg = _training_config_from_params(params.training)
    dtype_policy = _dtype_policy_from_params(params.datatypes)
    model = _training_model_from_params(params.model)
    workload = model.build_training_workload(
        cfg,
        hw,
        input_shape=(cfg.tokens, params.model.d_model),
        dtype_policy=dtype_policy,
    )
    return workload, (model, cfg, dtype_policy)


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
    workload, training_context = _workload_from_params(params.workload, hw)
    bare = workload.chain
    diagnostics = None
    if params.planner.recompute and training_context is not None:
        model, cfg, dtype_policy = training_context
        chain, workload, diagnostics = _plan_with_recompute(
            params.planner, model, hw, cfg, dtype_policy, cap_bytes,
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
    model: TrainingWorkloadBuilder,
    hw: HardwareSpec,
    cfg: TrainingConfig,
    dtype_policy: DTypePolicy,
    cap_bytes: int | None,
) -> tuple[TaskChain, Workload, PressureFitDiagnostics | None]:
    """Select recompute levels, then return the chosen annotated chain.

    Recompute options come from the workload's compute-block rewrite table.
    The selector is still per saved activation instance because memory pressure
    and transfer blame are instance-specific.
    """

    input_shape = (cfg.tokens, model.input_dim)
    base = model.build_training_workload(
        cfg,
        hw,
        input_shape=input_shape,
        dtype_policy=dtype_policy,
    )
    rewrites = base.metadata["recompute_rewrites"]

    def build_variant(levels) -> TaskChain:
        return model.build_training_workload(
            cfg,
            hw,
            input_shape=input_shape,
            dtype_policy=dtype_policy,
            recompute=dict(levels),
        ).chain

    result = plan_with_recompute(
        build_variant,
        rewrites,
        lambda b: _apply_policy(planner, b, cap_bytes),
        max_wall_s=RECOMPUTE_MAX_WALL_S,
    )
    selected_workload = model.build_training_workload(
        cfg,
        hw,
        input_shape=input_shape,
        dtype_policy=dtype_policy,
        recompute=dict(result.levels),
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
        summary = compute_workload_summary(workload, log)
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
