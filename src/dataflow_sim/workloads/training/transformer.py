"""Forward + backward training workloads.

`build_layerwise_training_chain` returns a bare chain: compute tasks with
inputs/outputs/runtime, all model state on host, and no memory-management
triggers. A policy annotates that bare chain before simulation.

Task graph (per training step k, accumulation round j, layer index i):

    f_k_j_0    = (deps=[input_k_j, W_0],       out=[A_k_j_0, y_k_j_0])
    f_k_j_i>0  = (deps=[y_k_j_{i-1}, W_i],     out=[A_k_j_i, y_k_j_i])
    head_k_j   = (deps=[y_k_j_{L-1}, W_head],  out=[dy_head_k_j, dW_head_k])
    r_k_j_i    = (deps=[A_k_j_i, W_i],         out=[], runtime=0)
    b_k_j_i    = (deps=[upstream, A_k_j_i, W_i], out=[dy_k_j_i, dW_k_i])

where upstream = dy_head for i=L-1 else dy_{i+1}.

When a layer instance (k, j, i) is marked for recomputation, the saved
activation is not produced by the forward pass; the recompute slot
re-produces it from the layer's input right before backward:

    f_k_j_i    = (deps=[in_act, W_i],          out=[y_k_j_i])
    r_k_j_i    = (deps=[in_act, W_i],          out=[A_k_j_i], runtime=R)

where in_act = input_k_j for i=0 else y_k_j_{i-1}. The layer input's
liveness therefore extends from the forward pass to the recompute slot;
planners handle its residency like any other object.

The first accumulation round in a step produces `dW_k_i`; later accumulation
rounds in that same step consume and mutate it. If an optimizer mode is
enabled, the chain appends one post-accumulation optimizer task per layer in
each step. A caller can also ask the policy to return the updated weights and
optimizer states to host after the whole chain:

    step_k_i = (deps=[dW_k_i, W_i, O_i], mutates=[W_i, O_i])

`W_i`, `W_head`, and `O_i` are persistent across steps; per-microbatch data and
per-step gradients carry step/accumulation indices.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Mapping

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.workloads.common.hardware import (
    HardwareSpec,
    gbs_to_bytes_per_microsecond,
)
from dataflow_sim.workloads.common.recompute import RecomputeOption, RecomputeRewrite
from dataflow_sim.workloads.common.workload import Workload
from dataflow_sim.workloads.models.transformer import TransformerSpec
from dataflow_sim.workloads.training.optimizers import OptimizerMode


@dataclass(frozen=True)
class TrainingConfig:
    seqlen: int
    num_seqs: int
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer: OptimizerMode = "none"
    final_model_state_on_host: bool = False


def build_layerwise_training_chain(
    L: int,
    *,
    input_size: int = 16,
    weight_size: int = 64,
    activation_size: int = 32,
    grad_size: int = 32,
    head_weight_size: int = 64,
    optimizer_state_size: int = 0,
    fwd_runtime: int = 10,
    head_runtime: int = 2,
    optimizer_runtime: int = 0,
    bandwidth_h2d: int = 8,
    bandwidth_d2h: int = 8,
    layer_output_size: int | None = None,
    bwd_runtime: int | None = None,
    grad_accum_rounds: int = 1,
    num_steps: int = 1,
    final_model_state_on_host: bool = False,
    recompute: frozenset[tuple[int, int, int]] = frozenset(),
    recompute_runtime: int = 0,
) -> TaskChain:
    """Build the structural skeleton of an L-layer training chain.

    Initial memory: the first input object on device; all other inputs and all
    persistent model state (`W_i`, `W_head`, and optionally `O_i`) on host.
    Per-step gradients (`dW_<step>_<layer>`) are produced by the first
    accumulation round of their step, not loaded from host. No triggers. No
    `device_capacity` set. A policy is required before the chain can run.

    `layer_output_size` (= `y_i` / `dy_i` size) defaults to `activation_size`
    for backwards compat with existing tests. The transformer-driven workload
    sizes A_i and y_i differently and overrides this.

    `bwd_runtime` defaults to `2 * fwd_runtime` for backwards compat.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if grad_accum_rounds < 1:
        raise ValueError("grad_accum_rounds must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if optimizer_state_size < 0:
        raise ValueError("optimizer_state_size must be >= 0")
    if optimizer_runtime < 0:
        raise ValueError("optimizer_runtime must be >= 0")
    if bwd_runtime is None:
        bwd_runtime = 2 * fwd_runtime
    if layer_output_size is None:
        layer_output_size = activation_size

    # --- initial memory: persistent model state on host; first input on device ---
    initial: list[Object] = []
    for k in range(num_steps):
        for j in range(grad_accum_rounds):
            initial.append(
                Object(
                    id=f"input_{k}_{j}",
                    size=input_size,
                    location="device" if k == 0 and j == 0 else "host",
                    type="activation",
                )
            )
    for i in range(L):
        initial.append(Object(id=f"W_{i}", size=weight_size, location="host", type="weight"))
        if optimizer_state_size > 0:
            initial.append(
                Object(id=f"O_{i}", size=optimizer_state_size, location="host", type="optimizer")
            )
    initial.append(Object(id="W_head", size=head_weight_size, location="host", type="weight"))

    tasks: list[Task] = []

    def input_id(k: int, j: int) -> str:
        return f"input_{k}_{j}"

    def round_id(base: str, k: int, j: int) -> str:
        return f"{base}_{k}_{j}"

    def layer_round_id(prefix: str, k: int, j: int, i: int) -> str:
        return f"{prefix}_{k}_{j}_{i}"

    def step_grad_id(k: int, i: int) -> str:
        return f"dW_{k}_{i}"

    def step_head_grad_id(k: int) -> str:
        return f"dW_head_{k}"

    for k in range(num_steps):
        for j in range(grad_accum_rounds):
            # --- forward ---
            for i in range(L):
                in_act = (
                    input_id(k, j)
                    if i == 0
                    else layer_round_id("y", k, j, i - 1)
                )
                f_outputs = [
                    OutputAlloc(
                        id=layer_round_id("y", k, j, i),
                        size=layer_output_size,
                        type="activation",
                    ),
                ]
                if (k, j, i) not in recompute:
                    f_outputs.insert(0, OutputAlloc(
                        id=layer_round_id("A", k, j, i),
                        size=activation_size,
                        type="activation",
                    ))
                tasks.append(
                    Task(
                        id=layer_round_id("f", k, j, i),
                        inputs=[in_act, f"W_{i}"],
                        outputs=f_outputs,
                        runtime=fwd_runtime,
                    )
                )

            # --- head ---
            head_grad = step_head_grad_id(k)
            head_inputs = [layer_round_id("y", k, j, L - 1), "W_head"]
            head_outputs = [
                OutputAlloc(
                    id=round_id("dy_head", k, j),
                    size=grad_size,
                    type="gradient",
                )
            ]
            head_mutates: list[str] = []
            if j == 0:
                head_outputs.append(
                    OutputAlloc(
                        id=head_grad,
                        size=head_weight_size,
                        type="gradient",
                    )
                )
            else:
                head_inputs.append(head_grad)
                head_mutates.append(head_grad)
            tasks.append(
                Task(
                    id=round_id("head", k, j),
                    inputs=head_inputs,
                    outputs=head_outputs,
                    runtime=head_runtime,
                    mutates_inputs=head_mutates,
                )
            )

            # --- backward: r_i then b_i, from L-1 down to 0 ---
            for i in range(L - 1, -1, -1):
                upstream = (
                    round_id("dy_head", k, j)
                    if i == L - 1
                    else layer_round_id("dy", k, j, i + 1)
                )
                if (k, j, i) in recompute:
                    r_in_act = (
                        input_id(k, j)
                        if i == 0
                        else layer_round_id("y", k, j, i - 1)
                    )
                    tasks.append(
                        Task(
                            id=layer_round_id("r", k, j, i),
                            inputs=[r_in_act, f"W_{i}"],
                            outputs=[
                                OutputAlloc(
                                    id=layer_round_id("A", k, j, i),
                                    size=activation_size,
                                    type="activation",
                                ),
                            ],
                            runtime=recompute_runtime,
                        )
                    )
                else:
                    tasks.append(
                        Task(
                            id=layer_round_id("r", k, j, i),
                            inputs=[layer_round_id("A", k, j, i), f"W_{i}"],
                            outputs=[],
                            runtime=0,
                        )
                    )

                grad_id = step_grad_id(k, i)
                b_inputs = [upstream, layer_round_id("A", k, j, i), f"W_{i}"]
                b_outputs = [
                    OutputAlloc(
                        id=layer_round_id("dy", k, j, i),
                        size=grad_size,
                        type="gradient",
                    )
                ]
                b_mutates: list[str] = []
                if j == 0:
                    b_outputs.append(
                        OutputAlloc(
                            id=grad_id,
                            size=weight_size,
                            type="gradient",
                        )
                    )
                else:
                    b_inputs.append(grad_id)
                    b_mutates.append(grad_id)
                tasks.append(
                    Task(
                        id=layer_round_id("b", k, j, i),
                        inputs=b_inputs,
                        outputs=b_outputs,
                        runtime=bwd_runtime,
                        mutates_inputs=b_mutates,
                    )
                )

        if optimizer_state_size > 0:
            for i in range(L):
                tasks.append(
                    Task(
                        id=f"step_{k}_{i}",
                        inputs=[step_grad_id(k, i), f"W_{i}", f"O_{i}"],
                        outputs=[],
                        runtime=optimizer_runtime,
                        mutates_inputs=[f"W_{i}", f"O_{i}"],
                    )
                )

    final_locations: dict[str, str] = {}
    if optimizer_state_size > 0 and final_model_state_on_host:
        for i in range(L):
            final_locations[f"W_{i}"] = "host"
            final_locations[f"O_{i}"] = "host"

    return TaskChain(
        initial_memory=initial,
        tasks=tasks,
        bandwidth_h2d=bandwidth_h2d,
        bandwidth_d2h=bandwidth_d2h,
        final_locations=final_locations,
        device_capacity=None,
    )


def build_transformer_training_workload(
    spec: TransformerSpec,
    hw: HardwareSpec,
    cfg: TrainingConfig,
    recompute: Mapping[str, int] | None = None,
) -> Workload:
    """Build a bare training chain from transformer model dimensions +
    hardware specs + training params.

    The returned chain is bare: it has task topology, object sizes, runtimes,
    and bandwidths, but no policy annotations. Metadata includes the
    per-sub-op breakdown used by the web UI and analysis scripts, plus the
    recompute rewrite table (`metadata["recompute_rewrites"]`) declaring the
    discrete recompute options per saved-activation object.

    `recompute` maps saved-activation object ids (e.g. "A_0_0_5") to a
    chosen option level. Level 0 (the default) saves the full activation;
    level 1 saves nothing and re-produces it in the layer's recompute slot.
    """
    # Local import to avoid hard dependency at module load.
    from dataflow_sim.workloads.models.transformer import (
        activation_bytes,
        head_breakdown,
        head_microseconds,
        head_weight_bytes,
        input_bytes,
        layer_bwd_breakdown,
        layer_bwd_microseconds,
        layer_fwd_breakdown,
        layer_fwd_microseconds,
        layer_output_bytes,
        layer_recompute_microseconds,
        layer_weight_bytes,
        optimizer_state_bytes_per_layer,
        optimizer_step_breakdown,
        optimizer_step_microseconds,
    )

    fwd_us = layer_fwd_microseconds(spec, hw, cfg)
    bwd_us = layer_bwd_microseconds(spec, hw, cfg)
    head_us = head_microseconds(spec, hw, cfg)
    opt_us = optimizer_step_microseconds(spec, hw, cfg)
    opt_state_bytes = optimizer_state_bytes_per_layer(spec, cfg.optimizer)
    act_bytes = activation_bytes(spec, cfg)
    recompute_us = layer_recompute_microseconds(spec, hw, cfg)
    bw_link = gbs_to_bytes_per_microsecond(hw.interconnect_bw_gbs)

    options = (
        RecomputeOption(level=0, saved_bytes=act_bytes, recompute_us=0, label="save-full"),
        RecomputeOption(level=1, saved_bytes=0, recompute_us=recompute_us, label="recompute-full"),
    )
    rewrites: list[RecomputeRewrite] = []
    recompute_instances: set[tuple[int, int, int]] = set()
    levels = dict(recompute or {})
    for k in range(cfg.num_steps):
        for j in range(cfg.grad_accum_rounds):
            for i in range(spec.n_layers):
                obj_id = f"A_{k}_{j}_{i}"
                rewrites.append(RecomputeRewrite(
                    object_id=obj_id,
                    f_task_id=f"f_{k}_{j}_{i}",
                    r_task_id=f"r_{k}_{j}_{i}",
                    options=options,
                ))
                level = levels.pop(obj_id, 0)
                if level not in (0, 1):
                    raise ValueError(
                        f"unsupported recompute level {level} for {obj_id!r}"
                    )
                if level == 1:
                    recompute_instances.add((k, j, i))
    if levels:
        raise ValueError(f"unknown recompute object ids: {sorted(levels)}")

    bare = build_layerwise_training_chain(
        spec.n_layers,
        input_size=input_bytes(spec, cfg),
        weight_size=layer_weight_bytes(spec),
        activation_size=act_bytes,
        layer_output_size=layer_output_bytes(spec, cfg),
        grad_size=layer_output_bytes(spec, cfg),
        head_weight_size=head_weight_bytes(spec),
        optimizer_state_size=opt_state_bytes,
        fwd_runtime=fwd_us,
        head_runtime=head_us,
        optimizer_runtime=opt_us,
        bandwidth_h2d=bw_link,
        bandwidth_d2h=bw_link,
        bwd_runtime=bwd_us,
        grad_accum_rounds=cfg.grad_accum_rounds,
        num_steps=cfg.num_steps,
        final_model_state_on_host=cfg.final_model_state_on_host,
        recompute=frozenset(recompute_instances),
        recompute_runtime=recompute_us,
    )
    breakdown = {
        "fwd": [t.asdict() for t in layer_fwd_breakdown(spec, hw, cfg)],
        "bwd": [t.asdict() for t in layer_bwd_breakdown(spec, hw, cfg)],
        "head": [t.asdict() for t in head_breakdown(spec, hw, cfg)],
        "optimizer": [t.asdict() for t in optimizer_step_breakdown(spec, hw, cfg)],
        "totals_us": {
            "layer_fwd": fwd_us,
            "layer_bwd": bwd_us,
            "head": head_us,
            "optimizer_step": opt_us,
            "layer_recompute": recompute_us,
        },
    }
    return Workload(chain=bare, metadata={
        "breakdown": breakdown,
        "recompute_rewrites": rewrites,
    })
