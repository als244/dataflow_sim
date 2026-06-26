"""Forward + backward training workloads.

`build_layerwise_training_chain` returns a bare chain: compute tasks with
inputs/outputs/runtime, all model state on backing, and no memory-management
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
optimizer states to backing after the whole chain:

    step_k_i = (deps=[dW_k_i, W_i, O_i], mutates=[W_i, O_i])

`W_i`, `W_head`, and `O_i` are persistent across steps; per-microbatch data and
per-step gradients carry step/accumulation indices.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

from typing import Mapping

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain
from dataflow_sim.workloads.common.hardware import (
    HardwareSpec,
    gbs_to_bytes_per_microsecond,
)
from dataflow_sim.workloads.common.recompute import RecomputeOption, RecomputeRewrite
from dataflow_sim.workloads.common.workload import Workload
from dataflow_sim.workloads.dataflow import (
    ComputeBlock,
    DataflowCost,
    DataflowMetrics,
    DataflowObject,
    DataflowOutput,
    DataflowProgram,
    DataflowTask,
    realize_dataflow_program,
)
from dataflow_sim.workloads.models.transformer import TransformerSpec
from dataflow_sim.workloads.training.optimizers import OptimizerMode


@dataclass(frozen=True)
class TrainingConfig:
    seqlen: int
    num_seqs: int
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer: OptimizerMode = "none"
    final_model_state_on_backing: bool = False


def _fixed_cost(name: str, runtime_us: int) -> DataflowCost:
    return DataflowCost(kind="fixed", name=name, runtime_us=runtime_us)


def _subop_cost(subop) -> DataflowCost:
    eff_map = {
        "matmul": "matmul",
        "attn_fwd": "attention_fwd",
        "attn_bwd": "attention_bwd",
        "none": "memory",
    }
    return DataflowCost(
        kind="roofline",
        name=subop.name,
        flops=subop.flops,
        effective_flops=(
            subop.effective_flops
            if subop.effective_flops is not None
            else subop.flops
        ),
        memory_bytes=subop.bytes,
        efficiency=eff_map[subop.eff_name],
        count=subop.count,
        compute_eff=subop.compute_eff,
        mem_eff=subop.mem_eff,
    )


def _recompute_subop_cost(subop) -> DataflowCost:
    # Recompute consumes hardware FLOPs but does not add useful model work.
    return _subop_cost(subop).model_copy(update={"effective_flops": 0})


def _sum_cost(name: str, subops) -> DataflowCost:
    terms = [_subop_cost(subop) for subop in subops]
    if not terms:
        return _fixed_cost(name, 0)
    return DataflowCost(kind="sum", name=name, terms=terms)


def _recompute_sum_cost(name: str, subops) -> DataflowCost:
    terms = [_recompute_subop_cost(subop) for subop in subops]
    if not terms:
        return _fixed_cost(name, 0)
    return DataflowCost(kind="sum", name=name, terms=terms)


def _cost_terms(cost: DataflowCost) -> list[DataflowCost]:
    if cost.kind == "sum":
        return list(cost.terms)
    return [cost]


def _transformer_variant(spec: TransformerSpec) -> str:
    return "moe" if spec.num_routed_experts > 0 else "dense"


def _layer_signature(spec: TransformerSpec) -> str:
    body = asdict(replace(spec, n_layers=1))
    digest = hashlib.sha1(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return f"{_transformer_variant(spec)}_{digest}"


def build_layerwise_training_program(
    L: int,
    *,
    input_size: int = 16,
    weight_size: int = 64,
    activation_size: int = 32,
    grad_size: int = 32,
    head_weight_size: int = 64,
    optimizer_state_size: int = 0,
    fwd_cost: DataflowCost | None = None,
    head_cost: DataflowCost | None = None,
    optimizer_cost: DataflowCost | None = None,
    layer_output_size: int | None = None,
    bwd_cost: DataflowCost | None = None,
    grad_accum_rounds: int = 1,
    num_steps: int = 1,
    final_model_state_on_backing: bool = False,
    recompute: frozenset[tuple[int, int, int]] = frozenset(),
    recompute_cost: DataflowCost | None = None,
    name: str = "layerwise-training",
    metadata: dict | None = None,
    metrics: DataflowMetrics | None = None,
) -> DataflowProgram:
    """Build a hardware-free layerwise training program.

    This mirrors `build_layerwise_training_chain`, but costs remain symbolic
    until the program is realized against a hardware profile.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if grad_accum_rounds < 1:
        raise ValueError("grad_accum_rounds must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if optimizer_state_size < 0:
        raise ValueError("optimizer_state_size must be >= 0")
    if layer_output_size is None:
        layer_output_size = activation_size
    fwd_cost = fwd_cost or _fixed_cost("layer_fwd", 10)
    head_cost = head_cost or _fixed_cost("head", 2)
    bwd_cost = bwd_cost or _fixed_cost("layer_bwd", 20)
    optimizer_cost = optimizer_cost or _fixed_cost("optimizer_step", 0)
    recompute_cost = recompute_cost or _fixed_cost("layer_recompute", 0)
    block_keys = {
        "forward": "layer_forward",
        "head": "head",
        "recompute_slot": "recompute_slot",
        "recompute": "layer_recompute",
        "backward": "layer_backward",
        "optimizer": "optimizer_step",
    }
    compute_blocks = [
        ComputeBlock(
            key=block_keys["forward"],
            name="Layer Forward",
            category="forward",
            subops=_cost_terms(fwd_cost),
        ),
        ComputeBlock(
            key=block_keys["head"],
            name="Head",
            category="head",
            subops=_cost_terms(head_cost),
        ),
        ComputeBlock(
            key=block_keys["recompute_slot"],
            name="Layer Recompute",
            category="recompute",
            subops=[_fixed_cost("layer_recompute", 0)],
        ),
        ComputeBlock(
            key=block_keys["recompute"],
            name="Layer Recompute",
            category="recompute",
            subops=_cost_terms(recompute_cost),
        ),
        ComputeBlock(
            key=block_keys["backward"],
            name="Layer Backward",
            category="backward",
            subops=_cost_terms(bwd_cost),
        ),
    ]
    if optimizer_state_size > 0:
        compute_blocks.append(
            ComputeBlock(
                key=block_keys["optimizer"],
                name="Optimizer Step",
                category="optimizer",
                subops=_cost_terms(optimizer_cost),
            )
        )

    objects: list[DataflowObject] = []
    for k in range(num_steps):
        for j in range(grad_accum_rounds):
            objects.append(
                DataflowObject(
                    id=f"input_{k}_{j}",
                    size_bytes=input_size,
                    initial_location="fast" if k == 0 and j == 0 else "backing",
                    role="activation",
                )
            )
    for i in range(L):
        objects.append(
            DataflowObject(
                id=f"W_{i}",
                size_bytes=weight_size,
                initial_location="backing",
                role="parameter",
            )
        )
        if optimizer_state_size > 0:
            objects.append(
                DataflowObject(
                    id=f"O_{i}",
                    size_bytes=optimizer_state_size,
                    initial_location="backing",
                    role="optimizer_state",
                )
            )
    objects.append(
        DataflowObject(
            id="W_head",
            size_bytes=head_weight_size,
            initial_location="backing",
            role="parameter",
        )
    )

    program_tasks: list[DataflowTask] = []

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
            for i in range(L):
                in_act = input_id(k, j) if i == 0 else layer_round_id("y", k, j, i - 1)
                outputs = [
                    DataflowOutput(
                        id=layer_round_id("y", k, j, i),
                        size_bytes=layer_output_size,
                        role="activation",
                    ),
                ]
                if (k, j, i) not in recompute:
                    outputs.insert(
                        0,
                        DataflowOutput(
                            id=layer_round_id("A", k, j, i),
                            size_bytes=activation_size,
                            role="activation",
                        ),
                    )
                program_tasks.append(
                    DataflowTask(
                        id=layer_round_id("f", k, j, i),
                        label=f"Step {k} Round {j} Layer {i} Forward",
                        group="forward",
                        compute_block_key=block_keys["forward"],
                        inputs=[in_act, f"W_{i}"],
                        outputs=outputs,
                    )
                )

            head_grad = step_head_grad_id(k)
            head_inputs = [layer_round_id("y", k, j, L - 1), "W_head"]
            head_outputs = [
                DataflowOutput(
                    id=round_id("dy_head", k, j),
                    size_bytes=grad_size,
                    role="gradient",
                )
            ]
            head_mutates: list[str] = []
            if j == 0:
                head_outputs.append(
                    DataflowOutput(
                        id=head_grad,
                        size_bytes=head_weight_size,
                        role="gradient",
                    )
                )
            else:
                head_inputs.append(head_grad)
                head_mutates.append(head_grad)
            program_tasks.append(
                DataflowTask(
                    id=round_id("head", k, j),
                    label=f"Step {k} Round {j} Head",
                    group="head",
                    compute_block_key=block_keys["head"],
                    inputs=head_inputs,
                    outputs=head_outputs,
                    mutates=head_mutates,
                )
            )

            for i in range(L - 1, -1, -1):
                upstream = (
                    round_id("dy_head", k, j)
                    if i == L - 1
                    else layer_round_id("dy", k, j, i + 1)
                )
                if (k, j, i) in recompute:
                    r_in_act = input_id(k, j) if i == 0 else layer_round_id("y", k, j, i - 1)
                    program_tasks.append(
                        DataflowTask(
                            id=layer_round_id("r", k, j, i),
                            label=f"Step {k} Round {j} Layer {i} Recompute",
                            group="recompute",
                            compute_block_key=block_keys["recompute"],
                            inputs=[r_in_act, f"W_{i}"],
                            outputs=[
                                DataflowOutput(
                                    id=layer_round_id("A", k, j, i),
                                    size_bytes=activation_size,
                                    role="activation",
                                ),
                            ],
                        )
                    )
                else:
                    r_in_act = input_id(k, j) if i == 0 else layer_round_id("y", k, j, i - 1)
                    program_tasks.append(
                        DataflowTask(
                            id=layer_round_id("r", k, j, i),
                            label=f"Step {k} Round {j} Layer {i} Recompute",
                            group="recompute",
                            compute_block_key=block_keys["recompute_slot"],
                            inputs=[r_in_act, f"W_{i}"],
                        )
                    )

                grad_id = step_grad_id(k, i)
                b_inputs = [upstream, layer_round_id("A", k, j, i), f"W_{i}"]
                b_outputs = [
                    DataflowOutput(
                        id=layer_round_id("dy", k, j, i),
                        size_bytes=grad_size,
                        role="gradient",
                    )
                ]
                b_mutates: list[str] = []
                if j == 0:
                    b_outputs.append(
                        DataflowOutput(
                            id=grad_id,
                            size_bytes=weight_size,
                            role="gradient",
                        )
                    )
                else:
                    b_inputs.append(grad_id)
                    b_mutates.append(grad_id)
                program_tasks.append(
                    DataflowTask(
                        id=layer_round_id("b", k, j, i),
                        label=f"Step {k} Round {j} Layer {i} Backward",
                        group="backward",
                        compute_block_key=block_keys["backward"],
                        inputs=b_inputs,
                        outputs=b_outputs,
                        mutates=b_mutates,
                    )
                )

        if optimizer_state_size > 0:
            for i in range(L):
                program_tasks.append(
                    DataflowTask(
                        id=f"step_{k}_{i}",
                        label=f"Step {k} Layer {i} Optimizer",
                        group="optimizer",
                        compute_block_key=block_keys["optimizer"],
                        inputs=[step_grad_id(k, i), f"W_{i}", f"O_{i}"],
                        mutates=[f"W_{i}", f"O_{i}"],
                    )
                )

    final_locations: dict[str, str] = {}
    if optimizer_state_size > 0 and final_model_state_on_backing:
        for i in range(L):
            final_locations[f"W_{i}"] = "backing"
            final_locations[f"O_{i}"] = "backing"

    return DataflowProgram(
        name=name,
        description="Layerwise training workload",
        metadata=metadata or {},
        metrics=metrics,
        objects=objects,
        compute_blocks=compute_blocks,
        tasks=program_tasks,
        final_locations=final_locations,
    )


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
    bandwidth_from_slow: int = 8,
    bandwidth_to_slow: int = 8,
    layer_output_size: int | None = None,
    bwd_runtime: int | None = None,
    grad_accum_rounds: int = 1,
    num_steps: int = 1,
    final_model_state_on_backing: bool = False,
    recompute: frozenset[tuple[int, int, int]] = frozenset(),
    recompute_runtime: int = 0,
) -> TaskChain:
    """Build the structural skeleton of an L-layer training chain.

    Initial memory: the first input object in fast memory; all other inputs and all
    persistent model state (`W_i`, `W_head`, and optionally `O_i`) on backing.
    Per-step gradients (`dW_<step>_<layer>`) are produced by the first
    accumulation round of their step, not loaded from backing. No triggers. No
    `fast_memory_capacity` set. A policy is required before the chain can run.

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

    # --- initial memory: persistent model state on backing; first input in fast memory ---
    initial: list[Object] = []
    for k in range(num_steps):
        for j in range(grad_accum_rounds):
            initial.append(
                Object(
                    id=f"input_{k}_{j}",
                    size=input_size,
                    location="fast" if k == 0 and j == 0 else "backing",
                    type="activation",
                )
            )
    for i in range(L):
        initial.append(Object(id=f"W_{i}", size=weight_size, location="backing", type="weight"))
        if optimizer_state_size > 0:
            initial.append(
                Object(id=f"O_{i}", size=optimizer_state_size, location="backing", type="optimizer")
            )
    initial.append(Object(id="W_head", size=head_weight_size, location="backing", type="weight"))

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
                    r_in_act = (
                        input_id(k, j)
                        if i == 0
                        else layer_round_id("y", k, j, i - 1)
                    )
                    tasks.append(
                        Task(
                            id=layer_round_id("r", k, j, i),
                            inputs=[r_in_act, f"W_{i}"],
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
    if optimizer_state_size > 0 and final_model_state_on_backing:
        for i in range(L):
            final_locations[f"W_{i}"] = "backing"
            final_locations[f"O_{i}"] = "backing"

    return TaskChain(
        initial_memory=initial,
        tasks=tasks,
        bandwidth_from_slow=bandwidth_from_slow,
        bandwidth_to_slow=bandwidth_to_slow,
        final_locations=final_locations,
        fast_memory_capacity=None,
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
    program = build_transformer_training_program(spec, cfg, recompute=recompute)
    workload = realize_dataflow_program(program, hw)

    # Local import to avoid hard dependency at module load.
    from dataflow_sim.workloads.models.transformer import (
        activation_bytes,
        head_breakdown,
        head_microseconds,
        layer_bwd_breakdown,
        layer_bwd_microseconds,
        layer_fwd_breakdown,
        layer_fwd_microseconds,
        layer_recompute_microseconds,
        optimizer_step_breakdown,
        optimizer_step_microseconds,
    )

    fwd_us = layer_fwd_microseconds(spec, hw, cfg)
    bwd_us = layer_bwd_microseconds(spec, hw, cfg)
    head_us = head_microseconds(spec, hw, cfg)
    opt_us = optimizer_step_microseconds(spec, hw, cfg)
    act_bytes = activation_bytes(spec, cfg)
    recompute_us = layer_recompute_microseconds(spec, hw, cfg)

    options = (
        RecomputeOption(level=0, saved_bytes=act_bytes, recompute_us=0, label="save-full"),
        RecomputeOption(level=1, saved_bytes=0, recompute_us=recompute_us, label="recompute-full"),
    )
    rewrites: list[RecomputeRewrite] = []
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
    if levels:
        raise ValueError(f"unknown recompute object ids: {sorted(levels)}")
    generic_breakdown = workload.metadata["breakdown"]
    breakdown = {
        "compute_blocks": generic_breakdown.get("compute_blocks", []),
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
    return Workload(chain=workload.chain, metadata={
        "breakdown": breakdown,
        "compute_blocks": workload.metadata["compute_blocks"],
        "recompute_rewrites": rewrites,
        "program": workload.metadata["program"],
        "preview": workload.metadata["preview"],
        "task_summaries": workload.metadata["task_summaries"],
        "metrics": workload.metadata.get("metrics"),
        "summary": {
            "kind": "training.transformer",
            "n_layers": spec.n_layers,
            "total_tokens": (
                cfg.seqlen
                * cfg.num_seqs
                * cfg.grad_accum_rounds
                * cfg.num_steps
            ),
            "grad_accum_rounds": cfg.grad_accum_rounds,
            "num_steps": cfg.num_steps,
            "metrics": workload.metadata.get("metrics"),
        },
    })


def build_transformer_training_program(
    spec: TransformerSpec,
    cfg: TrainingConfig,
    recompute: Mapping[str, int] | None = None,
) -> DataflowProgram:
    """Generate a generic DataflowProgram for transformer training."""
    from dataflow_sim.workloads.models.transformer import (
        activation_bytes,
        backward_subops,
        forward_subops,
        head_subops,
        head_weight_bytes,
        input_bytes,
        layer_output_bytes,
        layer_weight_bytes,
        optimizer_state_bytes_per_layer,
        optimizer_step_subops,
        recompute_subops,
    )

    levels = dict(recompute or {})
    recompute_instances: set[tuple[int, int, int]] = set()
    for k in range(cfg.num_steps):
        for j in range(cfg.grad_accum_rounds):
            for i in range(spec.n_layers):
                obj_id = f"A_{k}_{j}_{i}"
                level = levels.pop(obj_id, 0)
                if level not in (0, 1):
                    raise ValueError(
                        f"unsupported recompute level {level} for {obj_id!r}"
                    )
                if level == 1:
                    recompute_instances.add((k, j, i))
    if levels:
        raise ValueError(f"unknown recompute object ids: {sorted(levels)}")

    return build_layerwise_training_program(
        spec.n_layers,
        input_size=input_bytes(spec, cfg),
        weight_size=layer_weight_bytes(spec),
        activation_size=activation_bytes(spec, cfg),
        layer_output_size=layer_output_bytes(spec, cfg),
        grad_size=layer_output_bytes(spec, cfg),
        head_weight_size=head_weight_bytes(spec),
        optimizer_state_size=optimizer_state_bytes_per_layer(spec, cfg.optimizer),
        fwd_cost=_sum_cost("layer_fwd", forward_subops(spec, cfg)),
        bwd_cost=_sum_cost("layer_bwd", backward_subops(spec, cfg)),
        head_cost=_sum_cost("head", head_subops(spec, cfg)),
        optimizer_cost=_sum_cost("optimizer_step", optimizer_step_subops(spec, cfg)),
        grad_accum_rounds=cfg.grad_accum_rounds,
        num_steps=cfg.num_steps,
        final_model_state_on_backing=cfg.final_model_state_on_backing,
        recompute=frozenset(recompute_instances),
        recompute_cost=_recompute_sum_cost("layer_recompute", recompute_subops(spec, cfg)),
        name=f"transformer-training-{spec.n_layers}L",
        metadata={
            "kind": "training.transformer",
            "transformer": asdict(spec),
            "training": asdict(cfg),
        },
        metrics=DataflowMetrics(
            primary_unit="tokens",
            primary_count=(
                cfg.seqlen
                * cfg.num_seqs
                * cfg.grad_accum_rounds
                * cfg.num_steps
            ),
            metadata={
                "seqlen": cfg.seqlen,
                "num_seqs": cfg.num_seqs,
                "grad_accum_rounds": cfg.grad_accum_rounds,
                "num_steps": cfg.num_steps,
            },
        ),
    )


def build_heterogeneous_transformer_training_program(
    layer_specs: list[TransformerSpec],
    cfg: TrainingConfig,
    *,
    name: str = "heterogeneous-transformer-training",
    recompute: Mapping[str, int] | None = None,
) -> DataflowProgram:
    """Generate a block-based DataflowProgram for non-uniform transformers.

    Each entry in ``layer_specs`` is one layer instance. Distinct layer shapes
    get distinct compute blocks; repeated shapes share block summaries. The
    program remains a generic dataflow schema and can be exported/imported by
    the webapp like any other custom workload.
    """
    if not layer_specs:
        raise ValueError("layer_specs must contain at least one layer")
    first = layer_specs[0]
    for idx, spec in enumerate(layer_specs):
        if spec.d_model != first.d_model:
            raise ValueError(
                f"layer {idx} has d_model={spec.d_model}; heterogeneous "
                "transformer programs currently require a shared d_model"
            )
        if spec.vocab_size != first.vocab_size:
            raise ValueError(
                f"layer {idx} has vocab_size={spec.vocab_size}; head sharing "
                "currently requires a shared vocab_size"
            )

    from dataflow_sim.workloads.models.transformer import (
        activation_bytes,
        backward_subops,
        forward_subops,
        head_subops,
        head_weight_bytes,
        input_bytes,
        layer_output_bytes,
        layer_weight_bytes,
        optimizer_state_bytes_per_layer,
        optimizer_step_subops,
        recompute_subops,
    )

    levels = dict(recompute or {})
    recompute_instances: set[tuple[int, int, int]] = set()
    for k in range(cfg.num_steps):
        for j in range(cfg.grad_accum_rounds):
            for i in range(len(layer_specs)):
                obj_id = f"A_{k}_{j}_{i}"
                level = levels.pop(obj_id, 0)
                if level not in (0, 1):
                    raise ValueError(
                        f"unsupported recompute level {level} for {obj_id!r}"
                    )
                if level == 1:
                    recompute_instances.add((k, j, i))
    if levels:
        raise ValueError(f"unknown recompute object ids: {sorted(levels)}")

    objects: list[DataflowObject] = []
    for k in range(cfg.num_steps):
        for j in range(cfg.grad_accum_rounds):
            objects.append(
                DataflowObject(
                    id=f"input_{k}_{j}",
                    size_bytes=input_bytes(first, cfg),
                    initial_location="fast" if k == 0 and j == 0 else "backing",
                    role="activation",
                )
            )
    for i, spec in enumerate(layer_specs):
        objects.append(
            DataflowObject(
                id=f"W_{i}",
                size_bytes=layer_weight_bytes(spec),
                initial_location="backing",
                role="parameter",
            )
        )
        opt_bytes = optimizer_state_bytes_per_layer(spec, cfg.optimizer)
        if opt_bytes > 0:
            objects.append(
                DataflowObject(
                    id=f"O_{i}",
                    size_bytes=opt_bytes,
                    initial_location="backing",
                    role="optimizer_state",
                )
            )
    objects.append(
        DataflowObject(
            id="W_head",
            size_bytes=head_weight_bytes(first),
            initial_location="backing",
            role="parameter",
        )
    )

    blocks: dict[str, ComputeBlock] = {
        "recompute_slot": ComputeBlock(
            key="recompute_slot",
            name="Layer Recompute",
            category="recompute",
            subops=[_fixed_cost("layer_recompute", 0)],
        ),
        "transformer_head": ComputeBlock(
            key="transformer_head",
            name="Transformer Head",
            category="head",
            subops=_cost_terms(_sum_cost("head", head_subops(first, cfg))),
        ),
    }

    def block_key(spec: TransformerSpec, phase: str) -> str:
        signature = _layer_signature(spec)
        key = f"transformer_{signature}_{phase}"
        if key not in blocks:
            variant_name = "MoE" if _transformer_variant(spec) == "moe" else "Dense"
            if phase == "forward":
                cost = _sum_cost("layer_fwd", forward_subops(spec, cfg))
                display = f"Transformer {variant_name} Forward"
            elif phase == "backward":
                cost = _sum_cost("layer_bwd", backward_subops(spec, cfg))
                display = f"Transformer {variant_name} Backward"
            elif phase == "recompute":
                cost = _recompute_sum_cost("layer_recompute", recompute_subops(spec, cfg))
                display = f"Transformer {variant_name} Recompute"
            elif phase == "optimizer":
                cost = _sum_cost("optimizer_step", optimizer_step_subops(spec, cfg))
                display = f"Transformer {variant_name} Optimizer"
            else:
                raise ValueError(f"unknown transformer block phase {phase!r}")
            blocks[key] = ComputeBlock(
                key=key,
                name=display,
                category=phase,
                subops=_cost_terms(cost),
                metadata={
                    "variant": _transformer_variant(spec),
                    "transformer": asdict(spec),
                },
            )
        return key

    program_tasks: list[DataflowTask] = []

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

    for k in range(cfg.num_steps):
        for j in range(cfg.grad_accum_rounds):
            for i, spec in enumerate(layer_specs):
                in_act = input_id(k, j) if i == 0 else layer_round_id("y", k, j, i - 1)
                outputs = [
                    DataflowOutput(
                        id=layer_round_id("y", k, j, i),
                        size_bytes=layer_output_bytes(spec, cfg),
                        role="activation",
                    ),
                ]
                if (k, j, i) not in recompute_instances:
                    outputs.insert(
                        0,
                        DataflowOutput(
                            id=layer_round_id("A", k, j, i),
                            size_bytes=activation_bytes(spec, cfg),
                            role="activation",
                        ),
                    )
                program_tasks.append(
                    DataflowTask(
                        id=layer_round_id("f", k, j, i),
                        label=f"Step {k} Round {j} Layer {i} Forward",
                        group="forward",
                        compute_block_key=block_key(spec, "forward"),
                        inputs=[in_act, f"W_{i}"],
                        outputs=outputs,
                    )
                )

            head_grad = step_head_grad_id(k)
            head_inputs = [layer_round_id("y", k, j, len(layer_specs) - 1), "W_head"]
            head_outputs = [
                DataflowOutput(
                    id=round_id("dy_head", k, j),
                    size_bytes=layer_output_bytes(first, cfg),
                    role="gradient",
                )
            ]
            head_mutates: list[str] = []
            if j == 0:
                head_outputs.append(
                    DataflowOutput(
                        id=head_grad,
                        size_bytes=head_weight_bytes(first),
                        role="gradient",
                    )
                )
            else:
                head_inputs.append(head_grad)
                head_mutates.append(head_grad)
            program_tasks.append(
                DataflowTask(
                    id=round_id("head", k, j),
                    label=f"Step {k} Round {j} Head",
                    group="head",
                    compute_block_key="transformer_head",
                    inputs=head_inputs,
                    outputs=head_outputs,
                    mutates=head_mutates,
                )
            )

            for i in range(len(layer_specs) - 1, -1, -1):
                spec = layer_specs[i]
                upstream = (
                    round_id("dy_head", k, j)
                    if i == len(layer_specs) - 1
                    else layer_round_id("dy", k, j, i + 1)
                )
                if (k, j, i) in recompute_instances:
                    r_in_act = input_id(k, j) if i == 0 else layer_round_id("y", k, j, i - 1)
                    program_tasks.append(
                        DataflowTask(
                            id=layer_round_id("r", k, j, i),
                            label=f"Step {k} Round {j} Layer {i} Recompute",
                            group="recompute",
                            compute_block_key=block_key(spec, "recompute"),
                            inputs=[r_in_act, f"W_{i}"],
                            outputs=[
                                DataflowOutput(
                                    id=layer_round_id("A", k, j, i),
                                    size_bytes=activation_bytes(spec, cfg),
                                    role="activation",
                                ),
                            ],
                        )
                    )
                else:
                    r_in_act = input_id(k, j) if i == 0 else layer_round_id("y", k, j, i - 1)
                    program_tasks.append(
                        DataflowTask(
                            id=layer_round_id("r", k, j, i),
                            label=f"Step {k} Round {j} Layer {i} Recompute",
                            group="recompute",
                            compute_block_key="recompute_slot",
                            inputs=[r_in_act, f"W_{i}"],
                        )
                    )

                grad_id = step_grad_id(k, i)
                b_inputs = [upstream, layer_round_id("A", k, j, i), f"W_{i}"]
                b_outputs = [
                    DataflowOutput(
                        id=layer_round_id("dy", k, j, i),
                        size_bytes=layer_output_bytes(spec, cfg),
                        role="gradient",
                    )
                ]
                b_mutates: list[str] = []
                if j == 0:
                    b_outputs.append(
                        DataflowOutput(
                            id=grad_id,
                            size_bytes=layer_weight_bytes(spec),
                            role="gradient",
                        )
                    )
                else:
                    b_inputs.append(grad_id)
                    b_mutates.append(grad_id)
                program_tasks.append(
                    DataflowTask(
                        id=layer_round_id("b", k, j, i),
                        label=f"Step {k} Round {j} Layer {i} Backward",
                        group="backward",
                        compute_block_key=block_key(spec, "backward"),
                        inputs=b_inputs,
                        outputs=b_outputs,
                        mutates=b_mutates,
                    )
                )

        if cfg.optimizer != "none":
            for i, spec in enumerate(layer_specs):
                program_tasks.append(
                    DataflowTask(
                        id=f"step_{k}_{i}",
                        label=f"Step {k} Layer {i} Optimizer",
                        group="optimizer",
                        compute_block_key=block_key(spec, "optimizer"),
                        inputs=[step_grad_id(k, i), f"W_{i}", f"O_{i}"],
                        mutates=[f"W_{i}", f"O_{i}"],
                    )
                )

    final_locations: dict[str, str] = {}
    if cfg.optimizer != "none" and cfg.final_model_state_on_backing:
        for i in range(len(layer_specs)):
            final_locations[f"W_{i}"] = "backing"
            final_locations[f"O_{i}"] = "backing"

    return DataflowProgram(
        name=name,
        description="Heterogeneous transformer training workload",
        metadata={
            "kind": "training.transformer.heterogeneous",
            "layers": [asdict(spec) for spec in layer_specs],
            "training": asdict(cfg),
        },
        metrics=DataflowMetrics(
            primary_unit="tokens",
            primary_count=(
                cfg.seqlen
                * cfg.num_seqs
                * cfg.grad_accum_rounds
                * cfg.num_steps
            ),
            metadata={
                "seqlen": cfg.seqlen,
                "num_seqs": cfg.num_seqs,
                "grad_accum_rounds": cfg.grad_accum_rounds,
                "num_steps": cfg.num_steps,
            },
        ),
        objects=objects,
        compute_blocks=list(blocks.values()),
        tasks=program_tasks,
        final_locations=final_locations,
    )


def build_example_heterogeneous_transformer_program(
    cfg: TrainingConfig | None = None,
) -> DataflowProgram:
    """Small dense+MoE example suitable for JSON export/import smoke tests."""
    cfg = cfg or TrainingConfig(seqlen=128, num_seqs=1, optimizer="none")
    dense = TransformerSpec(
        vocab_size=32_000,
        n_layers=1,
        d_model=512,
        head_dim=64,
        n_heads=8,
        n_kv_heads=8,
        expert_dim=2_048,
        num_shared_experts=1,
        num_routed_experts=0,
        top_k=0,
        qk_norm=True,
    )
    moe = replace(
        dense,
        expert_dim=1_536,
        num_shared_experts=1,
        num_routed_experts=8,
        top_k=2,
    )
    return build_heterogeneous_transformer_training_program(
        [dense, dense, moe, dense],
        cfg,
        name="example-heterogeneous-transformer",
    )
