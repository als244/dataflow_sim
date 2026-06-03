"""Forward + backward training chain for L layers — workload structure only.

`build_bare_training_chain` returns the *bare* chain: compute tasks with
inputs/outputs/runtime, all model state on host, **no** triggers. The bare
chain is not directly runnable — it requires a policy (e.g. sliding-window or
auto) to add trigger annotations and initial-device-pool entries.

`build_training_chain` is a backward-compatible wrapper that applies the
sliding-window policy by default, matching the API/behaviour shipped before
the Step-1 refactor in AUTOPOLICY.md.

Task graph (per layer index i, 0 <= i < L):

    f_0    = (deps=[input,   W_0],     out=[A_0, y_0])
    f_i>0  = (deps=[y_{i-1}, W_i],     out=[A_i, y_i])
    head   = (deps=[y_{L-1}, W_head, dW_head], out=[dy_head])
    r_i    = (deps=[A_i, W_i],         out=[],         runtime=0)
    b_i    = (deps=[upstream, A_i, W_i, dW_i], out=[dy_i])

where upstream = dy_head for i=L-1 else dy_{i+1}.
"""
from __future__ import annotations

from dataflow_sim.schema import Object, OutputAlloc, Task, TaskChain


def build_bare_training_chain(
    L: int,
    *,
    input_size: int = 16,
    weight_size: int = 64,
    activation_size: int = 32,
    grad_size: int = 32,
    head_weight_size: int = 64,
    fwd_runtime: int = 10,
    head_runtime: int = 2,
    bandwidth_h2d: int = 8,
    bandwidth_d2h: int = 8,
    layer_output_size: int | None = None,
    bwd_runtime: int | None = None,
) -> TaskChain:
    """Build the structural skeleton of an L-layer training chain.

    Initial memory: `input` on device; all weights and gradient buffers
    (`W_i, dW_i, W_head, dW_head`) on host only. No triggers. No
    `device_capacity` set. A policy is required before the chain can run.

    `layer_output_size` (= `y_i` / `dy_i` size) defaults to `activation_size`
    for backwards compat with existing tests. The transformer-driven workload
    sizes A_i and y_i differently and overrides this.

    `bwd_runtime` defaults to `2 * fwd_runtime` for backwards compat.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if bwd_runtime is None:
        bwd_runtime = 2 * fwd_runtime
    if layer_output_size is None:
        layer_output_size = activation_size

    # --- initial memory: model state on host; input on device ---
    initial: list[Object] = [
        Object(id="input", size=input_size, location="device", type="activation"),
    ]
    for i in range(L):
        initial.append(Object(id=f"W_{i}", size=weight_size, location="host", type="weight"))
        initial.append(Object(id=f"dW_{i}", size=weight_size, location="host", type="gradient"))
    initial.append(Object(id="W_head", size=head_weight_size, location="host", type="weight"))
    initial.append(Object(id="dW_head", size=head_weight_size, location="host", type="gradient"))

    tasks: list[Task] = []

    # --- forward ---
    for i in range(L):
        in_act = "input" if i == 0 else f"y_{i - 1}"
        tasks.append(
            Task(
                id=f"f_{i}",
                inputs=[in_act, f"W_{i}"],
                outputs=[
                    OutputAlloc(id=f"A_{i}", size=activation_size, type="activation"),
                    OutputAlloc(id=f"y_{i}", size=layer_output_size, type="activation"),
                ],
                runtime=fwd_runtime,
            )
        )

    # --- head ---
    tasks.append(
        Task(
            id="head",
            inputs=[f"y_{L - 1}", "W_head", "dW_head"],
            outputs=[OutputAlloc(id="dy_head", size=grad_size, type="gradient")],
            runtime=head_runtime,
            # head consumes dW_head as a buffer and writes the updated
            # head-weight gradient into it. Declaring this mutation lets
            # the planner emit the proper write-back (rather than a
            # release that would discard the gradient update).
            mutates_inputs=["dW_head"],
        )
    )

    # --- backward: r_i then b_i, from L-1 down to 0 ---
    for i in range(L - 1, -1, -1):
        upstream = "dy_head" if i == L - 1 else f"dy_{i + 1}"
        tasks.append(
            Task(
                id=f"r_{i}",
                inputs=[f"A_{i}", f"W_{i}"],
                outputs=[],
                runtime=0,
            )
        )
        tasks.append(
            Task(
                id=f"b_{i}",
                inputs=[upstream, f"A_{i}", f"W_{i}", f"dW_{i}"],
                outputs=[OutputAlloc(id=f"dy_{i}", size=grad_size, type="gradient")],
                runtime=bwd_runtime,
                # b_i writes the layer-i weight gradient into dW_i (which
                # arrives as a buffer containing the previous step's
                # accumulator). The planner must write dW_i back to host
                # after this task — releasing it would discard the update.
                mutates_inputs=[f"dW_{i}"],
            )
        )

    return TaskChain(
        initial_memory=initial,
        tasks=tasks,
        bandwidth_h2d=bandwidth_h2d,
        bandwidth_d2h=bandwidth_d2h,
        device_capacity=None,
    )


def build_training_chain(
    L: int,
    *,
    window_size: int = 2,
    device_capacity: int | None = None,
    input_size: int = 16,
    weight_size: int = 64,
    activation_size: int = 32,
    grad_size: int = 32,
    head_weight_size: int = 64,
    fwd_runtime: int = 10,
    head_runtime: int = 2,
    bandwidth_h2d: int = 8,
    bandwidth_d2h: int = 8,
) -> TaskChain:
    """Backward-compatible wrapper: bare chain + sliding-window policy.

    Matches the pre-refactor API. Equivalent to:
        bare = build_bare_training_chain(L, ...)
        return apply_sliding_window_policy(bare, window_size=..., device_capacity=...)
    """
    # Imported here to avoid a circular import (sliding_window imports schema
    # types that this module re-exports indirectly via TaskChain).
    from dataflow_sim.policy.sliding_window import apply_sliding_window_policy

    bare = build_bare_training_chain(
        L,
        input_size=input_size,
        weight_size=weight_size,
        activation_size=activation_size,
        grad_size=grad_size,
        head_weight_size=head_weight_size,
        fwd_runtime=fwd_runtime,
        head_runtime=head_runtime,
        bandwidth_h2d=bandwidth_h2d,
        bandwidth_d2h=bandwidth_d2h,
    )
    return apply_sliding_window_policy(
        bare,
        window_size=window_size,
        device_capacity=device_capacity,
    )


def build_transformer_bare_chain(spec, hw, cfg) -> "tuple[TaskChain, dict]":
    """Build a bare training chain from transformer model dimensions +
    hardware specs + training params. Returns the chain alongside a
    breakdown payload (sub-op timings for one representative layer + head).

    Args:
        spec: `core.workloads.transformer.TransformerSpec`
        hw:   `core.workloads.transformer.HardwareEnv`
        cfg:  `core.workloads.transformer.TrainingConfig`
    """
    # Local import to avoid hard dependency at module load.
    from dataflow_app.workloads.transformer import (
        activation_bytes,
        gbs_to_bytes_per_microsecond,
        head_breakdown,
        head_microseconds,
        head_weight_bytes,
        input_bytes,
        layer_bwd_breakdown,
        layer_bwd_microseconds,
        layer_fwd_breakdown,
        layer_fwd_microseconds,
        layer_output_bytes,
        layer_weight_bytes,
    )

    fwd_us = layer_fwd_microseconds(spec, hw, cfg)
    bwd_us = layer_bwd_microseconds(spec, hw, cfg)
    head_us = head_microseconds(spec, hw, cfg)
    bw_link = gbs_to_bytes_per_microsecond(hw.interconnect_bw_gbs)

    bare = build_bare_training_chain(
        spec.n_layers,
        input_size=input_bytes(spec, cfg),
        weight_size=layer_weight_bytes(spec),
        activation_size=activation_bytes(spec, cfg),
        layer_output_size=layer_output_bytes(spec, cfg),
        grad_size=layer_output_bytes(spec, cfg),
        head_weight_size=head_weight_bytes(spec),
        fwd_runtime=fwd_us,
        head_runtime=head_us,
        bandwidth_h2d=bw_link,
        bandwidth_d2h=bw_link,
        bwd_runtime=bwd_us,
    )
    breakdown = {
        "fwd": [t.asdict() for t in layer_fwd_breakdown(spec, hw, cfg)],
        "bwd": [t.asdict() for t in layer_bwd_breakdown(spec, hw, cfg)],
        "head": [t.asdict() for t in head_breakdown(spec, hw, cfg)],
        "totals_us": {
            "layer_fwd": fwd_us,
            "layer_bwd": bwd_us,
            "head": head_us,
        },
    }
    return bare, breakdown
