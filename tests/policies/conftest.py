"""Simulator-side test fixtures.

Provides minimal legacy bare-chain builders for policy tests that exercise
the original `f_i` / `b_i` task-id convention.

The `build_bare_training_chain` function is a local copy of the
training-chain builder used by belady_reactive / min_grow tests. It
mirrors the workload-side helper without importing it.
"""
from __future__ import annotations

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain


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
    bandwidth_from_slow: int = 8,
    bandwidth_to_slow: int = 8,
    layer_output_size: int | None = None,
    bwd_runtime: int | None = None,
) -> TaskChain:
    """Build the structural skeleton of an L-layer training chain.

    Initial memory: `input` on compute; all weights and gradient buffers
    (`W_i, dW_i, W_head, dW_head`) on backing only. No triggers. No
    `fast_memory_capacity` set. A policy is required before the chain can run.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if bwd_runtime is None:
        bwd_runtime = 2 * fwd_runtime
    if layer_output_size is None:
        layer_output_size = activation_size

    initial: list[Object] = [
        Object(id="input", size=input_size, location="fast", type="activation"),
    ]
    for i in range(L):
        initial.append(Object(id=f"W_{i}", size=weight_size, location="backing", type="weight"))
        initial.append(Object(id=f"dW_{i}", size=weight_size, location="backing", type="gradient"))
    initial.append(Object(id="W_head", size=head_weight_size, location="backing", type="weight"))
    initial.append(Object(id="dW_head", size=head_weight_size, location="backing", type="gradient"))

    tasks: list[Task] = []

    # forward
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

    # head
    tasks.append(
        Task(
            id="head",
            inputs=[f"y_{L - 1}", "W_head", "dW_head"],
            outputs=[OutputAlloc(id="dy_head", size=grad_size, type="gradient")],
            runtime=head_runtime,
            mutates_inputs=["dW_head"],
        )
    )

    # backward: r_i then b_i, from L-1 down to 0
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
                mutates_inputs=[f"dW_{i}"],
            )
        )

    return TaskChain(
        initial_memory=initial,
        tasks=tasks,
        bandwidth_from_slow=bandwidth_from_slow,
        bandwidth_to_slow=bandwidth_to_slow,
        fast_memory_capacity=None,
    )
