"""Example custom op helpers.

These are intentionally local to the example. To promote an op into the library,
move the helper into `src/dataflow_sim/workloads/ops/<phase>/<op_type>.py` and
re-export it from that phase package.
"""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost


def roofline(
    name: str,
    *,
    flops: int = 0,
    memory_bytes: int = 0,
    efficiency: str = "memory",
    effective_flops: int | None = None,
) -> DataflowCost:
    return DataflowCost(
        kind="roofline",
        name=name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency=efficiency,
        effective_flops=effective_flops,
    )


def dense_projection(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    flops = 2 * tokens * input_dim * output_dim
    memory_bytes = (
        tokens * input_dim
        + input_dim * output_dim
        + tokens * output_dim
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="matmul",
    )


def dense_input_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return dense_projection(
        name,
        tokens=tokens,
        input_dim=output_dim,
        output_dim=input_dim,
        bytes_per_element=bytes_per_element,
    )


def dense_weight_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    flops = 2 * tokens * input_dim * output_dim
    memory_bytes = (
        tokens * input_dim
        + tokens * output_dim
        + input_dim * output_dim
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="matmul",
    )


def fast_gelu(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    flops = 8 * tokens * dim
    memory_bytes = 3 * tokens * dim * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="memory",
    )


def fast_gelu_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    flops = 10 * tokens * dim
    memory_bytes = 4 * tokens * dim * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="memory",
    )


def residual_add(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return roofline(
        name,
        memory_bytes=3 * tokens * dim * bytes_per_element,
        efficiency="memory",
    )


def cross_entropy(
    name: str,
    *,
    tokens: int,
    classes: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    flops = 5 * tokens * classes
    memory_bytes = 3 * tokens * classes * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="memory",
    )


def optimizer_step(
    name: str,
    *,
    optimizer: str,
    param_count: int,
    bytes_per_element: int = 2,
) -> list[DataflowCost]:
    if optimizer == "none":
        return []
    if optimizer == "sgd":
        return [
            roofline(
                f"{name}_sgd",
                flops=param_count,
                memory_bytes=3 * param_count * bytes_per_element,
                efficiency="memory",
            )
        ]
    if optimizer == "adamw":
        return [
            roofline(
                f"{name}_adamw",
                flops=12 * param_count,
                memory_bytes=7 * param_count * bytes_per_element,
                efficiency="memory",
            )
        ]
    raise ValueError(
        "custom example supports optimizer none, sgd, or adamw; "
        f"got {optimizer!r}"
    )


def recompute_only(ops: list[DataflowCost]) -> list[DataflowCost]:
    """Mark compute FLOPs as recompute overhead instead of useful work."""
    return [
        op.model_copy(update={"effective_flops": 0})
        if op.kind == "roofline" and op.flops > 0
        else op
        for op in ops
    ]
