"""Backward matmul op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import roofline


def matmul_input_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int = 2,
    count: int = 1,
    memory_bytes: int | None = None,
) -> DataflowCost:
    return _matmul_grad(
        name,
        tokens=tokens,
        input_dim=input_dim,
        output_dim=output_dim,
        bytes_per_element=bytes_per_element,
        count=count,
        accumulator_elements=0,
        memory_bytes=memory_bytes,
    )


def matmul_weight_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int = 2,
    count: int = 1,
    accumulate: bool = True,
    memory_bytes: int | None = None,
) -> DataflowCost:
    return _matmul_grad(
        name,
        tokens=tokens,
        input_dim=input_dim,
        output_dim=output_dim,
        bytes_per_element=bytes_per_element,
        count=count,
        accumulator_elements=input_dim * output_dim if accumulate else 0,
        memory_bytes=memory_bytes,
    )


def _matmul_grad(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int,
    count: int,
    accumulator_elements: int,
    memory_bytes: int | None,
) -> DataflowCost:
    flops = 2 * tokens * input_dim * output_dim
    bytes_total = memory_bytes
    if bytes_total is None:
        bytes_total = (
            tokens * input_dim
            + input_dim * output_dim
            + tokens * output_dim
            + accumulator_elements
        ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=bytes_total,
        efficiency="matmul",
        count=count,
    )
