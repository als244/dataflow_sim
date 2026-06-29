"""Forward matmul-like op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import roofline


def matmul(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: int = 2,
    count: int = 1,
    accumulate: bool = False,
    memory_bytes: int | None = None,
) -> DataflowCost:
    flops = 2 * tokens * input_dim * output_dim
    bytes_total = memory_bytes
    if bytes_total is None:
        accumulator_elements = tokens * output_dim if accumulate else 0
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
