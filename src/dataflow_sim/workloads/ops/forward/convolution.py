"""Forward convolution op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import roofline


def depthwise_causal_conv1d(
    name: str,
    *,
    tokens: int,
    dim: int,
    kernel_size: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    flops = 2 * tokens * dim * kernel_size
    memory_bytes = (
        tokens * dim
        + dim * kernel_size
        + tokens * dim
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="matmul",
    )
