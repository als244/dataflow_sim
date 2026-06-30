"""Forward matmul-like op helpers."""
from __future__ import annotations

import math

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import OpDTypePolicy
from dataflow_sim.workloads.ops._common import matmul_efficiency, roofline


def matmul(
    name: str,
    *,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: float | OpDTypePolicy = 2,
    activation_bytes_per_element: float | None = None,
    weight_bytes_per_element: float | None = None,
    output_bytes_per_element: float | None = None,
    count: int = 1,
    accumulate: bool = False,
    memory_bytes: int | None = None,
    compute_precision: str = "bf16",
) -> DataflowCost:
    if isinstance(bytes_per_element, OpDTypePolicy):
        policy = bytes_per_element
        bytes_per_element = policy.activation_bpe
        activation_bytes_per_element = policy.activation_bpe
        weight_bytes_per_element = policy.weight_bpe
        output_bytes_per_element = policy.activation_bpe
        compute_precision = policy.compute_precision
    flops = 2 * tokens * input_dim * output_dim
    bytes_total = memory_bytes
    if bytes_total is None:
        act_bpe = bytes_per_element if activation_bytes_per_element is None else activation_bytes_per_element
        weight_bpe = bytes_per_element if weight_bytes_per_element is None else weight_bytes_per_element
        out_bpe = act_bpe if output_bytes_per_element is None else output_bytes_per_element
        accumulator_elements = tokens * output_dim if accumulate else 0
        bytes_total = math.ceil(
            tokens * input_dim * act_bpe
            + input_dim * output_dim * weight_bpe
            + tokens * output_dim * out_bpe
            + accumulator_elements * out_bpe
        )
    return roofline(
        name,
        flops=flops,
        memory_bytes=bytes_total,
        efficiency=matmul_efficiency(compute_precision),
        count=count,
    )
