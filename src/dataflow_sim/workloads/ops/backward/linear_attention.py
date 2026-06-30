"""Backward linear-attention op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op, roofline


def gated_delta_rule_grad(
    name: str,
    *,
    tokens: int,
    num_key_heads: int,
    key_head_dim: int,
    num_value_heads: int,
    value_head_dim: int,
    chunk_size: int = 64,
    bytes_per_element: int = 2,
) -> DataflowCost:
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    # Keep the backward estimate tied to the FLA-shaped forward model until
    # the backward kernels are split into explicit sub-ops.
    recurrent_flops = 6 * key_head_dim * value_head_dim
    chunk_local_flops = 2 * chunk_size * (key_head_dim + value_head_dim)
    forward_flops = tokens * num_value_heads * (recurrent_flops + chunk_local_flops)
    forward_bytes = (
        2 * tokens * key_dim
        + 4 * tokens * value_dim
        + 2 * tokens * num_value_heads
        + tokens * num_value_heads * chunk_size
    ) * bytes_per_element
    return roofline(
        name,
        flops=2 * forward_flops,
        memory_bytes=2 * forward_bytes,
        efficiency="attention_bwd",
    )


def gated_rms_norm_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 8 * tokens * dim * bytes_per_element)
