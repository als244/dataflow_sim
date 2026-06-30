"""Forward linear-attention op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op, roofline


def gated_delta_rule(
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
    # Serial DeltaNet costs 6*K*V per token/value-head. FLA's chunked path
    # also forms chunk-local qk and qk-v terms, adding 2*C*(K+V).
    recurrent_flops = 6 * key_head_dim * value_head_dim
    chunk_local_flops = 2 * chunk_size * (key_head_dim + value_head_dim)
    flops = tokens * num_value_heads * (recurrent_flops + chunk_local_flops)
    memory_bytes = (
        2 * tokens * key_dim
        + 4 * tokens * value_dim
        + 2 * tokens * num_value_heads
        + tokens * num_value_heads * chunk_size
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="attention_fwd",
    )


def gated_rms_norm(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 4 * tokens * dim * bytes_per_element)
