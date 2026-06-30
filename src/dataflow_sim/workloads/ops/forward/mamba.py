"""Forward Mamba-family op helpers."""
from __future__ import annotations

import math

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op, roofline


def _scan_counts(tokens: int, seqlen: int, chunk_size: int) -> tuple[int, int]:
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if tokens % seqlen != 0:
        raise ValueError("tokens must be divisible by seqlen")
    return tokens // seqlen, math.ceil(seqlen / chunk_size)


def mamba_chunk_scan(
    name: str,
    *,
    tokens: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    state_dim: int,
    n_groups: int,
    chunk_size: int,
    bytes_per_element: float = 2,
) -> DataflowCost:
    batch_size, num_chunks = _scan_counts(tokens, seqlen, chunk_size)
    intermediate_dim = num_heads * head_dim
    conv_state_dim = 2 * n_groups * state_dim

    flops = (
        2 * batch_size * num_chunks * chunk_size * chunk_size * num_heads * (state_dim + head_dim)
        + 4 * tokens * num_heads * head_dim * state_dim
        + 2 * batch_size * num_chunks * num_heads * head_dim * state_dim
        + 2 * tokens * intermediate_dim
    )
    memory_bytes = (
        tokens * (intermediate_dim + conv_state_dim + num_heads + intermediate_dim)
        + batch_size * num_chunks * chunk_size * chunk_size * num_heads
        + batch_size * num_chunks * num_heads * head_dim * state_dim
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="attention_fwd",
    )


def mamba_gated_rms_norm(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: float = 2,
) -> DataflowCost:
    return memory_op(name, 4 * tokens * dim * bytes_per_element)
