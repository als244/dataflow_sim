"""Backward attention op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import attention_score_terms, memory_op, roofline


def attention_grad(
    name: str,
    *,
    tokens: int,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
    bytes_per_element: int = 2,
) -> DataflowCost:
    score_terms = attention_score_terms(
        tokens,
        seqlen=seqlen,
        sequence_lengths=sequence_lengths,
    )
    flops = 5 * n_heads * head_dim * score_terms
    effective_flops = 4 * n_heads * head_dim * score_terms
    bytes_read = (
        tokens * (n_heads + 2 * n_kv_heads) * head_dim
        + tokens * n_heads * head_dim
    ) * bytes_per_element
    bytes_write = tokens * (n_heads + 2 * n_kv_heads) * head_dim * bytes_per_element
    return roofline(
        name,
        flops=flops,
        effective_flops=effective_flops,
        memory_bytes=bytes_read + bytes_write,
        efficiency="attention_bwd",
    )


def rope_grad(
    name: str,
    *,
    tokens: int,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(
        name,
        2 * tokens * head_dim * (n_heads + n_kv_heads) * bytes_per_element,
    )
