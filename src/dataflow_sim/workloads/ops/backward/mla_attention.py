"""Backward multi-head latent-attention op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import attention_score_terms, memory_op, roofline


def mla_rope_grad(
    name: str,
    *,
    tokens: int,
    rope_head_dim: int,
    n_heads: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(
        name,
        2 * tokens * rope_head_dim * (n_heads + 1) * bytes_per_element,
    )


def mla_attention_grad(
    name: str,
    *,
    tokens: int,
    n_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
    bytes_per_element: int = 2,
) -> DataflowCost:
    score_terms = attention_score_terms(
        tokens,
        seqlen=seqlen,
        sequence_lengths=sequence_lengths,
    )
    flops = n_heads * (2 * qk_head_dim + 3 * value_head_dim) * score_terms
    effective_flops = n_heads * (2 * qk_head_dim + 2 * value_head_dim) * score_terms
    memory_bytes = (
        tokens * n_heads * (3 * qk_head_dim + 3 * value_head_dim)
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        effective_flops=effective_flops,
        memory_bytes=memory_bytes,
        efficiency="attention_bwd",
    )
