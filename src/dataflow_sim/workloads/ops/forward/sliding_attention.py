"""Forward sliding-window attention op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import sliding_attention_score_terms, roofline


def sliding_attention(
    name: str,
    *,
    tokens: int,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    window_size: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
    bytes_per_element: int = 2,
) -> DataflowCost:
    score_terms = sliding_attention_score_terms(
        tokens,
        window_size=window_size,
        seqlen=seqlen,
        sequence_lengths=sequence_lengths,
    )
    flops = 2 * n_heads * head_dim * score_terms
    bytes_total = (
        tokens * (n_heads + 2 * n_kv_heads) * head_dim
        + tokens * n_heads * head_dim
    ) * bytes_per_element
    return roofline(
        name,
        flops=flops,
        memory_bytes=bytes_total,
        efficiency="attention_fwd",
    )
