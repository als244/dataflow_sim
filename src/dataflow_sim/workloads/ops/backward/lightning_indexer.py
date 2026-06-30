"""Backward Lightning Indexer op helpers for DeepSeek sparse attention."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import OpDTypePolicy
from dataflow_sim.workloads.ops._common import (
    matmul_efficiency,
    roofline,
    topk_attention_score_terms,
)


def lightning_index_score_grad(
    name: str,
    *,
    tokens: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
    bytes_per_element: float | OpDTypePolicy = 2,
    activation_bytes_per_element: float | None = None,
    indexer_activation_bytes_per_element: float | None = None,
    compute_precision: str = "fp8",
) -> DataflowCost:
    if isinstance(bytes_per_element, OpDTypePolicy):
        policy = bytes_per_element
        activation_bytes_per_element = policy.activation_bpe
        indexer_activation_bytes_per_element = policy.indexer_activation_bpe
        compute_precision = policy.indexer_compute_precision
    else:
        activation_bytes_per_element = (
            bytes_per_element
            if activation_bytes_per_element is None
            else activation_bytes_per_element
        )
        indexer_activation_bytes_per_element = (
            activation_bytes_per_element
            if indexer_activation_bytes_per_element is None
            else indexer_activation_bytes_per_element
        )
    selected_terms = topk_attention_score_terms(
        tokens,
        top_k=index_topk,
        seqlen=seqlen,
        sequence_lengths=sequence_lengths,
    )
    flops = 6 * selected_terms * index_n_heads * index_head_dim
    effective_flops = 4 * selected_terms * index_n_heads * index_head_dim
    memory_bytes = (
        selected_terms * indexer_activation_bytes_per_element
        + tokens * index_n_heads * index_head_dim * indexer_activation_bytes_per_element
        + tokens * index_head_dim * indexer_activation_bytes_per_element
        + tokens * index_n_heads * indexer_activation_bytes_per_element
    )
    return roofline(
        name,
        flops=flops,
        effective_flops=effective_flops,
        memory_bytes=memory_bytes,
        efficiency=matmul_efficiency(compute_precision),
    )
