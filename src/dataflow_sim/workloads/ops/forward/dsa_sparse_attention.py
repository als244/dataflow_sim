"""Forward DeepSeek sparse-attention op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import roofline, topk_attention_score_terms


def dsa_sparse_attention(
    name: str,
    *,
    tokens: int,
    n_heads: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    value_head_dim: int,
    index_topk: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
    bytes_per_element: float = 2,
) -> DataflowCost:
    selected_terms = topk_attention_score_terms(
        tokens,
        top_k=index_topk,
        seqlen=seqlen,
        sequence_lengths=sequence_lengths,
    )
    flops = n_heads * (2 * kv_lora_rank + rope_head_dim) * selected_terms
    memory_bytes = (
        tokens
        * n_heads
        * (2 * (kv_lora_rank + rope_head_dim) + 2 * value_head_dim)
        * bytes_per_element
    )
    return roofline(
        name,
        flops=flops,
        memory_bytes=memory_bytes,
        efficiency="attention_fwd",
    )
