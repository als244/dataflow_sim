"""Backward Mamba-family op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op, roofline
from dataflow_sim.workloads.ops.forward.mamba import mamba_chunk_scan


def mamba_chunk_scan_grad(
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
    forward = mamba_chunk_scan(
        name,
        tokens=tokens,
        seqlen=seqlen,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dim=state_dim,
        n_groups=n_groups,
        chunk_size=chunk_size,
        bytes_per_element=bytes_per_element,
    )
    return roofline(
        name,
        flops=2 * forward.flops,
        memory_bytes=2 * forward.memory_bytes,
        efficiency="attention_bwd",
    )


def mamba_gated_rms_norm_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: float = 2,
) -> DataflowCost:
    return memory_op(name, 8 * tokens * dim * bytes_per_element)
