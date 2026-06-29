"""Forward normalization op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def rms_norm(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 2 * tokens * dim * bytes_per_element)


def layer_norm(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return rms_norm(name, tokens=tokens, dim=dim, bytes_per_element=bytes_per_element)


def qk_norm(
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
