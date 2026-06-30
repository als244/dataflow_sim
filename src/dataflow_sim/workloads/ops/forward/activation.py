"""Forward activation op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def swiglu(
    name: str,
    *,
    tokens: int,
    expert_dim: int,
    branches: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 3 * tokens * expert_dim * branches * bytes_per_element)


def silu(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 3 * tokens * dim * bytes_per_element)


def gelu(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 3 * tokens * dim * bytes_per_element)


def relu2(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: float = 2,
) -> DataflowCost:
    return memory_op(name, 3 * tokens * dim * bytes_per_element)


def gated_multiply(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 3 * tokens * dim * bytes_per_element)
