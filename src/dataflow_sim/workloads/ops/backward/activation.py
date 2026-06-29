"""Backward activation op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def swiglu_grad(
    name: str,
    *,
    tokens: int,
    expert_dim: int,
    branches: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 5 * tokens * expert_dim * branches * bytes_per_element)


def silu_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 5 * tokens * dim * bytes_per_element)


def gelu_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 5 * tokens * dim * bytes_per_element)
