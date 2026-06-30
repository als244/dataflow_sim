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
    bytes_per_element: float = 2,
    activation_bytes_per_element: float | None = None,
    gradient_bytes_per_element: float | None = None,
) -> DataflowCost:
    act_bpe = bytes_per_element if activation_bytes_per_element is None else activation_bytes_per_element
    grad_bpe = bytes_per_element if gradient_bytes_per_element is None else gradient_bytes_per_element
    return memory_op(
        name,
        tokens * expert_dim * branches * (2 * act_bpe + 3 * grad_bpe),
    )


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


def relu2_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: float = 2,
) -> DataflowCost:
    return memory_op(name, 5 * tokens * dim * bytes_per_element)


def gated_multiply_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 5 * tokens * dim * bytes_per_element)
