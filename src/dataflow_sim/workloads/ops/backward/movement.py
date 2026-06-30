"""Backward movement and reduction op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import Efficiency, memory_op


def scatter_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    fanout: int,
    bytes_per_element: float = 2,
    input_bytes_per_element: float | None = None,
    output_bytes_per_element: float | None = None,
    efficiency: Efficiency = "memory",
) -> DataflowCost:
    in_bpe = bytes_per_element if input_bytes_per_element is None else input_bytes_per_element
    out_bpe = bytes_per_element if output_bytes_per_element is None else output_bytes_per_element
    return memory_op(
        name,
        tokens * dim * in_bpe + tokens * fanout * dim * out_bpe,
        efficiency=efficiency,
    )


def gather_grad(
    name: str,
    *,
    tokens: int,
    dim: int,
    fanin: int,
    bytes_per_element: float = 2,
    input_bytes_per_element: float | None = None,
    output_bytes_per_element: float | None = None,
    efficiency: Efficiency = "memory",
) -> DataflowCost:
    in_bpe = bytes_per_element if input_bytes_per_element is None else input_bytes_per_element
    out_bpe = bytes_per_element if output_bytes_per_element is None else output_bytes_per_element
    return memory_op(
        name,
        tokens * fanin * dim * in_bpe + tokens * dim * out_bpe,
        efficiency=efficiency,
    )


def reduce_grad(
    name: str,
    *,
    elements: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 2 * elements * bytes_per_element)
