"""Forward movement and reduction op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def scatter(
    name: str,
    *,
    tokens: int,
    dim: int,
    fanout: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, tokens * (1 + fanout) * dim * bytes_per_element)


def gather(
    name: str,
    *,
    tokens: int,
    dim: int,
    fanin: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, tokens * (1 + fanin) * dim * bytes_per_element)


def reduce(
    name: str,
    *,
    elements: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 2 * elements * bytes_per_element)


def memory(
    name: str,
    *,
    bytes_total: int,
) -> DataflowCost:
    return memory_op(name, bytes_total)
