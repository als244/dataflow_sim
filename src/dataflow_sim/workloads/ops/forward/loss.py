"""Forward loss op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def cross_entropy(
    name: str,
    *,
    tokens: int,
    vocab_size: int,
    bytes_per_element: int = 2,
) -> DataflowCost:
    return memory_op(name, 2 * tokens * vocab_size * bytes_per_element)
