"""SGD optimizer op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def sgd_step(name: str, *, weight_bytes: int) -> DataflowCost:
    return memory_op(name, 3 * weight_bytes)
