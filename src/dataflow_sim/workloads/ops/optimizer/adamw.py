"""AdamW optimizer op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def adamw_step(name: str, *, weight_bytes: int) -> DataflowCost:
    return memory_op(name, 7 * weight_bytes)
