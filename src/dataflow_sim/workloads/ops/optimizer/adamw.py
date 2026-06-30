"""AdamW optimizer op helpers."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import memory_op


def adamw_step(
    name: str,
    *,
    weight_bytes: int,
    gradient_bytes: int | None = None,
    optimizer_state_bytes: int | None = None,
) -> DataflowCost:
    gradient = weight_bytes if gradient_bytes is None else gradient_bytes
    state = 2 * weight_bytes if optimizer_state_bytes is None else optimizer_state_bytes
    return memory_op(name, 2 * weight_bytes + gradient + 2 * state)
