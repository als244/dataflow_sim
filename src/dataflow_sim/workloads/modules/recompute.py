"""Recompute phase helper modules."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import fixed


def zero_recompute_slot() -> list[DataflowCost]:
    return [fixed("layer_recompute", 0)]
