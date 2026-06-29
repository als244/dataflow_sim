"""Optimizer step module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.dimensions import (
    TransformerDimensions,
    layer_weight_matrices,
    params_per_layer,
)
from dataflow_sim.workloads.ops import optimizer as opt_ops


class OptimizerStep(DataflowModule):
    def __init__(self, dims: TransformerDimensions, optimizer: opt_ops.OptimizerMode) -> None:
        super().__init__(name="OptimizerStep")
        self.dims = dims
        self.optimizer = optimizer

    def state_bytes(self, *, bytes_per_element: int = 2) -> int:
        return opt_ops.optimizer_state_bytes(
            params_per_layer(self.dims) * bytes_per_element,
            self.optimizer,
        )

    def step_ops(self, *, bytes_per_element: int = 2) -> list[DataflowCost]:
        if self.optimizer == "none":
            return []
        weight_bytes = params_per_layer(self.dims) * bytes_per_element
        if self.optimizer == "adamw":
            return [opt_ops.adamw_step("adamw_step", weight_bytes=weight_bytes)]
        if self.optimizer == "muon":
            return [
                opt_ops.muon_step(
                    "muon_step",
                    matrices=layer_weight_matrices(self.dims),
                    bytes_per_element=bytes_per_element,
                )
            ]
        if self.optimizer == "sgd":
            return [opt_ops.sgd_step("sgd_step", weight_bytes=weight_bytes)]
        raise ValueError(f"unknown optimizer mode: {self.optimizer!r}")

    def recompute_ops(self, *, bytes_per_element: int = 2) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.step_ops(bytes_per_element=bytes_per_element)
        ]
