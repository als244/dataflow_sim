"""Optimizer step module."""
from __future__ import annotations

from dataclasses import replace
import math

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import (
    DTypePolicy,
    DataflowModule,
    OpDTypePolicy,
    ParallelismConfig,
    dtype_nbytes,
)
from dataflow_sim.workloads.modules.dimensions import (
    TransformerDimensions,
    layer_weight_matrices,
    params_per_layer,
)
from dataflow_sim.workloads.ops import optimizer as opt_ops


def optimizer_ops_for_matrices(
    name: str,
    *,
    matrices: list[opt_ops.OptimizerMatrix],
    optimizer: opt_ops.OptimizerMode,
    bytes_per_element: float | OpDTypePolicy = 2,
) -> list[DataflowCost]:
    if optimizer == "none":
        return []
    policy = (
        bytes_per_element
        if isinstance(bytes_per_element, OpDTypePolicy)
        else OpDTypePolicy.from_single_bpe(bytes_per_element)
    )
    local_matrices = local_optimizer_matrices(matrices, policy)
    weight_bytes = matrix_weight_bytes(matrices, policy)
    gradient_bytes = matrix_gradient_bytes(matrices, policy)
    state_bytes = optimizer_state_bytes_for_matrices(matrices, optimizer, policy)
    if optimizer == "adamw":
        return [
            opt_ops.adamw_step(
                "adamw_step",
                weight_bytes=weight_bytes,
                gradient_bytes=gradient_bytes,
                optimizer_state_bytes=state_bytes,
            )
        ]
    if optimizer == "muon":
        return [
            opt_ops.muon_matrix_step(
                f"{matrix.name}_muon_step",
                matrix=matrix,
                bytes_per_element=policy.optimizer_state_bpe,
            )
            for matrix in local_matrices
        ]
    if optimizer == "sgd":
        return [
            opt_ops.sgd_step(
                "sgd_step",
                weight_bytes=weight_bytes,
                gradient_bytes=gradient_bytes,
            )
        ]
    raise ValueError(f"unknown optimizer mode for {name}: {optimizer!r}")


def _ep_group_size(
    policy: DTypePolicy | OpDTypePolicy,
    parallelism: ParallelismConfig | None = None,
) -> int:
    if parallelism is not None:
        return parallelism.ep_group_size
    return policy.ep_group_size if isinstance(policy, OpDTypePolicy) else 1


def local_matrix_count(
    matrix: opt_ops.OptimizerMatrix,
    policy: DTypePolicy | OpDTypePolicy,
    parallelism: ParallelismConfig | None = None,
) -> int:
    ep_group_size = _ep_group_size(policy, parallelism)
    if not matrix.ep_sharded:
        return matrix.count
    if matrix.count % ep_group_size != 0:
        raise ValueError(
            f"ep_group_size={ep_group_size} must divide routed expert count "
            f"{matrix.count} for {matrix.name}"
        )
    return matrix.count // ep_group_size


def local_optimizer_matrices(
    matrices: list[opt_ops.OptimizerMatrix],
    policy: DTypePolicy | OpDTypePolicy,
    parallelism: ParallelismConfig | None = None,
) -> list[opt_ops.OptimizerMatrix]:
    return [
        replace(matrix, count=local_matrix_count(matrix, policy, parallelism))
        for matrix in matrices
    ]


def matrix_weight_bytes(
    matrices: list[opt_ops.OptimizerMatrix],
    policy: DTypePolicy | OpDTypePolicy,
    parallelism: ParallelismConfig | None = None,
) -> int:
    if isinstance(policy, DTypePolicy):
        default_bpe = dtype_nbytes(policy.param)
        expert_bpe = dtype_nbytes(policy.expert_param)
    else:
        default_bpe = policy.weight_bpe
        expert_bpe = policy.expert_weight_bpe
    return math.ceil(
        sum(
            matrix.rows
            * matrix.cols
            * local_matrix_count(matrix, policy, parallelism)
            * (expert_bpe if matrix.expert else default_bpe)
            for matrix in matrices
        )
    )


def matrix_gradient_bytes(
    matrices: list[opt_ops.OptimizerMatrix],
    policy: DTypePolicy | OpDTypePolicy,
    parallelism: ParallelismConfig | None = None,
) -> int:
    bpe = dtype_nbytes(policy.gradient) if isinstance(policy, DTypePolicy) else policy.gradient_bpe
    return math.ceil(
        sum(
            matrix.rows
            * matrix.cols
            * local_matrix_count(matrix, policy, parallelism)
            for matrix in matrices
        )
        * bpe
    )


def optimizer_state_bytes_for_matrices(
    matrices: list[opt_ops.OptimizerMatrix],
    optimizer: opt_ops.OptimizerMode,
    policy: DTypePolicy | OpDTypePolicy,
    parallelism: ParallelismConfig | None = None,
) -> int:
    if optimizer in {"none", "sgd"}:
        return 0
    state_factor = 2 if optimizer == "adamw" else 1
    bpe = (
        dtype_nbytes(policy.optimizer_state)
        if isinstance(policy, DTypePolicy)
        else policy.optimizer_state_bpe
    )
    return math.ceil(
        state_factor
        * sum(
            matrix.rows
            * matrix.cols
            * local_matrix_count(matrix, policy, parallelism)
            for matrix in matrices
        )
        * bpe
    )


class OptimizerStep(DataflowModule):
    def __init__(self, dims: TransformerDimensions, optimizer: opt_ops.OptimizerMode) -> None:
        super().__init__(name="OptimizerStep")
        self.dims = dims
        self.optimizer = optimizer

    def state_bytes(self, *, bytes_per_element: float | OpDTypePolicy = 2) -> int:
        return optimizer_state_bytes_for_matrices(
            layer_weight_matrices(self.dims),
            self.optimizer,
            bytes_per_element
            if isinstance(bytes_per_element, OpDTypePolicy)
            else OpDTypePolicy.from_single_bpe(bytes_per_element),
        )

    def step_ops(self, *, bytes_per_element: float | OpDTypePolicy = 2) -> list[DataflowCost]:
        return optimizer_ops_for_matrices(
            "optimizer_step",
            matrices=layer_weight_matrices(self.dims),
            optimizer=self.optimizer,
            bytes_per_element=bytes_per_element,
        )

    def recompute_ops(self, *, bytes_per_element: int = 2) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.step_ops(bytes_per_element=bytes_per_element)
        ]
