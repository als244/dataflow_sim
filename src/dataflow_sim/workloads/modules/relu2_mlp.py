"""Dense ReLU2 MLP module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.ops import optimizer as opt_ops


class ReLU2MLP(DataflowModule):
    def __init__(self, *, d_model: int, intermediate_size: int) -> None:
        super().__init__(name="ReLU2MLP")
        self.d_model = d_model
        self.intermediate_size = intermediate_size

    def optimizer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        return [
            opt_ops.OptimizerMatrix("mlp_up", self.d_model, self.intermediate_size),
            opt_ops.OptimizerMatrix("mlp_down", self.intermediate_size, self.d_model),
        ]

    @staticmethod
    def _policy(bytes_per_element: float | OpDTypePolicy) -> OpDTypePolicy:
        return (
            bytes_per_element
            if isinstance(bytes_per_element, OpDTypePolicy)
            else OpDTypePolicy.from_single_bpe(bytes_per_element)
        )

    def forward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
        accumulate: bool = True,
    ) -> list[DataflowCost]:
        policy = self._policy(bytes_per_element)
        return [
            fwd.matmul(
                "mlp_up",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.intermediate_size,
                bytes_per_element=policy,
            ),
            fwd.relu2(
                "relu2",
                tokens=tokens,
                dim=self.intermediate_size,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.matmul(
                "mlp_down",
                tokens=tokens,
                input_dim=self.intermediate_size,
                output_dim=self.d_model,
                bytes_per_element=policy,
                accumulate=accumulate,
            ),
        ]

    def dgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        policy = self._policy(bytes_per_element)
        return [
            bwd.matmul_input_grad(
                "mlp_down_dgrad",
                tokens=tokens,
                input_dim=self.intermediate_size,
                output_dim=self.d_model,
                bytes_per_element=policy,
            ),
            bwd.relu2_grad(
                "relu2_bwd",
                tokens=tokens,
                dim=self.intermediate_size,
                bytes_per_element=policy.activation_bpe,
            ),
            bwd.matmul_input_grad(
                "mlp_up_dgrad",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.intermediate_size,
                bytes_per_element=policy,
            ),
        ]

    def wgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        policy = self._policy(bytes_per_element)
        return [
            bwd.matmul_weight_grad(
                "mlp_down_wgrad",
                tokens=tokens,
                input_dim=self.intermediate_size,
                output_dim=self.d_model,
                bytes_per_element=policy,
            ),
            bwd.matmul_weight_grad(
                "mlp_up_wgrad",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.intermediate_size,
                bytes_per_element=policy,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        return self.dgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element) + self.wgrad_ops(
            tokens=tokens,
            bytes_per_element=bytes_per_element,
        )

    def recompute_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
                accumulate=False,
            )
            if op.name != "mlp_down"
        ]
