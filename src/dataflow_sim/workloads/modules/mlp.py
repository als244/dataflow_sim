"""Dense SwiGLU MLP module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.ops import optimizer as opt_ops


class SwiGLUMLP(DataflowModule):
    def __init__(self, dims: TransformerDimensions) -> None:
        super().__init__(name="SwiGLUMLP")
        self.dims = dims

    @property
    def count(self) -> int:
        return self.dims.num_shared_experts

    def optimizer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        dims = self.dims
        if self.count <= 0:
            return []
        return [
            opt_ops.OptimizerMatrix("shared_mlp_gate", dims.d_model, dims.expert_dim, self.count),
            opt_ops.OptimizerMatrix("shared_mlp_up", dims.d_model, dims.expert_dim, self.count),
            opt_ops.OptimizerMatrix("shared_mlp_down", dims.expert_dim, dims.d_model, self.count),
        ]

    def forward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        if self.count <= 0:
            return []
        return [
            fwd.rms_norm(
                "ffn_norm",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "shared_mlp_up",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.expert_dim,
                bytes_per_element=bytes_per_element,
                count=self.count,
            ),
            fwd.swiglu(
                "swiglu",
                tokens=tokens,
                expert_dim=dims.expert_dim,
                branches=self.count,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "shared_mlp_down",
                tokens=tokens,
                input_dim=dims.expert_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
                count=self.count,
                accumulate=True,
            ),
        ]

    def dgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        if self.count <= 0:
            return []
        return [
            bwd.matmul_input_grad(
                "shared_mlp_down_dgrad",
                tokens=tokens,
                input_dim=dims.expert_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
                count=self.count,
            ),
            bwd.swiglu_grad(
                "swiglu_bwd",
                tokens=tokens,
                expert_dim=dims.expert_dim,
                branches=self.count,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "shared_mlp_up_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.expert_dim,
                bytes_per_element=bytes_per_element,
                count=self.count,
            ),
        ]

    def wgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        if self.count <= 0:
            return []
        return [
            bwd.matmul_weight_grad(
                "shared_mlp_down_wgrad",
                tokens=tokens,
                input_dim=dims.expert_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
                count=self.count,
            ),
            bwd.matmul_weight_grad(
                "shared_mlp_up_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.expert_dim,
                bytes_per_element=bytes_per_element,
                count=self.count,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return (
            self.dgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
            + self.wgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
        )

    def recompute_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        if self.count <= 0:
            return []
        return [
            fwd.rms_norm(
                "ffn_norm",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ).model_copy(update={"effective_flops": 0}),
            fwd.matmul(
                "shared_mlp_up",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.expert_dim,
                bytes_per_element=bytes_per_element,
                count=self.count,
            ).model_copy(update={"effective_flops": 0}),
            fwd.swiglu(
                "swiglu",
                tokens=tokens,
                expert_dim=dims.expert_dim,
                branches=self.count,
                bytes_per_element=bytes_per_element,
            ).model_copy(update={"effective_flops": 0}),
        ]
