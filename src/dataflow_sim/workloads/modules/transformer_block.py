"""Transformer block composition module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.dense_attention import DenseAttention
from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.modules.mlp import SwiGLUMLP
from dataflow_sim.workloads.modules.moe import MoE
from dataflow_sim.workloads.ops import optimizer as opt_ops


class TransformerBlock(DataflowModule):
    def __init__(self, dims: TransformerDimensions) -> None:
        super().__init__(name="TransformerBlock")
        self.dims = dims
        self.attention = DenseAttention(dims)
        self.feed_forward: SwiGLUMLP | MoE
        if dims.num_routed_experts > 0 and dims.top_k > 0:
            self.feed_forward = MoE(dims)
        else:
            self.feed_forward = SwiGLUMLP(dims)

    def optimizer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        return self.attention.optimizer_matrices() + self.feed_forward.optimizer_matrices()

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return (
            self.attention.forward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
            + self.feed_forward.forward_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        )

    def backward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return (
            self.feed_forward.dgrad_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
            + self.attention.dgrad_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
            + self.feed_forward.wgrad_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
            + self.attention.wgrad_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        )

    def recompute_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return (
            self.attention.recompute_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
            + self.feed_forward.recompute_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        )
