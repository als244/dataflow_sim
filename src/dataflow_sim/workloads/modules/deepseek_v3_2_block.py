"""DeepSeek-V3.2 block composition module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.modules.deepseek_v3_2_dimensions import DeepSeekV32Dimensions
from dataflow_sim.workloads.modules.dsa_sparse_attention import DSASparseAttention, IndexerMode
from dataflow_sim.workloads.modules.mlp import SwiGLUMLP
from dataflow_sim.workloads.modules.moe import MoE


class DeepSeekV32Block(DataflowModule):
    def __init__(
        self,
        dims: DeepSeekV32Dimensions,
        *,
        dense_ffn: bool,
        indexer_mode: IndexerMode = "full",
    ) -> None:
        super().__init__(name="DeepSeekV32Block")
        self.dims = dims
        self.dense_ffn = dense_ffn
        self.indexer_mode = indexer_mode
        self.attention = DSASparseAttention(dims, indexer_mode=indexer_mode)
        ffn_dims = dims.ffn_dimensions(dense=dense_ffn)
        self.feed_forward = SwiGLUMLP(ffn_dims) if dense_ffn else MoE(ffn_dims)

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
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
        bytes_per_element: float | OpDTypePolicy = 2,
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
        bytes_per_element: float | OpDTypePolicy = 2,
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
