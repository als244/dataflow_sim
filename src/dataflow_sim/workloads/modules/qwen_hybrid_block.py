"""Qwen hybrid block composition module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.mlp import SwiGLUMLP
from dataflow_sim.workloads.modules.moe import MoE
from dataflow_sim.workloads.modules.qwen_hybrid_dimensions import QwenHybridDimensions
from dataflow_sim.workloads.modules.qwen_hybrid_full_attention import QwenHybridFullAttention
from dataflow_sim.workloads.modules.qwen_hybrid_linear_attention import QwenHybridLinearAttention


class QwenHybridBlock(DataflowModule):
    def __init__(self, dims: QwenHybridDimensions, layer_type: str) -> None:
        super().__init__(name="QwenHybridBlock")
        self.dims = dims
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.attention = QwenHybridLinearAttention(dims)
        elif layer_type == "full_attention":
            self.attention = QwenHybridFullAttention(dims)
        else:
            raise ValueError(f"unknown Qwen hybrid layer type {layer_type!r}")
        ffn_dims = dims.ffn_dimensions()
        self.feed_forward = MoE(ffn_dims) if dims.is_moe else SwiGLUMLP(ffn_dims)

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        if self.layer_type == "linear_attention":
            attention_ops = self.attention.forward_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        else:
            attention_ops = self.attention.forward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
        return attention_ops + self.feed_forward.forward_ops(
            tokens=tokens,
            bytes_per_element=bytes_per_element,
        )

    def backward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        if self.layer_type == "linear_attention":
            attention_dgrad = self.attention.dgrad_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
            attention_wgrad = self.attention.wgrad_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        else:
            attention_dgrad = self.attention.dgrad_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
            attention_wgrad = self.attention.wgrad_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        return (
            self.feed_forward.dgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
            + attention_dgrad
            + self.feed_forward.wgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
            + attention_wgrad
        )

    def recompute_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        if self.layer_type == "linear_attention":
            attention_ops = self.attention.recompute_ops(
                tokens=tokens,
                bytes_per_element=bytes_per_element,
            )
        else:
            attention_ops = self.attention.recompute_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
        return attention_ops + self.feed_forward.recompute_ops(
            tokens=tokens,
            bytes_per_element=bytes_per_element,
        )
