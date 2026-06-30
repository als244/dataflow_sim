"""GPT-OSS block composition module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.gpt_oss_attention import GPTOSSAttention
from dataflow_sim.workloads.modules.gpt_oss_dimensions import GPTOSSDimensions
from dataflow_sim.workloads.modules.moe import MoE


class GPTOSSBlock(DataflowModule):
    def __init__(self, dims: GPTOSSDimensions, layer_type: str) -> None:
        super().__init__(name="GPTOSSBlock")
        self.dims = dims
        self.layer_type = layer_type
        self.attention = GPTOSSAttention(dims, layer_type)
        self.feed_forward = MoE(dims.ffn_dimensions())

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
            self.feed_forward.dgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
            + self.attention.dgrad_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
            + self.feed_forward.wgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
            + self.attention.wgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
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
