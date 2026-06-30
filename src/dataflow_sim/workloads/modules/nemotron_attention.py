"""Nemotron-H dense attention mixer module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.nemotron_dimensions import NemotronDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class NemotronAttention(DataflowModule):
    def __init__(self, dims: NemotronDimensions) -> None:
        super().__init__(name="NemotronAttention")
        self.dims = dims

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        return [
            fwd.matmul(
                "q_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_q_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "k_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "v_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.attention(
                "attn",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "o_proj",
                tokens=tokens,
                input_dim=dims.attention_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
                accumulate=True,
            ),
        ]

    def dgrad_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        return [
            bwd.matmul_input_grad(
                "o_proj_dgrad",
                tokens=tokens,
                input_dim=dims.attention_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.attention_grad(
                "attn_bwd",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "v_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "k_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "q_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_q_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def wgrad_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        dims = self.dims
        return [
            bwd.matmul_weight_grad(
                "o_proj_wgrad",
                tokens=tokens,
                input_dim=dims.attention_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "v_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "k_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "q_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.attention_q_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return self.dgrad_ops(
            tokens=tokens,
            seqlen=seqlen,
            bytes_per_element=bytes_per_element,
        ) + self.wgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)

    def recompute_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
        ]
