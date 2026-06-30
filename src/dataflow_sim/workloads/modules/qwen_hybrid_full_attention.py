"""Qwen hybrid gated full-attention module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.qwen_hybrid_dimensions import QwenHybridDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class QwenHybridFullAttention(DataflowModule):
    def __init__(self, dims: QwenHybridDimensions) -> None:
        super().__init__(name="QwenHybridFullAttention")
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
            fwd.rms_norm("attn_norm", tokens=tokens, dim=dims.d_model, bytes_per_element=bytes_per_element),
            fwd.matmul(
                "q_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.full_q_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "k_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.full_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "v_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.full_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.memory(
                "q_norm",
                bytes_total=2 * tokens * dims.full_q_dim * bytes_per_element,
            ),
            fwd.memory(
                "k_norm",
                bytes_total=2 * tokens * dims.full_kv_dim * bytes_per_element,
            ),
            fwd.rope(
                "rope",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
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
            fwd.gated_multiply(
                "sigmoid_gate_mul",
                tokens=tokens,
                dim=dims.full_q_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "o_proj",
                tokens=tokens,
                input_dim=dims.full_q_dim,
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
                input_dim=dims.full_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.gated_multiply_grad(
                "sigmoid_gate_mul_bwd",
                tokens=tokens,
                dim=dims.full_q_dim,
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
            bwd.rope_grad(
                "rope_bwd",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
                bytes_per_element=bytes_per_element,
            ),
            fwd.memory(
                "k_norm_bwd",
                bytes_total=4 * tokens * dims.full_kv_dim * bytes_per_element,
            ),
            fwd.memory(
                "q_norm_bwd",
                bytes_total=4 * tokens * dims.full_q_dim * bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "v_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.full_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "k_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.full_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "q_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.full_q_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.rms_norm_grad("attn_norm_bwd", tokens=tokens, dim=dims.d_model, bytes_per_element=bytes_per_element),
        ]

    def wgrad_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        dims = self.dims
        return [
            bwd.matmul_weight_grad(
                "o_proj_wgrad",
                tokens=tokens,
                input_dim=dims.full_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "v_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.full_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "k_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.full_kv_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "q_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.full_q_dim,
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
