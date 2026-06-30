"""Qwen hybrid Gated DeltaNet token-mixing module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.qwen_hybrid_dimensions import QwenHybridDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class QwenHybridLinearAttention(DataflowModule):
    def __init__(self, dims: QwenHybridDimensions) -> None:
        super().__init__(name="QwenHybridLinearAttention")
        self.dims = dims

    def forward_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        dims = self.dims
        return [
            fwd.rms_norm("attn_norm", tokens=tokens, dim=dims.d_model, bytes_per_element=bytes_per_element),
            fwd.matmul(
                "in_proj_qkvz",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.linear_key_dim + 2 * dims.linear_value_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "in_proj_ba",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.linear_num_value_heads,
                bytes_per_element=bytes_per_element,
            ),
            fwd.depthwise_causal_conv1d(
                "causal_conv1d",
                tokens=tokens,
                dim=dims.linear_conv_dim,
                kernel_size=dims.linear_conv_kernel_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.gated_delta_rule(
                "gated_delta_rule",
                tokens=tokens,
                num_key_heads=dims.linear_num_key_heads,
                key_head_dim=dims.linear_key_head_dim,
                num_value_heads=dims.linear_num_value_heads,
                value_head_dim=dims.linear_value_head_dim,
                chunk_size=dims.gdn_chunk_size,
                bytes_per_element=bytes_per_element,
            ),
            fwd.gated_rms_norm(
                "gated_rms_norm",
                tokens=tokens,
                dim=dims.linear_value_dim,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "linear_out_proj",
                tokens=tokens,
                input_dim=dims.linear_value_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
                accumulate=True,
            ),
        ]

    def dgrad_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        dims = self.dims
        return [
            bwd.matmul_input_grad(
                "linear_out_proj_dgrad",
                tokens=tokens,
                input_dim=dims.linear_value_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.gated_rms_norm_grad(
                "gated_rms_norm_bwd",
                tokens=tokens,
                dim=dims.linear_value_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.gated_delta_rule_grad(
                "gated_delta_rule_bwd",
                tokens=tokens,
                num_key_heads=dims.linear_num_key_heads,
                key_head_dim=dims.linear_key_head_dim,
                num_value_heads=dims.linear_num_value_heads,
                value_head_dim=dims.linear_value_head_dim,
                chunk_size=dims.gdn_chunk_size,
                bytes_per_element=bytes_per_element,
            ),
            bwd.depthwise_causal_conv1d_grad(
                "causal_conv1d_bwd",
                tokens=tokens,
                dim=dims.linear_conv_dim,
                kernel_size=dims.linear_conv_kernel_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "in_proj_ba_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.linear_num_value_heads,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "in_proj_qkvz_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.linear_key_dim + 2 * dims.linear_value_dim,
                bytes_per_element=bytes_per_element,
            ),
            bwd.rms_norm_grad("attn_norm_bwd", tokens=tokens, dim=dims.d_model, bytes_per_element=bytes_per_element),
        ]

    def wgrad_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        dims = self.dims
        return [
            bwd.matmul_weight_grad(
                "linear_out_proj_wgrad",
                tokens=tokens,
                input_dim=dims.linear_value_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "in_proj_ba_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.linear_num_value_heads,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "in_proj_qkvz_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=2 * dims.linear_key_dim + 2 * dims.linear_value_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def backward_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        return self.dgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element) + self.wgrad_ops(
            tokens=tokens,
            bytes_per_element=bytes_per_element,
        )

    def recompute_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(tokens=tokens, bytes_per_element=bytes_per_element)
        ]
