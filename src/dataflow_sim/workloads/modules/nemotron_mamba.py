"""Nemotron-H Mamba2 mixer module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.modules.nemotron_dimensions import NemotronDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class NemotronMamba(DataflowModule):
    def __init__(self, dims: NemotronDimensions) -> None:
        super().__init__(name="NemotronMamba")
        self.dims = dims

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
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        return [
            fwd.matmul(
                "mamba_in_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.mamba_projection_dim,
                bytes_per_element=policy,
            ),
            fwd.depthwise_causal_conv1d(
                "mamba_depthwise_conv1d",
                tokens=tokens,
                dim=dims.mamba_conv_dim,
                kernel_size=dims.conv_kernel,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.silu(
                "mamba_silu",
                tokens=tokens,
                dim=dims.mamba_conv_dim,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.mamba_chunk_scan(
                "mamba_chunk_scan",
                tokens=tokens,
                seqlen=seqlen,
                num_heads=dims.mamba_num_heads,
                head_dim=dims.mamba_head_dim,
                state_dim=dims.ssm_state_size,
                n_groups=dims.n_groups,
                chunk_size=dims.mamba_chunk_size,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.mamba_gated_rms_norm(
                "mamba_gated_rms_norm",
                tokens=tokens,
                dim=dims.mamba_intermediate_dim,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.matmul(
                "mamba_out_proj",
                tokens=tokens,
                input_dim=dims.mamba_intermediate_dim,
                output_dim=dims.d_model,
                bytes_per_element=policy,
                accumulate=True,
            ),
        ]

    def dgrad_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        return [
            bwd.matmul_input_grad(
                "mamba_out_proj_dgrad",
                tokens=tokens,
                input_dim=dims.mamba_intermediate_dim,
                output_dim=dims.d_model,
                bytes_per_element=policy,
            ),
            bwd.mamba_gated_rms_norm_grad(
                "mamba_gated_rms_norm_bwd",
                tokens=tokens,
                dim=dims.mamba_intermediate_dim,
                bytes_per_element=policy.activation_bpe,
            ),
            bwd.mamba_chunk_scan_grad(
                "mamba_chunk_scan_bwd",
                tokens=tokens,
                seqlen=seqlen,
                num_heads=dims.mamba_num_heads,
                head_dim=dims.mamba_head_dim,
                state_dim=dims.ssm_state_size,
                n_groups=dims.n_groups,
                chunk_size=dims.mamba_chunk_size,
                bytes_per_element=policy.activation_bpe,
            ),
            bwd.silu_grad(
                "mamba_silu_bwd",
                tokens=tokens,
                dim=dims.mamba_conv_dim,
                bytes_per_element=policy.activation_bpe,
            ),
            bwd.depthwise_causal_conv1d_grad(
                "mamba_depthwise_conv1d_bwd",
                tokens=tokens,
                dim=dims.mamba_conv_dim,
                kernel_size=dims.conv_kernel,
                bytes_per_element=policy.activation_bpe,
            ),
            bwd.matmul_input_grad(
                "mamba_in_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.mamba_projection_dim,
                bytes_per_element=policy,
            ),
        ]

    def wgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        return [
            bwd.matmul_weight_grad(
                "mamba_out_proj_wgrad",
                tokens=tokens,
                input_dim=dims.mamba_intermediate_dim,
                output_dim=dims.d_model,
                bytes_per_element=policy,
            ),
            bwd.matmul_weight_grad(
                "mamba_in_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.mamba_projection_dim,
                bytes_per_element=policy,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
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
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
        ]
