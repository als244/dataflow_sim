"""Language modeling head and loss module."""
from __future__ import annotations

import math

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.modules.dimensions import TransformerDimensions, head_params
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.ops import optimizer as opt_ops


class LanguageModelingHead(DataflowModule):
    def __init__(self, dims: TransformerDimensions) -> None:
        super().__init__(name="LanguageModelingHead")
        self.dims = dims

    def optimizer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        return [
            opt_ops.OptimizerMatrix("lm_head", self.dims.d_model, self.dims.vocab_size)
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
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        head_weight_bytes = head_params(dims) * policy.weight_bpe
        head_bytes = math.ceil(
            head_weight_bytes + 2 * tokens * dims.d_model * policy.activation_bpe
        )
        return [
            fwd.rms_norm(
                "final_norm",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.matmul(
                "head_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.vocab_size,
                bytes_per_element=policy.activation_bpe,
                weight_bytes_per_element=policy.weight_bpe,
                output_bytes_per_element=policy.activation_bpe,
                memory_bytes=head_bytes,
                compute_precision=policy.compute_precision,
            ),
            fwd.cross_entropy(
                "cross_entropy",
                tokens=tokens,
                vocab_size=dims.vocab_size,
                bytes_per_element=policy.activation_bpe,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        head_dgrad_bytes = math.ceil(
            head_params(dims) * policy.weight_bpe
            + 2 * tokens * dims.d_model * policy.activation_bpe
        )
        head_wgrad_bytes = math.ceil(
            head_params(dims) * policy.gradient_bpe
            + 2 * tokens * dims.d_model * policy.activation_bpe
        )
        return [
            bwd.matmul_input_grad(
                "head_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.vocab_size,
                bytes_per_element=policy.activation_bpe,
                weight_bytes_per_element=policy.weight_bpe,
                upstream_gradient_bytes_per_element=policy.activation_bpe,
                input_gradient_bytes_per_element=policy.activation_bpe,
                memory_bytes=head_dgrad_bytes,
                compute_precision=policy.compute_precision,
            ),
            bwd.matmul_weight_grad(
                "head_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.vocab_size,
                bytes_per_element=policy.activation_bpe,
                activation_bytes_per_element=policy.activation_bpe,
                upstream_gradient_bytes_per_element=policy.activation_bpe,
                parameter_gradient_bytes_per_element=policy.gradient_bpe,
                memory_bytes=head_wgrad_bytes,
                compute_precision=policy.compute_precision,
            ),
            bwd.rms_norm_grad(
                "final_norm_bwd",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=policy.activation_bpe,
            ),
        ]

    def recompute_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(tokens=tokens, bytes_per_element=bytes_per_element)
        ]
