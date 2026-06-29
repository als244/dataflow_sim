"""Language modeling head and loss module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.dimensions import TransformerDimensions, head_params
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class LanguageModelingHead(DataflowModule):
    def __init__(self, dims: TransformerDimensions) -> None:
        super().__init__(name="LanguageModelingHead")
        self.dims = dims

    def forward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        head_weight_bytes = head_params(dims) * bytes_per_element
        head_bytes = head_weight_bytes + 2 * tokens * dims.d_model * bytes_per_element
        return [
            fwd.rms_norm(
                "final_norm",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "head_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.vocab_size,
                bytes_per_element=bytes_per_element,
                memory_bytes=head_bytes,
            ),
            fwd.cross_entropy(
                "cross_entropy",
                tokens=tokens,
                vocab_size=dims.vocab_size,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        head_weight_bytes = head_params(dims) * bytes_per_element
        head_bytes = head_weight_bytes + 2 * tokens * dims.d_model * bytes_per_element
        return [
            bwd.matmul_input_grad(
                "head_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.vocab_size,
                bytes_per_element=bytes_per_element,
                memory_bytes=head_bytes,
            ),
            bwd.matmul_weight_grad(
                "head_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.vocab_size,
                bytes_per_element=bytes_per_element,
                memory_bytes=head_bytes,
            ),
            bwd.rms_norm_grad(
                "final_norm_bwd",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def recompute_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return [
            op.model_copy(update={"effective_flops": 0})
            for op in self.forward_ops(tokens=tokens, bytes_per_element=bytes_per_element)
        ]
