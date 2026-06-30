"""Dense transformer attention module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd
from dataflow_sim.workloads.ops import optimizer as opt_ops


class DenseAttention(DataflowModule):
    def __init__(self, dims: TransformerDimensions) -> None:
        super().__init__(name="DenseAttention")
        self.dims = dims

    def optimizer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        dims = self.dims
        return [
            opt_ops.OptimizerMatrix("q_proj", dims.d_model, dims.n_heads * dims.head_dim),
            opt_ops.OptimizerMatrix("k_proj", dims.d_model, dims.n_kv_heads * dims.head_dim),
            opt_ops.OptimizerMatrix("v_proj", dims.d_model, dims.n_kv_heads * dims.head_dim),
            opt_ops.OptimizerMatrix("attn_proj", dims.n_heads * dims.head_dim, dims.d_model),
        ]

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        qkv_out_dim = (dims.n_heads + 2 * dims.n_kv_heads) * dims.head_dim
        attn_in_dim = dims.n_heads * dims.head_dim
        ops: list[DataflowCost] = [
            fwd.rms_norm(
                "attn_norm",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            fwd.matmul(
                "qkv_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=qkv_out_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]
        if dims.qk_norm:
            ops.append(
                fwd.qk_norm(
                    "qk_norm",
                    tokens=tokens,
                    head_dim=dims.head_dim,
                    n_heads=dims.n_heads,
                    n_kv_heads=dims.n_kv_heads,
                    bytes_per_element=bytes_per_element,
                )
            )
        ops.extend(
            [
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
                fwd.matmul(
                    "attn_proj",
                    tokens=tokens,
                    input_dim=attn_in_dim,
                    output_dim=dims.d_model,
                    bytes_per_element=bytes_per_element,
                    accumulate=True,
                ),
            ]
        )
        return ops

    def dgrad_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        qkv_out_dim = (dims.n_heads + 2 * dims.n_kv_heads) * dims.head_dim
        attn_in_dim = dims.n_heads * dims.head_dim
        ops: list[DataflowCost] = [
            bwd.rms_norm_grad(
                "ffn_norm_bwd",
                tokens=tokens,
                dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "attn_proj_dgrad",
                tokens=tokens,
                input_dim=attn_in_dim,
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
            bwd.rope_grad(
                "rope_bwd",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
                bytes_per_element=bytes_per_element,
            ),
        ]
        if dims.qk_norm:
            ops.append(
                bwd.qk_norm_grad(
                    "qk_norm_bwd",
                    tokens=tokens,
                    head_dim=dims.head_dim,
                    n_heads=dims.n_heads,
                    n_kv_heads=dims.n_kv_heads,
                    bytes_per_element=bytes_per_element,
                )
            )
        ops.extend(
            [
                bwd.matmul_input_grad(
                    "qkv_proj_dgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=qkv_out_dim,
                    bytes_per_element=bytes_per_element,
                ),
                bwd.rms_norm_grad(
                    "attn_norm_bwd",
                    tokens=tokens,
                    dim=dims.d_model,
                    bytes_per_element=bytes_per_element,
                ),
            ]
        )
        return ops

    def wgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        qkv_out_dim = (dims.n_heads + 2 * dims.n_kv_heads) * dims.head_dim
        attn_in_dim = dims.n_heads * dims.head_dim
        return [
            bwd.matmul_weight_grad(
                "attn_proj_wgrad",
                tokens=tokens,
                input_dim=attn_in_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "qkv_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=qkv_out_dim,
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
        return (
            self.dgrad_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
            + self.wgrad_ops(tokens=tokens, bytes_per_element=bytes_per_element)
        )

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
