"""GPT-OSS attention module with full and sliding-window variants."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.gpt_oss_dimensions import GPTOSSDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class GPTOSSAttention(DataflowModule):
    def __init__(self, dims: GPTOSSDimensions, layer_type: str) -> None:
        super().__init__(name="GPTOSSAttention")
        if layer_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(f"unknown GPT-OSS attention layer type {layer_type!r}")
        self.dims = dims
        self.layer_type = layer_type

    def _attention_op(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int,
    ) -> DataflowCost:
        dims = self.dims
        if self.layer_type == "sliding_attention":
            return fwd.sliding_attention(
                "sliding_attn",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
                window_size=dims.sliding_window,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
        return fwd.attention(
            "attn",
            tokens=tokens,
            head_dim=dims.head_dim,
            n_heads=dims.n_heads,
            n_kv_heads=dims.n_kv_heads,
            seqlen=seqlen,
            bytes_per_element=bytes_per_element,
        )

    def _attention_grad_op(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int,
    ) -> DataflowCost:
        dims = self.dims
        if self.layer_type == "sliding_attention":
            return bwd.sliding_attention_grad(
                "sliding_attn_bwd",
                tokens=tokens,
                head_dim=dims.head_dim,
                n_heads=dims.n_heads,
                n_kv_heads=dims.n_kv_heads,
                window_size=dims.sliding_window,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            )
        return bwd.attention_grad(
            "attn_bwd",
            tokens=tokens,
            head_dim=dims.head_dim,
            n_heads=dims.n_heads,
            n_kv_heads=dims.n_kv_heads,
            seqlen=seqlen,
            bytes_per_element=bytes_per_element,
        )

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
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
                output_dim=dims.qkv_dim,
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
                self._attention_op(
                    tokens=tokens,
                    seqlen=seqlen,
                    bytes_per_element=bytes_per_element,
                ),
                fwd.matmul(
                    "attn_proj",
                    tokens=tokens,
                    input_dim=dims.attn_q_dim,
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
                input_dim=dims.attn_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            self._attention_grad_op(
                tokens=tokens,
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
                    output_dim=dims.qkv_dim,
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
        return [
            bwd.matmul_weight_grad(
                "attn_proj_wgrad",
                tokens=tokens,
                input_dim=dims.attn_q_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "qkv_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.qkv_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def recompute_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return self.forward_ops(
            tokens=tokens,
            seqlen=seqlen,
            bytes_per_element=bytes_per_element,
        )
