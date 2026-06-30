"""DeepSeek/Kimi multi-head latent-attention module."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule
from dataflow_sim.workloads.modules.deepseek_dimensions import DeepSeekDimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class MLAAttention(DataflowModule):
    def __init__(self, dims: DeepSeekDimensions) -> None:
        super().__init__(name="MLAAttention")
        self.dims = dims

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        ops: list[DataflowCost] = [
            fwd.rms_norm("attn_norm", tokens=tokens, dim=dims.d_model, bytes_per_element=bytes_per_element)
        ]
        if dims.q_lora_rank > 0:
            ops.extend(
                [
                    fwd.matmul(
                        "q_a_proj",
                        tokens=tokens,
                        input_dim=dims.d_model,
                        output_dim=dims.q_lora_rank,
                        bytes_per_element=bytes_per_element,
                    ),
                    fwd.memory(
                        "q_a_norm",
                        bytes_total=2 * tokens * dims.q_lora_rank * bytes_per_element,
                    ),
                    fwd.matmul(
                        "q_b_proj",
                        tokens=tokens,
                        input_dim=dims.q_lora_rank,
                        output_dim=dims.q_dim,
                        bytes_per_element=bytes_per_element,
                    ),
                ]
            )
        else:
            ops.append(
                fwd.matmul(
                    "q_proj",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.q_dim,
                    bytes_per_element=bytes_per_element,
                )
            )
        ops.extend(
            [
                fwd.matmul(
                    "kv_a_proj_with_mqa",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.kv_lora_rank + dims.qk_rope_head_dim,
                    bytes_per_element=bytes_per_element,
                ),
                fwd.memory(
                    "kv_a_norm",
                    bytes_total=2 * tokens * dims.kv_lora_rank * bytes_per_element,
                ),
                fwd.matmul(
                    "kv_b_proj",
                    tokens=tokens,
                    input_dim=dims.kv_lora_rank,
                    output_dim=dims.n_heads * (dims.qk_nope_head_dim + dims.v_head_dim),
                    bytes_per_element=bytes_per_element,
                ),
                fwd.mla_rope(
                    "mla_rope",
                    tokens=tokens,
                    rope_head_dim=dims.qk_rope_head_dim,
                    n_heads=dims.n_heads,
                    bytes_per_element=bytes_per_element,
                ),
                fwd.mla_attention(
                    "mla_attn",
                    tokens=tokens,
                    n_heads=dims.n_heads,
                    qk_head_dim=dims.qk_head_dim,
                    value_head_dim=dims.v_head_dim,
                    seqlen=seqlen,
                    bytes_per_element=bytes_per_element,
                ),
                fwd.matmul(
                    "o_proj",
                    tokens=tokens,
                    input_dim=dims.o_dim,
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
            bwd.matmul_input_grad(
                "o_proj_dgrad",
                tokens=tokens,
                input_dim=dims.o_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.mla_attention_grad(
                "mla_attn_bwd",
                tokens=tokens,
                n_heads=dims.n_heads,
                qk_head_dim=dims.qk_head_dim,
                value_head_dim=dims.v_head_dim,
                seqlen=seqlen,
                bytes_per_element=bytes_per_element,
            ),
            bwd.mla_rope_grad(
                "mla_rope_bwd",
                tokens=tokens,
                rope_head_dim=dims.qk_rope_head_dim,
                n_heads=dims.n_heads,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "kv_b_proj_dgrad",
                tokens=tokens,
                input_dim=dims.kv_lora_rank,
                output_dim=dims.n_heads * (dims.qk_nope_head_dim + dims.v_head_dim),
                bytes_per_element=bytes_per_element,
            ),
            fwd.memory(
                "kv_a_norm_bwd",
                bytes_total=4 * tokens * dims.kv_lora_rank * bytes_per_element,
            ),
            bwd.matmul_input_grad(
                "kv_a_proj_with_mqa_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.kv_lora_rank + dims.qk_rope_head_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]
        if dims.q_lora_rank > 0:
            ops.extend(
                [
                    bwd.matmul_input_grad(
                        "q_b_proj_dgrad",
                        tokens=tokens,
                        input_dim=dims.q_lora_rank,
                        output_dim=dims.q_dim,
                        bytes_per_element=bytes_per_element,
                    ),
                    fwd.memory(
                        "q_a_norm_bwd",
                        bytes_total=4 * tokens * dims.q_lora_rank * bytes_per_element,
                    ),
                    bwd.matmul_input_grad(
                        "q_a_proj_dgrad",
                        tokens=tokens,
                        input_dim=dims.d_model,
                        output_dim=dims.q_lora_rank,
                        bytes_per_element=bytes_per_element,
                    ),
                ]
            )
        else:
            ops.append(
                bwd.matmul_input_grad(
                    "q_proj_dgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.q_dim,
                    bytes_per_element=bytes_per_element,
                )
            )
        ops.append(
            bwd.rms_norm_grad("attn_norm_bwd", tokens=tokens, dim=dims.d_model, bytes_per_element=bytes_per_element)
        )
        return ops

    def wgrad_ops(self, *, tokens: int, bytes_per_element: int = 2) -> list[DataflowCost]:
        dims = self.dims
        ops = [
            bwd.matmul_weight_grad(
                "o_proj_wgrad",
                tokens=tokens,
                input_dim=dims.o_dim,
                output_dim=dims.d_model,
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "kv_b_proj_wgrad",
                tokens=tokens,
                input_dim=dims.kv_lora_rank,
                output_dim=dims.n_heads * (dims.qk_nope_head_dim + dims.v_head_dim),
                bytes_per_element=bytes_per_element,
            ),
            bwd.matmul_weight_grad(
                "kv_a_proj_with_mqa_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.kv_lora_rank + dims.qk_rope_head_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]
        if dims.q_lora_rank > 0:
            ops.extend(
                [
                    bwd.matmul_weight_grad(
                        "q_b_proj_wgrad",
                        tokens=tokens,
                        input_dim=dims.q_lora_rank,
                        output_dim=dims.q_dim,
                        bytes_per_element=bytes_per_element,
                    ),
                    bwd.matmul_weight_grad(
                        "q_a_proj_wgrad",
                        tokens=tokens,
                        input_dim=dims.d_model,
                        output_dim=dims.q_lora_rank,
                        bytes_per_element=bytes_per_element,
                    ),
                ]
            )
        else:
            ops.append(
                bwd.matmul_weight_grad(
                    "q_proj_wgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.q_dim,
                    bytes_per_element=bytes_per_element,
                )
            )
        return ops

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
