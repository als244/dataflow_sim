"""DeepSeek-V3.2 DeepSeek Sparse Attention module."""
from __future__ import annotations

from typing import Literal

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.modules.deepseek_v3_2_dimensions import DeepSeekV32Dimensions
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


IndexerMode = Literal["full", "shared"]


class DSASparseAttention(DataflowModule):
    def __init__(self, dims: DeepSeekV32Dimensions, *, indexer_mode: IndexerMode = "full") -> None:
        super().__init__(name="DSASparseAttention")
        if indexer_mode not in ("full", "shared"):
            raise ValueError(f"unknown DSA indexer mode: {indexer_mode!r}")
        self.dims = dims
        self.indexer_mode = indexer_mode

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
        ops = [
            fwd.rms_norm("attn_norm", tokens=tokens, dim=dims.d_model, bytes_per_element=policy.activation_bpe),
            fwd.matmul(
                "q_a_proj",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.q_lora_rank,
                bytes_per_element=policy,
            ),
            fwd.memory(
                "q_a_norm",
                bytes_total=2 * tokens * dims.q_lora_rank * policy.activation_bpe,
            ),
            fwd.matmul(
                "q_b_proj",
                tokens=tokens,
                input_dim=dims.q_lora_rank,
                output_dim=dims.q_dim,
                bytes_per_element=policy,
            ),
        ]
        if self.indexer_mode == "full":
            ops.extend(
                [
                    fwd.matmul(
                        "index_q_b_proj",
                        tokens=tokens,
                        input_dim=dims.q_lora_rank,
                        output_dim=dims.index_q_dim,
                        bytes_per_element=policy,
                        output_bytes_per_element=policy.indexer_activation_bpe,
                    ),
                    fwd.matmul(
                        "index_k_proj",
                        tokens=tokens,
                        input_dim=dims.d_model,
                        output_dim=dims.index_head_dim,
                        bytes_per_element=policy,
                        output_bytes_per_element=policy.indexer_activation_bpe,
                    ),
                    fwd.matmul(
                        "index_weight_proj",
                        tokens=tokens,
                        input_dim=dims.d_model,
                        output_dim=dims.index_n_heads,
                        bytes_per_element=policy,
                        output_bytes_per_element=policy.indexer_activation_bpe,
                    ),
                    fwd.lightning_index_score(
                        "lightning_index_score",
                        tokens=tokens,
                        index_n_heads=dims.index_n_heads,
                        index_head_dim=dims.index_head_dim,
                        index_topk=dims.index_topk,
                        seqlen=seqlen,
                        bytes_per_element=policy,
                        save_selected_scores=dims.train_indexer,
                    ),
                ]
            )
        ops.extend([
            fwd.matmul(
                "kv_a_proj_with_mqa",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.kv_lora_rank + dims.qk_rope_head_dim,
                bytes_per_element=policy,
            ),
            fwd.memory(
                "kv_a_norm",
                bytes_total=2 * tokens * dims.kv_lora_rank * policy.activation_bpe,
            ),
            fwd.matmul(
                "kv_b_proj",
                tokens=tokens,
                input_dim=dims.kv_lora_rank,
                output_dim=dims.n_heads * (dims.qk_nope_head_dim + dims.v_head_dim),
                bytes_per_element=policy,
            ),
            fwd.mla_rope(
                "dsa_rope",
                tokens=tokens,
                rope_head_dim=dims.qk_rope_head_dim,
                n_heads=dims.n_heads,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.dsa_sparse_attention(
                "dsa_sparse_attn",
                tokens=tokens,
                n_heads=dims.n_heads,
                kv_lora_rank=dims.kv_lora_rank,
                rope_head_dim=dims.qk_rope_head_dim,
                value_head_dim=dims.v_head_dim,
                index_topk=dims.index_topk,
                seqlen=seqlen,
                bytes_per_element=policy.activation_bpe,
            ),
            fwd.matmul(
                "o_proj",
                tokens=tokens,
                input_dim=dims.o_dim,
                output_dim=dims.d_model,
                bytes_per_element=policy,
                accumulate=True,
            ),
        ])
        return ops

    def dgrad_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        ops = [
            bwd.matmul_input_grad(
                "o_proj_dgrad",
                tokens=tokens,
                input_dim=dims.o_dim,
                output_dim=dims.d_model,
                bytes_per_element=policy,
            ),
            bwd.dsa_sparse_attention_grad(
                "dsa_sparse_attn_bwd",
                tokens=tokens,
                n_heads=dims.n_heads,
                kv_lora_rank=dims.kv_lora_rank,
                rope_head_dim=dims.qk_rope_head_dim,
                value_head_dim=dims.v_head_dim,
                index_topk=dims.index_topk,
                seqlen=seqlen,
                bytes_per_element=policy.activation_bpe,
            ),
        ]
        if self.indexer_mode == "full" and dims.train_indexer:
            ops.append(
                bwd.lightning_index_score_grad(
                    "lightning_index_score_bwd",
                    tokens=tokens,
                    index_n_heads=dims.index_n_heads,
                    index_head_dim=dims.index_head_dim,
                    index_topk=dims.index_topk,
                    seqlen=seqlen,
                    bytes_per_element=policy,
                )
            )
        ops.extend([
            bwd.mla_rope_grad(
                "dsa_rope_bwd",
                tokens=tokens,
                rope_head_dim=dims.qk_rope_head_dim,
                n_heads=dims.n_heads,
                bytes_per_element=policy.activation_bpe,
            ),
            bwd.matmul_input_grad(
                "kv_b_proj_dgrad",
                tokens=tokens,
                input_dim=dims.kv_lora_rank,
                output_dim=dims.n_heads * (dims.qk_nope_head_dim + dims.v_head_dim),
                bytes_per_element=policy,
            ),
            fwd.memory(
                "kv_a_norm_bwd",
                bytes_total=4 * tokens * dims.kv_lora_rank * policy.activation_bpe,
            ),
            bwd.matmul_input_grad(
                "kv_a_proj_with_mqa_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.kv_lora_rank + dims.qk_rope_head_dim,
                bytes_per_element=policy,
            ),
            bwd.matmul_input_grad(
                "q_b_proj_dgrad",
                tokens=tokens,
                input_dim=dims.q_lora_rank,
                output_dim=dims.q_dim,
                bytes_per_element=policy,
            ),
            fwd.memory(
                "q_a_norm_bwd",
                bytes_total=4 * tokens * dims.q_lora_rank * policy.activation_bpe,
            ),
            bwd.matmul_input_grad(
                "q_a_proj_dgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.q_lora_rank,
                bytes_per_element=policy,
            ),
            bwd.rms_norm_grad("attn_norm_bwd", tokens=tokens, dim=dims.d_model, bytes_per_element=policy.activation_bpe),
        ])
        return ops

    def wgrad_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        dims = self.dims
        policy = self._policy(bytes_per_element)
        ops = [
            bwd.matmul_weight_grad(
                "o_proj_wgrad",
                tokens=tokens,
                input_dim=dims.o_dim,
                output_dim=dims.d_model,
                bytes_per_element=policy,
            ),
            bwd.matmul_weight_grad(
                "kv_b_proj_wgrad",
                tokens=tokens,
                input_dim=dims.kv_lora_rank,
                output_dim=dims.n_heads * (dims.qk_nope_head_dim + dims.v_head_dim),
                bytes_per_element=policy,
            ),
            bwd.matmul_weight_grad(
                "kv_a_proj_with_mqa_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.kv_lora_rank + dims.qk_rope_head_dim,
                bytes_per_element=policy,
            ),
            bwd.matmul_weight_grad(
                "q_b_proj_wgrad",
                tokens=tokens,
                input_dim=dims.q_lora_rank,
                output_dim=dims.q_dim,
                bytes_per_element=policy,
            ),
            bwd.matmul_weight_grad(
                "q_a_proj_wgrad",
                tokens=tokens,
                input_dim=dims.d_model,
                output_dim=dims.q_lora_rank,
                bytes_per_element=policy,
            ),
        ]
        if self.indexer_mode == "full" and dims.train_indexer:
            ops.extend([
                bwd.matmul_weight_grad(
                    "index_q_b_proj_wgrad",
                    tokens=tokens,
                    input_dim=dims.q_lora_rank,
                    output_dim=dims.index_q_dim,
                    bytes_per_element=policy,
                    upstream_gradient_bytes_per_element=policy.indexer_activation_bpe,
                ),
                bwd.matmul_weight_grad(
                    "index_k_proj_wgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.index_head_dim,
                    bytes_per_element=policy,
                    upstream_gradient_bytes_per_element=policy.indexer_activation_bpe,
                ),
                bwd.matmul_weight_grad(
                    "index_weight_proj_wgrad",
                    tokens=tokens,
                    input_dim=dims.d_model,
                    output_dim=dims.index_n_heads,
                    bytes_per_element=policy,
                    upstream_gradient_bytes_per_element=policy.indexer_activation_bpe,
                ),
            ])
        return ops

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
