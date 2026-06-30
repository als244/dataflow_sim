"""Dimensions for DeepSeek-V3.2 DSA/MLA models."""
from __future__ import annotations

from dataclasses import dataclass
import math

from dataflow_sim.workloads.dataflow_builder import DTypePolicy, dtype_nbytes
from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.ops import optimizer as opt_ops
from dataflow_sim.workloads.ops._common import topk_attention_score_terms


INDEX_BYTES = 4


@dataclass(frozen=True)
class DeepSeekV32Dimensions:
    vocab_size: int
    n_layers: int
    d_model: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    expert_dim: int
    num_shared_experts: int
    num_routed_experts: int
    top_k: int
    intermediate_size: int
    first_k_dense_replace: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    qk_norm: bool = True
    train_indexer: bool = True

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.qk_head_dim

    @property
    def index_q_dim(self) -> int:
        return self.index_n_heads * self.index_head_dim

    @property
    def o_dim(self) -> int:
        return self.n_heads * self.v_head_dim

    def is_dense_layer(self, index: int) -> bool:
        return index < self.first_k_dense_replace

    def ffn_dimensions(self, *, dense: bool) -> TransformerDimensions:
        if dense:
            return TransformerDimensions(
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                d_model=self.d_model,
                head_dim=self.head_dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                expert_dim=self.intermediate_size,
                num_shared_experts=1,
                num_routed_experts=0,
                top_k=0,
                qk_norm=self.qk_norm,
            )
        return TransformerDimensions(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            d_model=self.d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            expert_dim=self.expert_dim,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            top_k=self.top_k,
            qk_norm=self.qk_norm,
        )

    def attention_matrices(self, *, include_indexer: bool = True) -> list[opt_ops.OptimizerMatrix]:
        matrices: list[opt_ops.OptimizerMatrix] = []

        def add(
            name: str,
            rows: int,
            cols: int,
            count: int = 1,
            *,
            expert: bool = False,
            ep_sharded: bool = False,
        ) -> None:
            if rows > 0 and cols > 0 and count > 0:
                matrices.append(
                    opt_ops.OptimizerMatrix(name, rows, cols, count, expert, ep_sharded)
                )

        add("q_a_proj", self.d_model, self.q_lora_rank)
        add("q_b_proj", self.q_lora_rank, self.q_dim)
        if include_indexer:
            add("index_q_b_proj", self.q_lora_rank, self.index_q_dim)
            add("index_k_proj", self.d_model, self.index_head_dim)
            add("index_weight_proj", self.d_model, self.index_n_heads)
        add("kv_a_proj_with_mqa", self.d_model, self.kv_lora_rank + self.qk_rope_head_dim)
        add("kv_b_proj", self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        add("o_proj", self.o_dim, self.d_model)
        return matrices

    def ffn_matrices(self, *, dense: bool) -> list[opt_ops.OptimizerMatrix]:
        ffn_dims = self.ffn_dimensions(dense=dense)
        matrices: list[opt_ops.OptimizerMatrix] = []

        def add(
            name: str,
            rows: int,
            cols: int,
            count: int = 1,
            *,
            expert: bool = False,
            ep_sharded: bool = False,
        ) -> None:
            if rows > 0 and cols > 0 and count > 0:
                matrices.append(
                    opt_ops.OptimizerMatrix(name, rows, cols, count, expert, ep_sharded)
                )

        is_moe = not dense and ffn_dims.num_routed_experts > 0 and ffn_dims.top_k > 0
        add("shared_mlp_gate", ffn_dims.d_model, ffn_dims.expert_dim, ffn_dims.num_shared_experts, expert=is_moe)
        add("shared_mlp_up", ffn_dims.d_model, ffn_dims.expert_dim, ffn_dims.num_shared_experts, expert=is_moe)
        add("shared_mlp_down", ffn_dims.expert_dim, ffn_dims.d_model, ffn_dims.num_shared_experts, expert=is_moe)
        add(
            "routed_mlp_gate",
            ffn_dims.d_model,
            ffn_dims.expert_dim,
            ffn_dims.num_routed_experts,
            expert=is_moe,
            ep_sharded=is_moe,
        )
        add(
            "routed_mlp_up",
            ffn_dims.d_model,
            ffn_dims.expert_dim,
            ffn_dims.num_routed_experts,
            expert=is_moe,
            ep_sharded=is_moe,
        )
        add(
            "routed_mlp_down",
            ffn_dims.expert_dim,
            ffn_dims.d_model,
            ffn_dims.num_routed_experts,
            expert=is_moe,
            ep_sharded=is_moe,
        )
        return matrices

    def layer_matrices(
        self,
        *,
        dense: bool,
        trainable_only: bool = False,
        include_indexer: bool = True,
    ) -> list[opt_ops.OptimizerMatrix]:
        include_indexer_matrices = include_indexer and (self.train_indexer or not trainable_only)
        return (
            self.attention_matrices(include_indexer=include_indexer_matrices)
            + self.ffn_matrices(dense=dense)
        )

    def params_per_layer(self, *, dense: bool, include_indexer: bool = True) -> int:
        return sum(
            matrix.rows * matrix.cols * matrix.count
            for matrix in self.layer_matrices(dense=dense, include_indexer=include_indexer)
        )

    def trainable_params_per_layer(self, *, dense: bool, include_indexer: bool = True) -> int:
        return sum(
            matrix.rows * matrix.cols * matrix.count
            for matrix in self.layer_matrices(
                dense=dense,
                trainable_only=True,
                include_indexer=include_indexer,
            )
        )

    def saved_activation_width(self, *, dense: bool, include_indexer: bool = True) -> int:
        ffn_dims = self.ffn_dimensions(dense=dense)
        ffn_width = 2 * (ffn_dims.num_shared_experts + ffn_dims.top_k) * ffn_dims.expert_dim
        attn_width = (
            self.q_lora_rank
            + self.q_dim
            + self.kv_lora_rank
            + self.qk_rope_head_dim
            + self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
            + self.o_dim
        )
        if include_indexer and self.train_indexer:
            attn_width += self.indexer_saved_activation_width()
        return attn_width + 2 * self.d_model + ffn_width

    def indexer_saved_activation_width(self) -> int:
        if not self.train_indexer:
            return 0
        return self.index_q_dim + self.index_head_dim + self.index_n_heads

    def saved_activation_bytes(
        self,
        *,
        tokens: int,
        seqlen: int,
        dense: bool,
        policy: DTypePolicy,
        include_indexer: bool = True,
    ) -> int:
        indexer_width = self.indexer_saved_activation_width() if include_indexer else 0
        regular_width = self.saved_activation_width(
            dense=dense,
            include_indexer=include_indexer,
        ) - indexer_width
        activation_bytes = (
            tokens
            * regular_width
            * dtype_nbytes(policy.activation)
        )
        indexer_activation_bytes = (
            tokens
            * indexer_width
            * dtype_nbytes(policy.indexer_activation)
        )
        selection_bytes = 0
        if include_indexer:
            selected_terms = topk_attention_score_terms(
                tokens,
                top_k=self.index_topk,
                seqlen=seqlen,
            )
            selection_bytes = selected_terms * INDEX_BYTES
            if self.train_indexer:
                selection_bytes += selected_terms * dtype_nbytes(policy.indexer_activation)
        return math.ceil(activation_bytes + indexer_activation_bytes + selection_bytes)
