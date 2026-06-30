"""Dimensions for OpenAI GPT-OSS models."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.ops import optimizer as opt_ops


@dataclass(frozen=True)
class GPTOSSDimensions:
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
    sliding_window: int
    layer_types: tuple[str, ...]
    qk_norm: bool = False

    @property
    def attn_q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def attn_kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.attn_q_dim + 2 * self.attn_kv_dim

    def ffn_dimensions(self) -> TransformerDimensions:
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

    def layer_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        matrices: list[opt_ops.OptimizerMatrix] = []
        is_moe = self.num_routed_experts > 0 and self.top_k > 0

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

        add("q_proj", self.d_model, self.attn_q_dim)
        add("k_proj", self.d_model, self.attn_kv_dim)
        add("v_proj", self.d_model, self.attn_kv_dim)
        add("attn_proj", self.attn_q_dim, self.d_model)
        add("shared_mlp_gate", self.d_model, self.expert_dim, self.num_shared_experts, expert=is_moe)
        add("shared_mlp_up", self.d_model, self.expert_dim, self.num_shared_experts, expert=is_moe)
        add("shared_mlp_down", self.expert_dim, self.d_model, self.num_shared_experts, expert=is_moe)
        add(
            "routed_mlp_gate",
            self.d_model,
            self.expert_dim,
            self.num_routed_experts,
            expert=is_moe,
            ep_sharded=is_moe,
        )
        add(
            "routed_mlp_up",
            self.d_model,
            self.expert_dim,
            self.num_routed_experts,
            expert=is_moe,
            ep_sharded=is_moe,
        )
        add(
            "routed_mlp_down",
            self.expert_dim,
            self.d_model,
            self.num_routed_experts,
            expert=is_moe,
            ep_sharded=is_moe,
        )
        return matrices

    def params_per_layer(self) -> int:
        return sum(matrix.rows * matrix.cols * matrix.count for matrix in self.layer_matrices())

    def saved_activation_width(self) -> int:
        ffn_width = 2 * (self.num_shared_experts + self.top_k) * self.expert_dim
        attn_width = 2 * self.attn_q_dim + 2 * self.attn_kv_dim
        return attn_width + 2 * self.d_model + ffn_width
