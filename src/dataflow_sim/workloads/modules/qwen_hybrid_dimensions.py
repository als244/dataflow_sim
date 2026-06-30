"""Dimensions for Qwen3.5/Qwen3.6 hybrid models."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.ops import optimizer as opt_ops


@dataclass(frozen=True)
class QwenHybridDimensions:
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
    layer_types: tuple[str, ...]
    full_attention_interval: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    gdn_chunk_size: int = 64
    qk_norm: bool = True
    router_aux_loss_coef: float = 0.0
    mtp_num_hidden_layers: int = 0

    @property
    def is_moe(self) -> bool:
        return self.num_routed_experts > 0 and self.top_k > 0

    @property
    def full_q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def full_kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

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

    def attention_matrices(self, layer_type: str) -> list[opt_ops.OptimizerMatrix]:
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

        if layer_type == "linear_attention":
            add("in_proj_q", self.d_model, self.linear_key_dim)
            add("in_proj_k", self.d_model, self.linear_key_dim)
            add("in_proj_v", self.d_model, self.linear_value_dim)
            add("in_proj_z", self.d_model, self.linear_value_dim)
            add("in_proj_ba", self.d_model, 2 * self.linear_num_value_heads)
            add("causal_conv1d", self.linear_conv_dim, self.linear_conv_kernel_dim)
            add("linear_out_proj", self.linear_value_dim, self.d_model)
            return matrices
        if layer_type == "full_attention":
            add("q_proj", self.d_model, self.full_q_dim)
            add("q_gate_proj", self.d_model, self.full_q_dim)
            add("k_proj", self.d_model, self.full_kv_dim)
            add("v_proj", self.d_model, self.full_kv_dim)
            add("o_proj", self.full_q_dim, self.d_model)
            return matrices
        raise ValueError(f"unknown Qwen hybrid layer type {layer_type!r}")

    def ffn_matrices(self) -> list[opt_ops.OptimizerMatrix]:
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

        add("shared_mlp_gate", self.d_model, self.expert_dim, self.num_shared_experts, expert=self.is_moe)
        add("shared_mlp_up", self.d_model, self.expert_dim, self.num_shared_experts, expert=self.is_moe)
        add("shared_mlp_down", self.expert_dim, self.d_model, self.num_shared_experts, expert=self.is_moe)
        add(
            "routed_mlp_gate",
            self.d_model,
            self.expert_dim,
            self.num_routed_experts,
            expert=self.is_moe,
            ep_sharded=self.is_moe,
        )
        add(
            "routed_mlp_up",
            self.d_model,
            self.expert_dim,
            self.num_routed_experts,
            expert=self.is_moe,
            ep_sharded=self.is_moe,
        )
        add(
            "routed_mlp_down",
            self.expert_dim,
            self.d_model,
            self.num_routed_experts,
            expert=self.is_moe,
            ep_sharded=self.is_moe,
        )
        return matrices

    def layer_matrices(self, layer_type: str) -> list[opt_ops.OptimizerMatrix]:
        return self.attention_matrices(layer_type) + self.ffn_matrices()

    def params_per_layer(self, layer_type: str) -> int:
        return sum(matrix.rows * matrix.cols * matrix.count for matrix in self.layer_matrices(layer_type))

    def saved_activation_width(self, layer_type: str) -> int:
        ffn_width = 2 * (self.num_shared_experts + self.top_k) * self.expert_dim
        if layer_type == "linear_attention":
            attn_width = (
                2 * self.linear_key_dim
                + 2 * self.linear_value_dim
                + self.linear_num_value_heads * self.gdn_chunk_size
                + self.linear_value_dim
                + 2 * self.linear_num_value_heads
            )
        elif layer_type == "full_attention":
            attn_width = 3 * self.full_q_dim + 2 * self.full_kv_dim
        else:
            raise ValueError(f"unknown Qwen hybrid layer type {layer_type!r}")
        return attn_width + 2 * self.d_model + ffn_width
