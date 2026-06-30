"""Shared transformer-family dimensions and byte/count helpers."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.workloads.ops import optimizer as opt_ops


@dataclass(frozen=True)
class TransformerDimensions:
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
    qk_norm: bool = True


def layer_weight_matrices(dims: TransformerDimensions) -> list[opt_ops.OptimizerMatrix]:
    matrices: list[opt_ops.OptimizerMatrix] = []

    is_moe = dims.num_routed_experts > 0 and dims.top_k > 0

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

    d = dims.d_model
    hd = dims.head_dim
    add("q_proj", d, dims.n_heads * hd)
    add("k_proj", d, dims.n_kv_heads * hd)
    add("v_proj", d, dims.n_kv_heads * hd)
    add("attn_proj", dims.n_heads * hd, d)
    add("shared_mlp_gate", d, dims.expert_dim, dims.num_shared_experts, expert=is_moe)
    add("shared_mlp_up", d, dims.expert_dim, dims.num_shared_experts, expert=is_moe)
    add("shared_mlp_down", dims.expert_dim, d, dims.num_shared_experts, expert=is_moe)
    add(
        "routed_mlp_gate",
        d,
        dims.expert_dim,
        dims.num_routed_experts,
        expert=is_moe,
        ep_sharded=is_moe,
    )
    add(
        "routed_mlp_up",
        d,
        dims.expert_dim,
        dims.num_routed_experts,
        expert=is_moe,
        ep_sharded=is_moe,
    )
    add(
        "routed_mlp_down",
        dims.expert_dim,
        d,
        dims.num_routed_experts,
        expert=is_moe,
        ep_sharded=is_moe,
    )
    return matrices


def params_per_layer(dims: TransformerDimensions) -> int:
    return sum(matrix.rows * matrix.cols * matrix.count for matrix in layer_weight_matrices(dims))


def active_params_per_layer(dims: TransformerDimensions) -> int:
    attn = dims.head_dim * (2 * dims.n_heads + 2 * dims.n_kv_heads)
    mlp = 3 * dims.expert_dim * (dims.num_shared_experts + dims.top_k)
    return dims.d_model * (attn + mlp)


def head_params(dims: TransformerDimensions) -> int:
    return dims.d_model * dims.vocab_size


def layer_activation_elements_per_token(dims: TransformerDimensions) -> int:
    return (
        dims.head_dim * (2 * dims.n_heads + 2 * dims.n_kv_heads)
        + 2 * dims.d_model
        + 2 * (dims.num_shared_experts + dims.top_k) * dims.expert_dim
    )
