"""Dimensions for NVIDIA Nemotron-H hybrid models."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.workloads.modules.dimensions import TransformerDimensions
from dataflow_sim.workloads.ops import optimizer as opt_ops


_LAYER_TYPE_ALIASES = {
    "M": "mamba",
    "mamba": "mamba",
    "*": "attention",
    "attention": "attention",
    "E": "moe",
    "moe": "moe",
    "-": "mlp",
    "mlp": "mlp",
}


def normalize_nemotron_layer_type(layer_type: str) -> str:
    try:
        return _LAYER_TYPE_ALIASES[layer_type]
    except KeyError as exc:
        raise ValueError(f"unknown Nemotron-H layer type {layer_type!r}") from exc


def parse_hybrid_pattern(pattern: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(pattern, str):
        return tuple(
            normalize_nemotron_layer_type(token)
            for token in pattern
            if not token.isspace()
        )
    return tuple(normalize_nemotron_layer_type(token) for token in pattern)


@dataclass(frozen=True)
class NemotronDimensions:
    vocab_size: int
    n_layers: int
    d_model: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    expert_dim: int
    shared_expert_dim: int
    num_shared_experts: int
    num_routed_experts: int
    top_k: int
    intermediate_size: int
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    conv_kernel: int
    mamba_chunk_size: int
    n_groups: int
    layer_types: tuple[str, ...]
    hybrid_override_pattern: str
    qk_norm: bool = False

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.n_layers:
            raise ValueError("layer_types length must match n_layers")

    @property
    def attention_q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def attention_kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def mamba_intermediate_dim(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_dim(self) -> int:
        return self.mamba_intermediate_dim + 2 * self.n_groups * self.ssm_state_size

    @property
    def mamba_projection_dim(self) -> int:
        return 2 * self.mamba_intermediate_dim + 2 * self.n_groups * self.ssm_state_size + self.mamba_num_heads

    @property
    def is_moe(self) -> bool:
        return self.num_routed_experts > 0 and self.top_k > 0

    def head_dimensions(self) -> TransformerDimensions:
        return TransformerDimensions(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            d_model=self.d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            expert_dim=max(self.expert_dim, 1),
            num_shared_experts=0,
            num_routed_experts=0,
            top_k=0,
            qk_norm=self.qk_norm,
        )

    def _add_matrix(
        self,
        matrices: list[opt_ops.OptimizerMatrix],
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

    def mamba_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        matrices: list[opt_ops.OptimizerMatrix] = []
        self._add_matrix(matrices, "mamba_in_proj", self.d_model, self.mamba_projection_dim)
        self._add_matrix(matrices, "mamba_depthwise_conv1d", self.mamba_conv_dim, self.conv_kernel)
        self._add_matrix(matrices, "mamba_out_proj", self.mamba_intermediate_dim, self.d_model)
        return matrices

    def attention_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        matrices: list[opt_ops.OptimizerMatrix] = []
        self._add_matrix(matrices, "q_proj", self.d_model, self.attention_q_dim)
        self._add_matrix(matrices, "k_proj", self.d_model, self.attention_kv_dim)
        self._add_matrix(matrices, "v_proj", self.d_model, self.attention_kv_dim)
        self._add_matrix(matrices, "o_proj", self.attention_q_dim, self.d_model)
        return matrices

    def mlp_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        matrices: list[opt_ops.OptimizerMatrix] = []
        self._add_matrix(matrices, "mlp_up", self.d_model, self.intermediate_size)
        self._add_matrix(matrices, "mlp_down", self.intermediate_size, self.d_model)
        return matrices

    def moe_matrices(self) -> list[opt_ops.OptimizerMatrix]:
        matrices: list[opt_ops.OptimizerMatrix] = []
        self._add_matrix(
            matrices,
            "routed_mlp_up",
            self.d_model,
            self.expert_dim,
            self.num_routed_experts,
            expert=True,
            ep_sharded=True,
        )
        self._add_matrix(
            matrices,
            "routed_mlp_down",
            self.expert_dim,
            self.d_model,
            self.num_routed_experts,
            expert=True,
            ep_sharded=True,
        )
        self._add_matrix(
            matrices,
            "shared_mlp_up",
            self.d_model,
            self.shared_expert_dim,
            self.num_shared_experts,
            expert=True,
        )
        self._add_matrix(
            matrices,
            "shared_mlp_down",
            self.shared_expert_dim,
            self.d_model,
            self.num_shared_experts,
            expert=True,
        )
        return matrices

    def layer_matrices(self, layer_type: str) -> list[opt_ops.OptimizerMatrix]:
        normalized = normalize_nemotron_layer_type(layer_type)
        if normalized == "mamba":
            return self.mamba_matrices()
        if normalized == "attention":
            return self.attention_matrices()
        if normalized == "moe":
            return self.moe_matrices()
        if normalized == "mlp":
            return self.mlp_matrices()
        raise ValueError(f"unknown Nemotron-H layer type {layer_type!r}")

    def params_per_layer(self, layer_type: str) -> int:
        return sum(matrix.rows * matrix.cols * matrix.count for matrix in self.layer_matrices(layer_type))

    def saved_activation_width(self, layer_type: str) -> int:
        normalized = normalize_nemotron_layer_type(layer_type)
        if normalized == "mamba":
            return (
                self.mamba_projection_dim
                + self.mamba_conv_dim
                + self.mamba_num_heads * self.mamba_chunk_size
                + 2 * self.mamba_intermediate_dim
                + 2 * self.d_model
            )
        if normalized == "attention":
            return self.attention_q_dim + 2 * self.attention_kv_dim + self.attention_q_dim + 2 * self.d_model
        if normalized == "moe":
            routed_width = self.top_k * (self.expert_dim + self.d_model)
            shared_width = self.num_shared_experts * (self.shared_expert_dim + self.d_model)
            return routed_width + shared_width + 2 * self.d_model
        if normalized == "mlp":
            return 2 * self.intermediate_size + 2 * self.d_model
        raise ValueError(f"unknown Nemotron-H layer type {layer_type!r}")
