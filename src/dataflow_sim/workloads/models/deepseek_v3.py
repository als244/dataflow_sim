"""DeepSeek-V3 model-family definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models._config import load_model_dims
from dataflow_sim.workloads.modules import (
    DeepSeekBlock,
    DeepSeekDimensions,
    LanguageModelingHead,
    head_params,
    optimizer_ops_for_matrices,
)
from dataflow_sim.workloads.modules.optimizer import (
    matrix_gradient_bytes,
    matrix_weight_bytes,
    optimizer_state_bytes_for_matrices,
)
from dataflow_sim.workloads.training_builder import (
    TrainingBuilder,
    TrainingHeadSpec,
    TrainingLayerSpec,
)


_DEEPSEEK_FIELDS = {
    "vocab_size",
    "n_layers",
    "d_model",
    "head_dim",
    "n_heads",
    "n_kv_heads",
    "expert_dim",
    "num_shared_experts",
    "num_routed_experts",
    "top_k",
    "intermediate_size",
    "first_k_dense_replace",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "routed_scaling_factor",
    "scoring_func",
    "qk_norm",
}

_ALIASES = {
    "671b-37b": "deepseek_v3_671B-37B",
    "deepseek-v3": "deepseek_v3_671B-37B",
    "deepseek_v3": "deepseek_v3_671B-37B",
    "deepseek_v3_671b-37b": "deepseek_v3_671B-37B",
    "deepseek_v3_671B-37B": "deepseek_v3_671B-37B",
}


@dataclass(frozen=True)
class DeepSeekConfig:
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
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    qk_norm: bool = True
    preset_name: str = "custom"

    @classmethod
    def from_model_dims(cls, key: str, **overrides: Any) -> "DeepSeekConfig":
        body = load_model_dims(key)
        kwargs = {field: body[field] for field in _DEEPSEEK_FIELDS if field in body}
        kwargs.update(overrides)
        kwargs.setdefault("preset_name", key)
        return cls(**kwargs)

    def dimensions(self) -> DeepSeekDimensions:
        return DeepSeekDimensions(
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
            intermediate_size=self.intermediate_size,
            first_k_dense_replace=self.first_k_dense_replace,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            routed_scaling_factor=self.routed_scaling_factor,
            scoring_func=self.scoring_func,
            qk_norm=self.qk_norm,
        )


@dataclass(frozen=True)
class DeepSeekV3Config(DeepSeekConfig):
    @classmethod
    def preset(cls, scale: str = "671B-37B", **overrides: Any) -> "DeepSeekV3Config":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(f"unknown DeepSeek-V3 scale {scale!r}; use 671B-37B")
        return cls.from_model_dims(key, **overrides)


def _layer_spec(index: int, dims: DeepSeekDimensions) -> TrainingLayerSpec:
    dense = dims.is_dense_layer(index)
    block = DeepSeekBlock(dims, dense_ffn=dense)
    variant = "dense_prefix" if dense else "moe_suffix"
    block_key = f"deepseek.{variant}_block"
    block_name = "DeepSeek Dense Prefix Block" if dense else "DeepSeek MoE Block"
    matrices = dims.layer_matrices(dense=dense)
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=dims.params_per_layer(dense=dense),
        saved_activation_width=dims.saved_activation_width(dense=dense),
        forward_ops=(
            lambda tokens, seqlen, bpe, block=block: block.forward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bpe,
            )
        ),
        backward_ops=(
            lambda tokens, seqlen, bpe, block=block: block.backward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bpe,
            )
        ),
        recompute_ops=(
            lambda tokens, seqlen, bpe, block=block: block.recompute_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bpe,
            )
        ),
        optimizer_ops=(
            lambda optimizer, bpe, matrices=matrices: optimizer_ops_for_matrices(
                "deepseek_optimizer",
                matrices=matrices,
                optimizer=optimizer,
                bytes_per_element=bpe,
            )
        ),
        parameter_bytes=lambda policy, parallelism, matrices=matrices: matrix_weight_bytes(
            matrices,
            policy,
            parallelism,
        ),
        gradient_bytes=lambda policy, parallelism, matrices=matrices: matrix_gradient_bytes(
            matrices,
            policy,
            parallelism,
        ),
        optimizer_state_bytes=(
            lambda optimizer, policy, parallelism, matrices=matrices: optimizer_state_bytes_for_matrices(
                matrices,
                optimizer,
                policy,
                parallelism,
            )
        ),
        block_key=block_key,
        block_name=block_name,
        optimizer_block_key=f"{block_key}.optimizer_step",
        metadata={
            "deepseek": asdict(dims),
            "dense_ffn": dense,
            "variant": variant,
        },
    )


def _head_spec(dims: DeepSeekDimensions) -> TrainingHeadSpec:
    head_dims = dims.ffn_dimensions(dense=False)
    head = LanguageModelingHead(head_dims)
    return TrainingHeadSpec(
        name="head",
        input_dim=dims.d_model,
        param_count=head_params(head_dims),
        forward_ops=(
            lambda tokens, bpe, head=head: head.forward_ops(
                tokens=tokens,
                bytes_per_element=bpe,
            )
        ),
        backward_ops=(
            lambda tokens, bpe, head=head: head.backward_ops(
                tokens=tokens,
                bytes_per_element=bpe,
            )
        ),
        block_key="lm_head",
        block_name="LM Head",
        metadata={"deepseek": asdict(dims)},
    )


class DeepSeekForTraining(TrainingBuilder):
    metadata_kind = "training.deepseek.modular"

    def __init__(self, config: DeepSeekConfig, *, family_name: str) -> None:
        self.config = config
        self.dims = config.dimensions()
        super().__init__(
            family_name=family_name,
            metadata_kind=self.metadata_kind,
            preset_name=config.preset_name,
            layers=[
                _layer_spec(index, self.dims)
                for index in range(config.n_layers)
            ],
            head=_head_spec(self.dims),
            model_metadata={"deepseek": asdict(self.dims)},
        )


class DeepSeekV3ForTraining(DeepSeekForTraining):
    family_name = "deepseek_v3"

    def __init__(self, config: DeepSeekV3Config) -> None:
        super().__init__(config, family_name=self.family_name)
