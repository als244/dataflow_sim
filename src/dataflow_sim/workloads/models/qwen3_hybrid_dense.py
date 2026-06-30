"""Qwen3.5/Qwen3.6 hybrid dense model-family definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models._config import load_model_dims
from dataflow_sim.workloads.modules import (
    LanguageModelingHead,
    QwenHybridBlock,
    QwenHybridDimensions,
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


_QWEN_HYBRID_FIELDS = {
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
    "layer_types",
    "full_attention_interval",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "gdn_chunk_size",
    "qk_norm",
    "router_aux_loss_coef",
    "mtp_num_hidden_layers",
}

_DENSE_ALIASES = {
    "9b": "qwen3_5_9B",
    "qwen3_5_9b": "qwen3_5_9B",
    "qwen3.5-9b": "qwen3_5_9B",
    "27b": "qwen3_5_27B",
    "qwen3_5_27b": "qwen3_5_27B",
    "qwen3.5-27b": "qwen3_5_27B",
    "qwen3_6_27b": "qwen3_5_27B",
    "qwen3.6-27b": "qwen3_5_27B",
    "qwen3_5_9B": "qwen3_5_9B",
    "qwen3_5_27B": "qwen3_5_27B",
    "qwen3_6_27B": "qwen3_5_27B",
}


def _layer_pattern(n_layers: int, full_attention_interval: int) -> tuple[str, ...]:
    if full_attention_interval <= 0:
        raise ValueError("full_attention_interval must be positive")
    return tuple(
        "full_attention" if (index + 1) % full_attention_interval == 0 else "linear_attention"
        for index in range(n_layers)
    )


@dataclass(frozen=True)
class QwenHybridConfig:
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
    layer_types: tuple[str, ...] | None = None
    preset_name: str = "custom"

    @classmethod
    def from_model_dims(cls, key: str, **overrides: Any) -> "QwenHybridConfig":
        body = load_model_dims(key)
        kwargs = {field: body[field] for field in _QWEN_HYBRID_FIELDS if field in body}
        kwargs.update(overrides)
        kwargs.setdefault("preset_name", key)
        layer_types = kwargs.get("layer_types")
        if layer_types is None:
            kwargs["layer_types"] = _layer_pattern(
                int(kwargs["n_layers"]),
                int(kwargs["full_attention_interval"]),
            )
        else:
            kwargs["layer_types"] = tuple(layer_types)
        return cls(**kwargs)

    def dimensions(self) -> QwenHybridDimensions:
        layer_types = self.layer_types or _layer_pattern(
            self.n_layers,
            self.full_attention_interval,
        )
        if len(layer_types) != self.n_layers:
            raise ValueError("layer_types length must match n_layers")
        return QwenHybridDimensions(
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
            layer_types=tuple(layer_types),
            full_attention_interval=self.full_attention_interval,
            linear_num_key_heads=self.linear_num_key_heads,
            linear_key_head_dim=self.linear_key_head_dim,
            linear_num_value_heads=self.linear_num_value_heads,
            linear_value_head_dim=self.linear_value_head_dim,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            gdn_chunk_size=self.gdn_chunk_size,
            qk_norm=self.qk_norm,
            router_aux_loss_coef=self.router_aux_loss_coef,
            mtp_num_hidden_layers=self.mtp_num_hidden_layers,
        )


@dataclass(frozen=True)
class QwenHybridDenseConfig(QwenHybridConfig):
    @classmethod
    def preset(cls, scale: str = "9B", **overrides: Any) -> "QwenHybridDenseConfig":
        key = _DENSE_ALIASES.get(scale, _DENSE_ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                f"unknown Qwen hybrid dense scale {scale!r}; use 9B, 27B, or qwen3_5_27B"
            )
        return cls.from_model_dims(key, **overrides)


def _layer_spec(index: int, dims: QwenHybridDimensions) -> TrainingLayerSpec:
    layer_type = dims.layer_types[index]
    block = QwenHybridBlock(dims, layer_type)
    variant = "moe" if dims.is_moe else "dense"
    attention = "linear" if layer_type == "linear_attention" else "full"
    block_key = f"qwen_hybrid.{variant}.{attention}_block"
    block_name = f"Qwen Hybrid {attention.title()} {variant.upper()} Block"
    matrices = dims.layer_matrices(layer_type)
    param_count = dims.params_per_layer(layer_type)
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=param_count,
        saved_activation_width=dims.saved_activation_width(layer_type),
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
                "qwen_hybrid_optimizer",
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
            "qwen_hybrid": asdict(dims),
            "layer_type": layer_type,
            "variant": variant,
        },
    )


def _head_spec(dims: QwenHybridDimensions) -> TrainingHeadSpec:
    head_dims = dims.ffn_dimensions()
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
        metadata={"qwen_hybrid": asdict(dims)},
    )


class QwenHybridForTraining(TrainingBuilder):
    metadata_kind = "training.qwen_hybrid.modular"

    def __init__(self, config: QwenHybridConfig, *, family_name: str) -> None:
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
            model_metadata={"qwen_hybrid": asdict(self.dims)},
        )


class QwenHybridDenseForTraining(QwenHybridForTraining):
    family_name = "qwen3_hybrid_dense"

    def __init__(self, config: QwenHybridDenseConfig) -> None:
        super().__init__(config, family_name=self.family_name)
