"""OpenAI GPT-OSS model-family definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models._config import load_model_dims
from dataflow_sim.workloads.modules import (
    GPTOSSBlock,
    GPTOSSDimensions,
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


_GPT_OSS_FIELDS = {
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
    "sliding_window",
    "layer_types",
    "qk_norm",
}

_ALIASES = {
    "20b": "gpt_oss_20B",
    "gpt-oss-20b": "gpt_oss_20B",
    "gpt_oss_20b": "gpt_oss_20B",
    "gpt_oss_20B": "gpt_oss_20B",
    "120b": "gpt_oss_120B",
    "gpt-oss-120b": "gpt_oss_120B",
    "gpt_oss_120b": "gpt_oss_120B",
    "gpt_oss_120B": "gpt_oss_120B",
}

_FIELD_ALIASES = {
    "hidden_size": "d_model",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "intermediate_size": "expert_dim",
    "num_local_experts": "num_routed_experts",
    "num_experts_per_tok": "top_k",
}


def _layer_pattern(n_layers: int) -> tuple[str, ...]:
    return tuple(
        "sliding_attention" if index % 2 == 0 else "full_attention"
        for index in range(n_layers)
    )


def _config_kwargs(body: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, value in body.items():
        normalized = _FIELD_ALIASES.get(key, key)
        if normalized in _GPT_OSS_FIELDS:
            kwargs[normalized] = value
    for key, value in overrides.items():
        normalized = _FIELD_ALIASES.get(key, key)
        kwargs[normalized] = value
    return kwargs


@dataclass(frozen=True)
class GPTOSSConfig:
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
    qk_norm: bool = False
    layer_types: tuple[str, ...] | None = None
    preset_name: str = "custom"

    @classmethod
    def from_model_dims(cls, key: str, **overrides: Any) -> "GPTOSSConfig":
        body = load_model_dims(key)
        kwargs = _config_kwargs(body, overrides)
        kwargs.setdefault("preset_name", key)
        layer_types = kwargs.get("layer_types")
        if layer_types is None:
            kwargs["layer_types"] = _layer_pattern(int(kwargs["n_layers"]))
        else:
            kwargs["layer_types"] = tuple(layer_types)
        return cls(**kwargs)

    @classmethod
    def preset(cls, scale: str = "120B", **overrides: Any) -> "GPTOSSConfig":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                f"unknown GPT-OSS scale {scale!r}; use 20B, 120B, or a gpt_oss_* preset key"
            )
        return cls.from_model_dims(key, **overrides)

    def dimensions(self) -> GPTOSSDimensions:
        layer_types = self.layer_types or _layer_pattern(self.n_layers)
        if len(layer_types) != self.n_layers:
            raise ValueError(
                "layer_types length must match n_layers; "
                f"got {len(layer_types)} entries for {self.n_layers} layers"
            )
        unknown = sorted(set(layer_types) - {"sliding_attention", "full_attention"})
        if unknown:
            raise ValueError(f"unknown GPT-OSS layer types: {unknown}")
        return GPTOSSDimensions(
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
            sliding_window=self.sliding_window,
            layer_types=tuple(layer_types),
            qk_norm=self.qk_norm,
        )


def _layer_spec(index: int, dims: GPTOSSDimensions) -> TrainingLayerSpec:
    layer_type = dims.layer_types[index]
    block = GPTOSSBlock(dims, layer_type)
    variant = "sliding_attention" if layer_type == "sliding_attention" else "full_attention"
    label = "Sliding Attention" if layer_type == "sliding_attention" else "Full Attention"
    matrices = dims.layer_matrices()
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=dims.params_per_layer(),
        saved_activation_width=dims.saved_activation_width(),
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
                "gpt_oss_optimizer",
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
        block_key=f"gpt_oss.{variant}_moe_block",
        block_name=f"GPT-OSS {label} MoE Block",
        optimizer_block_key=f"gpt_oss.{variant}_moe_block.optimizer_step",
        metadata={
            "gpt_oss": asdict(dims),
            "layer_type": layer_type,
        },
    )


def _head_spec(dims: GPTOSSDimensions) -> TrainingHeadSpec:
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
        metadata={"gpt_oss": asdict(dims)},
    )


class GPTOSSForTraining(TrainingBuilder):
    family_name = "gpt_oss"
    metadata_kind = "training.gpt_oss.modular"

    def __init__(self, config: GPTOSSConfig) -> None:
        self.config = config
        self.dims = config.dimensions()
        super().__init__(
            family_name=self.family_name,
            metadata_kind=self.metadata_kind,
            preset_name=config.preset_name,
            layers=[_layer_spec(index, self.dims) for index in range(config.n_layers)],
            head=_head_spec(self.dims),
            model_metadata={"gpt_oss": asdict(self.dims)},
        )
