"""NVIDIA Nemotron-H model-family definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models._config import load_model_dims
from dataflow_sim.workloads.modules import (
    LanguageModelingHead,
    NemotronBlock,
    NemotronDimensions,
    head_params,
    optimizer_ops_for_matrices,
)
from dataflow_sim.workloads.modules.nemotron_dimensions import parse_hybrid_pattern
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


_NEMOTRON_FIELDS = {
    "vocab_size",
    "n_layers",
    "d_model",
    "head_dim",
    "n_heads",
    "n_kv_heads",
    "expert_dim",
    "shared_expert_dim",
    "num_shared_experts",
    "num_routed_experts",
    "top_k",
    "intermediate_size",
    "mamba_num_heads",
    "mamba_head_dim",
    "ssm_state_size",
    "conv_kernel",
    "mamba_chunk_size",
    "n_groups",
    "hybrid_override_pattern",
    "qk_norm",
}

_ALIASES = {
    "30b-a3b": "nemotron3_nano_30B-A3B",
    "nano": "nemotron3_nano_30B-A3B",
    "nano-30b-a3b": "nemotron3_nano_30B-A3B",
    "nemotron3_nano_30b-a3b": "nemotron3_nano_30B-A3B",
    "nemotron3_nano_30B-A3B": "nemotron3_nano_30B-A3B",
    "120b-a12b": "nemotron3_super_120B-A12B",
    "super": "nemotron3_super_120B-A12B",
    "super-120b-a12b": "nemotron3_super_120B-A12B",
    "nemotron3_super_120b-a12b": "nemotron3_super_120B-A12B",
    "nemotron3_super_120B-A12B": "nemotron3_super_120B-A12B",
    "550b-a55b": "nemotron3_ultra_550B-A55B",
    "ultra": "nemotron3_ultra_550B-A55B",
    "ultra-550b-a55b": "nemotron3_ultra_550B-A55B",
    "nemotron3_ultra_550b-a55b": "nemotron3_ultra_550B-A55B",
    "nemotron3_ultra_550B-A55B": "nemotron3_ultra_550B-A55B",
}

_FIELD_ALIASES = {
    "hidden_size": "d_model",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "moe_intermediate_size": "expert_dim",
    "moe_shared_expert_intermediate_size": "shared_expert_dim",
    "n_shared_experts": "num_shared_experts",
    "n_routed_experts": "num_routed_experts",
    "num_experts_per_tok": "top_k",
    "chunk_size": "mamba_chunk_size",
}


def _config_kwargs(body: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, value in body.items():
        normalized = _FIELD_ALIASES.get(key, key)
        if normalized in _NEMOTRON_FIELDS:
            kwargs[normalized] = value
    for key, value in overrides.items():
        normalized = _FIELD_ALIASES.get(key, key)
        kwargs[normalized] = value
    return kwargs


@dataclass(frozen=True)
class NemotronHConfig:
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
    hybrid_override_pattern: str
    qk_norm: bool = False
    preset_name: str = "custom"

    @classmethod
    def from_model_dims(cls, key: str, **overrides: Any) -> "NemotronHConfig":
        body = load_model_dims(key)
        kwargs = _config_kwargs(body, overrides)
        kwargs.setdefault("preset_name", key)
        if "hybrid_override_pattern" not in kwargs and "layers_block_type" in body:
            reverse = {"mamba": "M", "attention": "*", "moe": "E", "mlp": "-"}
            kwargs["hybrid_override_pattern"] = "".join(
                reverse.get(str(layer), str(layer))
                for layer in body["layers_block_type"]
            )
        return cls(**kwargs)

    @classmethod
    def preset(cls, scale: str = "nano", **overrides: Any) -> "NemotronHConfig":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                "unknown Nemotron-H scale "
                f"{scale!r}; use nano, super, ultra, or a nemotron3_* preset key"
            )
        return cls.from_model_dims(key, **overrides)

    def dimensions(self) -> NemotronDimensions:
        layer_types = parse_hybrid_pattern(self.hybrid_override_pattern)
        if len(layer_types) != self.n_layers:
            raise ValueError(
                "hybrid_override_pattern length must match n_layers; "
                f"got {len(layer_types)} entries for {self.n_layers} layers"
            )
        return NemotronDimensions(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            d_model=self.d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            expert_dim=self.expert_dim,
            shared_expert_dim=self.shared_expert_dim,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            mamba_num_heads=self.mamba_num_heads,
            mamba_head_dim=self.mamba_head_dim,
            ssm_state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel,
            mamba_chunk_size=self.mamba_chunk_size,
            n_groups=self.n_groups,
            layer_types=layer_types,
            hybrid_override_pattern=self.hybrid_override_pattern,
            qk_norm=self.qk_norm,
        )


def _layer_spec(index: int, dims: NemotronDimensions) -> TrainingLayerSpec:
    layer_type = dims.layer_types[index]
    block = NemotronBlock(dims, layer_type)
    block_key = f"nemotron_h.{layer_type}_block"
    block_name = f"Nemotron {layer_type.title()} Block"
    matrices = dims.layer_matrices(layer_type)
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=dims.params_per_layer(layer_type),
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
                "nemotron_h_optimizer",
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
            "nemotron_h": asdict(dims),
            "layer_type": layer_type,
        },
    )


def _head_spec(dims: NemotronDimensions) -> TrainingHeadSpec:
    head_dims = dims.head_dimensions()
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
        metadata={"nemotron_h": asdict(dims)},
    )


class NemotronHForTraining(TrainingBuilder):
    family_name = "nemotron_h"
    metadata_kind = "training.nemotron_h.modular"

    def __init__(self, config: NemotronHConfig) -> None:
        self.config = config
        self.dims = config.dimensions()
        super().__init__(
            family_name=self.family_name,
            metadata_kind=self.metadata_kind,
            preset_name=config.preset_name,
            layers=[
                _layer_spec(index, self.dims)
                for index in range(config.n_layers)
            ],
            head=_head_spec(self.dims),
            model_metadata={"nemotron_h": asdict(self.dims)},
        )
