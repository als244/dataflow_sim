"""Z.AI GLM-5.2 IndexShare model-family definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from dataflow_sim.workloads.models._config import load_model_dims
from dataflow_sim.workloads.models.deepseek_v3_2 import (
    _DEEPSEEK_V32_FIELDS,
    _FIELD_ALIASES,
    _config_kwargs,
    DeepSeekV32Config,
)
from dataflow_sim.workloads.modules import (
    DeepSeekV32Block,
    DeepSeekV32Dimensions,
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


IndexerMode = Literal["full", "shared"]

_GLM52_FIELDS = _DEEPSEEK_V32_FIELDS | {
    "index_topk_freq",
    "index_skip_topk_offset",
}

_ALIASES = {
    "5.2": "glm_5_2_744B-40B",
    "glm-5.2": "glm_5_2_744B-40B",
    "glm5.2": "glm_5_2_744B-40B",
    "glm_5_2": "glm_5_2_744B-40B",
    "glm_5_2_744b-40b": "glm_5_2_744B-40B",
    "glm_5_2_744B-40B": "glm_5_2_744B-40B",
}


def _glm52_config_kwargs(body: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    kwargs = _config_kwargs(body, overrides)
    for source in (body, overrides):
        for key, value in source.items():
            normalized = _FIELD_ALIASES.get(key, key)
            if normalized in _GLM52_FIELDS:
                kwargs[normalized] = value
    kwargs.setdefault("index_topk_freq", 4)
    kwargs.setdefault("index_skip_topk_offset", 3)
    return kwargs


@dataclass(frozen=True)
class GLM52Config(DeepSeekV32Config):
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3

    @classmethod
    def from_model_dims(cls, key: str, **overrides: Any) -> "GLM52Config":
        body = load_model_dims(key)
        kwargs = _glm52_config_kwargs(body, overrides)
        kwargs.setdefault("preset_name", key)
        return cls(**kwargs)

    @classmethod
    def preset(cls, scale: str = "5.2", **overrides: Any) -> "GLM52Config":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(f"unknown GLM-5.2 scale {scale!r}; use 5.2")
        return cls.from_model_dims(key, **overrides)

    def dimensions(self) -> DeepSeekV32Dimensions:
        if self.index_topk_freq <= 0:
            raise ValueError("index_topk_freq must be positive")
        if self.index_skip_topk_offset < 0:
            raise ValueError("index_skip_topk_offset cannot be negative")
        return super().dimensions()

    def indexer_mode(self, layer_index: int) -> IndexerMode:
        if layer_index < self.first_k_dense_replace:
            return "full"
        first_sparse_full = self.first_k_dense_replace + self.index_skip_topk_offset
        if layer_index >= first_sparse_full and (
            layer_index - first_sparse_full
        ) % self.index_topk_freq == 0:
            return "full"
        return "shared"

    def indexer_modes(self) -> tuple[IndexerMode, ...]:
        return tuple(self.indexer_mode(index) for index in range(self.n_layers))


def _layer_spec(index: int, config: GLM52Config, dims: DeepSeekV32Dimensions) -> TrainingLayerSpec:
    dense = dims.is_dense_layer(index)
    indexer_mode = config.indexer_mode(index)
    include_indexer = indexer_mode == "full"
    block = DeepSeekV32Block(dims, dense_ffn=dense, indexer_mode=indexer_mode)
    ffn_variant = "dense" if dense else "moe"
    indexer_variant = "full_index" if include_indexer else "shared_index"
    variant = f"{ffn_variant}_{indexer_variant}"
    block_key = f"glm_5_2.{variant}_block"
    ffn_label = "Dense" if dense else "MoE"
    indexer_label = "Full-Index" if include_indexer else "Shared-Index"
    block_name = f"GLM-5.2 {ffn_label} {indexer_label} Block"
    matrices = dims.layer_matrices(dense=dense, include_indexer=include_indexer)
    trainable_matrices = dims.layer_matrices(
        dense=dense,
        trainable_only=True,
        include_indexer=include_indexer,
    )
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=dims.params_per_layer(dense=dense, include_indexer=include_indexer),
        gradient_count=dims.trainable_params_per_layer(
            dense=dense,
            include_indexer=include_indexer,
        ),
        saved_activation_width=dims.saved_activation_width(
            dense=dense,
            include_indexer=include_indexer,
        ),
        saved_activation_bytes=(
            lambda tokens, seqlen, policy, dims=dims, dense=dense, include_indexer=include_indexer: (
                dims.saved_activation_bytes(
                    tokens=tokens,
                    seqlen=seqlen,
                    dense=dense,
                    policy=policy,
                    include_indexer=include_indexer,
                )
            )
        ),
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
            lambda optimizer, bpe, matrices=trainable_matrices: optimizer_ops_for_matrices(
                "glm_5_2_optimizer",
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
        gradient_bytes=(
            lambda policy, parallelism, matrices=trainable_matrices: matrix_gradient_bytes(
                matrices,
                policy,
                parallelism,
            )
        ),
        optimizer_state_bytes=(
            lambda optimizer, policy, parallelism, matrices=trainable_matrices: optimizer_state_bytes_for_matrices(
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
            "glm_5_2": asdict(config),
            "dense_ffn": dense,
            "indexer_mode": indexer_mode,
            "variant": variant,
        },
    )


def _head_spec(dims: DeepSeekV32Dimensions) -> TrainingHeadSpec:
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
        metadata={"glm_5_2": asdict(dims)},
    )


class GLM52ForTraining(TrainingBuilder):
    family_name = "glm_5_2"
    metadata_kind = "training.glm_5_2.indexshare.modular"

    def __init__(self, config: GLM52Config) -> None:
        self.config = config
        self.dims = config.dimensions()
        model_metadata = {
            "glm_5_2": {
                **asdict(self.dims),
                "index_topk_freq": config.index_topk_freq,
                "index_skip_topk_offset": config.index_skip_topk_offset,
                "indexer_modes": config.indexer_modes(),
            }
        }
        super().__init__(
            family_name=self.family_name,
            metadata_kind=self.metadata_kind,
            preset_name=config.preset_name,
            layers=[_layer_spec(index, config, self.dims) for index in range(config.n_layers)],
            head=_head_spec(self.dims),
            model_metadata=model_metadata,
        )
