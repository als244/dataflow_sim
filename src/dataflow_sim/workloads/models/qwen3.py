"""Qwen3 dense model-family definitions for the modular workload builder."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models._config import TransformerFamilyConfig
from dataflow_sim.workloads.modules import (
    TransformerBlock,
    TransformerDimensions,
    LanguageModelingHead,
    head_params,
    layer_activation_elements_per_token,
    optimizer_ops_for_matrices,
    params_per_layer,
)
from dataflow_sim.workloads.training_builder import (
    TrainingBuilder,
    TrainingHeadSpec,
    TrainingLayerSpec,
)


_ALIASES = {
    "4b": "qwen3_4B",
    "qwen3_4b": "qwen3_4B",
    "qwen3_4B": "qwen3_4B",
    "8b": "qwen3_8B",
    "qwen3_8b": "qwen3_8B",
    "qwen3_8B": "qwen3_8B",
    "32b": "qwen3_32B",
    "qwen3_32b": "qwen3_32B",
    "qwen3_32B": "qwen3_32B",
}


@dataclass(frozen=True)
class Qwen3Config(TransformerFamilyConfig):
    @classmethod
    def preset(cls, scale: str = "8B", **overrides: Any) -> "Qwen3Config":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                f"unknown Qwen3 dense scale {scale!r}; use a known preset or "
                "start from Qwen3Config.preset('8B', ...) with explicit overrides"
            )
        return cls.from_model_dims(key, **overrides)


def _layer_spec(index: int, dims: TransformerDimensions) -> TrainingLayerSpec:
    block = TransformerBlock(dims)
    matrices = block.optimizer_matrices()
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=params_per_layer(dims),
        saved_activation_width=layer_activation_elements_per_token(dims),
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
                "transformer_block_optimizer",
                matrices=matrices,
                optimizer=optimizer,
                bytes_per_element=bpe,
            )
        ),
        block_key="transformer_block",
        block_name="Transformer Block",
        metadata={"transformer": asdict(dims)},
    )


def _head_spec(dims: TransformerDimensions) -> TrainingHeadSpec:
    head = LanguageModelingHead(dims)
    return TrainingHeadSpec(
        name="head",
        input_dim=dims.d_model,
        param_count=head_params(dims),
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
        metadata={"transformer": asdict(dims)},
    )


class Qwen3ForTraining(TrainingBuilder):
    family_name = "qwen3"
    metadata_kind = "training.transformer.qwen3.modular"

    def __init__(self, config: Qwen3Config) -> None:
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
            model_metadata={"transformer": asdict(self.dims)},
        )
