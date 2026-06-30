"""Qwen3 MoE model-family definitions for the modular workload builder."""
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


_ALIASES = {
    "30b-3b": "qwen3_moe_30B-3B",
    "30bx3b": "qwen3_moe_30B-3B",
    "30b-a3b": "qwen3_moe_30B-3B",
    "qwen3_moe_30b-3b": "qwen3_moe_30B-3B",
    "qwen3_moe_30bx3b": "qwen3_moe_30B-3B",
    "qwen3_moe_30b-a3b": "qwen3_moe_30B-3B",
    "qwen3_moe_30B-3B": "qwen3_moe_30B-3B",
    "235b-a22b": "qwen3_moe_235B-A22B",
    "235b-22b": "qwen3_moe_235B-A22B",
    "235b-a22B": "qwen3_moe_235B-A22B",
    "qwen3_moe_235b-a22b": "qwen3_moe_235B-A22B",
    "qwen3_moe_235b-22b": "qwen3_moe_235B-A22B",
    "qwen3_moe_235B-A22B": "qwen3_moe_235B-A22B",
}


@dataclass(frozen=True)
class Qwen3MoEConfig(TransformerFamilyConfig):
    @classmethod
    def preset(cls, scale: str = "30B-3B", **overrides: Any) -> "Qwen3MoEConfig":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                f"unknown Qwen3 MoE scale {scale!r}; use a known preset or "
                "start from Qwen3MoEConfig.preset('30B-3B', ...) or "
                "Qwen3MoEConfig.preset('235B-A22B', ...) with explicit overrides"
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


class Qwen3MoEForTraining(TrainingBuilder):
    family_name = "qwen3_moe"
    metadata_kind = "training.transformer.qwen3_moe.modular"

    def __init__(self, config: Qwen3MoEConfig) -> None:
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
