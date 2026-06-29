"""Example custom model built from custom modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from dataflow_sim.workloads.training_builder import (
    TrainingBuilder,
    TrainingHeadSpec,
    TrainingLayerSpec,
)

from custom_modules import ClassifierHead, MixerBlock


@dataclass(frozen=True)
class TinyMixerConfig:
    n_layers: int = 4
    d_model: int = 512
    hidden_dim: int = 2048
    classes: int = 32_000
    wide_every: int = 0


class TinyMixerForTraining(TrainingBuilder):
    """A custom architecture that owns its ordered module list."""

    def __init__(self, config: TinyMixerConfig) -> None:
        self.config = config
        layers = [
            self._layer_spec(index, config)
            for index in range(config.n_layers)
        ]
        head = self._head_spec(config)
        super().__init__(
            family_name="tiny_mixer",
            metadata_kind="training.custom.tiny_mixer",
            preset_name="custom",
            layers=layers,
            head=head,
            model_metadata={"tiny_mixer": asdict(config)},
        )

    @staticmethod
    def _layer_spec(index: int, config: TinyMixerConfig) -> TrainingLayerSpec:
        hidden_dim = config.hidden_dim
        if config.wide_every > 0 and (index + 1) % config.wide_every == 0:
            hidden_dim *= 2
        block = MixerBlock(d_model=config.d_model, hidden_dim=hidden_dim)
        return TrainingLayerSpec(
            name=f"mixer_{index}",
            input_dim=config.d_model,
            output_dim=config.d_model,
            param_count=block.param_count,
            saved_activation_width=block.saved_activation_width,
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
                lambda optimizer, bpe, block=block: block.optimizer_ops(
                    optimizer,
                    bytes_per_element=bpe,
                )
            ),
            block_key=f"tiny_mixer_block.h{hidden_dim}",
            block_name=f"Tiny Mixer Block h={hidden_dim}",
            metadata={"hidden_dim": hidden_dim},
        )

    @staticmethod
    def _head_spec(config: TinyMixerConfig) -> TrainingHeadSpec:
        head = ClassifierHead(d_model=config.d_model, classes=config.classes)
        return TrainingHeadSpec(
            name="classifier",
            input_dim=config.d_model,
            param_count=head.param_count,
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
            block_key="tiny_classifier_head",
            block_name="Tiny Classifier Head",
            optimizer_ops=(
                lambda optimizer, bpe, head=head: head.optimizer_ops(
                    optimizer,
                    bytes_per_element=bpe,
                )
            ),
            metadata={"classes": config.classes},
        )
