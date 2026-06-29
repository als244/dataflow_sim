"""Example modules composed from local custom ops."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.workloads.dataflow import DataflowCost

import custom_ops as ops


@dataclass(frozen=True)
class MixerBlock:
    """A tiny MLP-style block over `[tokens, d_model]` activations."""

    d_model: int
    hidden_dim: int

    @property
    def param_count(self) -> int:
        return 2 * self.d_model * self.hidden_dim

    @property
    def saved_activation_width(self) -> int:
        return self.d_model + self.hidden_dim

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        del seqlen
        return [
            ops.dense_projection(
                "mixer_up",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.hidden_dim,
                bytes_per_element=bytes_per_element,
            ),
            ops.fast_gelu(
                "mixer_gelu",
                tokens=tokens,
                dim=self.hidden_dim,
                bytes_per_element=bytes_per_element,
            ),
            ops.dense_projection(
                "mixer_down",
                tokens=tokens,
                input_dim=self.hidden_dim,
                output_dim=self.d_model,
                bytes_per_element=bytes_per_element,
            ),
            ops.residual_add(
                "mixer_residual",
                tokens=tokens,
                dim=self.d_model,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        del seqlen
        return [
            ops.residual_add(
                "mixer_residual_bwd",
                tokens=tokens,
                dim=self.d_model,
                bytes_per_element=bytes_per_element,
            ),
            ops.dense_input_grad(
                "mixer_down_dgrad",
                tokens=tokens,
                input_dim=self.hidden_dim,
                output_dim=self.d_model,
                bytes_per_element=bytes_per_element,
            ),
            ops.dense_weight_grad(
                "mixer_down_wgrad",
                tokens=tokens,
                input_dim=self.hidden_dim,
                output_dim=self.d_model,
                bytes_per_element=bytes_per_element,
            ),
            ops.fast_gelu_grad(
                "mixer_gelu_bwd",
                tokens=tokens,
                dim=self.hidden_dim,
                bytes_per_element=bytes_per_element,
            ),
            ops.dense_input_grad(
                "mixer_up_dgrad",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.hidden_dim,
                bytes_per_element=bytes_per_element,
            ),
            ops.dense_weight_grad(
                "mixer_up_wgrad",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.hidden_dim,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def recompute_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        del seqlen
        return ops.recompute_only(
            [
                ops.dense_projection(
                    "mixer_up_recompute",
                    tokens=tokens,
                    input_dim=self.d_model,
                    output_dim=self.hidden_dim,
                    bytes_per_element=bytes_per_element,
                ),
                ops.fast_gelu(
                    "mixer_gelu_recompute",
                    tokens=tokens,
                    dim=self.hidden_dim,
                    bytes_per_element=bytes_per_element,
                ),
            ]
        )

    def optimizer_ops(
        self,
        optimizer: str,
        *,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return ops.optimizer_step(
            "mixer",
            optimizer=optimizer,
            param_count=self.param_count,
            bytes_per_element=bytes_per_element,
        )


@dataclass(frozen=True)
class ClassifierHead:
    d_model: int
    classes: int

    @property
    def param_count(self) -> int:
        return self.d_model * self.classes

    def forward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return [
            ops.dense_projection(
                "classifier_logits",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.classes,
                bytes_per_element=bytes_per_element,
            ),
            ops.cross_entropy(
                "classifier_loss",
                tokens=tokens,
                classes=self.classes,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def backward_ops(
        self,
        *,
        tokens: int,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return [
            ops.dense_input_grad(
                "classifier_dgrad",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.classes,
                bytes_per_element=bytes_per_element,
            ),
            ops.dense_weight_grad(
                "classifier_wgrad",
                tokens=tokens,
                input_dim=self.d_model,
                output_dim=self.classes,
                bytes_per_element=bytes_per_element,
            ),
        ]

    def optimizer_ops(
        self,
        optimizer: str,
        *,
        bytes_per_element: int = 2,
    ) -> list[DataflowCost]:
        return ops.optimizer_step(
            "classifier",
            optimizer=optimizer,
            param_count=self.param_count,
            bytes_per_element=bytes_per_element,
        )
