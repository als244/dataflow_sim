"""Nemotron-H residual block composition."""
from __future__ import annotations

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.dataflow_builder import DataflowModule, OpDTypePolicy
from dataflow_sim.workloads.modules.nemotron_attention import NemotronAttention
from dataflow_sim.workloads.modules.nemotron_dimensions import (
    NemotronDimensions,
    normalize_nemotron_layer_type,
)
from dataflow_sim.workloads.modules.nemotron_mamba import NemotronMamba
from dataflow_sim.workloads.modules.relu2_mlp import ReLU2MLP
from dataflow_sim.workloads.modules.relu2_moe import ReLU2MoE
from dataflow_sim.workloads.ops import backward as bwd
from dataflow_sim.workloads.ops import forward as fwd


class NemotronBlock(DataflowModule):
    def __init__(self, dims: NemotronDimensions, layer_type: str) -> None:
        super().__init__(name="NemotronBlock")
        self.dims = dims
        self.layer_type = normalize_nemotron_layer_type(layer_type)
        if self.layer_type == "mamba":
            self.mixer = NemotronMamba(dims)
        elif self.layer_type == "attention":
            self.mixer = NemotronAttention(dims)
        elif self.layer_type == "moe":
            self.mixer = ReLU2MoE(dims)
        elif self.layer_type == "mlp":
            self.mixer = ReLU2MLP(
                d_model=dims.d_model,
                intermediate_size=dims.intermediate_size,
            )
        else:
            raise ValueError(f"unknown Nemotron-H layer type {layer_type!r}")

    @staticmethod
    def _policy(bytes_per_element: float | OpDTypePolicy) -> OpDTypePolicy:
        return (
            bytes_per_element
            if isinstance(bytes_per_element, OpDTypePolicy)
            else OpDTypePolicy.from_single_bpe(bytes_per_element)
        )

    def forward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        policy = self._policy(bytes_per_element)
        ops = [
            fwd.rms_norm(
                "block_norm",
                tokens=tokens,
                dim=self.dims.d_model,
                bytes_per_element=policy.activation_bpe,
            )
        ]
        if self.layer_type in {"mamba", "attention"}:
            ops.extend(
                self.mixer.forward_ops(
                    tokens=tokens,
                    seqlen=seqlen,
                    bytes_per_element=policy,
                )
            )
        else:
            ops.extend(
                self.mixer.forward_ops(
                    tokens=tokens,
                    bytes_per_element=policy,
                )
            )
        return ops

    def backward_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        policy = self._policy(bytes_per_element)
        if self.layer_type in {"mamba", "attention"}:
            dgrad = self.mixer.dgrad_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=policy,
            )
            wgrad = self.mixer.wgrad_ops(tokens=tokens, bytes_per_element=policy)
        else:
            dgrad = self.mixer.dgrad_ops(tokens=tokens, bytes_per_element=policy)
            wgrad = self.mixer.wgrad_ops(tokens=tokens, bytes_per_element=policy)
        return (
            dgrad
            + [
                bwd.rms_norm_grad(
                    "block_norm_bwd",
                    tokens=tokens,
                    dim=self.dims.d_model,
                    bytes_per_element=policy.activation_bpe,
                )
            ]
            + wgrad
        )

    def recompute_ops(
        self,
        *,
        tokens: int,
        seqlen: int,
        bytes_per_element: float | OpDTypePolicy = 2,
    ) -> list[DataflowCost]:
        policy = self._policy(bytes_per_element)
        ops = [
            fwd.rms_norm(
                "block_norm",
                tokens=tokens,
                dim=self.dims.d_model,
                bytes_per_element=policy.activation_bpe,
            ).model_copy(update={"effective_flops": 0})
        ]
        if self.layer_type in {"mamba", "attention"}:
            ops.extend(
                self.mixer.recompute_ops(
                    tokens=tokens,
                    seqlen=seqlen,
                    bytes_per_element=policy,
                )
            )
        else:
            ops.extend(
                self.mixer.recompute_ops(
                    tokens=tokens,
                    bytes_per_element=policy,
                )
            )
        return ops
