"""Shared model-family config helpers.

This module is intentionally limited to model definition concerns: preset
loading, dimension overrides, and conversion into module-friendly dimensions.
Training-loop scheduling lives in `dataflow_sim.workloads.training_builder`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from importlib.resources import files
from typing import Any, TypeVar

from dataflow_sim.workloads.modules import TransformerDimensions


_TRANSFORMER_FIELDS = {
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
    "qk_norm",
}


ConfigT = TypeVar("ConfigT", bound="TransformerFamilyConfig")


def load_model_dims(key: str) -> dict[str, Any]:
    raw = json.loads(
        (files("dataflow_sim.workloads.models") / "model_dims.json").read_text()
    )
    if key not in raw:
        known = ", ".join(sorted(raw))
        raise ValueError(f"unknown model preset {key!r}; known presets: {known}")
    return dict(raw[key])


@dataclass(frozen=True)
class TransformerFamilyConfig:
    """Base config for dense and MoE transformer-family presets.

    Family-specific config classes inherit this and provide a small `preset()`
    alias table. Users can then do `Llama3Config.preset("8B", n_layers=80)`.
    """

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
    qk_norm: bool = True
    preset_name: str = "custom"

    @classmethod
    def from_model_dims(cls: type[ConfigT], key: str, **overrides: Any) -> ConfigT:
        body = load_model_dims(key)
        kwargs = {field: body[field] for field in _TRANSFORMER_FIELDS if field in body}
        kwargs.update(overrides)
        kwargs.setdefault("preset_name", key)
        return cls(**kwargs)

    def with_overrides(self: ConfigT, **overrides: Any) -> ConfigT:
        return replace(self, **overrides)

    def dimensions(self) -> TransformerDimensions:
        return TransformerDimensions(
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
            qk_norm=self.qk_norm,
        )
