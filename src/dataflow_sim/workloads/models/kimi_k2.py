"""Kimi-K2 model-family definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataflow_sim.workloads.models.deepseek_v3 import (
    DeepSeekConfig,
    DeepSeekForTraining,
)


_ALIASES = {
    "1t-32b": "kimi_k2_1T-32B",
    "kimi-k2": "kimi_k2_1T-32B",
    "kimi_k2": "kimi_k2_1T-32B",
    "kimi_k2_1t-32b": "kimi_k2_1T-32B",
    "kimi_k2_1T-32B": "kimi_k2_1T-32B",
}


@dataclass(frozen=True)
class KimiK2Config(DeepSeekConfig):
    @classmethod
    def preset(cls, scale: str = "1T-32B", **overrides: Any) -> "KimiK2Config":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(f"unknown Kimi-K2 scale {scale!r}; use 1T-32B")
        return cls.from_model_dims(key, **overrides)


class KimiK2ForTraining(DeepSeekForTraining):
    family_name = "deepseek_v3"

    def __init__(self, config: KimiK2Config) -> None:
        super().__init__(config, family_name=self.family_name)
