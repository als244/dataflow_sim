"""Qwen3.5/Qwen3.6 hybrid MoE model-family definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataflow_sim.workloads.models.qwen3_hybrid_dense import (
    QwenHybridConfig,
    QwenHybridForTraining,
)


_MOE_ALIASES = {
    "35b-a3b": "qwen3_5_35B-A3B",
    "35b-3b": "qwen3_5_35B-A3B",
    "qwen3_5_35b-a3b": "qwen3_5_35B-A3B",
    "qwen3.5-35b-a3b": "qwen3_5_35B-A3B",
    "qwen3_6_35b-a3b": "qwen3_5_35B-A3B",
    "qwen3.6-35b-a3b": "qwen3_5_35B-A3B",
    "122b-a10b": "qwen3_5_122B-A10B",
    "qwen3_5_122b-a10b": "qwen3_5_122B-A10B",
    "qwen3.5-122b-a10b": "qwen3_5_122B-A10B",
    "397b-a17b": "qwen3_5_397B-A17B",
    "qwen3_5_397b-a17b": "qwen3_5_397B-A17B",
    "qwen3.5-397b-a17b": "qwen3_5_397B-A17B",
    "qwen3_5_35B-A3B": "qwen3_5_35B-A3B",
    "qwen3_6_35B-A3B": "qwen3_5_35B-A3B",
    "qwen3_5_122B-A10B": "qwen3_5_122B-A10B",
    "qwen3_5_397B-A17B": "qwen3_5_397B-A17B",
}


@dataclass(frozen=True)
class QwenHybridMoEConfig(QwenHybridConfig):
    @classmethod
    def preset(cls, scale: str = "35B-A3B", **overrides: Any) -> "QwenHybridMoEConfig":
        key = _MOE_ALIASES.get(scale, _MOE_ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                f"unknown Qwen hybrid MoE scale {scale!r}; use 35B-A3B, 122B-A10B, or 397B-A17B"
            )
        return cls.from_model_dims(key, **overrides)


class QwenHybridMoEForTraining(QwenHybridForTraining):
    family_name = "qwen3_hybrid_moe"

    def __init__(self, config: QwenHybridMoEConfig) -> None:
        super().__init__(config, family_name=self.family_name)
