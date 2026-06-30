"""Z.AI GLM-5 presets for the DeepSeek-V3.2 DSA architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataflow_sim.workloads.models.deepseek_v3_2 import (
    DeepSeekV32Config,
    DeepSeekV32ForTraining,
)


_ALIASES = {
    "5": "glm_5_744B-40B",
    "glm-5": "glm_5_744B-40B",
    "glm5": "glm_5_744B-40B",
    "glm_5": "glm_5_744B-40B",
    "glm_5_744b-40b": "glm_5_744B-40B",
    "glm_5_744B-40B": "glm_5_744B-40B",
    "5.1": "glm_5_744B-40B",
    "glm-5.1": "glm_5_744B-40B",
    "glm5.1": "glm_5_744B-40B",
    "glm_5_1": "glm_5_744B-40B",
    "glm_5_1_744b-40b": "glm_5_744B-40B",
    "glm_5_1_744B-40B": "glm_5_744B-40B",
}


@dataclass(frozen=True)
class GLM5Config(DeepSeekV32Config):
    @classmethod
    def preset(cls, scale: str = "5", **overrides: Any) -> "GLM5Config":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(f"unknown GLM-5 scale {scale!r}; use 5 or 5.1")
        return cls.from_model_dims(key, **overrides)


class GLM5ForTraining(DeepSeekV32ForTraining):
    family_name = "deepseek_v3_2"

    def __init__(self, config: GLM5Config) -> None:
        super().__init__(config)
