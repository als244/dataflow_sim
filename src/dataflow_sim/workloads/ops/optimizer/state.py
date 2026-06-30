"""Optimizer state sizing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptimizerMode = Literal["none", "adamw", "muon", "sgd"]


@dataclass(frozen=True)
class OptimizerMatrix:
    name: str
    rows: int
    cols: int
    count: int = 1
    expert: bool = False
    ep_sharded: bool = False


def optimizer_state_bytes(weight_bytes: int, optimizer: OptimizerMode) -> int:
    if optimizer in {"none", "sgd"}:
        return 0
    if optimizer == "adamw":
        return 2 * weight_bytes
    if optimizer == "muon":
        return weight_bytes
    raise ValueError(f"unknown optimizer mode: {optimizer!r}")
