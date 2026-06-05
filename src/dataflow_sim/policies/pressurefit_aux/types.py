"""Shared PressureFit candidate portfolio types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PressureFitPortfolioMode = Literal["auto", "fast", "full"]


@dataclass(frozen=True)
class _CandidateSpec:
    """One bounded alternative passed through the shared PressureFit pipeline."""
    name: str
    family: str
    seed_key: str = "base"
    seed: str = "base"
    pack_inbound: bool = False
    extend_inbound: bool = False
    respect_interval_start: bool = False
    latest_inbound: bool = False
    reserve_pressure: int = 0
    protected_initial: frozenset[str] = frozenset()
    skip_reason: str | None = None
    pre_error: Exception | None = None
    fallback_only: bool = False


@dataclass(frozen=True)
class _PortfolioPlan:
    requested_mode: PressureFitPortfolioMode
    effective_mode: str
    fast_portfolio: bool
    specs: list[_CandidateSpec]


@dataclass(frozen=True)
class _InitialProtectionJob:
    oid: str
    release_t: int
    deadline: int
    tau: int
    first_use: int
    size: int
    residency_cost: int


_IntervalSet = dict[str, list[tuple[int, int]]]
_SeedPortfolio = dict[str, _IntervalSet]
