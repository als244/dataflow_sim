"""Diagnostics data structures for PressureFit."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.policy.pressurefit_aux.core import _Facts
from dataflow_sim.policy.pressurefit_aux.portfolio import _CandidateSpec


@dataclass(frozen=True)
class PressureFitCandidateDiagnostic:
    name: str
    family: str
    status: str
    selected: bool
    makespan_us: int | None
    wall_time_s: float
    error: str | None = None
    pack_h2d: bool | None = None
    extend_h2d: bool = False
    respect_interval_start: bool = False
    latest_h2d: bool = False
    reserve_pressure: int = 0
    protected_count: int = 0
    protected_bytes: int = 0
    seed: str = "base"


@dataclass(frozen=True)
class PressureFitDiagnostics:
    portfolio_mode: str
    effective_portfolio_mode: str
    fast_portfolio: bool
    planning_time_s: float
    task_count: int
    object_count: int
    device_capacity: int | None
    candidate_count: int
    valid_candidate_count: int
    selected_candidate: str
    selected_makespan_us: int
    candidates: list[PressureFitCandidateDiagnostic]


def _candidate_diagnostic(
    facts: _Facts,
    spec: _CandidateSpec,
    *,
    status: str,
    wall_time_s: float = 0.0,
    makespan_us: int | None = None,
    error: str | None = None,
) -> PressureFitCandidateDiagnostic:
    protected = set(spec.protected_initial)
    return PressureFitCandidateDiagnostic(
        name=spec.name,
        family=spec.family,
        status=status,
        selected=False,
        makespan_us=makespan_us,
        wall_time_s=wall_time_s,
        error=error,
        pack_h2d=spec.pack_h2d,
        extend_h2d=spec.extend_h2d,
        respect_interval_start=spec.respect_interval_start,
        latest_h2d=spec.latest_h2d,
        reserve_pressure=spec.reserve_pressure,
        protected_count=len(protected),
        protected_bytes=sum(facts.sizes[oid] for oid in protected),
        seed=spec.seed,
    )
