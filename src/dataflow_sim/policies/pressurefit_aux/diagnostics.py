"""Diagnostics data structures for PressureFit."""
from __future__ import annotations

from dataclasses import dataclass

from dataflow_sim.policies.pressurefit_aux.types import _ScheduleSpec


@dataclass(frozen=True)
class PressureFitCandidateDiagnostic:
    """Outcome of one inbound-schedule variant."""
    name: str
    status: str
    selected: bool
    makespan_us: int | None
    wall_time_s: float
    error: str | None = None
    pack_inbound: bool = False
    extend_inbound: bool = False
    respect_interval_start: bool = False
    clamp_inbound: bool = False


@dataclass(frozen=True)
class PressureFitDiagnostics:
    planning_time_s: float
    task_count: int
    object_count: int
    fast_memory_capacity: int | None
    candidate_count: int
    valid_candidate_count: int
    selected_candidate: str
    selected_makespan_us: int
    candidates: list[PressureFitCandidateDiagnostic]


def _candidate_diagnostic(
    spec: _ScheduleSpec,
    *,
    status: str,
    wall_time_s: float = 0.0,
    makespan_us: int | None = None,
    error: str | None = None,
) -> PressureFitCandidateDiagnostic:
    return PressureFitCandidateDiagnostic(
        name=spec.name,
        status=status,
        selected=False,
        makespan_us=makespan_us,
        wall_time_s=wall_time_s,
        error=error,
        pack_inbound=spec.pack_inbound,
        extend_inbound=spec.extend_inbound,
        respect_interval_start=spec.respect_interval_start,
        clamp_inbound=spec.clamp_inbound,
    )
