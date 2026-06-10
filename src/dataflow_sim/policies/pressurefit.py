"""PressureFit policy.

Standalone planner built around deterministic greedy pressure reduction:
  1. derive schema-level facts from the TaskChain;
  2. choose initial residency and build one seed interval set from liveness
     anchors;
  3. cut optional non-anchor residency gaps until every boundary satisfies
     the capacity inequality (pressure reduction);
  4. emit release/offload/prefetch triggers from the reduced intervals under
     each of three inbound schedules (packed-fifo, interval-entry,
     latest-safe);
  5. verify each annotated chain with the simulator, translating bounded
     capacity contradictions back into boundary pressure and re-reducing;
  6. return the fastest valid annotated chain.

The three inbound schedules are the policy's only branching: residency
planning is shared and deterministic, but the best moment to fire a prefetch
depends on FIFO congestion vs. memory pressure, which the analytic model
cannot rank without replay. Packed-fifo coordinates transfers backward from
their deadlines (best under inbound congestion), interval-entry extends
interval entries earlier when strict pressure allows (best when lead time is
scarce), and latest-safe places each transfer independently as late as
possible (most conservative arrivals; the variant that survives extreme
pressure).

The policy is name-agnostic. It uses object source availability, size, uses,
producer, and explicit mutation metadata.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

from dataflow_sim.policies._common import (
    _compute_ideal_starts,
    _object_sizes,
    _object_uses_by_task_idx,
)
from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _build_facts,
)
from dataflow_sim.policies.pressurefit_aux.diagnostics import (
    PressureFitCandidateDiagnostic,
    PressureFitDiagnostics,
    _candidate_diagnostic,
)
from dataflow_sim.policies.pressurefit_aux.emit import (
    _emit_chain,
)
from dataflow_sim.policies.pressurefit_aux.inbound_schedules import _extend_inbound_lead_time
from dataflow_sim.policies.pressurefit_aux.seeds import (
    _copy_intervals,
    _initial_residency,
    _pressure_initial_placement,
)
from dataflow_sim.policies.pressurefit_aux.types import _IntervalSet, _ScheduleSpec
from dataflow_sim.policies.pressurefit_aux.physical_repair import (
    _PHYSICAL_REPAIR_LIMIT,
    _apply_physical_repair,
)
from dataflow_sim.policies.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.core.schema import TaskChain
from dataflow_sim.engine.simulator import run as simulator_run

# The three inbound schedules, in tie-break priority order: when two produce
# the same simulated makespan, the earlier entry is selected.
_SCHEDULES: tuple[_ScheduleSpec, ...] = (
    _ScheduleSpec("packed-fifo", pack_inbound=True),
    _ScheduleSpec(
        "interval-entry",
        extend_inbound=True,
        respect_interval_start=True,
    ),
    _ScheduleSpec("latest-safe"),
)


@dataclass(frozen=True)
class _CandidateResult:
    makespan_us: int
    chain: TaskChain
    name: str


def apply_pressurefit_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
) -> TaskChain:
    """Return an annotated chain using the standalone PressureFit policy."""
    chain, _diagnostics = plan_pressurefit_policy(
        bare, device_capacity=device_capacity,
    )
    return chain


def plan_pressurefit_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
) -> tuple[TaskChain, PressureFitDiagnostics]:
    """Return an annotated chain and per-schedule planning diagnostics.

    The algorithm spine is:
      facts -> seed intervals -> shared pressure reduction -> three inbound
      schedules -> fastest valid annotated chain.
    """
    planning_start = time.perf_counter()
    if device_capacity is not None:
        bare = replace(bare, device_capacity=device_capacity)

    ideal = _compute_ideal_starts(bare)
    sizes = _object_sizes(bare)
    uses_by_task = _object_uses_by_task_idx(bare, ideal)
    initial_device = _pressure_initial_placement(
        bare, bare.device_capacity, sizes, uses_by_task,
    )
    facts = _build_facts(bare)
    seed = _initial_residency(facts, initial_device)

    results, candidate_diagnostics, first_error = _evaluate_schedules(
        bare, facts, seed,
    )

    if not results:
        assert first_error is not None
        raise first_error
    results.sort(key=lambda x: x.makespan_us)
    selected = results[0]
    selected_candidates = [
        replace(diag, selected=diag.name == selected.name)
        for diag in candidate_diagnostics
    ]
    diagnostics = PressureFitDiagnostics(
        planning_time_s=time.perf_counter() - planning_start,
        task_count=facts.n,
        object_count=len(facts.sizes),
        device_capacity=bare.device_capacity,
        candidate_count=len(selected_candidates),
        valid_candidate_count=sum(
            1 for diag in selected_candidates if diag.status == "valid"
        ),
        selected_candidate=selected.name,
        selected_makespan_us=selected.makespan_us,
        candidates=selected_candidates,
    )
    return selected.chain, diagnostics


def _reduce_intervals(
    bare: TaskChain,
    facts: _Facts,
    intervals: _IntervalSet,
    spec: _ScheduleSpec,
    extra_pressure: list[int],
) -> None:
    _reduce_to_fit(facts, intervals, bare.device_capacity, extra_pressure)
    if spec.extend_inbound:
        _extend_inbound_lead_time(
            facts, intervals, bare.device_capacity, bare.bandwidth_h2d,
            extra_pressure,
        )


def _simulated_makespan_us(annotated: TaskChain) -> int:
    log = simulator_run(annotated, snapshots=False)
    return max(iv.end for iv in log.task_intervals)


def _verify_schedule_plan(
    bare: TaskChain,
    facts: _Facts,
    seed: _IntervalSet,
    spec: _ScheduleSpec,
) -> tuple[int, TaskChain]:
    intervals = _copy_intervals(seed)
    extra_pressure = [0] * (facts.n + 1)
    _reduce_intervals(bare, facts, intervals, spec, extra_pressure)

    for _ in range(_PHYSICAL_REPAIR_LIMIT):
        annotated = _emit_chain(
            bare, facts, intervals,
            pack_inbound=spec.pack_inbound,
            respect_interval_start=spec.respect_interval_start,
        )
        try:
            return _simulated_makespan_us(annotated), annotated
        except ValueError as e:
            repaired = _apply_physical_repair(
                str(e), bare, facts, intervals, extra_pressure,
            )
            if not repaired:
                raise
            _reduce_intervals(bare, facts, intervals, spec, extra_pressure)

    annotated = _emit_chain(
        bare, facts, intervals,
        pack_inbound=spec.pack_inbound,
        respect_interval_start=spec.respect_interval_start,
    )
    return _simulated_makespan_us(annotated), annotated


def _evaluate_schedules(
    bare: TaskChain,
    facts: _Facts,
    seed: _IntervalSet,
) -> tuple[list[_CandidateResult], list[PressureFitCandidateDiagnostic], Exception | None]:
    results: list[_CandidateResult] = []
    diagnostics: list[PressureFitCandidateDiagnostic] = []
    first_error: Exception | None = None

    for spec in _SCHEDULES:
        t0 = time.perf_counter()
        try:
            makespan, annotated = _verify_schedule_plan(bare, facts, seed, spec)
            wall = time.perf_counter() - t0
            results.append(_CandidateResult(makespan, annotated, spec.name))
            diagnostics.append(_candidate_diagnostic(
                spec,
                status="valid",
                wall_time_s=wall,
                makespan_us=makespan,
            ))
        except Exception as e:
            wall = time.perf_counter() - t0
            if first_error is None:
                first_error = e
            diagnostics.append(_candidate_diagnostic(
                spec,
                status="error",
                wall_time_s=wall,
                error=f"{type(e).__name__}: {e}",
            ))

    return results, diagnostics, first_error
