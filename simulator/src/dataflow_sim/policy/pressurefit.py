"""PressureFit policy.

Standalone planner built around deterministic greedy pressure reduction:
  1. derive schema-level facts from the TaskChain;
  2. build seed interval sets from liveness anchors;
  3. construct a bounded portfolio of heuristic candidate specs;
  4. for each spec, cut optional non-anchor residency gaps until every
     boundary satisfies the capacity inequality;
  5. emit release/offload/prefetch triggers from the reduced intervals;
  6. verify each annotated chain with the simulator;
  7. return the valid candidate with the lowest makespan.

The policy is name-agnostic. It uses object source availability, size, uses,
producer, and explicit mutation metadata.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

from dataflow_sim.policy._common import (
    _compute_ideal_starts,
    _object_sizes,
    _object_uses_by_task_idx,
)
from dataflow_sim.policy.pressurefit_aux.core import (
    _Facts,
    _build_facts,
)
from dataflow_sim.policy.pressurefit_aux.diagnostics import (
    PressureFitCandidateDiagnostic,
    PressureFitDiagnostics,
    _candidate_diagnostic,
)
from dataflow_sim.policy.pressurefit_aux.emit import (
    _emit_chain,
)
from dataflow_sim.policy.pressurefit_aux.inbound_schedules import _extend_inbound_lead_time
from dataflow_sim.policy.pressurefit_aux.initial_protection import (
    _initial_protection_headroom,
    _initial_protection_jobs_from_probe,
    _initial_protection_sets,
    _select_initial_protection_set,
)
from dataflow_sim.policy.pressurefit_aux.portfolio import (
    _build_candidate_portfolio,
)
from dataflow_sim.policy.pressurefit_aux.seeds import (
    _build_candidate_seeds,
    _copy_intervals,
    _initial_residency,
    _pressure_initial_placement,
)
from dataflow_sim.policy.pressurefit_aux.types import (
    PressureFitPortfolioMode,
    _CandidateSpec,
    _InitialProtectionJob,
)
from dataflow_sim.policy.pressurefit_aux.physical_repair import (
    _PHYSICAL_REPAIR_LIMIT,
    _apply_physical_repair,
)
from dataflow_sim.policy.pressurefit_aux.reducer import _reduce_to_fit
from dataflow_sim.schema import TaskChain
from dataflow_sim.simulator import run as simulator_run


@dataclass(frozen=True)
class _CandidateResult:
    makespan_us: int
    chain: TaskChain
    name: str


def apply_pressurefit_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
    refinement_iters: int = 0,
    portfolio_mode: PressureFitPortfolioMode = "auto",
) -> TaskChain:
    """Return an annotated chain using the standalone PressureFit policy."""
    chain, _diagnostics = plan_pressurefit_policy(
        bare,
        device_capacity=device_capacity,
        refinement_iters=refinement_iters,
        portfolio_mode=portfolio_mode,
    )
    return chain


def plan_pressurefit_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
    refinement_iters: int = 0,
    portfolio_mode: PressureFitPortfolioMode = "auto",
) -> tuple[TaskChain, PressureFitDiagnostics]:
    """Return an annotated chain and candidate-level planning diagnostics.

    The algorithm spine is:
      facts -> seed intervals -> candidate specs -> shared verification
      pipeline -> fastest valid annotated chain.
    """
    del refinement_iters  # kept for API symmetry; this policy does not search.
    if portfolio_mode not in ("auto", "fast", "full"):
        raise ValueError(f"unknown pressurefit portfolio_mode: {portfolio_mode!r}")
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
    base_intervals = _initial_residency(facts, initial_device)
    seeds = _build_candidate_seeds(
        facts,
        base_intervals,
        bare.bandwidth_h2d,
        bare.bandwidth_d2h,
    )

    portfolio = _build_candidate_portfolio(
        bare, facts, sizes, uses_by_task, initial_device, seeds, portfolio_mode,
    )
    results, candidate_diagnostics, first_error = _evaluate_candidate_portfolio(
        bare, facts, seeds, portfolio.specs,
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
        portfolio_mode=portfolio_mode,
        effective_portfolio_mode=portfolio.effective_mode,
        fast_portfolio=portfolio.fast_portfolio,
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


def _reduce_candidate_intervals(
    bare: TaskChain,
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    spec: _CandidateSpec,
    extra_pressure: list[int],
    protected_initial: set[str],
) -> None:
    _reduce_to_fit(
        facts, intervals, bare.device_capacity, extra_pressure,
        protected_initial=protected_initial,
    )
    if spec.extend_inbound:
        _extend_inbound_lead_time(
            facts, intervals, bare.device_capacity, bare.bandwidth_h2d,
            extra_pressure,
        )


def _emit_candidate_chain(
    bare: TaskChain,
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    spec: _CandidateSpec,
) -> TaskChain:
    return _emit_chain(
        bare, facts, intervals,
        pack_inbound=spec.pack_inbound,
        respect_interval_start=spec.respect_interval_start,
        latest_inbound=spec.latest_inbound,
    )


def _simulated_makespan_us(annotated: TaskChain) -> int:
    log = simulator_run(annotated, snapshots=False)
    return max(iv.end for iv in log.task_intervals)


def _verify_candidate_plan(
    bare: TaskChain,
    facts: _Facts,
    seeds: dict[str, dict[str, list[tuple[int, int]]]],
    spec: _CandidateSpec,
) -> tuple[int, TaskChain]:
    intervals = _copy_intervals(seeds[spec.seed_key])
    protected_initial = set(spec.protected_initial)
    extra_pressure = [spec.reserve_pressure] * (facts.n + 1)
    _reduce_candidate_intervals(
        bare, facts, intervals, spec, extra_pressure, protected_initial,
    )

    for _ in range(_PHYSICAL_REPAIR_LIMIT):
        annotated = _emit_candidate_chain(bare, facts, intervals, spec)
        try:
            return _simulated_makespan_us(annotated), annotated
        except ValueError as e:
            repaired = _apply_physical_repair(
                str(e),
                bare,
                facts,
                intervals,
                extra_pressure,
                spec.reserve_pressure,
            )
            if not repaired:
                raise
            _reduce_candidate_intervals(
                bare, facts, intervals, spec, extra_pressure, protected_initial,
            )

    annotated = _emit_candidate_chain(bare, facts, intervals, spec)
    return _simulated_makespan_us(annotated), annotated


def _evaluate_candidate_portfolio(
    bare: TaskChain,
    facts: _Facts,
    seeds: dict[str, dict[str, list[tuple[int, int]]]],
    specs: list[_CandidateSpec],
) -> tuple[list[_CandidateResult], list[PressureFitCandidateDiagnostic], Exception | None]:
    results: list[_CandidateResult] = []
    diagnostics: list[PressureFitCandidateDiagnostic] = []
    first_error: Exception | None = None

    for spec in specs:
        if spec.skip_reason is not None:
            diagnostics.append(_candidate_diagnostic(
                facts, spec, status="skipped", error=spec.skip_reason,
            ))
            continue
        if spec.fallback_only and results:
            diagnostics.append(_candidate_diagnostic(
                facts,
                spec,
                status="skipped",
                error="fallback candidate skipped because earlier candidates succeeded",
            ))
            continue
        if spec.pre_error is not None:
            if first_error is None:
                first_error = spec.pre_error
            diagnostics.append(_candidate_diagnostic(
                facts,
                spec,
                status="error",
                error=f"{type(spec.pre_error).__name__}: {spec.pre_error}",
            ))
            continue

        t0 = time.perf_counter()
        try:
            makespan, annotated = _verify_candidate_plan(bare, facts, seeds, spec)
            wall = time.perf_counter() - t0
            results.append(_CandidateResult(makespan, annotated, spec.name))
            diagnostics.append(_candidate_diagnostic(
                facts,
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
                facts,
                spec,
                status="error",
                wall_time_s=wall,
                error=f"{type(e).__name__}: {e}",
            ))

    return results, diagnostics, first_error
