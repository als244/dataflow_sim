"""Candidate portfolio orchestration for PressureFit."""
from __future__ import annotations

from dataflow_sim.policy._common import _UseEvent
from dataflow_sim.policy.pressurefit_aux.candidate_specs import (
    _add_base_specs,
    _add_cold_admission_spec,
    _add_initial_protection_specs,
    _add_reserve_spec,
    _resolve_portfolio_mode,
)
from dataflow_sim.policy.pressurefit_aux.core import _Facts
from dataflow_sim.policy.pressurefit_aux.types import (
    PressureFitPortfolioMode,
    _CandidateSpec,
    _PortfolioPlan,
    _SeedPortfolio,
)
from dataflow_sim.schema import TaskChain


def _build_candidate_portfolio(
    bare: TaskChain,
    facts: _Facts,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    initial_device: set[str],
    seeds: _SeedPortfolio,
    requested_mode: PressureFitPortfolioMode,
) -> _PortfolioPlan:
    """Construct the bounded candidate list."""
    effective_mode, fast_portfolio, minimal_fast = _resolve_portfolio_mode(
        facts, requested_mode,
    )
    specs: list[_CandidateSpec] = []

    _add_base_specs(specs, minimal_fast)
    _add_reserve_spec(specs, facts, bare.device_capacity, minimal_fast)
    _add_cold_admission_spec(
        specs, bare, facts, sizes, uses_by_task, initial_device, seeds,
        minimal_fast,
    )
    _add_initial_protection_specs(
        specs, facts, bare.device_capacity, bare.bandwidth_h2d,
        fast_portfolio, effective_mode,
    )

    return _PortfolioPlan(
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        fast_portfolio=fast_portfolio,
        specs=specs,
    )
