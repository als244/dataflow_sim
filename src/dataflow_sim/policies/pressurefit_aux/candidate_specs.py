"""Candidate-spec family assembly for PressureFit."""
from __future__ import annotations

from dataflow_sim.policies._common import _UseEvent
from dataflow_sim.policies.pressurefit_aux.core import _Facts
from dataflow_sim.policies.pressurefit_aux.initial_protection import (
    _initial_protection_sets,
)
from dataflow_sim.policies.pressurefit_aux.seeds import (
    _initial_residency,
    _pressure_initial_placement,
)
from dataflow_sim.policies.pressurefit_aux.types import (
    PressureFitPortfolioMode,
    _CandidateSpec,
    _SeedPortfolio,
)
from dataflow_sim.core.schema import TaskChain


def _resolve_portfolio_mode(
    facts: _Facts,
    requested_mode: PressureFitPortfolioMode,
) -> tuple[str, bool, bool]:
    auto_would_use_fast = _use_fast_portfolio(facts)
    fast_requested = (
        requested_mode == "fast"
        or (requested_mode == "auto" and auto_would_use_fast)
    )
    minimal_fast = fast_requested and _use_minimal_fast_portfolio(facts)
    effective_mode = (
        "fast-minimal" if minimal_fast
        else "fast" if fast_requested
        else "full"
    )
    return effective_mode, effective_mode != "full", minimal_fast


def _add_base_specs(specs: list[_CandidateSpec], minimal_fast: bool) -> None:
    if minimal_fast:
        specs.extend([
            _CandidateSpec("base-unpacked", "base", pack_inbound=False),
            _CandidateSpec(
                "source-gap-unpacked",
                "source-gap-trim",
                seed_key="source-gap",
                seed="source-gap-trim",
                pack_inbound=False,
            ),
            _CandidateSpec(
                "base-latest-inbound",
                "base",
                pack_inbound=True,
                latest_inbound=True,
                fallback_only=True,
            ),
            _skip_spec(
                "base-packed-fifo",
                "base",
                "fast-minimal portfolio skips secondary inbound schedules",
                pack_inbound=True,
            ),
            _skip_spec(
                "base-interval-entry",
                "base",
                "fast-minimal portfolio skips secondary inbound schedules",
                extend_inbound=True,
                respect_interval_start=True,
            ),
        ])
        return

    specs.extend([
        _CandidateSpec("base-packed-fifo", "base", pack_inbound=True),
        _CandidateSpec("base-unpacked", "base", pack_inbound=False),
        _CandidateSpec(
            "source-gap-unpacked",
            "source-gap-trim",
            seed_key="source-gap",
            seed="source-gap-trim",
            pack_inbound=False,
        ),
        _CandidateSpec(
            "base-interval-entry",
            "base",
            pack_inbound=False,
            extend_inbound=True,
            respect_interval_start=True,
        ),
        _CandidateSpec(
            "base-latest-inbound",
            "base",
            pack_inbound=True,
            latest_inbound=True,
        ),
    ])


def _add_reserve_spec(
    specs: list[_CandidateSpec],
    facts: _Facts,
    device_capacity: int | None,
    minimal_fast: bool,
) -> None:
    reserve_pressure = max(facts.next_outputs) if device_capacity is not None else 0
    if minimal_fast:
        specs.append(_skip_spec(
            "reserve-next-output",
            "slack-reserve",
            "fast-minimal portfolio skips secondary pressure seeds",
        ))
    elif reserve_pressure > 0:
        specs.append(_CandidateSpec(
            "reserve-next-output",
            "slack-reserve",
            pack_inbound=True,
            reserve_pressure=reserve_pressure,
        ))
    else:
        specs.append(_skip_spec(
            "reserve-next-output",
            "slack-reserve",
            "no finite-cap next-output reserve pressure",
        ))


def _add_cold_admission_spec(
    specs: list[_CandidateSpec],
    bare: TaskChain,
    facts: _Facts,
    sizes: dict[str, int],
    uses_by_task: dict[str, list[_UseEvent]],
    initial_device: set[str],
    seeds: _SeedPortfolio,
    minimal_fast: bool,
) -> None:
    if minimal_fast:
        specs.append(_skip_spec(
            "cold-admission-packed",
            "cold-admission",
            "fast-minimal portfolio skips secondary pressure seeds",
            seed="cold-admission",
            seed_key="cold-admission",
        ))
        return
    if bare.device_capacity is None:
        specs.append(_skip_spec(
            "cold-admission-packed",
            "cold-admission",
            "unlimited capacity",
            seed="cold-admission",
            seed_key="cold-admission",
        ))
        return

    cold_admission_cap = bare.device_capacity // 2
    if cold_admission_cap <= 0:
        specs.append(_skip_spec(
            "cold-admission-packed",
            "cold-admission",
            "cold admission cap is zero",
            seed="cold-admission",
            seed_key="cold-admission",
        ))
        return

    try:
        cold_initial = _pressure_initial_placement(
            bare, cold_admission_cap, sizes, uses_by_task,
        )
        if cold_initial == initial_device:
            specs.append(_skip_spec(
                "cold-admission-packed",
                "cold-admission",
                "cold initial set equals base initial set",
                seed="cold-admission",
                seed_key="cold-admission",
            ))
        else:
            seeds["cold-admission"] = _initial_residency(facts, cold_initial)
            specs.append(_CandidateSpec(
                "cold-admission-packed",
                "cold-admission",
                seed_key="cold-admission",
                seed="cold-admission",
                pack_inbound=True,
            ))
    except Exception as e:
        specs.append(_CandidateSpec(
            "cold-admission-packed",
            "cold-admission",
            seed_key="cold-admission",
            seed="cold-admission",
            pack_inbound=True,
            pre_error=e,
        ))


def _add_initial_protection_specs(
    specs: list[_CandidateSpec],
    facts: _Facts,
    device_capacity: int | None,
    inbound_bw: int | None,
    fast_portfolio: bool,
    effective_mode: str,
) -> None:
    protected_sets = _initial_protection_sets(facts, device_capacity, inbound_bw)
    if fast_portfolio:
        if protected_sets:
            specs.append(_skip_spec(
                "initial-protection-frontiers",
                "initial-protection",
                f"{effective_mode} portfolio omits {len(protected_sets)} frontier sets",
                seed="protected-initial",
                seed_key="all-host",
            ))
        return

    seen_protected_sets: set[frozenset[str]] = set()
    max_source_object_size = max(
        (facts.sizes[oid] for oid in facts.host_ids if facts.uses.get(oid)),
        default=0,
    )
    for protected_idx, protected in enumerate(protected_sets):
        frozen = frozenset(protected)
        if not frozen or frozen in seen_protected_sets:
            continue
        seen_protected_sets.add(frozen)
        protected_bytes = sum(facts.sizes[oid] for oid in protected)
        common = {
            "seed_key": "all-host",
            "seed": "protected-initial",
            "protected_initial": frozen,
            "extend_inbound": True,
        }
        if protected_idx == 0 or protected_bytes <= max_source_object_size:
            specs.append(_CandidateSpec(
                f"protected-{protected_idx}-packed-extended",
                "initial-protection",
                pack_inbound=True,
                **common,
            ))
        else:
            specs.append(_skip_spec(
                f"protected-{protected_idx}-packed-extended",
                "initial-protection",
                "protected set exceeds max single source-object size",
                pack_inbound=True,
                **common,
            ))
        specs.append(_CandidateSpec(
            f"protected-{protected_idx}-unpacked-extended",
            "initial-protection",
            pack_inbound=False,
            **common,
        ))
        specs.append(_CandidateSpec(
            f"protected-{protected_idx}-interval-entry",
            "initial-protection",
            pack_inbound=False,
            respect_interval_start=True,
            **common,
        ))


def _skip_spec(
    name: str,
    family: str,
    reason: str,
    *,
    seed: str = "base",
    seed_key: str = "base",
    **kwargs,
) -> _CandidateSpec:
    return _CandidateSpec(
        name=name,
        family=family,
        seed=seed,
        seed_key=seed_key,
        skip_reason=reason,
        **kwargs,
    )


def _use_fast_portfolio(facts: _Facts) -> bool:
    """Use a smaller candidate portfolio for very long chains."""
    return facts.n > 256 or len(facts.sizes) > 512


def _use_minimal_fast_portfolio(facts: _Facts) -> bool:
    """Use the primary fast candidate for very large interactive chains."""
    return facts.n > 4096 or len(facts.sizes) > 4096
