"""Simulator-error repair support for PressureFit.

Pressure reduction uses an analytic boundary model. If the simulator later
finds a physical capacity failure, this module translates that failure into
extra pressure at the relevant task-start boundary. The main planner then runs
the same reducer again with the stronger boundary requirement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _modeled_boundary_need,
)
from dataflow_sim.core.schema import TaskChain

_PHYSICAL_REPAIR_LIMIT = 12

_DEADLOCK_RE = re.compile(
    r"task '([^']+)' deadlocked .* compute (\d+)\+(\d+) bytes > cap (\d+)"
)
_OUTPUT_NEED_RE = re.compile(
    r"task '([^']+)' cannot satisfy fast memory need of (\d+) bytes "
    r"\(current free \+ all scheduled offloads = (\d+), capacity=(\d+)\)"
)
_MISSING_INPUTS_RE = re.compile(
    r"task '([^']+)' deadlocked .* inputs \[([^\]]+)\] not live on compute"
)


@dataclass(frozen=True)
class _PhysicalPressureNeed:
    task_id: str
    actual_need: int
    overage: int


def _apply_physical_repair(
    msg: str,
    bare: TaskChain,
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    extra_pressure: list[int],
) -> bool:
    """Convert one simulator failure into extra pressure for one boundary."""
    physical = _physical_pressure_from_error(msg, facts, bare)
    if physical is None or bare.fast_memory_capacity is None:
        return False
    boundary_idx, observed_need, observed_overage = physical
    modeled_need = _modeled_boundary_need(facts, intervals, boundary_idx)
    required_extra = observed_need - modeled_need
    if required_extra <= extra_pressure[boundary_idx]:
        required_extra = extra_pressure[boundary_idx] + observed_overage
    extra_pressure[boundary_idx] = max(1, required_extra)
    return True


def _physical_pressure_from_error(
    msg: str,
    facts: _Facts,
    bare: TaskChain,
) -> tuple[int, int, int] | None:
    """Translate simulator capacity failures into boundary repair pressure.

    Returns `(boundary_index, observed_compute_need, observed_overage)`.
    `boundary_index == task_idx`, representing boundary `task_idx - 1` in
    PressureFit's +1 boundary-array convention.
    """
    need = _parse_physical_pressure_need(msg, facts, bare)
    if need is None:
        return None
    boundary_idx = _task_start_boundary_idx(bare, facts, need.task_id)
    if boundary_idx is None:
        return None
    return boundary_idx, need.actual_need, max(1, need.overage)


def _parse_physical_pressure_need(
    msg: str,
    facts: _Facts,
    bare: TaskChain,
) -> _PhysicalPressureNeed | None:
    for parser in (
        _parse_deadlock_capacity_need,
        _parse_output_reservation_need,
    ):
        need = parser(msg)
        if need is not None:
            return need
    return _parse_missing_input_need(msg, facts, bare.fast_memory_capacity)


def _parse_deadlock_capacity_need(msg: str) -> _PhysicalPressureNeed | None:
    """Parse simulator deadlock: current compute bytes plus requested bytes."""
    m = _DEADLOCK_RE.search(msg)
    if not m:
        return None
    task_id, used_s, need_s, cap_s = m.groups()
    actual_need = int(used_s) + int(need_s)
    return _PhysicalPressureNeed(
        task_id=task_id,
        actual_need=actual_need,
        overage=actual_need - int(cap_s),
    )


def _parse_output_reservation_need(msg: str) -> _PhysicalPressureNeed | None:
    """Parse simulator failure to reserve outputs before a task starts."""
    m = _OUTPUT_NEED_RE.search(msg)
    if not m:
        return None
    task_id, need_s, free_s, cap_s = m.groups()
    overage = int(need_s) - int(free_s)
    return _PhysicalPressureNeed(
        task_id=task_id,
        actual_need=int(cap_s) + overage,
        overage=overage,
    )


def _parse_missing_input_need(
    msg: str,
    facts: _Facts,
    fast_memory_capacity: int | None,
) -> _PhysicalPressureNeed | None:
    """Parse deadlock where queued transfers left task inputs unavailable."""
    m = _MISSING_INPUTS_RE.search(msg)
    if not m or fast_memory_capacity is None:
        return None
    task_id, inputs_s = m.groups()
    missing = re.findall(r"'([^']+)'", inputs_s)
    overage = sum(facts.sizes.get(oid, 0) for oid in missing)
    if overage <= 0:
        return None
    return _PhysicalPressureNeed(
        task_id=task_id,
        actual_need=fast_memory_capacity + overage,
        overage=overage,
    )


def _task_start_boundary_idx(
    bare: TaskChain,
    facts: _Facts,
    task_id: str,
) -> int | None:
    task_idx = next((i for i, task in enumerate(bare.tasks) if task.id == task_id), None)
    if task_idx is None:
        return None
    boundary_idx = task_idx  # boundary (task_idx - 1) stored as +1.
    if not (0 <= boundary_idx <= facts.n):
        return None
    return boundary_idx
