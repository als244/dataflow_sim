from typing import Callable

from dataflow_sim.policies.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policies.max_reduce import apply_max_reduce_policy
from dataflow_sim.policies.min_grow import apply_min_grow_policy
from dataflow_sim.policies.pressurefit import (
    PressureFitCandidateDiagnostic,
    PressureFitDiagnostics,
    apply_pressurefit_policy,
    plan_pressurefit_policy,
)
from dataflow_sim.policies.roundtrip_planner import apply_roundtrip_planner_policy
from dataflow_sim.policies.sliding_window import apply_sliding_window_policy
from dataflow_sim.core.schema import TaskChain

__all__ = [
    "apply_sliding_window_policy",
    "apply_belady_reactive_policy",
    "apply_roundtrip_planner_policy",
    "apply_max_reduce_policy",
    "apply_min_grow_policy",
    "apply_pressurefit_policy",
    "plan_pressurefit_policy",
    "PressureFitCandidateDiagnostic",
    "PressureFitDiagnostics",
    "get_all_policies",
]


PolicyFn = Callable[[TaskChain], TaskChain]


def get_all_policies() -> list[tuple[str, PolicyFn]]:
    """Return the canonical list of selectable policies as (name, fn) pairs.

    Each fn takes a bare TaskChain (with fast_memory_capacity already set) and
    returns the annotated chain. Adapters here paper over per-policy kwarg
    differences so callers can iterate uniformly — useful for sweeps,
    comparisons, and any "run every policy" workflow.

    Adding a new policy: import it above and append to the list below.
    """
    return [
        (
            "sliding_window",
            lambda b: apply_sliding_window_policy(
                b, window_size=2, fast_memory_capacity=b.fast_memory_capacity,
            ),
        ),
        (
            "belady_reactive",
            lambda b: apply_belady_reactive_policy(b, fast_memory_capacity=b.fast_memory_capacity),
        ),
        (
            "roundtrip_planner",
            lambda b: apply_roundtrip_planner_policy(b, fast_memory_capacity=b.fast_memory_capacity),
        ),
        ("max_reduce", apply_max_reduce_policy),
        ("min_grow", apply_min_grow_policy),
        ("pressurefit", apply_pressurefit_policy),
    ]
