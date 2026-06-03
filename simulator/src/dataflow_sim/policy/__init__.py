from typing import Callable

from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policy.max_reduce import apply_max_reduce_policy
from dataflow_sim.policy.min_grow import apply_min_grow_policy
from dataflow_sim.policy.pressurefit import apply_pressurefit_policy
from dataflow_sim.policy.race_best import apply_race_best_policy
from dataflow_sim.policy.roundtrip_planner import apply_roundtrip_planner_policy
from dataflow_sim.policy.sliding_window import apply_sliding_window_policy
from dataflow_sim.schema import TaskChain

__all__ = [
    "apply_sliding_window_policy",
    "apply_belady_reactive_policy",
    "apply_roundtrip_planner_policy",
    "apply_race_best_policy",
    "apply_max_reduce_policy",
    "apply_min_grow_policy",
    "apply_pressurefit_policy",
    "get_all_policies",
]


PolicyFn = Callable[[TaskChain], TaskChain]


def get_all_policies() -> list[tuple[str, PolicyFn]]:
    """Return the canonical list of selectable policies as (name, fn) pairs.

    Each fn takes a bare TaskChain (with device_capacity already set) and
    returns the annotated chain. Adapters here paper over per-policy kwarg
    differences so callers can iterate uniformly — useful for sweeps,
    comparisons, and any "run every policy" workflow.

    Adding a new policy: import it above and append to the list below.
    """
    return [
        (
            "sliding_window",
            lambda b: apply_sliding_window_policy(
                b, window_size=2, device_capacity=b.device_capacity,
            ),
        ),
        (
            "belady_reactive",
            lambda b: apply_belady_reactive_policy(b, device_capacity=b.device_capacity),
        ),
        (
            "roundtrip_planner",
            lambda b: apply_roundtrip_planner_policy(b, device_capacity=b.device_capacity),
        ),
        (
            "race_best",
            lambda b: apply_race_best_policy(b, device_capacity=b.device_capacity),
        ),
        ("max_reduce", apply_max_reduce_policy),
        ("min_grow", apply_min_grow_policy),
        ("pressurefit", apply_pressurefit_policy),
    ]
