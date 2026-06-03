"""Race wrapper: run V2 reactive Belady AND V3 round-trip planner, pick the
plan with the lower makespan.

Both planners are run independently. Failures in either branch are caught so
a planner crash doesn't sink the race; if only one survives, that one wins.
If both crash, the V2 exception is surfaced (V2 is the safer fallback and its
errors are usually more informative about workload infeasibility).
"""
from __future__ import annotations

from dataflow_sim.schema import TaskChain
from dataflow_sim.policy._common import _try_makespan
from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policy.roundtrip_planner import apply_roundtrip_planner_policy


def apply_race_best_policy(
    bare: TaskChain,
    *,
    device_capacity: int | None = None,
    refinement_iters: int = 20,
) -> TaskChain:
    """Build BOTH plans and return the one whose simulated makespan is lower.

    On failures:
      * if only V2 succeeds, return V2's chain;
      * if only V3 succeeds, return V3's chain;
      * if both fail, re-raise the V2 exception (V2 errors are typically the
        most informative about workload infeasibility).
    """
    v2_chain: TaskChain | None = None
    v2_err: BaseException | None = None
    try:
        v2_chain = apply_belady_reactive_policy(
            bare,
            device_capacity=device_capacity,
            refinement_iters=refinement_iters,
        )
    except Exception as e:
        v2_err = e

    v3_chain: TaskChain | None = None
    v3_err: BaseException | None = None
    try:
        v3_chain = apply_roundtrip_planner_policy(
            bare,
            device_capacity=device_capacity,
            refinement_iters=refinement_iters,
        )
    except Exception as e:
        v3_err = e

    if v2_chain is None and v3_chain is None:
        # Surface the V2 error (more informative about feasibility).
        assert v2_err is not None
        raise v2_err

    if v2_chain is None:
        assert v3_chain is not None
        return v3_chain
    if v3_chain is None:
        return v2_chain

    v2_ms = _try_makespan(v2_chain, refinement_iters)
    v3_ms = _try_makespan(v3_chain, refinement_iters)

    # If a makespan probe fails for one branch, prefer the other.
    if v2_ms is None and v3_ms is None:
        return v2_chain  # both probes failed; default to V2
    if v2_ms is None:
        return v3_chain
    if v3_ms is None:
        return v2_chain

    return v2_chain if v2_ms <= v3_ms else v3_chain
