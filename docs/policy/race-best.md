# race_best

A meta-policy that composes `belady_reactive` (V2) and `roundtrip_planner` (V3): builds both annotated chains independently, simulates each, and returns the one with the lower makespan. Provides robustness — when one planner fails on a workload but the other succeeds, the race transparently returns the survivor — at the cost of running two planners per call.

## Mechanism

1. Run `apply_belady_reactive_policy(bare, ...)` inside a try/except, capturing the chain or the exception.
2. Run `apply_roundtrip_planner_policy(bare, ...)` inside a try/except, capturing the chain or the exception.
3. Resolve based on which branches survived:
   - Both succeeded: probe each chain's makespan via `_try_makespan`. Return the lower; on tie, prefer V2.
   - Only one succeeded: return that one.
   - Both failed: re-raise the **V2 exception** (V2's errors are typically more informative about workload infeasibility than V3's).
4. Makespan probes themselves can fail (e.g., simulator rejects the plan). If a probe fails for one branch, prefer the other; if both probes fail, default to V2.

The two planners share `bare`, `device_capacity`, and `refinement_iters` arguments; they do not share intermediate state, and their output chains are independent.

## When it wins / when it loses

| Regime | Outcome |
|---|---|
| Workloads where V2 and V3 trade wins across capacities | Wins — picks the better of the two per call. |
| Cap regimes where one planner is brittle (V3 cascade failures, V2 capacity errors) | Wins — survivor's plan is used; user sees no failure. |
| Single-planner-dominant regimes | Marginal — pays 2× planning cost for the same answer. |
| Both planners fail | Loses — re-raises the V2 exception (V2-fallback behavior). |

## Why the fallback raises V2's exception, not V3's

V2 (`belady_reactive`) is the broader-coverage baseline: it handles a wider slice of cap regimes and workload shapes, so when it fails the failure usually indicates a genuine workload-level problem (infeasible capacity, malformed chain, eviction-set exhaustion). Its exceptions tend to carry concrete diagnostic info about *why* the workload can't be planned. V3 (`roundtrip_planner`) failures, by contrast, are more often feasibility/packing edge cases inside the round-trip search itself — brittle in narrower ways and less informative when surfaced standalone. So when both planners crash on the same input, the V2 exception is the more useful one to hand back to the caller.

## Cost / benefit

`race_best` runs BOTH planners and then replays each chain through the simulator to measure makespan, so it costs roughly 2× planning + 2× simulation versus picking a single planner up front. Worth it when:

- You don't know which planner suits your workload (exploratory runs, sweeps across new shapes/capacities).
- You want a robustness safety net so a single-planner crash doesn't sink the run.

Not worth it when:

- You've already benchmarked your workload and one planner consistently wins — just call it directly.
- Planning time matters (interactive use, tight loops, large sweeps where 2× adds up).

## Extensibility

The race pattern generalizes cleanly: future policies like `reduce_from_max` and `min_grow` could be folded into the race with no changes to the base policies themselves. Each policy is self-contained (takes `bare` + standard kwargs, returns a `TaskChain`), and the race wrapper only needs to know how to call them and probe a makespan — it picks the lower makespan and otherwise treats each policy as a black box. Adding a fourth or fifth contender is mostly bookkeeping (more try/except branches, more makespan probes, same resolution logic).

## Implementation

`simulator/src/dataflow_sim/policy/race_best.py` — entry: `apply_race_best_policy(bare, *, device_capacity=None, refinement_iters=20)`.
