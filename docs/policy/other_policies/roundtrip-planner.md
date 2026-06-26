# roundtrip_planner

Constructively enumerates offload/prefetch round-trips over the oracle reference stream and packs them onto to-slow/from-slow timelines before the simulator runs. For each object, gaps between consecutive uses that fit `tau_to_slow + tau_from_slow` become candidate round-trips; the planner ranks them by `size × gap_length`, then greedily commits only those that actually relieve a predicted capacity overflow. Mandatory first-use prefetches for backing-only objects are packed in a second pass. Implemented in `src/dataflow_sim/policies/roundtrip_planner.py`.

## Mechanism

1. **Reference stream** — `_object_uses_by_task_idx` produces per-object sorted `_UseEvent`s from ideal start times.
2. **Initial placement** — reuses the slack-based initial-pool heuristic (dropping initial weights just shifts bytes to prefetches; net zero).
3. **Enumeration** — `_enumerate_roundtrips` emits a `_RoundTrip` for each consecutive-use pair whose gap fits offload + prefetch. `_enumerate_first_use_prefetches` emits mandatory prefetches for backing-only objects.
4. **Ranking** — `(-size × gap_length, -span_tasks, prev_use_task_idx)`: bigger objects over longer windows first, deterministic tiebreak by earliest use.
5. **Packing** (`_pack_roundtrips`, two passes):
   - **Pass 1 (round-trips)** — for each candidate in priority order, find the latest valid `(k_off, k_pre)` with free to-slow/from-slow stream slots. **Skip unless** some boundary in `[k_off+1, k_pre]` has `overflow_at(k) = bps[k] + first_use_demand_at(k) + next_outputs[k] − cap > 0`. This demand-driven filter is what prevents inflating makespan at loose caps.
   - **Pass 2 (first-use prefetches)** — walk back from `first_use_task_idx − 1`, place at the latest k that satisfies stream feasibility, arrival deadline, and cap budget across `[k, last_use]`. Fallback to latest stream-feasible k if nothing fits.
6. **Structural releases** — append last-use releases for every compute-resident object, mirroring `belady_reactive`'s GC.
7. **Verification** — chain runs through `_verify_and_refine` (Phase 5 trigger-shift safety net).

## When it wins

- **Moderate caps with long forward→backward gaps**: round-trips fit into natural idle intervals, freeing capacity-bytes-time that `belady_reactive`'s eviction can't reclaim. Validated wins: L=5 cap=600 (−10 ticks), L=10 cap=800 (−14 ticks).
- **Loose caps**: demand-driven filter commits zero round-trips; resulting plan matches `belady_reactive` minus its unnecessary trailing offloads. Ties `belady_reactive` everywhere it doesn't win.

## Limitations

- **Greedy at tight caps**: per-candidate fit-or-skip can miss feasible plans that reactive eviction finds. NP-hard in general (multi-resource, capacity-constrained, FIFO stream contention). At tight caps prefer `belady_reactive` or `pressurefit` instead.
- **Fixed recompute level** `k=0`; no joint compute/transfer tradeoff.
- **Oracle-dependent**: needs the full reference stream; no online variant.
- **Pass order is load-bearing**: first-use-before-round-trips inflates bps and bunches prefetches into bad slots.
