# max_reduce — clean general formulation

Implementation: [`src/dataflow_sim/policies/max_reduce.py`](../../../src/dataflow_sim/policies/max_reduce.py).

`max_reduce` is a from-first-principles auto policy that decouples *what* is resident from *when* transfers fire. This document defines the inputs, assumptions, outputs, invariants, and algorithm precisely enough that a reader can predict what the planner will do on any input.

## Motivation: why earlier reactive/constructive approaches fall short

A common failure mode shows up across the earlier reactive (`belady_reactive`) and constructive (`roundtrip_planner`) policies:

* **`belady_reactive`** reasons about state at simulation time via a "shadow simulator", choosing each eviction reactively. The reactive choice is **greedy-late** — it picks the latest boundary where an offload still completes by the deadline — which is correct for makespan but wastes the to_slow stream while compute is doing other things. When the cascade-eviction loop inside `_ensure_prefetch_v2` can't free enough at any prefetch boundary, it *throws* during plan construction (not just at simulation time); a caller comparing belady's makespan to a constructive planner's never gets to see belady's failure as a clean `None` because the exception escapes.
* **`roundtrip_planner`** does up-front constructive packing of "round-trip" candidates onto the streams. It only enumerates `(use_i → use_{i+1})` gaps, so it misses the longest gap of all — between an activation's **production** and its **first use** — which forces a paired reactive policy to handle activations badly. Its first-use prefetch fallback can also place a prefetch at an earlier boundary if it can't fit at the latest, cascading the over-commit forward until the chain becomes infeasible.
* The shadow simulator's `predict_schedule` (and the real simulator's) treats the FIFO transfer queue as un-blocked — assumes each queued transfer starts as soon as the previous ends. In reality a queued from_slow head can be **blocked on compute capacity** until a to_slow completes and frees bytes. Both planners and the real simulator used this optimistic prediction and produced plans that the simulator later couldn't execute.

**Root cause across both prior approaches:** the planners blend three concerns — *when* an object should be on compute, *how* triggers achieve that, and *whether* the streams can actually deliver it — and resolve them together, in slightly inconsistent ways, leading to combinations where the planner's notion of "feasible" doesn't match the simulator's. `max_reduce` factors these into three sequential phases.

## Approach

max_reduce decouples the three concerns into three sequential phases:

1. **Residency** (memory). Decide *when* each object is on compute. Care only about boundary capacity. No streams, no times-in-microseconds.
2. **Triggers** (mechanism). Derive the precise releases/offloads/prefetches/pre-placement that realize the residency. Local rewrite of the residency layout, no scheduling.
3. **Streams** (timing). The real simulator is the timing arbiter. The simulator hard-enforces the cap invariant; if max_reduce's plan would over-commit at any moment, the simulator raises.

Each phase has well-defined inputs, outputs, and invariants. No cross-phase fallbacks; if a phase can't satisfy its invariant, the planner raises.

## Inputs

A `TaskChain` with:

* `tasks`: ordered list `t_0, …, t_{n-1}`. Each task has:
  * `inputs` (list of obj ids consumed, *read-only by default*),
  * `outputs` (list of `OutputAlloc(id, size, location, type)` produced — outputs always introduce fresh ids),
  * `mutates_inputs` (subset of `inputs` whose contents are *modified* by this task — see Mutation below),
  * `runtime` (µs).
* `initial_memory`: list of `Object(id, size, location, type)` present before any task runs. Each is `location ∈ {compute, backing}`.
* `fast_memory_capacity`: int bytes (`None` = unlimited).
* `bandwidth_from_slow`, `bandwidth_to_slow`: int bytes/µs.
* Optional `backing_memory_capacity`.

## Assumptions

* **Static, known chain.** All task runtimes, sizes, and dependencies are known up front. No data-dependent control flow.
* **Read-only-by-default + explicit mutation.** A task does NOT modify the byte contents of its inputs *unless* the input id appears in `mutates_inputs`. Any input listed in `mutates_inputs` will have its compute-resident bytes updated by the task, so a subsequent release would discard the update. max_reduce does not know about domain names such as "gradient" or "state"; it just acts on whatever `mutates_inputs` declares.
* **Outputs introduce fresh ids.** No task's `outputs` reuses an existing object's id. So every object has a single producer (either initial-memory or one task).
* **One to_slow FIFO + one from_slow FIFO.** Transfers on the same direction serialize; the two directions run in parallel.
* **Boundary semantics.** Boundary `k` is the snapshot *after* task `k`'s end-of-task triggers fire (releases, offloads, prefetches enqueued). Boundary `-1` is the simulator's initial state (just `initial_memory` plus any pre-placed objects).

## Output

An annotated `TaskChain` with:

* `initial_memory` extended with compute-residency copies of backing objects that max_reduce chose to pre-place.
* Each `Task` carries `releases_after`, `offload_after`, `prefetch_after` lists realizing max_reduce's plan.

## Invariants the output must satisfy

1. **Capacity.** At every boundary `k`, the compute pool size plus the size of next-task compute-located outputs is ≤ `fast_memory_capacity`. (max_reduce reasons about this in *logical* residency; the simulator enforces the physical version with its hard `pool > cap` assertion.)
2. **Input liveness.** For every task `t_k`, every input is live on compute at boundary `k-1`.
3. **Output room.** At boundary `k-1`, the compute has room for `t_k`'s compute-located outputs.
4. **Trigger validity.** Every `release` and `offload` references an object live on compute at that boundary; every `prefetch` references an object with a live backing source.
5. **Mutation preservation.** For every backing-initial object that is mutated at any point in the chain, the final residency interval's departure trigger is an **offload** (writeback) — never a release. After the chain ends, every mutated backing-initial object is live on backing with the updated bytes.
6. **Simulator equivalence.** Running the annotated chain through the simulator produces no errors and the resulting timeline matches the residency plan up to transit-time stalls (which the simulator handles via its drain loop).

## Phase 1 — Residency

Goal: decide, for each object `o`, the set of boundaries at which `o` is live on compute. This set is a union of contiguous intervals `[a, b]` (inclusive of both ends). Phase 1 cares about memory only — no transfers, no times.

### Object roles

For each object `o`:

* `producer(o)`: task index that produces `o` as an output, or `-1` if `o` is in `initial_memory`.
* `uses(o)`: sorted list of task indices that consume `o` as input.
* `backing_source(o)`: `True` iff `o` is in `initial_memory` with `location == "backing"`.
* `is_mutated(o)`: `True` iff any task `t` in the chain has `o ∈ t.mutates_inputs`.
* `appears_at(o)`: boundary at which `o` first becomes available on the compute side without any planner action — `-1` for compute-initial, `producer(o)` for task outputs, "via prefetch only" for backing-initial (max_reduce decides whether to pre-place via Phase 1's reduction).

### Mandatory boundaries

For each `o`:

* **Use boundaries:** `{ u - 1 : u ∈ uses(o) }`. `o` must be live at each of these for the consuming task to start.
* **Production boundary:** if `producer(o) ≥ 0`, then `producer(o)` is mandatory.
* **Initial boundary:** if `o` is compute-initial, then `-1` is mandatory.

### Feasibility lower bound

Let `min_pool[k] = Σ_o {size(o) : k ∈ Mandatory(o)} + reserved_outputs[k+1]`. If `min_pool[k] > cap` for some `k`, the chain is infeasible and Phase 1 raises.

### Starting from the MAX plan

Begin with each object having one residency interval covering its entire active span:

* For backing-initial / compute-initial `o` with uses: `[-1, last_use(o) - 1]`.
* For task-output `o` with uses: `[producer(o), last_use(o) - 1]`.
* For object with no use (rare): single point at `appears_at(o)`.

Note: the MAX residency for a mutated backing-initial object is the SAME shape as for a clean backing-initial object. The mutation status only matters in Phase 2 (trigger type). Phase 1 only sees sizes and boundaries.

If `pool[k] + reserved_outputs[k+1] ≤ cap` for every `k`, Phase 1 is done.

### Reduction loop

While there exists `k` with `pool[k] + reserved_outputs[k+1] > cap`:

1. Find the most-overloaded boundary `k*`.
2. From objects "in pool" at `k*`, find an **eligible victim** — an object `v` whose residency covers `k*` AND can be split to exclude `k*` while keeping all of `v`'s anchors (production + use boundaries) covered.
3. Rank eligible victims by `(stream_cost, not_drop_init, -first_use, -size, -gap_length)`:
   * `stream_cost`:
     * For `drop_init` evictions (un-pre-placing): always `0`. The only added cost is **1 from_slow** (just-in-time prefetch when the obj is first needed). The writeback offload for mutated objs happens regardless of pre-placement, so it doesn't count.
     * For mid-life splits: `0` if release-eligible (backing source AND never mutated), `1` otherwise (round-trip = to_slow + from_slow).
   * `not_drop_init = 0` for drop-init evictions, `1` for mid-life splits.
   * `-first_use`: prefer evicting objects whose FIRST use is **latest** in the chain. Keeping pre-placed objs that are used SOON saves more (avoids forward from_slow contention where the stream is busy); late-first-use objs can be prefetched in backward where from_slow has slack.
   * `-size` and `-gap_length` as final tie-breakers favor big bytes and long freed stretches.

   Why first-use matters: the value of pre-placing an obj X is "we don't have to prefetch X". That value is the same per-obj (1 from_slow saved). But the *cost* of needing to prefetch X depends on **when** — if X is used early in forward, the prefetch competes with from_slow for forward-pass weights; if X is used late in backward, from_slow has plenty of slack. So pre-place early-use objs first; defer late-use objs to runtime prefetch. The prior "stream_cost preferring non-mutated" was a category error — it favored evicting weights (low cost-to-evict per dimension) when the right thing was to keep weights pre-placed (their pre-placement is high-value because they're used soon).
4. Split `v`'s covering interval `[a, b]` at the gap surrounding `k*`:
   * `anchors_le = {c ∈ anchors(v) ∩ [a, b] : c ≤ k* - 1}`,
   * `anchors_ge = {c ∈ anchors(v) ∩ [a, b] : c ≥ k* + 1}`,
   * New intervals: `[a, max(anchors_le)]` and `[min(anchors_ge), b]`. If either side has no anchors, drop that piece entirely (= un-pre-place or kill final tail).
5. Recompute `pool`.

Termination: each iteration strictly increases the number of distinct residency intervals; the count is bounded above by total `(uses + production)` anchors across all objects.

### Effective residency for pool tracking

When computing `pool[k]`, max_reduce accounts for one piece of simulator-side timing:

* **Prefetch arrival on the left edge.** For any interval `[a, b]` whose start `a` is NOT `-1` AND not the producer boundary, the simulator creates the compute entry the moment the from_slow trigger fires (= end of task `a - 1`) and the entry persists through from_slow completion. So the obj actually occupies compute bytes from boundary `a - 1` onward, not just from boundary `a`. max_reduce extends each prefetched interval's *effective* start by one boundary leftward when computing pool size.

max_reduce deliberately does NOT model to_slow transit time in Phase 1 — the simulator's drain loop stalls compute when a queued to_slow is still occupying bytes, which preserves correctness without making Phase 1 simulate the streams.

## Phase 2 — Triggers

Goal: turn each object's residency intervals into concrete `initial_compute` pre-placement and per-task triggers.

For each object `o` with residency intervals `[a_1, b_1] < [a_2, b_2] < … < [a_m, b_m]`:

**Initial placement.** If `a_1 == -1`:
* If `backing_source(o)`: add `o.id` to `initial_compute`.
* If compute-initial: already on compute, no action.

**Per-interval arrivals.** For each interval `i = 1..m`:
* If `i == 1` and `a_1 == -1`: initial placement (no trigger).
* If `i == 1` and `a_1 == producer(o) ≥ 0`: natural production (no trigger).
* Otherwise: emit `prefetch_after(o.id)` on `task[a_i - 1]`.

**Per-interval departures.** For each interval `[a, b]`, the trigger fires at the EARLIEST task at which `o` is no longer needed in this interval. Compute:

* `use_tasks_in = { u ∈ uses(o) : a ≤ u - 1 ≤ b }`
* `production_in = { producer(o) }` if `producer(o)` is in `[a, b]`, else `{}`
* `fire_task = max(use_tasks_in ∪ production_in)`

If `fire_task` is the production task (no use in interval), the trigger fires on the SAME task that produces the object — step-7 marks the output live, then step-9 (offload) immediately queues the to_slow. This is the key fix in max_reduce for the activation-offload-too-late bug: an activation produced by `f_i` with no immediate use should have its offload fire at `f_i`'s end, not at `f_{i+1}`'s end.

If `fire_task >= n` (= the obj naturally outlives the chain), no trigger is needed.

**Trigger type.** Determine whether `o` is *dirty* at end of this interval:

* `mutations_in = { t : t in interval AND o ∈ tasks[t].mutates_inputs }`. (For `t` to be "in interval", `t - 1 ∈ [a, b]`, i.e., this interval's residency covers the use that mutates.)
* `dirty_after = (mutations_in is non-empty)`.

Pick the trigger:

* `is_last = (i == m)`.
* Case A — `dirty_after AND has_backing_source`: emit `offload_after(o.id)` on `task[fire_task]`. The mutation made the backing copy stale; we must write back.
* Case B — `dirty_after AND not has_backing_source`: emit `offload_after(o.id)` on `task[fire_task]`. The mutation needs to be preserved for the next interval's prefetch.
* Case C — `not dirty_after AND has_backing_source`: emit `release_after(o.id)`. Backing copy is identical to the fast-memory copy — drop the fast side without a to_slow.
* Case D — `not dirty_after AND not has_backing_source AND not is_last`: emit `offload_after(o.id)`. No backing source AND another interval will need it back — must round-trip.
* Case E — `not dirty_after AND not has_backing_source AND is_last`: emit `release_after(o.id)`. Object dies; no need to write to backing.

The Phase 2 invariant **mutation preservation** falls out of cases A/B: any interval ending dirty emits an offload, so the backing eventually catches up. By the time the final interval ends, the backing has the latest bytes.

## Phase 3 — EDF prefetch scheduling on the from_slow FIFO

Naively, max_reduce emits each prefetch trigger one task before its consumer (`task[a-1]`). That's correct when the from_slow stream is idle — but when several prefetches share the FIFO, the queued transfers complete *after* their consumers' start times, and the simulator stalls compute waiting for the from_slow. Sliding-window avoids this by pre-firing dW prefetches several tasks early; max_reduce needs the same idea, generalized.

**Principle:** when transfers share a FIFO stream and each has a known deadline, **issue them in deadline order as early as the DAG and memory cap allow** — not as late as memory permits. The FIFO orders delivery; the planner's job is to push enough work in early enough that the queue is never empty when a deadline arrives.

**Algorithm (EDF-backward packing):**

1. Enumerate prefetched intervals (those that fire prefetch triggers in Phase 2). Each has a deadline = `task_end[a-1]` (the consumer's start time in ideal-runtime µs) and `tau = ceil(size / bw_from_slow)`.
2. Sort by deadline DESC.
3. Walk in this order. Maintain `next_start_t` (the start time of the most-recently-processed event, initially `+∞`).
   * `end_t = min(deadline, next_start_t)`
   * `start_t = end_t - tau`
   * `fire_task = max k with task_end[k] ≤ start_t`
   * `ideal_new_a = max(fire_task + 1, earliest_a)` where `earliest_a` is bounded by the obj's producer or prior-interval-end to keep the residency valid.
4. If `ideal_new_a < current_a`, extend the interval left to `(ideal_new_a, b)` — but only if the extension doesn't push pool over cap at any of the newly-occupied boundaries. Walk forward from `ideal_new_a` toward `current_a` and pick the earliest target that fits.

Extending an interval left = extra residency on compute = more pool pressure during the extension. Phase 3 respects the cap (via the same pool model as Phase 1), so a tight cap may bound how much lead time a prefetch gets. Untight caps allow extensions up to the EDF target.

The simulator remains the timing arbiter — if max_reduce's plan is logically feasible but the streams can't drain fast enough, the simulator extends makespan via compute stalls. The simulator hard-enforces the cap invariant; over-commits raise `task X deadlocked: compute pool > cap, no offloads in flight`.

## Termination & complexity

* Phase 1: at most `O(Σ|uses|)` iterations, each `O(n × #objects)` for the overflow scan and victim selection. For a chain with `n = O(L)` tasks and `#objects = O(L)`, this is `O(L³)` — tractable for `L < 200`.
* Phase 2: `O(Σ|intervals|) = O(n × #objects)`.

## What max_reduce does NOT do

* No stream-aware co-scheduling (which prefetches overlap which offloads). The planner emits triggers; the simulator's FIFOs decide order.
* No bandwidth-aware initial placement decisions — Phase 1 reduces purely on memory.
* No automatic re-prioritization of victims based on stream pressure.

These keep max_reduce simple and predictable. Layered optimizations (a "Phase 4" stream-feedback loop) are future work, not part of the base.

## Comparison to prior versions

| Aspect | belady_reactive | roundtrip_planner | max_reduce |
|---|---|---|---|
| Decision style | reactive at sim time | constructive up-front | constructive up-front |
| Eviction trigger | when pool[k] binds during shadow walk | overflow at any covered boundary in `bps` | overflow at any boundary in `pool` (with prefetch-arrival left edge) |
| Activation offload timing | reactive eviction, greedy-late | round-trip pack of consecutive-use gaps | residency reduction makes activations production-only intervals → offload trigger fires AT the production task |
| Production→first-use gap | implicit via reactive eviction | enumerated as a special candidate | implicit — production boundary is `appears_at`, eviction can split anywhere |
| Cross-phase fallback | roundtrip_planner → belady_reactive in `apply_auto_policy` | falls through to belady_reactive if roundtrip_planner fails | none — each phase succeeds or raises with a precise reason |
| Writeback handling | added as post-pass keyed on `type=="gradient"` | added as post-pass keyed on `type=="gradient"` | first-class via `Task.mutates_inputs` — no name-matching, fully general |
| Lines of code | ~700 | ~700 | ~280 |
