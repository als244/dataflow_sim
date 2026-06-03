> **Policy:** `max_reduce` (was: V4) — implementation in `simulator/src/dataflow_sim/policy/max_reduce.py`. This is the LIVE design doc.

# Auto-policy V4 — clean general formulation

V1/V2/V3 grew organically and accumulated subtle interactions that broke at edge cases. V4 is a re-derivation from first principles. This document defines the inputs, assumptions, outputs, invariants, and algorithm precisely enough that someone reading it can predict what the planner will do on any input.

## Why prior versions failed

A common failure mode runs across V1/V2/V3:

* **V2** reasons about state at simulation time via a "shadow simulator", choosing each eviction reactively. The reactive choice is **greedy-late** — it picks the latest boundary where an offload still completes by the deadline — which is correct for makespan but wastes the d2h stream while compute is doing other things. When the cascade-eviction loop inside `_ensure_prefetch_v2` can't free enough at any prefetch boundary, it *throws* during plan construction (not just at simulation time); a caller comparing V2's makespan to V3's never gets to see V2's failure as "v2_ms = None" because the exception escaped.
* **V3** does up-front constructive packing of "round-trip" candidates onto the streams. It only enumerated `(use_i → use_{i+1})` gaps, so it missed the longest gap of all — between an activation's **production** and its **first use** — which forced V2's reactive eviction to handle activations badly. Its first-use prefetch fallback can also place a prefetch at an earlier boundary if it can't fit at the latest, cascading the over-commit forward until the chain becomes infeasible.
* The shadow simulator's `predict_schedule` (and the real simulator's, as it turned out) treats the FIFO transfer queue as un-blocked — assumes each queued transfer starts as soon as the previous ends. In reality a queued h2d head can be **blocked on device capacity** until a d2h completes and frees bytes. Both planners and the real simulator used this optimistic prediction and produced plans that the simulator later couldn't execute.

**Root cause across all three:** the planners blend three concerns — *when* an object should be on device, *how* triggers achieve that, and *whether* the streams can actually deliver it — and resolve them together, in slightly inconsistent ways, leading to combinations where the planner's notion of "feasible" doesn't match the simulator's. Each fix shifted the inconsistency rather than removing it.

## Approach

V4 decouples the three concerns into three sequential phases:

1. **Residency** (memory). Decide *when* each object is on device. Care only about boundary capacity. No streams, no times-in-microseconds.
2. **Triggers** (mechanism). Derive the precise releases/offloads/prefetches/pre-placement that realize the residency. Local rewrite of the residency layout, no scheduling.
3. **Streams** (timing). The real simulator is the timing arbiter. The simulator hard-enforces the cap invariant; if V4's plan would over-commit at any moment, the simulator raises.

Each phase has well-defined inputs, outputs, and invariants. No cross-phase fallbacks; if a phase can't satisfy its invariant, the planner raises.

## Inputs

A `TaskChain` with:

* `tasks`: ordered list `t_0, …, t_{n-1}`. Each task has:
  * `inputs` (list of obj ids consumed, *read-only by default*),
  * `outputs` (list of `OutputAlloc(id, size, location, type)` produced — outputs always introduce fresh ids),
  * `mutates_inputs` (subset of `inputs` whose contents are *modified* by this task — see Mutation below),
  * `runtime` (µs).
* `initial_memory`: list of `Object(id, size, location, type)` present before any task runs. Each is `location ∈ {device, host}`.
* `device_capacity`: int bytes (`None` = unlimited).
* `bandwidth_h2d`, `bandwidth_d2h`: int bytes/µs.
* Optional `host_capacity`.

## Assumptions

* **Static, known chain.** All task runtimes, sizes, and dependencies are known up front. No data-dependent control flow.
* **Read-only-by-default + explicit mutation.** A task does NOT modify the byte contents of its inputs *unless* the input id appears in `mutates_inputs`. Mutation is the generalization of the prior "gradient writeback" convention: any input listed in `mutates_inputs` will have its device-resident bytes updated by the task, so a subsequent release would discard the update. In the transformer training chain, `b_i.mutates_inputs = [dW_i]` and `head.mutates_inputs = [dW_head]` — but V4 doesn't know about the names "dW" or "gradient"; it just acts on whatever `mutates_inputs` declares.
* **Outputs introduce fresh ids.** No task's `outputs` reuses an existing object's id. So every object has a single producer (either initial-memory or one task).
* **One d2h FIFO + one h2d FIFO.** Transfers on the same direction serialize; the two directions run in parallel.
* **Boundary semantics.** Boundary `k` is the snapshot *after* task `k`'s end-of-task triggers fire (releases, offloads, prefetches enqueued). Boundary `-1` is the simulator's initial state (just `initial_memory` plus any pre-placed objects).

## Output

An annotated `TaskChain` with:

* `initial_memory` extended with device-residency copies of host objects that V4 chose to pre-place.
* Each `Task` carries `releases_after`, `offload_after`, `prefetch_after` lists realizing V4's plan.

## Invariants the output must satisfy

1. **Capacity.** At every boundary `k`, the device pool size plus the size of next-task device-located outputs is ≤ `device_capacity`. (V4 reasons about this in *logical* residency; the simulator enforces the physical version with its hard `pool > cap` assertion.)
2. **Input liveness.** For every task `t_k`, every input is live on device at boundary `k-1`.
3. **Output room.** At boundary `k-1`, the device has room for `t_k`'s device-located outputs.
4. **Trigger validity.** Every `release` and `offload` references an object live on device at that boundary; every `prefetch` references an object with a live host source.
5. **Mutation preservation.** For every host-initial object that is mutated at any point in the chain, the final residency interval's departure trigger is an **offload** (writeback) — never a release. After the chain ends, every mutated host-initial object is live on host with the updated bytes.
6. **Simulator equivalence.** Running the annotated chain through the simulator produces no errors and the resulting timeline matches the residency plan up to transit-time stalls (which the simulator handles via its drain loop).

## Phase 1 — Residency

Goal: decide, for each object `o`, the set of boundaries at which `o` is live on device. This set is a union of contiguous intervals `[a, b]` (inclusive of both ends). Phase 1 cares about memory only — no transfers, no times.

### Object roles

For each object `o`:

* `producer(o)`: task index that produces `o` as an output, or `-1` if `o` is in `initial_memory`.
* `uses(o)`: sorted list of task indices that consume `o` as input.
* `host_source(o)`: `True` iff `o` is in `initial_memory` with `location == "host"`.
* `is_mutated(o)`: `True` iff any task `t` in the chain has `o ∈ t.mutates_inputs`.
* `appears_at(o)`: boundary at which `o` first becomes available on the device side without any planner action — `-1` for device-initial, `producer(o)` for task outputs, "via prefetch only" for host-initial (V4 decides whether to pre-place via Phase 1's reduction).

### Mandatory boundaries

For each `o`:

* **Use boundaries:** `{ u - 1 : u ∈ uses(o) }`. `o` must be live at each of these for the consuming task to start.
* **Production boundary:** if `producer(o) ≥ 0`, then `producer(o)` is mandatory.
* **Initial boundary:** if `o` is device-initial, then `-1` is mandatory.

### Feasibility lower bound

Let `min_pool[k] = Σ_o {size(o) : k ∈ Mandatory(o)} + reserved_outputs[k+1]`. If `min_pool[k] > cap` for some `k`, the chain is infeasible and Phase 1 raises.

### Starting from the MAX plan

Begin with each object having one residency interval covering its entire active span:

* For host-initial / device-initial `o` with uses: `[-1, last_use(o) - 1]`.
* For task-output `o` with uses: `[producer(o), last_use(o) - 1]`.
* For object with no use (rare): single point at `appears_at(o)`.

Note: the MAX residency for a mutated host-initial object is the SAME shape as for a clean host-initial object. The mutation status only matters in Phase 2 (trigger type). Phase 1 only sees sizes and boundaries.

If `pool[k] + reserved_outputs[k+1] ≤ cap` for every `k`, Phase 1 is done.

### Reduction loop

While there exists `k` with `pool[k] + reserved_outputs[k+1] > cap`:

1. Find the most-overloaded boundary `k*`.
2. From objects "in pool" at `k*`, find an **eligible victim** — an object `v` whose residency covers `k*` AND can be split to exclude `k*` while keeping all of `v`'s anchors (production + use boundaries) covered.
3. Rank eligible victims by `(stream_cost, not_drop_init, -first_use, -size, -gap_length)`:
   * `stream_cost`:
     * For `drop_init` evictions (un-pre-placing): always `0`. The only added cost is **1 h2d** (just-in-time prefetch when the obj is first needed). The writeback offload for mutated objs happens regardless of pre-placement, so it doesn't count.
     * For mid-life splits: `0` if release-eligible (host source AND never mutated), `1` otherwise (round-trip = d2h + h2d).
   * `not_drop_init = 0` for drop-init evictions, `1` for mid-life splits.
   * `-first_use`: prefer evicting objects whose FIRST use is **latest** in the chain. Keeping pre-placed objs that are used SOON saves more (avoids forward h2d contention where the stream is busy); late-first-use objs can be prefetched in backward where h2d has slack.
   * `-size` and `-gap_length` as final tie-breakers favor big bytes and long freed stretches.

   Why first-use matters: the value of pre-placing an obj X is "we don't have to prefetch X". That value is the same per-obj (1 h2d saved). But the *cost* of needing to prefetch X depends on **when** — if X is used early in forward, the prefetch competes with h2d for forward-pass weights; if X is used late in backward, h2d has plenty of slack. So pre-place early-use objs first; defer late-use objs to runtime prefetch. The prior "stream_cost preferring non-mutated" was a category error — it favored evicting weights (low cost-to-evict per dimension) when the right thing was to keep weights pre-placed (their pre-placement is high-value because they're used soon).
4. Split `v`'s covering interval `[a, b]` at the gap surrounding `k*`:
   * `anchors_le = {c ∈ anchors(v) ∩ [a, b] : c ≤ k* - 1}`,
   * `anchors_ge = {c ∈ anchors(v) ∩ [a, b] : c ≥ k* + 1}`,
   * New intervals: `[a, max(anchors_le)]` and `[min(anchors_ge), b]`. If either side has no anchors, drop that piece entirely (= un-pre-place or kill final tail).
5. Recompute `pool`.

Termination: each iteration strictly increases the number of distinct residency intervals; the count is bounded above by total `(uses + production)` anchors across all objects.

### Effective residency for pool tracking

When computing `pool[k]`, V4 accounts for one piece of simulator-side timing:

* **Prefetch arrival on the left edge.** For any interval `[a, b]` whose start `a` is NOT `-1` AND not the producer boundary, the simulator creates the device entry the moment the h2d trigger fires (= end of task `a - 1`) and the entry persists through h2d completion. So the obj actually occupies device bytes from boundary `a - 1` onward, not just from boundary `a`. V4 extends each prefetched interval's *effective* start by one boundary leftward when computing pool size.

V4 deliberately does NOT model d2h transit time in Phase 1 — the simulator's drain loop stalls compute when a queued d2h is still occupying bytes, which preserves correctness without making Phase 1 simulate the streams.

## Phase 2 — Triggers

Goal: turn each object's residency intervals into concrete `initial_device` pre-placement and per-task triggers.

For each object `o` with residency intervals `[a_1, b_1] < [a_2, b_2] < … < [a_m, b_m]`:

**Initial placement.** If `a_1 == -1`:
* If `host_source(o)`: add `o.id` to `initial_device`.
* If device-initial: already on device, no action.

**Per-interval arrivals.** For each interval `i = 1..m`:
* If `i == 1` and `a_1 == -1`: initial placement (no trigger).
* If `i == 1` and `a_1 == producer(o) ≥ 0`: natural production (no trigger).
* Otherwise: emit `prefetch_after(o.id)` on `task[a_i - 1]`.

**Per-interval departures.** For each interval `[a, b]`, the trigger fires at the EARLIEST task at which `o` is no longer needed in this interval. Compute:

* `use_tasks_in = { u ∈ uses(o) : a ≤ u - 1 ≤ b }`
* `production_in = { producer(o) }` if `producer(o)` is in `[a, b]`, else `{}`
* `fire_task = max(use_tasks_in ∪ production_in)`

If `fire_task` is the production task (no use in interval), the trigger fires on the SAME task that produces the object — step-7 marks the output live, then step-9 (offload) immediately queues the d2h. This is the key fix in V4 for the activation-offload-too-late bug: an activation produced by `f_i` with no immediate use should have its offload fire at `f_i`'s end, not at `f_{i+1}`'s end.

If `fire_task >= n` (= the obj naturally outlives the chain), no trigger is needed.

**Trigger type.** Determine whether `o` is *dirty* at end of this interval:

* `mutations_in = { t : t in interval AND o ∈ tasks[t].mutates_inputs }`. (For `t` to be "in interval", `t - 1 ∈ [a, b]`, i.e., this interval's residency covers the use that mutates.)
* `dirty_after = (mutations_in is non-empty)`.

Pick the trigger:

* `is_last = (i == m)`.
* Case A — `dirty_after AND has_host_source`: emit `offload_after(o.id)` on `task[fire_task]`. The mutation made the host copy stale; we must write back.
* Case B — `dirty_after AND not has_host_source`: emit `offload_after(o.id)` on `task[fire_task]`. The mutation needs to be preserved for the next interval's prefetch.
* Case C — `not dirty_after AND has_host_source`: emit `release_after(o.id)`. Host copy is identical to device — drop the device side without a d2h.
* Case D — `not dirty_after AND not has_host_source AND not is_last`: emit `offload_after(o.id)`. No host source AND another interval will need it back — must round-trip.
* Case E — `not dirty_after AND not has_host_source AND is_last`: emit `release_after(o.id)`. Object dies; no need to write to host.

The Phase 2 invariant **mutation preservation** falls out of cases A/B: any interval ending dirty emits an offload, so the host eventually catches up. By the time the final interval ends, the host has the latest bytes.

## Phase 3 — EDF prefetch scheduling on the h2d FIFO

Naively, V4 emits each prefetch trigger one task before its consumer (`task[a-1]`). That's correct when the h2d stream is idle — but when several prefetches share the FIFO, the queued transfers complete *after* their consumers' start times, and the simulator stalls compute waiting for the h2d. Sliding-window avoids this by pre-firing dW prefetches several tasks early; V4 needs the same idea, generalized.

**Principle:** when transfers share a FIFO stream and each has a known deadline, **issue them in deadline order as early as the DAG and memory cap allow** — not as late as memory permits. The FIFO orders delivery; the planner's job is to push enough work in early enough that the queue is never empty when a deadline arrives.

**Algorithm (EDF-backward packing):**

1. Enumerate prefetched intervals (those that fire prefetch triggers in Phase 2). Each has a deadline = `task_end[a-1]` (the consumer's start time in ideal-runtime µs) and `tau = ceil(size / bw_h2d)`.
2. Sort by deadline DESC.
3. Walk in this order. Maintain `next_start_t` (the start time of the most-recently-processed event, initially `+∞`).
   * `end_t = min(deadline, next_start_t)`
   * `start_t = end_t - tau`
   * `fire_task = max k with task_end[k] ≤ start_t`
   * `ideal_new_a = max(fire_task + 1, earliest_a)` where `earliest_a` is bounded by the obj's producer or prior-interval-end to keep the residency valid.
4. If `ideal_new_a < current_a`, extend the interval left to `(ideal_new_a, b)` — but only if the extension doesn't push pool over cap at any of the newly-occupied boundaries. Walk forward from `ideal_new_a` toward `current_a` and pick the earliest target that fits.

Extending an interval left = extra residency on device = more pool pressure during the extension. Phase 3 respects the cap (via the same pool model as Phase 1), so a tight cap may bound how much lead time a prefetch gets. Untight caps allow extensions up to the EDF target.

The simulator remains the timing arbiter — if V4's plan is logically feasible but the streams can't drain fast enough, the simulator extends makespan via compute stalls. The simulator hard-enforces the cap invariant; over-commits raise `task X deadlocked: device pool > cap, no offloads in flight`.

## Termination & complexity

* Phase 1: at most `O(Σ|uses|)` iterations, each `O(n × #objects)` for the overflow scan and victim selection. For a transformer training chain (`n = 3L+1`, `#objects = O(L)`), this is `O(L³)` — tractable for `L < 200`.
* Phase 2: `O(Σ|intervals|) = O(n × #objects)`.

## What V4 does NOT do

* No stream-aware co-scheduling (which prefetches overlap which offloads). The planner emits triggers; the simulator's FIFOs decide order.
* No bandwidth-aware initial placement decisions — Phase 1 reduces purely on memory.
* No automatic re-prioritization of victims based on stream pressure.

These keep V4 simple and predictable. Layered optimizations (a "Phase 4" stream-feedback loop) are future work, not part of the base.

## Comparison to prior versions

| Aspect | V2 | V3 | V4 |
|---|---|---|---|
| Decision style | reactive at sim time | constructive up-front | constructive up-front |
| Eviction trigger | when pool[k] binds during shadow walk | overflow at any covered boundary in `bps` | overflow at any boundary in `pool` (with prefetch-arrival left edge) |
| Activation offload timing | reactive eviction, greedy-late | round-trip pack of consecutive-use gaps | residency reduction makes activations production-only intervals → offload trigger fires AT the production task |
| Production→first-use gap | implicit via reactive eviction | enumerated as a special candidate | implicit — production boundary is `appears_at`, eviction can split anywhere |
| Cross-phase fallback | V3 → V2 in `apply_auto_policy` | falls through to V2 if V3 fails | none — each phase succeeds or raises with a precise reason |
| Writeback handling | added as post-pass keyed on `type=="gradient"` | added as post-pass keyed on `type=="gradient"` | first-class via `Task.mutates_inputs` — no name-matching, fully general |
| Lines of code | ~700 | ~700 | ~280 |
