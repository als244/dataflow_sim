# Principles of a good policy

What separates a correct, efficient scheduling policy from a buggy or wasteful one on this simulator. Each principle is a one-line rule with a **WHY** sentence; the principles are organized by what they protect (correctness, resource use, makespan) and the doc closes with a **Tensions** section listing places where principles legitimately conflict and a new policy must pick a stance.

This doc is policy-agnostic — it states the rules. For how each built-in policy resolves the tensions, see the per-policy docs in this directory.

---

## 1. Correctness

These are invariants. Violating any of them produces a buggy plan: the simulator raises, the workload silently loses data, or the chain deadlocks.

- **An object may be released only by a task that names it as input, and only when its compute entry is `live`.** WHY: simulator raises if the release key is absent or the entry is mid-transfer.
- **An object cannot be released if it has another use AND (backing lacks a copy OR object is dirty).** WHY: bare-releasing a dirty-only-on-compute object silently discards the mutation; a clean re-use requires a backing copy to re-prefetch from.
- **A mutated input must not be bare-released before a later use; if its declared final destination is backing, it must be offloaded after the last mutation.** WHY: dirty bytes only need preservation when some future consumer or terminal placement constraint needs them.
- **Every task input must be `live` on compute by task start** — either resident or via a prefetch whose from-slow (plus any blocking to-slow) completes before the earliest start. WHY: missing-input deadlock raise.
- **Free compute bytes + scheduled-to-slow reclaim must cover the task's compute-located output footprint at dispatch.** WHY: the simulator reserves output space at task start; insufficient headroom raises.
- **Backing pool + task's backing-located outputs must fit `backing_memory_capacity` at task start** (no backing stall mechanism exists). WHY: backing overflow raises immediately.
- **Prefetch of X is valid only when X's compute entry is absent or in-flight outbound;** offload of X is valid only when compute entry is `live` and any backing entry is `live` with matching size. WHY: stale, mid-flight, or size-mismatched transfer triggers raise.
- **Output ids must be fresh — no `(id, location)` collision with any existing pool entry.** WHY: output-key collision raises.
- **An offload of X is forbidden if a later task consumes X without an intervening re-prefetch being scheduled.** WHY: a task consuming a `pending_outbound` or `outbound` input with no re-prefetch raises.
- **Transit memory counts against `fast_memory_capacity`.** Bytes in states `inbound`, `pending_outbound`, and `outbound` all occupy compute pool until the transfer completes. WHY: ignoring transit footprint is how a plan that "looks feasible" produces runtime overflow.
- **Plans must terminate.** No policy may produce a chain that deadlocks with empty queues and missing inputs. WHY: the simulator's last-resort deadlock detector raises rather than hanging.

---

## 2. Resource conservation

These are optimality principles for the two scarce resources the simulator models: **cap bytes** (fast memory) and **stream time** (from-slow + to-slow bandwidth). Several principles span both — eviction strategies trade memory for stream time and vice versa.

### Memory

- **Release ASAP after last use in chain.** Attach the release to the task associated with the last use; if `final_locations[obj] == "backing"` and the latest bytes are only on compute, offload instead of release at the same anchor. WHY: every cycle of residency past last use is cap pressure on co-resident peers.
- **If Offloading, do ASAP after the task that produced or last mutated the object.** WHY: earlier anchors find cheaper to-slow slack and free the compute slot sooner; deferred offloads create cascading pressure.
- **Forbid double residency.** No need to prefetch an object already live on compute and no need to write-back a clean object to backing where an existing copy exists. WHY: two copies count twice against cap with zero correctness benefit.
- **Pin objects with reuse-distance ≈ 1** (e.g., heads, hot weights); never round-trip them. WHY: scheduling cost is pure waste when nearly every task needs them.
- **Pre-placement must be prefix-monotone in first-use time** — warm only a contiguous prefix of objects at t=0, bounded by `cap − first-task output reservation − first-task input footprint`. WHY: scattered pre-placement of late-used objects eats the early tasks' output headroom.
- **Pre-placement floor**: any object whose first-use time falls inside the cold-start from-slow lead must be resident at t=0. WHY: omitting these leaves the first few tasks stream-bound on otherwise-hideable transfers.
- **Pin an object across k consecutive uses iff peak across that span stays ≤ cap;** otherwise round-trip. WHY: avoids paying k transfer costs for shared weights when the cap allows.
- **Collapse output-into-input aliasing**: when task T's output feeds only T+1, align its birth with the release of an about-to-die input at the same task boundary. WHY: halves peak across that boundary via `releases_after` + `OutputAlloc` in the same trigger group.
- **Among co-resident objects under pressure, evict by farthest next use (Belady).** WHY: the only eviction rule with a known offline optimality bound.

### Stream

- **Prefer clean + backing-resident eviction (pure release) over offload when either could free the needed bytes.** WHY: zero stream cost vs full transfer; no need to waste writing back and occupying to-slow bandwidth.
- **Order prefetch triggers attached to the same anchor by consumer deadline** (earliest-needed first). WHY: per-stream queues are strict FIFO; trigger order = service order, unfixable later.
- **No-op round-trips are forbidden.** Never to-slow then from-slow the same object back into the same slot unless an intervening task needed the bytes. WHY: both transfers cost stream time for zero state change.
- **Saturate both streams in parallel.** Never park a transfer on the busy queue when the opposite stream is free. WHY: from-slow and to-slow are independent — serializing halves effective bandwidth.
- **Co-schedule dirty-object to-slow with a different-object from-slow.** WHY: opposite directions run concurrently, and to-slow completion can fire deferred prefetches in the same tick.
- **Before emitting a prefetch, verify projected compute residency at the transfer's *start* has ≥ `obj.size` headroom.** WHY: a capacity-blocked queue head freezes every later prefetch, including ones that would have fit.
- **Avoid chaining a prefetch onto a backing source that is the destination of an in-flight to-slow.** WHY: such prefetches become deferred, forcing a serialized `to-slow end → from-slow start` critical path.
- **Stream-aware lead times**: a prefetch's effective lead is `queue_drain + transfer`, not just `transfer`, when the from-slow queue is non-empty. WHY: ignoring queue depth produces correct-looking plans that stall in simulation.

---

## 3. Critical path / makespan

These principles are about where to spend resource budget so that every cycle of saving compounds into the headline number.

- **Makespan is the end of the last compute task.** Stream idle by itself is free; stream idle while compute waits on it is the only cost. WHY: the simulator pins makespan to `compute_busy_until`.
- **Lead time must cover transfer duration.** Anchor a prefetch for task T at producer P such that `sum(runtime strictly between P and T) ≥ from_slow_runtime + from_slow_queue_wait`. WHY: any less and compute stalls.
- **Anchor prefetch at the *latest* producer that still leaves enough slack — not the earliest.** WHY: earlier arrival squats on compute through unrelated tasks, evicting peers and bloating peak.
- **Off-critical-path work is free.** Spend releases / offloads / re-prefetches inside stall windows triggered by *other* objects. WHY: FIFO stream serialization only costs makespan when a critical transfer ends up queued behind them.
- **Pin objects whose next reference falls inside the next from-slow runtime; evict long-horizon "tourists."** WHY: far-future inputs can be re-fetched during hidden time without lengthening compute.
- **Don't bridge a use-gap longer than `(to-slow + from-slow)` by residency; don't round-trip a gap shorter than that.** WHY: this is the residency-vs-round-trip threshold — both directions of violation are common failure modes.
- **Tail effect**: triggers on the final task are bookkeeping (the final drain runs after `compute_busy_until` is set), but a late prefetch on the *second-to-last* task still extends makespan. WHY: optimize tail-adjacent transfers, not the tail itself.
- **Stall floor**: under tight cap, makespan ≥ `sum(compute) + max(0, unhidden_from_slow)` — minimize *unhidden* from-slow, not from-slow count. Under loose cap, `makespan = sum(compute)` is achievable. WHY: the cap regime determines whether transfer count or transfer hiding is the right objective.

---

## 4. Tensions

These are places where two principles legitimately conflict. A policy must pick a stance — there is no single right answer.

- **Release after last use in interval** vs **extend lifetime to avoid re-prefetch.** When is keeping an object resident across a long gap cheaper than discarding and re-loading it?
- **Pre-place to use cap fully** vs **leave room for outputs to accumulate.** A initial object placement strategy that perfectly fills the cap at t=0 leaves zero headroom for produced bytes; one that leaves too much headroom under-uses the cap. If objects are released or offloaded it reduces the cap space so not accounting for this will under-place initial (non-produced) objects and cost in terms of from-slow bandwidth potentially in critical path.
- **Saturate streams** vs **don't trigger transfers the simulator will defer.** A prefetch onto a backing source that's the destination of an in-flight to-slow looks like stream saturation but actually serializes the critical path.
- **Greedy-late offload anchoring** (belady-style) vs **greedy-early offload anchoring** (constructive-planner-style). Late anchoring keeps the object on compute longer (more reuse opportunity); early anchoring frees the slot sooner (more cap headroom).
- **MAX-shrink starting point** (start with everything resident, evict only under pressure) vs **MIN-grow starting point** (start with only forced residency, extend where it pays). Both have been tried; which fits a given regime is an empirical question.
- **Analytic ranking** vs **simulator replay as cost oracle.** An analytic 5-tuple key is fast but cannot model stream congestion or transit bytes; a simulator-as-oracle is accurate but costs replays per candidate.
- **"Trust the plan, raise on failure"** vs **"verify and repair."** A planner can either raise when its internal model fails, or replay the simulator and patch detected stalls. The first is simpler; the second is more robust.

---

## See also

- [docs/problem.md](../problem.md) — the formal scheduling problem these principles operate on.
- [docs/policy/README.md](README.md) — the decision table for the six built-in policies.
- Per-policy docs in this directory — how each policy resolves the tensions in §4.
