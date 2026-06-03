# min_grow — design doc

Implementation: [`simulator/src/dataflow_sim/policy/min_grow.py`](../../../simulator/src/dataflow_sim/policy/min_grow.py).

> **NOTE:** this is the original design spec; the live implementation has evolved (MIN-seeded over-shrink + beam search). Treat divergence as design history; the code is source of truth.

## 1. Overview

`min_grow` is a memory scheduling policy for the dataflow_sim that minimizes makespan by **searching over residency plans with simulator replay as the cost oracle**, then **deterministically deriving transfer triggers** from the chosen plan. It is a clean, independent rewrite — not a refactor of `max_reduce`. It targets chain-shaped task graphs initially (training workload), with a plan representation that generalizes cleanly to DAGs as future work.

**Algorithm direction: MAX → shrink** (starts from "everything kept resident as long as possible", reduces via simulator-scored beam search). The dual MIN → grow direction was considered and tested; MAX → shrink converges faster for both loose-cap (MAX is immediately optimal) and tight-cap configs (the natural shape of optimal plans is "MAX minus a few targeted evictions").

Four stages:
- **Phase A0 — Analytic pre-pass**: starting from MAX, greedily evict objects (Belady-first, static-peak as tiebreaker) by static-cap reduction alone until `static_peak ≤ cap`. Pure analytic — no simulator. Critical for scaling: for L=32 transformer with 30+GB host-init and 20GB cap, simulator-driven shrink from MAX is too slow (would burn entire time budget).
- **Phase A1 — Simulator-driven over-shrink**: from the static-feasible plan, iteratively try Belady-ranked further shrinks. Accept best NON-WORSE shrink per step (allows walking through plateaus where one un-pre-placement doesn't help but two-together do). Stops after `patience=6` consecutive non-improving steps. Captures the "shrinking initial objects can reduce makespan via reduced output-reservation contention" effect that analytic shrink misses.
- **Phase A2 — Beam search**: from over-shrink's plan, beam-explore both reductions AND extensions (re-pre-place where cap allows). Refines for local makespan improvements.
- **Phase B — Schedule derivation**: deterministic emission of `releases_after` / `offload_after` / `prefetch_after` triggers. Two key insights:
  1. **Smart prefetch placement**: fire on the latest task whose end gives the h2d enough lead time, walking past zero-runtime tasks (recompute slots) that would cause guaranteed stalls.
  2. **Earliest-possible offload**: for an interval `[a, b)`, the exit trigger fires right after the last use within the interval (not at boundary `b`). E.g., an activation produced by `f_i` and immediately offloaded — trigger goes on `f_i`, not `f_{i+1}`, so the d2h starts as soon as possible.

`min_grow` does not model stream congestion or in-flight transit analytically. The simulator already models both correctly; `min_grow` trusts its makespan verdict. This is the central design choice that distinguishes `min_grow` from `max_reduce`.

---

## 2. Motivation

`max_reduce` is best-or-tied on roughly half of sweep configs; the rest are won by `belady_reactive` / `roundtrip_planner` / `sliding_window`. Several architectural limitations all trace to **one root cause**: `max_reduce` uses a static analytic heuristic as the sole signal for placement decisions:

- Path-dependent reduce → the heuristic key locally evicts the "wrong" object and the cascade can't recover.
- EDF over-extension at loose caps → analytic model overestimates contention.
- D2H tail mis-modeling → analytic model underestimates transit-driven pool pressure.
- No global stream-load awareness → heuristic key is per-object, not whole-schedule.
- Uncoordinated writebacks → analytic model doesn't see d2h FIFO competition.

The pattern is: any analytic model `max_reduce` builds is approximate, and the approximation errors compound. The simulator is the only ground truth for makespan, and the simulator is **cheap** (~500ms–2s per L=32 replay). Use it directly.

The problem formulation, hardness analysis, and the case for "plan + schedule separation" are in [problem.md](../../problem.md); `min_grow` is the concrete instantiation of problem.md §10's three principles.

---

## 3. Problem recap (brief)

Per [problem.md](../../problem.md): given a DAG of compute tasks on a single compute stream + two FIFO transfer streams (H2D, D2H) + a hard byte cap, decide what's resident when, to minimize makespan. The training workload's compute DAG is a chain `f_1, ..., f_L, head, b_L, ..., b_1` (`2L+1` active tasks; recompute tasks `r_i` are zero-cost and ignored).

### 3.1 Input format (min_grow's input)

min_grow's public entry point is `apply_min_grow_policy(bare: TaskChain) -> TaskChain`. The input is a **bare** `TaskChain` as defined in `simulator/src/dataflow_sim/schema.py` — "bare" meaning the task list contains tasks whose `releases_after`, `offload_after`, `prefetch_after` lists are empty (no policy has been applied yet). The relevant fields, copied verbatim from `simulator/src/dataflow_sim/schema.py:22-83`:

```python
@dataclass(frozen=True)
class Object:                            # appears in TaskChain.initial_memory
    id: str
    size: int                            # bytes
    location: Location                   # "host" | "device" — defines what's pre-placed
    type: ObjectType                     # "weight" | "activation" | "gradient" | "optimizer" | "other"

@dataclass(frozen=True)
class OutputAlloc:                       # appears in Task.outputs
    id: str                              # always fresh — never re-uses an existing id
    size: int                            # bytes
    location: Location                   # destination location of the output
    type: ObjectType

@dataclass(frozen=True)
class Task:
    id: str
    inputs: list[str]                    # obj ids read by this task (read-only unless in mutates_inputs)
    outputs: list[OutputAlloc]           # fresh objects produced on-device by this task
    runtime: int                         # ticks; deterministic
    releases_after: list[str]            # EMPTY on input — min_grow fills this
    offload_after: list[TransferTrigger] # EMPTY on input — min_grow fills this
    prefetch_after: list[TransferTrigger]# EMPTY on input — min_grow fills this
    mutates_inputs: list[str]            # subset of `inputs`; these are modified in-place
                                         # (must be offloaded, not released)

@dataclass(frozen=True)
class TaskChain:
    initial_memory: list[Object]         # objects whose `location == "host"` are host-init;
                                         # objects whose `location == "device"` are pre-placed.
                                         # On a BARE chain, all entries are host-init (no pre-placement).
    tasks: list[Task]
    device_capacity: int | None          # hard byte cap on the device pool; None = unlimited
    host_capacity: int | None
    bandwidth_h2d: int | None            # bytes per tick; required for min_grow
    bandwidth_d2h: int | None            # bytes per tick; required for min_grow
```

**Object id space**: every `obj_id` referenced by `task.inputs`, `task.mutates_inputs`, or in a `TransferTrigger` must exist either in `TaskChain.initial_memory` or as some prior task's `outputs[i].id`. Schema validation is upstream — min_grow assumes well-formed input.

**Bare invariant**: on input, for every task `releases_after == [] and offload_after == [] and prefetch_after == []`. min_grow raises if not.

### 3.2 Output format (min_grow's output)

min_grow returns an **annotated** `TaskChain` of the same shape. The fields min_grow may modify:

- `initial_memory`: min_grow may append `Object(id=..., size=..., location="device", type=...)` entries to pre-place host-init objects on device. Existing host-init entries (with `location == "host"`) are preserved.
- For each task `t` in `tasks`: min_grow populates `t.releases_after`, `t.offload_after`, `t.prefetch_after` based on the chosen plan.

min_grow does NOT modify: task ids, task ordering, `inputs`, `outputs`, `runtime`, `mutates_inputs`, capacity fields, or bandwidth fields.

**Output invariants min_grow guarantees**:
- Every object referenced in any trigger has a valid id (exists in `initial_memory` or some task's `outputs`).
- No object is both released and offloaded by the same task.
- No mutated input is bare-released (would lose the update); mutated objects always exit residency via `offload_after`.
- Returned chain is feasible to simulate: `simulator.run(returned_chain)` completes without error. (Stalls are allowed; outright cap violations or unfulfilled prefetches are not.)

**Output failure mode**: if no feasible plan exists (e.g., forced footprint exceeds `device_capacity` at some boundary), min_grow raises `ValueError("infeasible: <reason>")` — same convention as max_reduce.

### 3.3 Forced facts (not subject to min_grow's optimization)

For every task `t`:
- All ids in `t.inputs` (which includes `t.mutates_inputs` as a subset) must be resident at `t`'s start.
- All ids in `{o.id for o in t.outputs}` must have their bytes accounted for in the pool at `t`'s start (the simulator reserves output bytes when the task begins; min_grow inherits this accounting).
- `t.runtime` is fixed; task order is fixed.

End-of-step:
- Every mutated host-resident object (any `o` whose id appears in some `t.mutates_inputs` and which has a host source) must be written back via `offload_after` before the final task ends, so the host copy reflects the mutation.

### 3.4 Decisions (the min_grow search space)

- Which host-initial objects to pre-place (append to `initial_memory` with `location="device"`).
- For each non-forced object-interval, whether to keep the object resident across it or let it leave the pool.
- For each resulting transfer (prefetch or offload), which task's trigger list to attach to (and thus its position in the FIFO).

---

## 4. Plan representation (precise)

A **plan** is a data structure:

```python
@dataclass
class Plan:
    intervals: dict[ObjId, list[Interval]]   # per-object residency intervals
    # Invariants:
    # 1. Each (a, b) ∈ intervals[o] satisfies -1 ≤ a < b ≤ n (n = task count)
    # 2. Intervals for a given object are disjoint, sorted by a
    # 3. For every task t and o ∈ t.inputs ∪ {alloc.id for alloc in t.outputs}:
    #    some interval (a, b) ∈ intervals[o] satisfies a ≤ t-1 and b ≥ t
    #    (the "forced residency at use" invariant; mutates_inputs ⊆ inputs so covered)

@dataclass
class Interval:
    a: int   # entry boundary: object becomes resident from boundary a (-1 = initial)
    b: int   # exit boundary: object remains resident until boundary b, leaves AT boundary b
```

**Boundary semantics** (matching max_reduce's convention from `AUTOmax_reduce.md:45`): boundary `k` is the state snapshot AFTER task `k`'s triggers fire (after task `k` ends and its releases/offload-enqueues/prefetch-enqueues execute). Boundary `-1` is the initial state before any task runs.

**An object's residency at boundary k** = `∃ (a, b) ∈ intervals[o]: a ≤ k < b`. (Note: `< b`, so an interval `(a, b)` means resident at boundaries `a, a+1, ..., b-1` but NOT at boundary `b`. The transition to non-residency happens AT boundary `b`.)

**"Residency" here means pool occupancy** — the object's bytes count against the cap at boundaries in `[a, b)`. It does NOT mean "the h2d has completed and the object is consumable". The h2d for a prefetched interval fires at task `a`'s end (boundary `a`) and may take many ticks to complete; during that time the object is in pool (counts against cap) but not yet usable. If a compute task tries to consume an object whose h2d is still in flight, the simulator stalls that task until h2d completes. Stalls are normal — min_grow minimizes them by extending intervals leftward (smaller `a`) so the prefetch fires earlier and runs in parallel with more compute. See §5.2.1 and §7.

**Pre-placement** = an object with `a = -1` in some interval.

**The MIN plan**: for every object `o` and every task `t` that uses `o` (i.e., `o ∈ t.inputs` or `o ∈ {alloc.id for alloc in t.outputs}`), the interval `[t-1, t]` is in `intervals[o]`. Nothing else. This is the smallest feasible plan structurally; it requires every object to be JIT-prefetched right before its use and immediately released after.

**The MAX plan**: every object has a single interval `[-1, n]` (resident from start to end). Always feasible if `sum(sizes) ≤ cap`, otherwise not.

### Why intervals (not per-boundary bitmaps)

Per-boundary bitmaps over `|objects| × |boundaries|` bits are equivalent information but harder to manipulate. Intervals make the natural operations cheap:
- "Extension": merge two adjacent intervals or extend an interval's `a` or `b`.
- "Eviction": split an interval at boundary `k` (removing residency over a sub-range).
- "Pre-placement decision": is there an interval starting at `-1`?

### Mutation tracking in the plan

The plan doesn't store mutation state per se — mutation is a fact about the workload (`task.mutates_inputs`), not the plan. When Phase B emits triggers, it determines whether an object is "host-equivalent" (releasable) by checking: has any task in the object's residency intervals so far mutated it? If yes, it's "dirty" and must be offloaded (writeback); if no and it has a host source, it can be released.

---

## 5. The actual decisions, enumerated

### 5.1 Phase A decisions (plan search)

The plan-search algorithm makes one type of decision at each step: **which extension to apply**. An "extension" is one of:

- **Add pre-placement**: change `intervals[o]` so that some interval starts at `-1` (object pre-placed at t=0). If no interval currently starts at `-1`, this adds a new interval `(-1, t)` where `t` is the first use of `o`.
- **Merge across gap**: object `o` currently has separate intervals `(a₁, b₁)` and `(a₂, b₂)` with `b₁ < a₂`. The extension merges them into `(a₁, b₂)`, making `o` continuously resident across the gap.
- **Extend interval right**: `(a, b)` becomes `(a, b')` for `b' > b` (object stays resident longer).
- **Extend interval left**: `(a, b)` becomes `(a', b)` for `a' < a` (object becomes resident earlier).

In Phase A's beam search, the candidate set at each step = all valid extensions of all plans in the beam that respect the static cap check. The search starts from MIN and only ADDS residency (never removes). This monotonic growth ensures convergence and makes the search space well-defined: at most `O(|objects| · n)` extensions are ever applicable.

**What the search does NOT decide**: which task to attach a trigger to, or the FIFO ordering. These are derived in Phase B from the residency intervals — they're not search decisions.

### 5.2 Phase B decisions (schedule derivation)

Given a fixed plan, Phase B makes **deterministic** decisions:

For each interval `(a, b) ∈ intervals[o]`:

1. **Entry handling** (object becomes resident at boundary `a`):
   - If `a == -1`: pre-place — add `o` to `initial_memory`. No trigger needed.
   - If `a > -1`: a prefetch must complete by boundary `a`. The prefetch trigger goes on the task whose end is "as late as possible while still leaving enough room for the h2d to complete by boundary `a`'s start time". See §5.2.1 below for the exact rule.

2. **Exit handling** (object leaves residency at boundary `b`; i.e., not in any interval at `b`):
   - If `b == n` (object alive at end of step) AND `o` was mutated in any task in `[a, b)`: emit `offload_after` on task `b-1` (writeback).
   - If `b == n` AND `o` was not mutated AND `o` has a host source: emit `releases_after` on task `b-1`.
   - If `b == n` AND `o` is an intermediate (no host source, not mutated, not used again): emit `releases_after` on task `b-1` (it's dead).
   - If `b < n` (object will be needed again later via another interval): emit `offload_after` (if mutated since `a`, or no host source) or `releases_after` (if host-source and not mutated). On task `b-1`.

3. **Intra-interval triggers**: none. Within `[a, b)` the object is resident and untouched by triggers (compute tasks may still mutate it, but that doesn't change the plan).

**5.2.1 — Plan intervals are pool-occupancy intervals**

The plan's intervals describe **pool occupancy**, not "h2d-completed-by" boundaries. An interval `(a, b)` for object `o` means: `o` counts against the cap at boundaries `a, a+1, ..., b-1`. Phase B's job is to emit triggers so the simulator achieves this occupancy pattern; whether the h2d behind it actually finishes "in time" for the consuming task is the simulator's concern, not the plan's.

**Trigger placement rule (deterministic)**:

For each interval `(a, b) ∈ intervals[o]`:
- **Entry trigger**:
  - `a == -1` → pre-place `o` (append to `initial_memory` with `location="device"`). No prefetch trigger.
  - `o` is produced by task `a` (i.e., `o.id ∈ tasks[a].outputs`) → no entry trigger. The output is created on-device by the task; pool entry is implicit at task `a`'s end.
  - Otherwise (`o` has a host source and needs h2d) → append `TransferTrigger(o.id)` to `tasks[a].prefetch_after`. The trigger fires at task `a`'s end (boundary `a`), the prefetch enqueues, and the object enters pool at boundary `a` per the simulator's "device entry created at transfer start" semantic.
- **Exit trigger**: as described in §5.2 (release or offload on `tasks[b-1]`).

**Why this is enough**:

That's the whole rule. No deadline calculation. No "is the prefetch going to make it in time" check. If the h2d doesn't complete by the consuming task's start, the simulator stalls the task — that's normal behavior, not an error.

min_grow's mechanism for AVOIDING stalls is at the plan-search level: **extend intervals leftward** (smaller `a`) to fire the prefetch earlier and give the h2d more time to overlap with compute. For a non-pre-placed object first used at task `t`, the MIN-plan interval `[t-1, t]` means the prefetch fires at task `t-1`'s end with zero compute to hide behind — task `t` will stall for the full h2d duration. The extension `[t-k, t]` fires the prefetch `k-1` tasks earlier, hiding the h2d behind `k-1` tasks of compute. If `Σ runtime over those k-1 tasks ≥ h2d_duration`, no stall.

The beam search evaluates these extensions via simulator replay. An extension that fully hides the h2d reduces makespan (good); an extension that adds pool pressure causing other transfers to defer increases makespan (bad). The simulator's verdict is the deciding signal.

**No schedule-infeasibility from trigger placement**: the only infeasibility min_grow raises is `respects_static_cap(MIN_plan) == False` (forced footprint exceeds cap at some boundary — the chain can't run at all). Every other plan, including ones the simulator will stall on, is a legal candidate.

**5.2.2 — FIFO ordering**

The simulator's FIFO order = the temporal order in which triggers fire (which equals the temporal order of attached tasks' end times). Phase B doesn't reorder across tasks: the natural ordering induced by trigger-task-id is what the simulator sees.

For multiple triggers attached to the same task (list order = FIFO order within that task's enqueue burst), Phase B sorts:
- `prefetch_after` by "first-use boundary of the prefetched object" ascending (EDF on h2d).
- `offload_after` by "interval entry boundary" ascending (oldest exits first).

Per-stream EDF is **provably optimal** for the within-stream scheduling sub-problem given fixed deadlines (Liu & Layland 1973). It is **not** globally optimal — the plan itself can be suboptimal, and joint h2d/d2h optimization is beyond EDF — but it's the right baseline.

---

## 6. Algorithm

### 6.1 Phase A: plan search (cold-start beam)

```
function v5_plan_search(bare_chain) -> Plan:
    plan = MIN_plan(bare_chain)
    if not respects_static_cap(plan):
        raise ValueError("infeasible: forced footprint exceeds cap at some boundary")

    best_makespan = score(plan)                       # one simulator replay
    best_plan = plan
    beam = [(plan, best_makespan)]

    while time_budget_remaining():
        candidates = []
        for (p, _) in beam:
            for ext in enumerate_extensions(p):
                p_new = apply_extension(p, ext)
                if respects_static_cap(p_new):
                    candidates.append(p_new)

        if not candidates:
            break

        ranked = top_k_by_analytic_bound(candidates, K_sim)
        scored = [(c, score(c)) for c in ranked]

        new_beam = top_k_by_makespan(scored, K_beam)
        new_best = min(m for _, m in new_beam)

        if new_best >= best_makespan:
            break   # diminishing returns
        beam = new_beam
        best_makespan, best_plan = new_best, new_beam[0][0]

    return best_plan
```

**Sub-routines**:

- `respects_static_cap(plan)`: at every boundary `k`, sum of sizes of objects whose plan-interval contains `k` ≤ cap. Pure analytic; cheap (`O(|objects| · n)`). Does NOT model in-flight transit (deliberate — see §7). This is the ONLY feasibility gate min_grow applies; if a plan passes this, it's a legal candidate even if the simulator will stall on it.

- `score(plan)`: derive triggers (Phase B), run `simulator.run(..., snapshots=False)`, and return makespan = `max(iv.end for iv in event_log.task_intervals)` plus `event_log.peak_device_bytes`. One lightweight simulator replay per call. Always returns a finite makespan for plans that pass `respects_static_cap` (stalls just inflate the makespan; they don't fail the simulation). The later pending-outbound stall repair pass may still use full snapshots because it inspects per-object memory state.

- `enumerate_extensions(plan)`: for each object `o` and each gap in `intervals[o]`, generate the extension that fills that gap (merge intervals). Also: if no interval starts at `-1`, generate pre-placement of `o`. Total candidates ≈ `O(|objects| · (uses per object))` ≈ `O(|objects| · L)` for the chain.

- `top_k_by_analytic_bound(candidates, K)`: rank by `compute_time + max(h2d_bytes_total / bw_h2d, d2h_bytes_total / bw_d2h)`. Loose upper bound on makespan; used only for prioritizing which candidates to spend simulator budget on. The actual score is from simulator, so analytic looseness doesn't affect correctness — only which candidates we evaluate first.

**Parameters** (knobs, not architectural):
- `K_beam` = 3 (initial)
- `K_sim` = 5 (initial)
- Time budget = 5–10s per config

**Cost model**: per beam step, `K_sim` simulator replays + `O(K_beam · |objects| · L)` analytic candidate generation. At 500ms–2s per L=32 replay, one step is 2.5–10s; we expect convergence in 2–5 steps for typical configs → 5–50s per config. If too slow, shrink K's.

### 6.2 Phase B: schedule derivation

```
function derive_schedule(plan, bare_chain) -> TaskChain:
    triggers = {t.id: TriggerSet() for t in bare_chain.tasks}
    initial_memory = list(bare_chain.initial_memory)   # copy

    for o, ivs in plan.intervals.items():
        for (a, b) in ivs:
            # Entry trigger (see §5.2.1)
            if a == -1:
                initial_memory.append(make_device_object(o))   # pre-place
            elif o.id in {alloc.id for alloc in bare_chain.tasks[a].outputs}:
                pass   # o is produced by task a; no entry trigger needed
            else:
                # host-source object, needs h2d. Trigger fires at task a's end;
                # the simulator handles h2d execution and any stall it causes.
                triggers[a].prefetches.append(TransferTrigger(o.id))

            # Exit trigger
            if b == n: continue   # end-of-step handling below
            mutated_in_interval = any(o.id in bare_chain.tasks[t].mutates_inputs
                                       for t in range(a + 1, b))
            if has_host_source(o) and not mutated_in_interval:
                triggers[b - 1].releases.append(o.id)
            else:
                triggers[b - 1].offloads.append(TransferTrigger(o.id))

    # End-of-step writebacks for mutated host-resident objects
    for o, ivs in plan.intervals.items():
        last_iv = ivs[-1]
        if last_iv.b == n and is_dirty_at_end(o, plan, bare_chain):
            triggers[n - 1].offloads.append(TransferTrigger(o.id))

    # Sort each task's trigger lists for in-task FIFO order (EDF on h2d, FIFO on d2h)
    for t_id, tset in triggers.items():
        tset.prefetches.sort(key=lambda tt: first_use_boundary(tt.obj_id, plan))
        tset.offloads.sort(key=lambda tt: original_entry_boundary(tt.obj_id, plan))

    return build_task_chain(bare_chain, initial_memory, triggers)
```

This is fully deterministic given a plan. No search, no heuristics with knobs, no infeasibility path — every plan that passes `respects_static_cap` produces a runnable TaskChain.

---

## 7. Treatment of stream congestion and transfer chaining

**Transfer chaining (offload completion frees bytes → unblocks deferred prefetch)**: the simulator handles this natively. When a prefetch can't fit (its bytes would push pool > cap), the simulator marks it deferred and re-checks each time the pool shrinks (a release or d2h completion). min_grow emits the prefetch trigger at a task end — what happens after that point is the simulator's job. min_grow does not try to "plan around" the deferral because the deferral itself is correct behavior: the prefetch fires at the earliest legal moment.

The consequence of deferral on makespan is observed as a stalled compute task downstream (the task that needs the deferred object waits). The simulator's drain loop accounts for this stall; the makespan it returns reflects it; min_grow's beam search penalizes the plan via score and prunes it.

**Stream congestion (FIFO depth, transit time, h2d/d2h competition for cap)**: min_grow does NOT model any of this analytically. The only analytic check min_grow makes is `respects_static_cap`, which sums sizes at task boundaries assuming all transfers have completed. This is a necessary-but-not-sufficient feasibility check — plans that pass might still be congestion-stalled at simulation time.

This is the deliberate central design choice of min_grow. The reasoning:
- max_reduce has 5 known bugs that all trace to inaccurate analytic congestion modeling (`_compute_d2h_tails`, `stream_cost` ranking, EDF lookahead extension).
- The simulator already models congestion correctly (FIFO, deferral, transit).
- One simulator replay costs less than a tenth of a second to a few seconds — affordable as the cost oracle.
- Trusting the simulator's verdict eliminates an entire class of "analytic model disagrees with reality" bugs.

The trade-off is that min_grow cannot prove a plan optimal without trying it; the search has to actually evaluate candidates. This is fine because beam search at K=3–5 keeps the simulator budget bounded.

**What about feasibility of the analytic pre-filter?** `respects_static_cap` can FALSELY accept a plan that simulation-stalls due to transit. That's fine — the simulator will reveal it as poor makespan. `respects_static_cap` is purely a pre-filter to skip "obviously infeasible" plans (sum-of-sizes > cap at some boundary), not a positive guarantee of feasibility.

**Why not also use the static cap check more aggressively (e.g., model d2h tails)?** Because max_reduce tried that exact thing and it was over-conservative (rejected configs that sliding handled). The principled simulator-only approach is more robust.

---

## 8. Worked example (L=2 transformer, tight cap)

Workload: `tasks = [f_1, f_2, head, b_2, b_1]` (5 active tasks). Objects:
- Host-init: `input, W_1, W_2, head_W`
- Produced: `A_1, A_2, y_1, y_2, dy_2, dy_1`
- Pre-allocated (mutated): `dW_1, dW_2, head_dW`

`task.inputs` / `task.outputs` / `task.mutates_inputs`:
- `f_1`: inputs={input, W_1}; outputs=[A_1, y_1]
- `f_2`: inputs={y_1, W_2}; outputs=[A_2, y_2]
- `head`: inputs={y_2, head_W, head_dW}; outputs=[dy_2]; mutates_inputs={head_dW}
- `b_2`: inputs={A_2, dy_2, W_2, dW_2}; outputs=[dy_1]; mutates_inputs={dW_2}
- `b_1`: inputs={A_1, dy_1, W_1, dW_1}; outputs=[]; mutates_inputs={dW_1}

(Mutated objects appear in `inputs` AND `mutates_inputs` — they're read-modify-write. `outputs` lists only fresh ids.)

**MIN plan** (forced residency only) for `W_1`: intervals `[(-1, 1), (3, 4)]` — resident at boundaries `-1, 0` for `f_1`'s use, gone at boundaries `1, 2, 3`, back resident at `3, 4` for `b_1`'s use. Same shape for `W_2`: `[(-1, 2), (2, 4)]` (used at `f_2` and `b_2`; these intervals are adjacent, so they merge to `[(-1, 4)]`).

**An extension candidate**: merge `W_1`'s two intervals into `[(-1, 4)]`. This makes `W_1` continuously resident — saving one h2d (re-prefetch of `W_1` before `b_1`), at the cost of `size[W_1]` bytes on device across boundaries 1, 2, 3.

If cap allows the extra `size[W_1]` bytes at those boundaries, the extension is feasible. Phase A evaluates: derive triggers (no re-prefetch trigger for `W_1`), run simulator, compare makespan. If makespan dropped, keep the extension.

**A trigger placement example**: for `W_2` in MIN plan with intervals `[(-1, 2), (2, 4)]`, the gap is empty (intervals are adjacent at boundary 2). No trigger needed at the boundary — `W_2` is continuously resident through boundary 2 (no exit/entry). The two intervals collapse to one in practice.

**A schedule-infeasibility example**: suppose Phase A proposes "pre-place `A_2` from `-1`" as an extension. But `A_2` is produced by `f_2` (not host-init) — there's no h2d for `A_2`. The "pre-place" extension is invalid for non-host-init objects; `enumerate_extensions` filters these out.

---

## 9. Edge cases

- **Forced footprint exceeds cap at some boundary**: `respects_static_cap(MIN_plan)` returns False. min_grow raises `ValueError("infeasible: forced footprint exceeds cap at boundary k")`. This is the ONLY way min_grow raises — the chain literally cannot run, even with perfect JIT. No recompute fallback in scope.
- **MIN plan runs with stalls**: this is the COMMON case, not an error. The simulator stalls compute tasks when their inputs aren't h2d-ready. Stall time is reflected in makespan. Beam search extends intervals leftward to reduce stalls.
- **Object with no host source and not produced on-device**: shouldn't happen in well-formed workload. If it does, schema validation should catch upstream of min_grow.
- **Object mutated multiple times across intervals**: the "is dirty" check is per-interval (any task in `(a, b)` that mutates `o`). If dirty in interval 1 and clean in interval 2 (no further mutation), interval 1's exit is offload, interval 2's exit is release — distinct decisions.
- **End-of-step writeback for mutated host-resident objects**: even if the final interval ends at `n` (object stays resident to end), if it was mutated, an offload trigger is added at task `n-1` (the last task) so the d2h is enqueued before step end. The simulator's drain ensures completion before the step terminates.
- **Empty beam after pre-filter**: no valid extensions exist (every extension would exceed cap somewhere). Return current best (might be MIN itself).
- **Simulator returns makespan worse after extension**: extension added pool pressure that caused other transfers to defer, ultimately costing more than it saved. Beam search rejects via the diminishing-returns check.

---

## 10. Comparison to max_reduce

| Aspect | max_reduce | min_grow |
|---|---|---|
| Start state | MAX (everything resident) | MIN (only forced) |
| Direction | Reduce (evict) | Grow (extend) |
| Cost signal | Static heuristic 5-tuple key | Simulator replay |
| Beam width | 1 (greedy single-pass) | K_beam (default 3) |
| Stream congestion model | Analytic (`_compute_d2h_tails`, `stream_cost`) | None — simulator handles |
| EDF extension | Phase 3 "lookahead" extends prefetch intervals | No extension; EDF inferred from plan |
| D2H scheduling | Implicit, triggered by exit boundaries | Same, EDF-sorted within task |
| Trigger placement | Computed during Phase 2 of reduce | Computed in Phase B after plan finalizes |
| Path-dependence | Yes (issue #1) | No (beam keeps top-K) |
| Loose-cap regression | Yes (issue #2) | No (no extension to over-apply) |

min_grow inherits from max_reduce only the **boundary convention** and the **schema primitives** (`TransferTrigger`, `Task.mutates_inputs`, `releases_after`/`offload_after`/`prefetch_after`). The algorithm shares no code.

---

## 11. Code structure

**Files**:
- `simulator/src/dataflow_sim/policy/min_grow.py` — min_grow implementation. Public: `apply_min_grow_policy(bare: TaskChain) -> TaskChain`. Internal: `Plan` dataclass, `Interval`, `MIN_plan`, `enumerate_extensions`, `apply_extension`, `respects_static_cap`, `analytic_bound`, `derive_schedule`, `score`.
- `docs/policy/other_policies/min-grow.md` — this doc.
- `simulator/tests/test_min_grow.py` — unit tests.
- `app/scripts/compare_policies.py` — adds min_grow to policy dict in `run_one()`; min_grow column in output table.

**Primitives reused** (unchanged):
- `simulator/src/dataflow_sim/schema.py` — `Task`, `TaskChain`, `Object`, `OutputAlloc`, `TransferTrigger`, boundary convention.
- `simulator/src/dataflow_sim/simulator.py` — `simulator.run(chain) -> EventLog`; makespan = `max(iv.end for iv in event_log.task_intervals)`. Use `snapshots=False` for scoring paths that need intervals and peak bytes but not the full event timeline.

**Not touched**:
- `simulator/src/dataflow_sim/policy/max_reduce.py`, `simulator/src/dataflow_sim/policy/belady_reactive.py`, `simulator/src/dataflow_sim/policy/sliding_window.py`. `min_grow` is independent; these stay as comparison baselines.

---

## 12. Verification

### 12.1 Unit tests (`simulator/tests/test_min_grow.py`)

- **MIN plan correctness**: 3-task synthetic chain. Verify only forced residency.
- **Extension enumeration**: on L=2 transformer, verify expected gap-merge candidates surfaced for `W_1` (the only object with a real gap in MIN).
- **Static cap pre-filter**: construct plan that exceeds cap at one boundary; verify pre-filter rejects.
- **Schedule derivation determinism**: fixed plan → fixed trigger set. Verify trigger placement, FIFO order, release-vs-offload classification.
- **End-to-end**: small config (L=3, tight cap), `apply_min_grow_policy` returns valid TaskChain, simulator runs to completion, makespan finite.
- **Infeasibility**: tiny cap below forced footprint → raises `ValueError`.

### 12.2 Sweep regression

Run `app/scripts/compare_policies.py` with min_grow added:
- Bar: min_grow makespan ≤ best-of-{belady_reactive, roundtrip_planner, max_reduce, sliding} on every (seqlen, num_seqs, cap) cell.
- Strict improvement: min_grow beats best-of-prior on at least the configs where max_reduce currently regresses (e.g., `sql=2048 M=1 cap=20GB`, currently +36.8% over belady_reactive).

### 12.3 Per-config timing

- Record min_grow planner wall-clock per config; assert ≤ 10s.

---

## 13. Risks and fallback

- **Cold-start beam may not match warm-started max_reduce** on configs where max_reduce happens to be near-optimal. max_reduce's 34/72 baseline is a hard floor. If min_grow falls short: introduce warm-start as Phase 0 (use belady_reactive's plan or max_reduce's plan as initial seed), with prior policies renamed into clearly-described "min_grow seeds" (belady_reactive → reactive-Belady seed; max_reduce → top-down-reduce seed). Sliding excluded as transformer-specific. Cold-start stays the primary design; warm-start is the documented escape hatch.
- **Per-replay simulator cost** higher than 500ms–2s estimate: K_beam/K_sim are knobs, shrink to fit.
- **Candidate enumeration on DAGs** needs more thought (forks/joins); chain-first explicit in scope.
- **Analytic upper bound for ranking** is loose; could mis-prioritize good candidates. Mitigation: simulator scores are truth — if a bad bound de-prioritizes a good candidate, it gets evaluated in a later beam step.

---

## 14. Out of scope (deferred)

- Recompute (`r_i` tasks); `r_i.runtime` stays 0.
- General DAG topology — chain-first; DAG extension is future work on same plan representation.
- CP-SAT verification oracle ([problem.md](../../problem.md) §7.3) — parallel workstream for bounding gap to provable optimum; not blocking min_grow.
- Warm-start from prior policies — held in reserve as fallback.
- Joint h2d/d2h optimization (EDF per-stream is optimal given fixed deadlines, but joint scheduling could shave more — out of scope).
