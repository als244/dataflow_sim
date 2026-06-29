# PressureFit

PressureFit is a standalone policy for linear `TaskChain`s. It plans compute
residency as intervals over task boundaries, removes optional residency where
modeled byte pressure exceeds capacity, emits release/offload/prefetch triggers
under each of four inbound schedules, verifies each annotated chain with the
simulator, and returns the fastest valid one.

The short version is:

```text
one seed interval set
    -> deterministic pressure reduction
    -> four inbound schedules (packed-fifo, packed-fit,
                               interval-entry, latest-safe)
    -> trigger emission per schedule
    -> simulator verification / bounded repair
    -> fastest valid plan
```

Residency planning is shared and deterministic. The only branching in the
policy is the inbound schedule: the best moment to fire a prefetch depends on
inbound-FIFO congestion vs. memory pressure, which the analytic model cannot
rank without replay, so the policy derives all four trigger placements and
lets the simulator pick.

Public entry point:

```text
apply_pressurefit_policy(bare, *, fast_memory_capacity=None)
```

Diagnostics entry point:

```text
plan_pressurefit_policy(...) -> (chain, diagnostics)
```

It runs the same planner and returns per-schedule status, timing, makespan,
and selection metadata.

## Design Contract

PressureFit is schema-driven. It uses:

- task order, runtimes, inputs, outputs, and compute output sizes;
- object sizes and initial source locations;
- producer/use positions;
- explicit mutation metadata from `Task.mutates_inputs`;
- inbound and outbound bandwidths and `fast_memory_capacity`.

It does not depend on object ids, object types, or task names.

## Core Algorithm: Pressure Reduction

Pressure reduction is the central algorithm in PressureFit. It makes the
residency decision; the inbound schedules only decide when the resulting
interval entries become prefetch triggers.

The pressure-reduction problem is:

```text
given:
    seed intervals S
    required anchors A
    object sizes size(o)
    compute capacity C
    output reservation Q(x) at each boundary x
    repair pressure E(x) at each boundary x

find:
    intervals P

such that:
    P is made only by cutting gaps out of S
    every required anchor in A remains covered
    for every boundary x:
        resident_P(x) + Q(x) + E(x) <= C
```

Here `resident_P(x)` is the sum of compute bytes for objects whose planned
intervals count resident at boundary `x`. `Q(x)` reserves memory for the next
task's compute outputs, and `E(x)` is extra pressure discovered by simulator
repair. The appendix defines these quantities and the exact boundary
accounting.

Operationally, pressure reduction repeatedly finds the most overloaded
boundary, chooses a legal non-anchor gap whose removal reduces pressure at that
boundary, splits that interval, and updates the boundary byte counts. It stops
when the capacity inequality above holds at every boundary, or raises
infeasible when no legal split can remove enough optional residency.

Pressure reduction is deterministic but greedy: its split ranking is a local
rule for reaching feasibility, not a proof of globally optimal residency.

## Core Vocabulary

This section names only the concepts needed to read the algorithm. The appendix
contains the precise definitions and ranking rules.

- **Boundary:** a point between tasks where the policy accounts for compute
  residency and output reservations. Boundary `-1` is the initial state.
- **Anchor:** a boundary where an object must be compute-resident: initial compute
  residency, production, or the boundary before a use.
- **Residency interval:** an inclusive boundary range `[a, b]` where an object
  is planned to be compute-resident.
- **Seed interval set:** the starting residency hypothesis before pressure
  reduction cuts optional gaps.
- **Pressure reduction:** the shared pass that cuts non-required residency gaps
  until the capacity inequality is satisfied at every boundary.
- **Pressure-fit interval set:** the output of pressure reduction, before
  trigger emission turns interval entries/exits into annotations.
- **Inbound schedule:** one rule for turning prefetched interval entries into
  `prefetch_after` triggers. PressureFit derives four.

## Algorithm Summary

The planner flow is:

1. **Build reference facts.** Compute ideal compute-stream times, object sizes,
   producers, uses, mutators, initial locations, final locations, and each next
   task's compute output reservation.
2. **Choose initial residency, build the seed, reduce once.** Select which
   backing-source objects start on compute (appendix: Initial Residency), build one
   seed interval set from liveness anchors, and run pressure reduction once —
   every schedule starts from the same pressure-fit interval set, so the base
   reduction is shared and each schedule works on a copy.
3. **For each of the four inbound schedules:**
   a. copy the shared pressure-fit interval set;
   b. for the interval-entry schedule, extend inbound interval starts earlier
      where strict pressure still fits;
   c. emit release, offload, prefetch, and initial-copy annotations;
   d. verify the annotated chain with the simulator;
   e. translate bounded simulator capacity contradictions into extra boundary
      pressure and rerun reduction when repair is possible.
4. **Return the fastest valid plan.** If no schedule verifies, raise the first
   planning error.

The simulator never sees the analytic model. It sees only the annotated
`TaskChain` produced by this pipeline.

## The Four Inbound Schedules

After pressure reduction, every non-initial, non-produced interval entry
becomes an inbound prefetch. All schedules start from the same inbound job for
each prefetched interval `[a, b]` of object `o`:

```text
first_use = first task that consumes o inside [a, b]
earliest = max(previous_interval_fire_task(o), producer_task(o), 0)
latest = first_use - 1
deadline = ideal_start(first_use)
inbound_runtime = ceil(size(o) / inbound_bandwidth)
```

The schedule chooses a trigger task `fire` in `[earliest, latest]`. The trigger
is emitted as `prefetch_after` on that task.

| Schedule | Exact placement rule | When it wins |
|---|---|---|
| `packed-fifo` | Treat all inbound jobs as one FIFO queue. Sort jobs from latest deadline to earliest deadline, then pack them backward so each job finishes before its consumer deadline and before the next later-packed inbound job. If no such trigger exists inside the job window, use `earliest`. | Inbound queue congestion is the main risk: multiple transfers would otherwise pile up near their consumers. Empirically the most common winner. |
| `packed-fit` | Packed FIFO with a pressure clamp. Firing on task `t` materializes destination bytes at boundaries `[t, a - 2]` that the interval model counts only from `a - 1`; the clamp slides each trigger later until every newly covered boundary still satisfies the strict capacity inequality, and commits accepted coverage so later-packed jobs see it. | Aggressive packing would over-commit bytes: tight caps where early arrivals strangle output reservations, and long chains where unclamped packing diverges in repair. The only packed schedule that is valid on every measured config. |
| `interval-entry` | First run the lead-time extension pass (below), which moves interval entries earlier where strict capacity allows. Then place each inbound independently at the latest `fire` such that `task_end(fire) + inbound_runtime <= deadline`, tightened so the trigger is no later than the task immediately before the planned interval entry. | Transfers need lead time the unmodified intervals don't provide. |
| `latest-safe` | Place each inbound independently at the latest `fire` in `[earliest, latest]` such that `task_end(fire) + inbound_runtime <= deadline`, assuming the inbound FIFO is otherwise idle. If none exists, use `latest`. | Memory pressure matters more than inbound congestion. The most conservative arrivals; the schedule that most often survives extreme cap tightness. |

`packed-fifo` and `packed-fit` are deliberate complements: unclamped packing
exploits the simulator's deferred destination allocation (a queued transfer
costs nothing until it starts, so early enqueues get event-driven backpressure
for free), while the clamp prevents the failure mode where an early transfer
*does* start and then strangles a later output reservation. Neither dominates
the other empirically, so both are derived.

A queue-aware middle ground was prototyped and rejected (2026-06): charging a
packed job's bytes from its estimated FIFO start (`max(enqueue, stream
cursor)`) instead of its trigger boundary interpolates between the two
variants rather than dominating either — on a 262-config grid it recovered
unclamped packing's congestion wins but reintroduced its repair divergence on
long tight chains (erroring on every 3,000+-task probe the strict clamp
solves) and gave up the strict clamp's conservative-rescuer wins. The strict
charge-from-trigger rule is what makes `packed-fit` the schedule that never
overcommits.

All schedules rely on simulator verification; a schedule that looks valid
analytically can be rejected or repaired when FIFO timing creates real
pressure. The schedules' plans usually agree at loose caps and diverge
under contention; the policy simply replays all of them and keeps the fastest
valid one (ties go to the table order above).

## Properties

- **General:** decisions are made from schema-level facts, not semantic object
  categories.
- **Fast:** four deterministic plans and four simulator replays; there is no
  search over residency plans.
- **Conservative first:** strict pressure reduction is preferred; timing relief
  is used only when strict reduction cannot make progress.
- **Standalone:** the policy does not fall back to another policy.

## Diagnostics

`plan_pressurefit_policy` returns a `PressureFitDiagnostics` object with:

- task/object counts and total planning wall time;
- number of valid schedules;
- selected schedule name and selected makespan;
- one row per schedule with status (`valid` or `error`), wall time, makespan
  when valid, and the schedule's control flags.

In the dataclass, JSON payload, and sweep CSV these rows are named
`candidates` (`candidate_count`, `selected_candidate`, `candidate_name`): a
candidate is exactly one inbound-schedule variant.

These diagnostics are observational. They do not alter the selected plan.

For repeatable comparisons across configs, run
`python scripts/canonical_sweep.py --pressurefit-candidates-out <csv>`, which
writes one row per config and schedule.

## Known Limits

- It can miss a faster chain that requires globally coordinated outbound and
  inbound stream placement rather than local interval pressure reduction.
- Physical verification repairs feasibility contradictions; it is not a general
  makespan optimizer.
- All four schedules aim to finish each transfer by its consumer's ideal
  start. A chain whose only feasible plan deliberately fires a prefetch at the
  consumer's immediate predecessor and stalls through it (delaying the transfer
  to preserve a predecessor's output reservation) is reported infeasible even
  though a valid annotation exists. No current benchmark config needs this;
  see Removed Alternatives.
- The policy is designed for ordered `TaskChain`s. A DAG scheduler would need a
  different boundary and deadline model.

## Removed Alternatives (2026-06)

Earlier versions evaluated a larger portfolio of candidate plans: alternative
seeds (source-gap trimming, cold admission, all-backing with protected initial
sets), a slack-reserve pressure variant, a latest-trigger schedule, and
`auto`/`fast`/`full` portfolio modes that gated which candidates ran by chain
size. A 286-config measurement (canonical llama3-8B grid on H100/RTX_5090 plus
an optimizer/grad-accum grid, all candidates evaluated) showed:

- the three schedules retained at that time achieved the full portfolio's best makespan on
  ~89% of feasible configs and covered every config some candidate solved;
- the removed candidates' wins were almost all under 2% (worst case ~10% on two
  tight-cap tiny-seqlen configs, won by initial-protection seeds);
- initial-protection candidates alone consumed ~68% of total planning time;
- `latest-safe` was the only schedule that survived several extreme-pressure
  and 3000+-task configs, which is why it is retained as the third variant.

The portfolio machinery (candidate families, portfolio modes, protected
initial sets) was removed in exchange for that measured regret. If a future
regime needs the dropped corners back, the better investment is an optimality
oracle (see docs/problem.md §7) rather than re-growing candidate families.

## Relation To Existing Policies

| Policy | Difference |
|---|---|
| `belady_reactive` | Walks forward and evicts reactively. `pressurefit` builds intervals first, then reduces overloaded boundaries. |
| `roundtrip_planner` | Constructs round-trip candidates up front. `pressurefit` derives round trips from interval cuts. |
| `max_reduce` | Starts from selective initial residency instead of universal initial residency. |
| `min_grow` | Avoids simulator-scored beam search over candidate residency plans. |

## Pseudocode

The planner flow itself is the Algorithm Summary above. The two passes with
non-obvious rules are pressure reduction and trigger emission:

```text
reduce_to_fit(intervals, extra_pressure):
    loop:
        pool = planned resident bytes at every boundary
        strict_overflow(x) =
            pool[x]
            + next_task_compute_outputs(x)
            + extra_pressure[x]
            - fast_memory_capacity

        worst = boundary with largest strict_overflow(x)

        if strict_overflow(worst) <= 0:
            return

        splits = legal_splits_at(worst, allow_timing_relief = false)

        if splits is empty:
            worst = boundary with largest relaxed overflow

            if relaxed_overflow(worst) <= 0:
                return

            splits = legal_splits_at(worst, allow_timing_relief = true)

        if splits is empty:
            raise infeasible

        split = best split by:
            transfer cost,
            clean initial drop,
            later first use,
            larger object,
            longer removed gap

        apply split
```

```text
emit_triggers(intervals, schedule):
    for each interval of each object:
        if interval starts at initial boundary:
            add initial compute copy when object has a backing source
        else if interval starts at object's producer:
            no arrival trigger is needed
        else:
            add inbound prefetch trigger per the schedule's placement rule

        if final_locations[object] == fast and this is the object's last interval:
            no exit trigger; the object stays resident at chain end
        else if interval exit contains a mutation and object has a later interval:
            add outbound offload trigger
        else if final_locations[object] == backing and backing lacks latest bytes:
            add outbound offload trigger
        else if object has no backing source and has a later interval:
            add outbound offload trigger
        else:
            add release trigger

    remove same-task release/offload/prefetch contradictions
    return annotated chain
```

## Appendix

### Implementation Module Map

- `pressurefit.py`: public entry points, the four schedule specs, simulator
  verification, fastest-valid selection;
- `pressurefit_aux/types.py`: the schedule-spec and interval-set types;
- `pressurefit_aux/core.py`: shared facts, interval accounting, and boundary
  helpers;
- `pressurefit_aux/seeds.py`: initial residency and seed interval construction;
- `pressurefit_aux/reducer.py`: deterministic greedy pressure reduction;
- `pressurefit_aux/emit.py`: interval-to-trigger emission;
- `pressurefit_aux/inbound_schedules.py`: inbound lead-time extension and
  inbound prefetch placement;
- `pressurefit_aux/physical_repair.py`: simulator-error interpretation and
  boundary pressure repair;
- `pressurefit_aux/diagnostics.py`: diagnostic result types.

### Boundary Model

The policy reasons over `n + 1` boundaries for `n` tasks. Boundary `-1` is the
initial state, boundary `0` is after task 0, and so on. An object use at task
`u` creates an anchor at boundary `u - 1`, because the object must be live before
task `u` starts. A produced object becomes live at its producer task's end; a
prefetched object occupies bytes one boundary earlier than its interval start
(see Byte Accounting).

### Byte Accounting

`resident_bytes(boundary)` is the sum of object sizes whose planned intervals
cover that boundary in the analytic model. It counts only compute-side bytes. It
does not include the next task's output reservation, slow memory, or
`physical_extra`.

For an interval `[a, b]`, the counted boundaries are:

- `a .. b` for initial-compute, initial-backing, and naturally produced intervals;
- `a - 1 .. b` for inbound-prefetched intervals, because the simulator allocates
  destination bytes when the transfer starts, not when it finishes.

`next_task_compute_outputs(boundary)` is the number of compute bytes that the next
task must reserve before it can start. It is added separately because outputs
are not part of any existing residency interval yet.

`physical_extra(boundary)` is a repair term, initially zero. It is not an object.
It means: "the analytic model must free at least this many more bytes at this
boundary because the simulator observed a real FIFO/capacity effect that the
static interval model missed." Examples are inbound destination bytes appearing when
a transfer starts, outbound source bytes staying live until transfer completion, or a
queue head blocked behind capacity.

### Seed Interval Set

In plain terms, the seed interval set is the starting guess for compute
residency: for each relevant object, one continuous interval that covers the
object's required anchors. It is intentionally simple.

The seed is not required to fit compute capacity. It is the input to pressure
reduction, which removes optional gaps until the strict capacity inequality can
be satisfied at every boundary.

Mathematically, the seed interval set `S` maps each planned object `o` to zero
or more inclusive boundary intervals — at most one per object in the seed;
pressure reduction may later split it into pieces. Let:

```text
producer(o)       = task index that produces o, or none
uses(o)           = sorted tasks that consume o
first_use(o)      = first task in uses(o)
last_use(o)       = last task in uses(o)
initial_compute(o) = true if o starts on compute
initial_backing(o)   = true if o starts on backing
initial_choice(o) = true if initial-residency selection places backing-source o
                    on compute at -1
```

Then the seed interval for object `o` is:

```text
if initial_backing(o) and uses(o) is empty:
    S[o] = []

elif initial_backing(o):
    start = -1 if initial_choice(o) else first_use(o) - 1
    end   = last_use(o) - 1
    S[o] = [[start, end]]

elif initial_compute(o):
    start = -1
    end   = last_use(o) - 1 if uses(o) is nonempty else -1
    S[o] = [[start, end]]

elif producer(o) exists:
    start = producer(o)
    end   = last_use(o) - 1 if uses(o) is nonempty else producer(o)
    S[o] = [[start, end]]

else:
    S[o] = []
```

Thus a seed interval is the smallest continuous interval, under the
initial-residency choice, that spans the object's required lifetime. It may
include optional residency between anchors; those optional stretches are exactly
what pressure reduction is allowed to cut.

### Pressure-Fit Interval Set

Pressure reduction takes the seed interval set `S` and produces a new interval
set `P`, where `P[o]` is the list of pressure-fit intervals for object `o`.
Each interval is an inclusive boundary range `[a, b]`. For each object:

1. **Subinterval property.** Every `[a, b]` in `P[o]` is contained in some
   `[a0, b0]` in `S[o]`:

   ```text
   a0 <= a <= b <= b0
   ```

   Pressure reduction can remove residency, but it does not create residency
   outside the seed.

2. **Anchor preservation.** Every required anchor remains covered by some
   interval in `P[o]`. Required anchors are:

   ```text
   -1                          if o is an initial compute object
   producer_task(o)            if o is produced by a task
   use_task(o) - 1             for every task that consumes o
   ```

   Removed gaps are therefore allowed only where there is no required anchor.

3. **Ordered pieces.** Intervals in `P[o]` are disjoint and ordered. Splitting
   one seed interval can produce a left piece, a right piece, both, or neither.

For boundary accounting, define:

```text
effective_start(o, [a, b]) =
    a - 1   if the interval starts from an inbound prefetch
    a       otherwise
```

An interval starts from an inbound prefetch when `a > -1` and `a` is not the
producer task for `o`. Object `o` is counted resident at boundary `x` when:

```text
effective_start(o, [a, b]) <= x <= b
```

The resident bytes of `P` at boundary `x` are:

```text
resident_P(x) =
    sum(size(o) for o where some interval in P[o] counts resident at x)
```

The strict capacity inequality is:

```text
resident_P(x)
+ next_task_compute_outputs(x)
+ physical_extra(x)
<= fast_memory_capacity
```

Pressure reduction first tries to produce a `P` satisfying this strict check at
every boundary. If no legal strict split can make progress, it retries against
**relaxed pressure**, which subtracts two kinds of bytes the static boundary
model is known to over-charge:

1. bytes that depart immediately after the current boundary;
2. inbound arrivals that are not needed by the next task and can wait until
   after that task reserves outputs.

In both cases, simulator verification is still the final feasibility check.

### Initial Residency

Initial residency is only a choice about boundary `-1`. It is not a promise that
the object will stay resident until its first use. Pressure reduction may cut an
initial interval if keeping it resident creates too much later pressure.

Finite-cap initial selection works in this order:

1. backing-source task-0 inputs are mandatory;
2. the policy computes a cold inbound FIFO estimate for every other used backing-source
   object, sorted by ascending first consumer task index;
3. for each object, `miss = max(0, estimated_inbound_end - first_consumer_start)`;
4. remaining objects are sorted by this exact key:

   ```text
   (
       first_consumer_task_index,
       inbound_slack,
       -miss,
       -object_size,
       object_id,
   )
   ```

5. `inbound_slack = max(0, first_consumer_start - task0_end - inbound_runtime)`;
6. objects are admitted in that order while initial compute bytes plus mandatory
   task-0 inputs plus task-0 compute outputs still fit.

### Split Legality

A split may remove only a gap that contains no anchor. The left piece must still
cover every production/use anchor before the removed gap, and the right piece
must still cover every anchor after the gap.

Split options are sorted by this exact key:

```text
(
    stream_cost,
    drop_initial_rank,
    -first_use_task_index,
    -object_size,
    -removed_gap_length,
)
```

Where:

- `stream_cost = 0` when the split either drops unused initial residency or can
  release a clean backing-source object;
- `stream_cost = 1` when the split needs an outbound offload to preserve bytes for a
  later interval;
- `drop_initial_rank = 0` when the split removes initial residency, otherwise
  `1`;
- larger `first_use_task_index`, larger `object_size`, and larger
  `removed_gap_length` are preferred because their negated values sort earlier.

Pressure reduction applies the split option with the lexicographically smallest
key.

### Inbound Lead-Time Extension

After pressure reduction, some prefetch intervals are capacity-feasible but too
late for the inbound FIFO. The lead-time pass (run only for the interval-entry
schedule) enumerates prefetched intervals, sorts them from latest deadline to
earliest deadline, and packs them backward. For each interval, it tries to move
the entry earlier and accepts the move only when every newly covered boundary
still satisfies the strict capacity inequality.

This changes only interval start positions. It does not change which objects
exist, which objects have backing sources, or which intervals are split.

### Physical Repair

The analytic model does not fully simulate the inbound and outbound FIFO queues. In the
real simulator:

- an inbound destination consumes compute bytes when the transfer starts;
- an outbound source keeps consuming compute bytes until the transfer completes;
- a queued transfer can block behind capacity at the queue head.

If the simulator reports a capacity contradiction, PressureFit translates the
error into extra pressure at the relevant boundary and reruns the same pressure
reduction pass.
This repair loop is bounded. It is for feasibility, not general makespan search.
