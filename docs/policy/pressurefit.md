# PressureFit

Fast standalone policy for linear `TaskChain`s. It plans device residency as
object intervals over task boundaries, cuts those intervals where byte pressure
is too high, tries bounded slack and H2D stall-relief candidate plans, then
emits release/offload/prefetch triggers with deadline-aware H2D ordering.

Implementation: `simulator/src/dataflow_sim/policy/pressurefit.py`.
Entry point: `apply_pressurefit_policy(bare, *, device_capacity=None,
refinement_iters=0)`. `refinement_iters` is accepted for API compatibility; the
policy does not use a tunable search budget.

## Design Contract

PressureFit is schema-driven. It uses:

- task order, runtimes, inputs, outputs, and device output sizes;
- object sizes and initial source locations;
- producer/use positions;
- explicit mutation metadata from `Task.mutates_inputs`;
- H2D/D2H bandwidths and `device_capacity`.

It does not depend on object ids, object types, or task names.

## Core Vocabulary

This section names only the concepts needed to read the algorithm. The appendix
contains the precise definitions and ranking rules.

- **Boundary:** a point between tasks where the policy accounts for device
  residency and output reservations. Boundary `-1` is the initial state.
- **Anchor:** a boundary where an object must be device-resident: initial device
  residency, production, or the boundary before a use.
- **Residency interval:** an inclusive boundary range `[a, b]` where an object
  is planned to be device-resident.
- **Seed interval set:** the candidate's starting residency hypothesis before
  pressure reduction cuts optional gaps.
- **Candidate spec:** one recipe for producing an annotated chain. It chooses
  a seed interval set, optional protected initial objects, optional reserve
  pressure, an H2D schedule, and whether to run H2D lead-time extension.
- **Pressure reduction:** the shared pass that cuts non-required residency gaps
  from a candidate spec until the modeled pressure fits.
- **Pressure-fit interval set:** the output of pressure reduction. It is the
  candidate's interval set after optional gaps have been removed, before trigger
  emission turns interval entries/exits into annotations.
- **Candidate plan:** the result of running one candidate spec through pressure
  reduction, optional H2D lead-time extension, trigger emission, physical
  repair, and simulator verification.

## Algorithm

The policy is best understood as a fixed portfolio of candidate specs. A spec
chooses the few things that are allowed to vary; the rest of the planner is
shared. This is the central separation:

- **Candidate-specific fields:** seed intervals, protected initial set, pressure
  reserve, H2D schedule, and whether H2D lead-time extension is enabled.
- **Common to every candidate:** pressure reduction, optional H2D lead-time
  extension, trigger emission, simulator verification, physical repair,
  and final makespan scoring.

The candidate-specific fields have these meanings:

| Field | Meaning | What it can change |
|---|---|---|
| `seed_intervals` | Map from object id to inclusive residency intervals before reduction. | The starting residency hypothesis, especially which host-source objects are initially resident. |
| `protected_initial` | Set of host-source object ids whose boundary-`-1` residency cannot be removed during pressure reduction. | Which initial objects are treated as non-droppable for this spec. It does not pin them forever. |
| `reserve_pressure` | Nonnegative bytes added to each boundary pressure check before comparing against capacity. | The amount of headroom pressure reduction must leave. It is not an object or runtime allocation. |
| `h2d_schedule` | Rule used when emitting prefetch triggers for interval entries that require H2D transfer. | Where H2D triggers are placed on the task chain from the pressure-fit interval set. |
| `extend_h2d` | Boolean enabling a pre-emission pass that moves some H2D interval starts earlier when strict capacity remains valid. | H2D lead time only. It does not split intervals or change object sources. |

The simulator never sees a candidate spec directly. It sees only the annotated
chain produced after pressure reduction, optional H2D lead-time extension,
trigger emission, and physical repair.

The common pipeline phases have these meanings:

| Phase | Input | Output | What it is allowed to change |
|---|---|---|---|
| Pressure reduction | Candidate `seed_intervals`, `protected_initial`, `reserve_pressure`, and any `physical_extra` from repair. | A pressure-fit interval set that satisfies the modeled capacity check. | Interval splits only. It does not choose a new seed or H2D schedule. |
| H2D lead-time extension | Pressure-fit interval set plus the same capacity model. | A pressure-fit interval set with some H2D entries moved earlier. | Interval starts for H2D-prefetched intervals only, and only when strict pressure remains valid. |
| Trigger emission | Pressure-fit interval set and the candidate `h2d_schedule`. | An annotated `TaskChain`. | Adds release, offload, prefetch, and initial-copy annotations. It does not replan residency. |
| Simulator verification | Annotated `TaskChain`. | A simulator makespan or a feasibility error. | Nothing; this phase observes the plan. |
| Physical repair | Simulator feasibility error plus current intervals. | Extra boundary pressure, followed by another reduction/emission attempt. | Adds conservative pressure at specific boundaries. It does not search for a faster plan. |

1. **Build reference facts.** Compute ideal compute-stream start/end times,
   object sizes, producers, uses, mutators, initial host/device sets, terminal
   placement constraints, and each next task's device output reservation.

2. **Build reusable seeds and protection sets.** Choose finite-cap initial residency,
   then build the base seed: one continuous interval per object from its
   anchors. Also build any auxiliary seeds or resource-derived protection sets
   used by the portfolio, such as a colder initial-residency seed and the
   initial-protection set.

3. **Generate candidate specs.** The current portfolio contains:

   - base candidate specs using the base seed with packed FIFO, latest-safe,
     interval-entry, and latest-trigger H2D schedules;
   - one slack-reserve candidate spec using the base seed;
   - one cold-admission candidate spec using a colder initial-residency seed;
   - initial-protection candidate specs chosen from H2D deadline demand and
     H2D-work frontiers.

   On large interactive chains, the portfolio skips the initial-protection
   frontier family and keeps the base/slack/cold-admission specs. This is a
   runtime guard for repeated-step chains, where each frontier candidate is
   much more expensive.

   The appendix enumerates every candidate family and its purpose.

4. **Run the common pipeline for each spec.** Copy the spec's seed, then run the
   same pressure reduction pass. While a boundary exceeds capacity, split one
   interval around that boundary. Pressure reduction first uses strict static
   pressure:

   ```text
   resident_bytes(boundary)
   + next_task_device_outputs(boundary)
   + physical_extra(boundary)
   + candidate_reserve(boundary)
   <= device_capacity
   ```

   If strict pressure has no legal split, pressure reduction may use timing
   relief for bytes that leave immediately after the boundary or H2D arrivals
   that can wait until after the next task reserves outputs.

5. **Optionally extend H2D lead time.** Some specs move H2D interval
   entries earlier when strict capacity permits. This pass changes only interval
   starts; it does not split intervals or change object sources.

6. **Emit and verify the candidate plan.** Convert interval entries/exits into
   release, offload, and prefetch triggers according to the spec's H2D schedule.
   Run the simulator. If the simulator reports a capacity
   contradiction caused by output reservation, missing live inputs, or stream
   queue timing, translate that contradiction into additional physical pressure,
   reduce again, and re-emit. The repair loop is bounded.

7. **Select the fastest valid plan.** Among candidate plans that verify, return
   the annotated chain with the lowest simulator makespan. If no spec produces a
   valid plan, raise the first planning error.

## Properties

- **General:** decisions are made from schema-level facts, not semantic object
  categories.
- **Fast:** there is no beam search over simulator-scored residency plans.
- **Bounded:** the candidate portfolio is fixed-size for a given chain; it does
  not search over arbitrary subsets of objects.
- **Conservative first:** strict pressure reduction is preferred; timing relief
  is used only when strict reduction cannot make progress.
- **Standalone:** the policy does not fall back to another policy. Its only
  makespan choice is between local H2D schedules derived from its own interval
  plan.

## Known Limits

- It can miss a faster chain that requires globally coordinated D2H/H2D stream
  placement rather than local interval pressure reduction.
- Physical verification repairs feasibility contradictions; it is not a general
  makespan optimizer.
- The policy is designed for ordered `TaskChain`s. A DAG scheduler would need a
  different boundary and deadline model.

## Relation To Existing Policies

| Policy | Difference |
|---|---|
| `belady_reactive` | Walks forward and evicts reactively. `pressurefit` builds intervals first, then reduces overloaded boundaries. |
| `roundtrip_planner` | Constructs round-trip candidates up front. `pressurefit` derives round trips from interval cuts. |
| `max_reduce` | Starts from selective initial residency instead of universal initial residency. |
| `min_grow` | Avoids simulator-scored beam search over candidate residency plans. |

## Pseudocode

```text
apply_pressurefit_policy(chain):
    if device_capacity override is provided:
        chain.device_capacity = override

    facts = build_facts(chain)
    initial = choose_initial_residency(chain, facts)
    base_intervals = build_initial_intervals(facts, initial)

    plans = []
    candidate_specs = build_candidate_specs(chain, facts, base_intervals)

    for spec in candidate_specs:
        plan = verify_candidate_plan(spec)
        if plan is valid:
            append plan to plans

    if plans is empty:
        raise first planning error

    return annotated chain from plan with lowest simulator makespan
```

```text
build_candidate_specs(chain, facts, base_intervals):
    specs = []

    specs += base_intervals with h2d_schedule = packed_fifo
    specs += base_intervals with h2d_schedule = latest_safe
    specs += base_intervals with h2d_schedule = interval_entry
    specs += base_intervals with h2d_schedule = latest_trigger

    specs += base_intervals with:
        reserve_pressure = max(next_task_device_outputs)
        h2d_schedule = packed_fifo

    specs += cold_admission_seed(device_capacity / 2) with:
        h2d_schedule = packed_fifo

    if chain is small enough for the full portfolio:
        for protected in select_initial_protection_sets(facts):
            specs += all_host_seed with:
                protected_initial = protected
                extend_h2d = true
                h2d_schedule in [packed_fifo, latest_safe, interval_entry]

    return specs
```

```text
verify_candidate_plan(spec):
    intervals = copy(spec.seed_intervals)
    protected_initial = spec.protected_initial
    extra_pressure = spec.reserve_pressure at every boundary

    reduce_to_fit(intervals, extra_pressure, protected_initial)

    if spec.extend_h2d:
        extend_h2d_lead_time(intervals, extra_pressure)

    repeat up to fixed repair limit:
        annotated = emit_triggers(intervals, spec.h2d_schedule)

        try:
            log = simulator_run(annotated, snapshots=False)
            return (makespan(log), annotated)

        if simulator error can be translated to physical pressure:
            extra_pressure[boundary] = required additional bytes
            reduce_to_fit(intervals, extra_pressure, protected_initial)

            if spec.extend_h2d:
                extend_h2d_lead_time(intervals, extra_pressure)
        else:
            raise error

    annotated = emit_triggers(intervals, spec.h2d_schedule)
    log = simulator_run(annotated, snapshots=False)
    return (makespan(log), annotated)
```

```text
reduce_to_fit(intervals, extra_pressure, protected_initial):
    loop:
        pool = planned resident bytes at every boundary
        worst = boundary with largest strict overflow

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
emit_triggers(intervals, h2d_schedule):
    for each interval of each object:
        if interval starts at initial boundary:
            add initial device copy when object has a host source
        else if interval starts at object's producer:
            no arrival trigger is needed
        else:
            add H2D prefetch trigger according to h2d_schedule

        if interval exit contains a mutation and object has a later interval:
            add D2H offload trigger
        else if final_locations[object] == host and host lacks latest bytes:
            add D2H offload trigger
        else if object has no host source and has a later interval:
            add D2H offload trigger
        else:
            add release trigger

    remove same-task release/offload/prefetch contradictions
    return annotated chain
```

## Appendix

### Boundary Model

The policy reasons over `n + 1` boundaries for `n` tasks. Boundary `-1` is the
initial state, boundary `0` is after task 0, and so on. An object use at task
`u` creates an anchor at boundary `u - 1`, because the object must be live before
task `u` starts.

Produced objects are different from prefetched objects. A produced object starts
at its producer task because it becomes live at that task's end. A prefetched
object that starts at interval boundary `a` is counted from boundary `a - 1`
when the H2D transfer can begin.

### Byte Accounting

`resident_bytes(boundary)` is the sum of object sizes whose planned intervals
cover that boundary in the analytic model. It counts only device-side bytes. It
does not include the next task's output reservation, host memory, or
`physical_extra`.

For an interval `[a, b]`, the counted boundaries are:

- `a .. b` for initial-device, initial-host, and naturally produced intervals;
- `a - 1 .. b` for H2D-prefetched intervals, because the simulator allocates H2D
  destination bytes when the transfer starts, not when it finishes.

`next_task_device_outputs(boundary)` is the number of device bytes that the next
task must reserve before it can start. It is added separately because outputs
are not part of any existing residency interval yet.

`physical_extra(boundary)` is a repair term, initially zero. It is not an object.
It means: "the analytic model must free at least this many more bytes at this
boundary because the simulator observed a real FIFO/capacity effect that the
static interval model missed." Examples are H2D destination bytes appearing when
a transfer starts, D2H source bytes staying live until transfer completion, or a
queue head blocked behind capacity.

### Seed Interval Set

In plain terms, a seed interval set is a candidate's starting guess for device
residency. It is intentionally simple: for each relevant object, start with one
continuous interval that covers the object's required anchors. Different
candidate specs use different seeds, mainly by changing which host-source
objects are assumed to be resident at boundary `-1`.

The seed is not required to fit device capacity. It is the input to pressure
reduction, which removes optional gaps until the modeled pressure fits.

Mathematically, a seed interval set `S` maps each planned object `o` to zero or
more inclusive boundary intervals:

```text
S[o] = [[a0, b0], [a1, b1], ...]
```

In the current policy, each seed initially contains at most one interval per
object. Let:

```text
producer(o)       = task index that produces o, or none
uses(o)           = sorted tasks that consume o
first_use(o)      = first task in uses(o)
last_use(o)       = last task in uses(o)
initial_device(o) = true if o starts on device
initial_host(o)   = true if o starts on host
initial_choice(o) = true if this candidate places host-source o on device at -1
```

The base, cold-admission, and initial-protection candidate families differ mainly
in this `initial_choice` predicate. Protected-initial candidates still use a
seed interval set; `protected_initial` only constrains which boundary-`-1`
pieces pressure reduction may remove from that seed.

Then the seed interval for object `o` is:

```text
if initial_host(o) and uses(o) is empty:
    S[o] = []

elif initial_host(o):
    start = -1 if initial_choice(o) else first_use(o) - 1
    end   = last_use(o) - 1
    S[o] = [[start, end]]

elif initial_device(o):
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

Thus a seed interval is the smallest continuous interval, under that candidate's
initial-residency choice, that spans the object's required lifetime. It may
include optional residency between anchors; those optional stretches are exactly
what pressure reduction is allowed to cut.

### Pressure-Fit Interval Set

In plain terms, a pressure-fit interval set is the result of taking a seed
interval set and cutting out some non-required residency gaps so the modeled
device bytes fit the capacity check. It does not invent new objects, new uses,
new producers, or a new H2D schedule. It only decides which optional stretches
of device residency to remove.

Mathematically, take the seed interval set `S` defined above as input. Pressure
reduction produces a new interval set `P`, where `P[o]` is the list of
pressure-fit intervals for object `o`. Each interval is an inclusive boundary
range `[a, b]`. For each object:

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
   -1                         if o is an initial device object
   producer_task(o)            if o is produced by a task
   use_task(o) - 1             for every task that consumes o
   -1                         if o is in protected_initial and S[o] covers -1
   ```

   Removed gaps are therefore allowed only where there is no required anchor.

3. **Ordered pieces.** Intervals in `P[o]` are disjoint and ordered. Splitting
   one seed interval can produce a left piece, a right piece, both, or neither.

For boundary accounting, define:

```text
effective_start(o, [a, b]) =
    a - 1   if the interval starts from an H2D prefetch
    a       otherwise
```

An interval starts from an H2D prefetch when `a > -1` and `a` is not the
producer task for `o`. Object `o` is counted resident at boundary `x` when:

```text
effective_start(o, [a, b]) <= x <= b
```

The resident bytes of `P` at boundary `x` are:

```text
resident_P(x) =
    sum(size(o) for o where some interval in P[o] counts resident at x)
```

The strict pressure check is:

```text
resident_P(x)
+ next_task_device_outputs(x)
+ physical_extra(x)
+ reserve_pressure
<= device_capacity
```

Pressure reduction first tries to produce a `P` satisfying this strict check at
every boundary. If no legal strict split can make progress, it may use the
relaxed check from the next section, which subtracts bytes that the static model
is known to over-count at that boundary. In both cases, simulator verification
is still the final feasibility check.

### Initial Residency

Initial residency is only a choice about boundary `-1`. It is not a promise that
the object will stay resident until its first use. Pressure reduction may cut an
initial interval if keeping it resident creates too much later pressure.

Finite-cap initial selection works in this order:

1. host-source task-0 inputs are mandatory;
2. the policy computes a cold H2D FIFO estimate for every other used host-source
   object, sorted by ascending first consumer task index;
3. for each object, `miss = max(0, estimated_h2d_end - first_consumer_start)`;
4. remaining objects are sorted by this exact key:

   ```text
   (
       first_consumer_task_index,
       h2d_slack,
       -miss,
       -object_size,
       object_id,
   )
   ```

5. `h2d_slack = max(0, first_consumer_start - task0_end - h2d_runtime)`;
6. objects are admitted in that order while initial device bytes plus mandatory
   task-0 inputs plus task-0 device outputs still fit.

### Split Legality

A split may remove only a gap that contains no anchor. The left piece must still
cover every production/use anchor before the removed gap, and the right piece
must still cover every anchor after the gap.

When a split removes initial residency for an object in `protected_initial`, that
split is skipped. This protection is used only by protected-initial candidate
plans. Baseline candidate plans pass an empty protected set.

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
  release a clean host-source object;
- `stream_cost = 1` when the split needs a D2H offload to preserve bytes for a
  later interval;
- `drop_initial_rank = 0` when the split removes initial residency, otherwise
  `1`;
- larger `first_use_task_index`, larger `object_size`, and larger
  `removed_gap_length` are preferred because their negated values sort earlier.

Pressure reduction applies the split option with the lexicographically smallest
key.

### Strict And Relaxed Pressure

Strict pressure is:

```text
planned_resident_bytes
+ next_task_device_outputs
+ physical_extra
```

Relaxed pressure subtracts two kinds of bytes that the static boundary model can
over-charge:

1. bytes that depart immediately after the current boundary;
2. H2D arrivals that are not needed by the next task and can wait until after
   that task reserves outputs.

Pressure reduction always tries strict pressure first. Relaxed pressure is used
only when no strict split can make progress.

### Candidate Plan Portfolio

The portfolio is a fixed list of local alternatives. Every candidate spec goes
through pressure reduction, trigger emission, and simulator verification
independently.

A candidate spec is exactly the tuple:

```text
(
    seed_intervals,
    protected_initial,
    reserve_pressure,
    extend_h2d,
    h2d_schedule,
)
```

Those fields are interpreted as follows:

| Field | Type | Operational effect |
|---|---|---|
| `seed_intervals` | object id -> list of inclusive `[start, end]` boundary intervals | Copied at the start of verification, then passed to pressure reduction. Different seeds give pressure reduction different starting assumptions. |
| `protected_initial` | set of object ids | Makes a split illegal if that split would remove boundary-`-1` residency for one of these objects. Later non-initial gaps may still be split normally. |
| `reserve_pressure` | bytes | Added as a baseline to every boundary's pressure check. Physical repair may add more pressure at individual boundaries, but may not go below this baseline. |
| `extend_h2d` | boolean | If true, run the H2D lead-time extension pass after each pressure-reduction attempt. |
| `h2d_schedule` | enum | Selects packed FIFO, latest-safe, interval-entry, or latest-trigger placement for H2D prefetch triggers. |

Only these fields vary across the portfolio. All candidate specs use the same
split legality rules, pressure reduction pass, trigger emission, simulator
verification, and physical repair loop.

| Family | Seed | Extra pressure | Protected initial set | H2D schedule(s) | Purpose |
|---|---|---|---|---|---|
| Base | normal finite-cap initial residency | none | none | packed FIFO, latest-safe, interval-entry, latest-trigger | Try the natural interval plan with different H2D trigger placement. |
| Slack reserve | base seed | `max(next_task_device_outputs)` at every boundary | none | packed FIFO | Leave output/FIFO headroom earlier than strict static pressure requires. |
| Cold admission | initial residency selected with half-cap admission budget | none | none | packed FIFO | Try one colder starting point without searching over initial subsets. |
| Initial protection | every used host-source object initially resident | none | deadline-demand and H2D-work frontier sets | packed FIFO for the first set and smallest byte-scale frontiers; latest-safe and interval-entry for all sets | Preserve selected boundary-`-1` residency when H2D demand or source-object timing suggests that dropping it may create FIFO stalls. Skipped by the large-chain fast portfolio. |

These families are stitched together only at the candidate-selection level. They
do not use separate correctness rules: all of them pass through the same
pressure reduction pass, trigger emission, simulator verification, and physical
repair loop.

### H2D Schedules

After pressure reduction produces a pressure-fit interval set, every
non-initial, non-produced interval entry becomes an H2D prefetch. The
candidate's H2D schedule determines where that prefetch trigger is emitted.

All schedules start from the same H2D job for each prefetched interval
`[a, b]` of object `o`:

```text
first_use = first task that consumes o inside [a, b]
earliest = max(previous_interval_fire_task(o), producer_task(o), 0)
latest = first_use - 1
deadline = ideal_start(first_use)
h2d_runtime = ceil(size(o) / h2d_bandwidth)
```

The schedule chooses a trigger task `fire` in `[earliest, latest]`. The trigger
is emitted as `prefetch_after` on that task.

| Schedule | Exact placement rule | Purpose |
|---|---|---|
| packed FIFO | Treat all H2D jobs as one FIFO queue. Sort jobs from latest deadline to earliest deadline, then pack them backward so each job finishes before its consumer deadline and before the next later-packed H2D job. If no such trigger exists inside the job window, use `earliest`. | Coordinate multiple H2D transfers that would otherwise pile up near their consumers. This is the default schedule when H2D queue congestion is the main risk. |
| latest-safe | Schedule each H2D independently. Pick the latest `fire` in `[earliest, latest]` such that `task_end(fire) + h2d_runtime <= deadline`, assuming the H2D FIFO is otherwise idle. If none exists, use `latest`. | Keep objects off device as long as possible. This is useful when memory pressure matters more than H2D FIFO congestion. |
| interval-entry | Use the latest-safe rule, but first tighten `latest` to `min(first_use - 1, a - 1)`. The trigger is therefore no later than the task immediately before the pressure-fit interval begins. | Respect the timing implied by the pressure-fit interval set, especially after H2D lead-time extension has intentionally moved an interval entry earlier. |
| latest-trigger | Use `fire = latest` for every H2D job, even when the transfer cannot finish by the consumer's ideal start. The consumer may stall while the transfer completes. | Preserve capacity for the task immediately before the consumer. This is useful when an early H2D destination would fit by itself but would make that predecessor's output reservation impossible. |

All schedules still rely on simulator verification; a schedule that looks valid
analytically can be rejected or repaired if FIFO timing creates real pressure.

### Slack-Reserve Candidate Plan

The slack-reserve candidate uses the same base intervals as the normal plan, but
initializes `extra_pressure(boundary)` to:

```text
reserve_bytes = max(next_task_device_outputs)
```

at every boundary. Pressure reduction therefore plans against:

```text
planned_resident_bytes
+ next_task_device_outputs
+ physical_extra
+ reserve_bytes
```

This reserve is deliberately conservative and bounded: there is only one such
candidate, and it uses packed FIFO H2D scheduling. It is not a correctness
requirement. Its purpose is to produce an alternate colder plan that may avoid
small stalls when real output reservations or queued transfers need room before
the static interval model would otherwise force that room to appear.

### Cold-Admission Candidate Plan

The cold-admission candidate reruns only the initial-residency selection with:

```text
admission_budget = floor(device_capacity / 2)
```

Task-0 host inputs remain mandatory. If they do not fit under the smaller
budget, the candidate is skipped. If the selected initial set is identical to
the normal finite-cap initial set, the candidate is skipped.

After initial residency is chosen, all later steps use the real
`device_capacity`. The smaller budget is not a runtime capacity and is not used
by pressure reduction except through the colder interval seed. This gives the
policy a single bounded alternative when filling the initial boundary too
aggressively creates later FIFO stalls.

### H2D Lead-Time Extension

After pressure reduction, some prefetch intervals are capacity-feasible but too
late for the H2D FIFO. The lead-time pass enumerates prefetched intervals, sorts
them from latest deadline to earliest deadline, and packs them backward. For
each interval, it tries to move the entry earlier and accepts the move only when
every newly covered boundary still satisfies the strict capacity check.

This changes only interval start positions. It does not change which objects
exist, which objects have host sources, or which intervals are split.

### Initial-Protection Candidate Plans

Initial-protection plans handle cases where pressure reduction's cheapest local
choice is to drop boundary-`-1` residency for host-source objects, but the
resulting H2D jobs may create FIFO stalls. The policy does not choose a fixed
number of objects. It builds a small set of candidate protected sets from object
sizes, transfer bandwidth, compute slack, reduced interval entries, first-use
positions, mutation metadata, and initial capacity.

The construction is:

1. Build a temporary all-host-source seed with every used host-source object
   initially resident.
2. Run pressure reduction on that seed.
3. For each host-source object whose boundary-`-1` interval was cut, build an
   initial-protection job for its first use:

   ```text
   first_use       = first task that consumes o
   [a, b]          = reduced interval containing first_use - 1
   release_time    = task_end(max(0, a - 1))
   deadline        = ideal_start(first_use)
   h2d_runtime     = ceil(size(o) / h2d_bandwidth)
   residency_cost  = size(o) * max(1, deadline)
   ```

   `release_time` is the earliest time implied by the reduced interval entry:
   if the object is not protected initially, an H2D transfer cannot safely begin
   before that entry without increasing modeled residency pressure.

4. Compute the bytes available for optional initial protection:

   ```text
   protection_headroom =
       device_capacity
     - initial_device_bytes
     - mandatory_task0_host_input_bytes
     - task0_device_outputs
   ```

5. Build the **deadline-deficit set**. Schedule unprotected protection jobs on a
   single H2D FIFO in deadline order,
   respecting each job's `release_time`:

   ```text
   start = max(h2d_cursor, release_time)
   end   = start + h2d_runtime
   miss  = max(0, end - deadline)
   ```

6. While any job misses its deadline and protection headroom remains, take the
   earliest missed deadline and choose one unprotected job with:

   ```text
   job.deadline <= earliest_missed_deadline
   job.size     <= remaining_protection_headroom
   ```

   The chosen job is the one with the smallest:

   ```text
   residency_cost / h2d_runtime
   ```

   with ties broken by larger `h2d_runtime`, earlier deadline, smaller size,
   then object id. Add that object to `protected_initial`, subtract its size
   from remaining headroom, and recompute H2D misses. The resulting set is one
   candidate protected set.

7. Build **H2D-work frontier sets**. A frontier set is a prefix of an ordered
   source-job list whose cumulative protected H2D runtime reaches a
   resource-derived scale point while still fitting `protection_headroom`.

   PressureFit currently uses three source-job orderings:

   - cut-demand order: jobs from step 3 sorted by earlier deadline, earlier
     first use, larger size, then object id;
   - clean tail order: host-source objects with no mutating task, sorted by
     later first use, larger size, then object id;
   - all-source tail order: all used host-source objects, sorted by later first
     use, larger size, then object id.

   For each ordering, define:

   ```text
   first_group = maximal prefix whose group_key equals group_key(first job)
   first_work  = sum(h2d_runtime(job) for job in first_group)
   total_work  = sum(h2d_runtime(job) for job in ordered jobs)
   horizon     = ceil(sqrt(first_work * total_work))
   ```

   The group key is `deadline` for cut-demand order and `first_use` for tail
   orders.

   The policy records the first urgency group and the immediately following
   urgency group. If the first group contains a single job, it also records
   prefixes at successive doubled H2D-work targets:

   ```text
   first_work, 2 * first_work, 4 * first_work, ...
   ```

   and stops after recording the first prefix whose cumulative H2D work reaches
   `horizon`. This is a logarithmic transfer-work frontier. The scale comes from
   transfer time and the ordered job list, not from a fixed number of objects.

8. For each nonempty deduplicated protected set, run candidate specs with the
   all-host-source seed, that `protected_initial` set, and H2D lead-time
   extension. The first protected set also tries packed FIFO H2D scheduling.
   Packed FIFO is also tried for any frontier whose protected bytes fit within
   the largest single used host-source object:

   ```text
   sum(size(o) for o in protected_initial)
   <= max(size(o) for used host-source object o)
   ```

   This keeps the packed-FIFO frontier bounded by a byte scale from the chain
   itself. All protected sets try latest-safe and interval-entry scheduling.

This is still bounded, but the bound comes from resources rather than object
count: protection prefixes stop at capacity or at a transfer-work horizon
derived from `first_work` and `total_work`.

### Physical Repair

The analytic model does not fully simulate the H2D and D2H FIFO queues. In the
real simulator:

- an H2D destination consumes device bytes when the transfer starts;
- a D2H source keeps consuming device bytes until the transfer completes;
- a queued transfer can block behind capacity at the queue head.

If the simulator reports a capacity contradiction, PressureFit translates the
error into extra pressure at the relevant boundary and reruns the same pressure
reduction pass.
This repair loop is bounded. It is for feasibility, not general makespan search.
