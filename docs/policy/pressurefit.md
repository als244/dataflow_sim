# PressureFit

PressureFit is a standalone policy for linear `TaskChain`s. It plans device
residency as intervals over task boundaries, removes optional residency where
modeled byte pressure exceeds capacity, emits release/offload/prefetch triggers,
then verifies the annotated chain with the simulator.

The short version is:

```text
candidate specs
    -> deterministic pressure reduction
    -> trigger emission
    -> simulator verification / bounded repair
    -> fastest valid candidate
```

The candidate specs are a small portfolio of starting assumptions and inbound
scheduling choices. The core pressure-reduction algorithm is shared by every
candidate; it is the part that actually decides which residency gaps are kept
or removed.

Public entry point:

```text
apply_pressurefit_policy(
    bare,
    *,
    device_capacity=None,
    refinement_iters=0,
    portfolio_mode="auto",
)
```

`refinement_iters` is accepted for API compatibility; PressureFit does not use a
tunable search budget.

Diagnostics entry point:

```text
plan_pressurefit_policy(...) -> (chain, diagnostics)
```

It runs the same planner and returns candidate-level status, timing, makespan,
and selection metadata.

## Design Contract

PressureFit is schema-driven. It uses:

- task order, runtimes, inputs, outputs, and device output sizes;
- object sizes and initial source locations;
- producer/use positions;
- explicit mutation metadata from `Task.mutates_inputs`;
- inbound and outbound bandwidths and `device_capacity`.

It does not depend on object ids, object types, or task names.

## Core Algorithm: Pressure Reduction

Pressure reduction is the central algorithm in PressureFit. Candidate specs
choose where the algorithm starts, and inbound schedules choose how the resulting
interval entries become prefetch triggers, but the residency decision itself is
made by pressure reduction.

The simplest way to read the policy is:

1. use a bounded set of heuristics to construct candidate specs;
2. for each spec, run the same deterministic greedy pressure-reduction pass;
3. emit transfer/release annotations from the reduced intervals;
4. verify each annotated chain in the simulator;
5. return the valid candidate with the lowest makespan.

The candidate specs are heuristic. They choose the initial interval hypothesis,
initial-residency/protection choices, reserve pressure, and inbound scheduling
style. Pressure reduction is deterministic once those choices are fixed, but it
is still greedy: its split ranking is a local rule for reaching feasibility, not
a proof of globally optimal residency.

The pressure-reduction problem is:

```text
given:
    seed intervals S
    required anchors A
    object sizes size(o)
    device capacity C
    output reservation Q(x) at each boundary x
    repair pressure E(x) at each boundary x
    candidate reserve R

find:
    intervals P

such that:
    P is made only by cutting gaps out of S
    every required anchor in A remains covered
    for every boundary x:
        resident_P(x) + Q(x) + E(x) + R <= C
```

Here `resident_P(x)` is the sum of device bytes for objects whose planned
intervals count resident at boundary `x`. `Q(x)` reserves memory for the next
task's device outputs, `E(x)` is extra pressure discovered by simulator repair,
and `R` is candidate-requested headroom. The appendix defines these quantities
and the exact boundary accounting.

Operationally, pressure reduction repeatedly finds the most overloaded
boundary, chooses a legal non-anchor gap whose removal reduces pressure at that
boundary, splits that interval, and updates the boundary byte counts. It stops
when the capacity inequality above holds at every boundary, or raises infeasible
when no legal split can remove enough optional residency.

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
  pressure, an inbound schedule, and whether to run inbound lead-time extension.
- **Pressure reduction:** the shared pass that cuts non-required residency gaps
  from a candidate spec until the capacity inequality is satisfied at every
  boundary.
- **Pressure-fit interval set:** the output of pressure reduction. It is the
  candidate's interval set after optional gaps have been removed, before trigger
  emission turns interval entries/exits into annotations.
- **Candidate plan:** the result of running one candidate spec through pressure
  reduction, optional inbound lead-time extension, trigger emission, physical
  repair, and simulator verification.

## Algorithm Summary

PressureFit deliberately separates the choices that vary across candidates from
the planning machinery that every candidate shares.

A candidate spec varies only these planning knobs:

- the seed interval set;
- the protected initial set;
- reserve pressure;
- inbound trigger scheduling;
- whether inbound lead-time extension is enabled.

Everything else is common. For each candidate, PressureFit:

1. copies the candidate's seed intervals;
2. runs pressure reduction to obtain a pressure-fit interval set;
3. optionally extends inbound interval starts earlier when strict pressure still
   fits;
4. emits release, offload, prefetch, and initial-copy annotations;
5. verifies the annotated chain with the simulator;
6. translates bounded simulator capacity contradictions into extra boundary
   pressure and reruns reduction when repair is possible;
7. scores the verified candidate by simulator makespan.

The simulator never sees a candidate spec directly. It sees only the annotated
`TaskChain` produced by this shared pipeline.

At a high level, the planner flow is:

1. **Build reference facts.** Compute ideal compute-stream times, object sizes,
   producers, uses, mutators, initial locations, final locations, and each next
   task's device output reservation.
2. **Build seed interval sets.** Construct the base seed from liveness anchors,
   plus auxiliary seeds for source-gap, cold-admission, and initial-protection
   candidates.
3. **Build the candidate portfolio.** Assemble a bounded list of candidate
   specs from the current portfolio mode.
4. **Evaluate candidates independently.** Run the shared pipeline above for each
   candidate.
5. **Return the fastest valid plan.** If no candidate verifies, raise the first
   planning error.

The appendix gives the exact candidate fields, candidate families, inbound
schedule rules, and pressure-reduction split ranking.

## Properties

- **General:** decisions are made from schema-level facts, not semantic object
  categories.
- **Fast:** there is no beam search over simulator-scored residency plans.
- **Bounded:** the candidate portfolio is fixed-size for a given chain; it does
  not search over arbitrary subsets of objects.
- **Conservative first:** strict pressure reduction is preferred; timing relief
  is used only when strict reduction cannot make progress.
- **Standalone:** the policy does not fall back to another policy. Its only
  makespan choice is between local inbound schedules derived from its own interval
  plan.

## Portfolio Modes And Diagnostics

`portfolio_mode` is a runtime/quality knob, not a correctness knob.

| Mode | Behavior | Intended use |
|---|---|---|
| `full` | Runs every candidate family, including initial-protection frontier specs. | Offline sweeps and small chains where best quality matters more than planner wall time. |
| `fast` | Runs base, source-gap, slack-reserve, and cold-admission families; skips initial-protection frontier specs. | Interactive or repeated-step chains where frontier candidates dominate planning time. |
| `auto` | Uses `full` below the large-chain guard and `fast` above it. | Default UI mode. |

In the current implementation, `auto` switches to `fast` when the chain has
more than 256 tasks or more than 512 distinct objects.

For very large chains, `auto` and `fast` may resolve to effective mode
`fast-minimal`. This evaluates `base-unpacked` and `source-gap-unpacked`, uses
`base-latest-inbound` only as a fallback if those fail, and records the remaining
secondary candidates as skipped. In the current implementation, this guard
starts above 4096 tasks or above 4096 distinct objects. `full` is the escape
hatch when exhaustive candidate comparison is desired.

`plan_pressurefit_policy` returns a `PressureFitDiagnostics` object with:

- requested and effective portfolio mode;
- task/object counts and total planning wall time;
- number of valid candidates;
- selected candidate name and selected makespan;
- one row per candidate with status (`valid`, `error`, or `skipped`),
  candidate family, wall time, makespan when valid, and candidate-specific
  fields.

These diagnostics are observational. They do not alter the selected plan.

For repeatable mode comparisons, run
`python app/scripts/pressurefit_mode_sweep.py --quick` or use `--compact` /
`--canonical` for broader grids. The script writes one row per config and mode,
including makespan, policy wall time, selected candidate, and candidate counts.

## Known Limits

- It can miss a faster chain that requires globally coordinated outbound and
  inbound stream placement rather than local interval pressure reduction.
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
plan_pressurefit_policy(chain):
    if device_capacity override is provided:
        chain.device_capacity = override

    facts = build_facts(chain)
    initial_device = choose_initial_residency(chain, facts)
    base_intervals = build_initial_intervals(facts, initial_device)
    seeds = build_candidate_seeds(chain, facts, base_intervals)
    portfolio = build_candidate_portfolio(
        chain, facts, initial_device, seeds, portfolio_mode
    )

    plans = []
    diagnostics = []

    for spec in portfolio.specs:
        if spec is marked skipped:
            append skipped diagnostic
            continue

        if spec is fallback-only and plans is not empty:
            append skipped diagnostic
            continue

        if spec has a precomputed construction error:
            append error diagnostic
            continue

        try:
            plan = verify_candidate_plan(spec, seeds[spec.seed_key])
            append plan and valid diagnostic
        except planning error:
            append error diagnostic

    if plans is empty:
        raise first planning error

    return annotated chain from plan with lowest simulator makespan, diagnostics
```

```text
build_candidate_portfolio(chain, facts, initial_device, seeds, portfolio_mode):
    effective_mode = resolve_portfolio_mode(portfolio_mode, facts)
    minimal_fast = effective_mode == fast-minimal
    specs = []

    add base specs:
        if minimal_fast:
            base-unpacked       -> seed_key=base, pack_inbound=false
            source-gap-unpacked -> seed_key=source-gap, pack_inbound=false
            base-latest-inbound -> seed_key=base,
                                   pack_inbound=true,
                                   latest_inbound=true,
                                   fallback_only=true
            record secondary base schedules as skipped
        else:
            base-packed-fifo    -> seed_key=base, pack_inbound=true
            base-unpacked       -> seed_key=base, pack_inbound=false
            source-gap-unpacked -> seed_key=source-gap, pack_inbound=false
            base-interval-entry -> seed_key=base,
                                   extend_inbound=true,
                                   respect_interval_start=true
            base-latest-inbound -> seed_key=base,
                                   pack_inbound=true,
                                   latest_inbound=true

    add reserve spec:
        if minimal_fast:
            record reserve-next-output as skipped
        else:
            seed_key=base
            reserve_pressure = max(next_task_device_outputs)
            pack_inbound=true

    add cold-admission spec when it differs from base:
        if minimal_fast:
            record cold-admission as skipped
        else:
            seed_key=cold-admission
            pack_inbound=true

    if effective_mode is fast or fast-minimal:
        record skipped initial-protection frontier diagnostic if applicable
        return specs

    if effective_mode is full:
        for protected in select_initial_protection_sets(facts):
            add protected specs:
                seed_key=all-host
                protected_initial=protected
                extend_inbound=true
                schedules in [packed FIFO, latest-safe, interval-entry]

    return specs
```

```text
verify_candidate_plan(spec):
    intervals = copy(seeds[spec.seed_key])
    protected_initial = spec.protected_initial
    extra_pressure = spec.reserve_pressure at every boundary

    reduce_to_fit(intervals, extra_pressure, protected_initial)

    if spec.extend_inbound:
        extend_inbound_lead_time(intervals, extra_pressure)

    repeat up to fixed repair limit:
        annotated = emit_triggers(intervals, spec schedule flags)

        try:
            log = simulator_run(annotated, snapshots=False)
            return (makespan(log), annotated)

        if simulator error can be translated to physical pressure:
            extra_pressure[boundary] = required additional bytes
            reduce_to_fit(intervals, extra_pressure, protected_initial)

            if spec.extend_inbound:
                extend_inbound_lead_time(intervals, extra_pressure)
        else:
            raise error

    annotated = emit_triggers(intervals, spec schedule flags)
    log = simulator_run(annotated, snapshots=False)
    return (makespan(log), annotated)
```

```text
reduce_to_fit(intervals, extra_pressure, protected_initial):
    loop:
        pool = planned resident bytes at every boundary
        strict_overflow(x) =
            pool[x]
            + next_task_device_outputs(x)
            + extra_pressure[x]
            - device_capacity

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
emit_triggers(intervals, spec schedule flags):
    for each interval of each object:
        if interval starts at initial boundary:
            add initial device copy when object has a host source
        else if interval starts at object's producer:
            no arrival trigger is needed
        else:
            add inbound prefetch trigger according to spec schedule flags

        if interval exit contains a mutation and object has a later interval:
            add outbound offload trigger
        else if final_locations[object] == host and host lacks latest bytes:
            add outbound offload trigger
        else if object has no host source and has a later interval:
            add outbound offload trigger
        else:
            add release trigger

    remove same-task release/offload/prefetch contradictions
    return annotated chain
```

## Appendix

### Implementation Module Map

- `pressurefit.py`: public entry points, candidate evaluation, simulator
  verification, fastest-valid selection;
- `pressurefit_aux/portfolio.py`: candidate portfolio orchestration;
- `pressurefit_aux/candidate_specs.py`: candidate-family assembly and portfolio
  mode selection;
- `pressurefit_aux/types.py`: shared candidate, portfolio, and interval types;
- `pressurefit_aux/core.py`: shared facts, interval accounting, and boundary
  helpers;
- `pressurefit_aux/seeds.py`: initial residency and seed interval construction;
- `pressurefit_aux/reducer.py`: deterministic greedy pressure reduction;
- `pressurefit_aux/emit.py`: interval-to-trigger emission;
- `pressurefit_aux/inbound_schedules.py`: inbound lead-time extension and
  inbound prefetch scheduling;
- `pressurefit_aux/initial_protection.py`: initial-residency protection
  frontier construction;
- `pressurefit_aux/physical_repair.py`: simulator-error interpretation and
  boundary pressure repair;
- `pressurefit_aux/diagnostics.py`: diagnostic result types and candidate
  diagnostic rows.

### Boundary Model

The policy reasons over `n + 1` boundaries for `n` tasks. Boundary `-1` is the
initial state, boundary `0` is after task 0, and so on. An object use at task
`u` creates an anchor at boundary `u - 1`, because the object must be live before
task `u` starts.

Produced objects are different from prefetched objects. A produced object starts
at its producer task because it becomes live at that task's end. A prefetched
object that starts at interval boundary `a` is counted from boundary `a - 1`
when the inbound transfer can begin.

### Byte Accounting

`resident_bytes(boundary)` is the sum of object sizes whose planned intervals
cover that boundary in the analytic model. It counts only device-side bytes. It
does not include the next task's output reservation, host memory, or
`physical_extra`.

For an interval `[a, b]`, the counted boundaries are:

- `a .. b` for initial-device, initial-host, and naturally produced intervals;
- `a - 1 .. b` for inbound-prefetched intervals, because the simulator allocates
  destination bytes when the transfer starts, not when it finishes.

`next_task_device_outputs(boundary)` is the number of device bytes that the next
task must reserve before it can start. It is added separately because outputs
are not part of any existing residency interval yet.

`physical_extra(boundary)` is a repair term, initially zero. It is not an object.
It means: "the analytic model must free at least this many more bytes at this
boundary because the simulator observed a real FIFO/capacity effect that the
static interval model missed." Examples are inbound destination bytes appearing when
a transfer starts, outbound source bytes staying live until transfer completion, or a
queue head blocked behind capacity.

### Seed Interval Set

In plain terms, a seed interval set is a candidate's starting guess for device
residency. It is intentionally simple: for each relevant object, start with one
continuous interval that covers the object's required anchors. Different
candidate specs use different seeds, mainly by changing which host-source
objects are assumed to be resident at boundary `-1`.

The seed is not required to fit device capacity. It is the input to pressure
reduction, which removes optional gaps until the strict capacity inequality can
be satisfied at every boundary.

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
device bytes satisfy the capacity inequality. It does not invent new objects,
new uses, new producers, or a new inbound schedule. It only decides which optional
stretches of device residency to remove.

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
+ next_task_device_outputs(x)
+ physical_extra(x)
+ candidate_reserve
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
2. the policy computes a cold inbound FIFO estimate for every other used host-source
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
- `stream_cost = 1` when the split needs an outbound offload to preserve bytes for a
  later interval;
- `drop_initial_rank = 0` when the split removes initial residency, otherwise
  `1`;
- larger `first_use_task_index`, larger `object_size`, and larger
  `removed_gap_length` are preferred because their negated values sort earlier.

Pressure reduction applies the split option with the lexicographically smallest
key.

### Strict And Relaxed Pressure

For interval set `P`, strict pressure at boundary `x` is:

```text
strict_pressure_P(x) =
    resident_P(x)
    + next_task_device_outputs(x)
    + physical_extra(x)
    + candidate_reserve
```

The corresponding overflow is:

```text
strict_overflow_P(x) = strict_pressure_P(x) - device_capacity
```

The strict capacity inequality is satisfied at `x` when
`strict_overflow_P(x) <= 0`.

Relaxed pressure subtracts two kinds of bytes that the static boundary model can
over-charge:

1. bytes that depart immediately after the current boundary;
2. inbound arrivals that are not needed by the next task and can wait until after
   that task reserves outputs.

Pressure reduction always tries strict pressure first. Relaxed pressure is used
only when no strict split can make progress.

### Candidate Plan Portfolio

The portfolio is a fixed list of local alternatives. Every candidate spec goes
through pressure reduction, trigger emission, and simulator verification
independently.

A candidate spec has identity fields plus control fields. The identity fields
are `name`, `family`, `seed_key`, and `seed`; they are used for diagnostics and
for choosing which seed interval set to copy. The control fields are:

```text
(
    protected_initial,
    reserve_pressure,
    extend_inbound,
    pack_inbound,
    respect_interval_start,
    latest_inbound,
)
```

Those control fields are interpreted as follows:

| Field | Type | Operational effect |
|---|---|---|
| `protected_initial` | set of object ids | Makes a split illegal if that split would remove boundary-`-1` residency for one of these objects. Later non-initial gaps may still be split normally. |
| `reserve_pressure` | bytes | Added as a baseline to every boundary's pressure check. Physical repair may add more pressure at individual boundaries, but may not go below this baseline. |
| `extend_inbound` | boolean | If true, run the inbound lead-time extension pass after each pressure-reduction attempt. |
| `pack_inbound` | boolean | If true, inbound jobs are assigned by the packed FIFO scheduler. If false, each inbound trigger is placed independently. |
| `respect_interval_start` | boolean | If true, independent inbound placement cannot fire later than the task immediately before the planned interval entry. |
| `latest_inbound` | boolean | If true with `pack_inbound`, packed jobs fire at their latest legal trigger instead of being packed backward by deadline. |

`skip_reason`, `pre_error`, and `fallback_only` are portfolio-control or
diagnostic fields. They do not define a different residency algorithm.

Only the seed identity and control fields above vary across the portfolio. All
candidate specs use the same split legality rules, pressure reduction pass,
trigger emission, simulator verification, and physical repair loop.

The named inbound schedules used in this document map to control fields as:

| Schedule | Control fields |
|---|---|
| packed FIFO | `pack_inbound=true` |
| latest-safe | `pack_inbound=false`, `respect_interval_start=false`, `latest_inbound=false` |
| interval-entry | `pack_inbound=false`, `respect_interval_start=true` |
| latest-trigger | `pack_inbound=true`, `latest_inbound=true` |

| Family | Seed | Extra pressure | Protected initial set | Inbound schedule(s) | Purpose |
|---|---|---|---|---|---|
| Base | normal finite-cap initial residency | none | none | packed FIFO, latest-safe, interval-entry, latest-trigger | Try the natural interval plan with different inbound trigger placement. |
| Source gap | base seed after long dirty source-state no-use gaps are split when the round trip fits | none | none | latest-safe | Avoid carrying updated source-state bytes through long no-use gaps when a bounded outbound and inbound round trip can fit inside the gap. |
| Slack reserve | base seed | `max(next_task_device_outputs)` at every boundary | none | packed FIFO | Leave output/FIFO headroom earlier than strict static pressure requires. |
| Cold admission | initial residency selected with half-cap admission budget | none | none | packed FIFO | Try one colder starting point without searching over initial subsets. |
| Initial protection | every used host-source object initially resident | none | deadline-demand and inbound-work frontier sets | packed FIFO for the first set and smallest byte-scale frontiers; latest-safe and interval-entry for all sets | Preserve selected boundary-`-1` residency when inbound demand or source-object timing suggests that dropping it may create FIFO stalls. Skipped by the large-chain fast portfolio. |

These families are stitched together only at the candidate-selection level. They
do not use separate correctness rules: all of them pass through the same
pressure reduction pass, trigger emission, simulator verification, and physical
repair loop.

Portfolio modes decide which rows from this table are evaluated:

- `full` evaluates all rows.
- `fast` evaluates base, source-gap, slack reserve, and cold admission only.
- `auto` resolves to `full` for small chains, `fast` for large chains, and
  `fast-minimal` for very large chains.

Candidate diagnostics report one row for every evaluated candidate and for
deliberately skipped family-level candidates. The diagnostic row name is stable
enough for measurement scripts and UI display, but it is not part of the
simulator schema.

### Inbound Schedules

After pressure reduction produces a pressure-fit interval set, every
non-initial, non-produced interval entry becomes an inbound prefetch. The
candidate's inbound schedule determines where that prefetch trigger is emitted.

All schedules start from the same inbound job for each prefetched interval
`[a, b]` of object `o`:

```text
first_use = first task that consumes o inside [a, b]
earliest = max(previous_interval_fire_task(o), producer_task(o), 0)
latest = first_use - 1
deadline = ideal_start(first_use)
inbound_runtime = ceil(size(o) / inbound_bandwidth)
```

The schedule chooses a trigger task `fire` in `[earliest, latest]`. The trigger
is emitted as `prefetch_after` on that task.

| Schedule | Exact placement rule | Purpose |
|---|---|---|
| packed FIFO | Treat all inbound jobs as one FIFO queue. Sort jobs from latest deadline to earliest deadline, then pack them backward so each job finishes before its consumer deadline and before the next later-packed inbound job. If no such trigger exists inside the job window, use `earliest`. | Coordinate multiple inbound transfers that would otherwise pile up near their consumers. This is the default schedule when inbound queue congestion is the main risk. |
| latest-safe | Schedule each inbound independently. Pick the latest `fire` in `[earliest, latest]` such that `task_end(fire) + inbound_runtime <= deadline`, assuming the inbound FIFO is otherwise idle. If none exists, use `latest`. | Keep objects off device as long as possible. This is useful when memory pressure matters more than inbound FIFO congestion. |
| interval-entry | Use the latest-safe rule, but first tighten `latest` to `min(first_use - 1, a - 1)`. The trigger is therefore no later than the task immediately before the pressure-fit interval begins. | Respect the timing implied by the pressure-fit interval set, especially after inbound lead-time extension has intentionally moved an interval entry earlier. |
| latest-trigger | Use `fire = latest` for every inbound job, even when the transfer cannot finish by the consumer's ideal start. The consumer may stall while the transfer completes. | Preserve capacity for the task immediately before the consumer. This is useful when an early inbound destination would fit by itself but would make that predecessor's output reservation impossible. |

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
candidate, and it uses packed FIFO inbound scheduling. It is not a correctness
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

### Inbound Lead-Time Extension

After pressure reduction, some prefetch intervals are capacity-feasible but too
late for the inbound FIFO. The lead-time pass enumerates prefetched intervals, sorts
them from latest deadline to earliest deadline, and packs them backward. For
each interval, it tries to move the entry earlier and accepts the move only when
every newly covered boundary still satisfies the strict capacity inequality.

This changes only interval start positions. It does not change which objects
exist, which objects have host sources, or which intervals are split.

### Initial-Protection Candidate Plans

Initial-protection plans handle cases where pressure reduction's cheapest local
choice is to drop boundary-`-1` residency for host-source objects, but the
resulting inbound jobs may create FIFO stalls. The policy does not choose a fixed
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
   inbound_runtime     = ceil(size(o) / inbound_bandwidth)
   residency_cost  = size(o) * max(1, deadline)
   ```

   `release_time` is the earliest time implied by the reduced interval entry:
   if the object is not protected initially, an inbound transfer cannot safely begin
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
   single inbound FIFO in deadline order,
   respecting each job's `release_time`:

   ```text
   start = max(inbound_cursor, release_time)
   end   = start + inbound_runtime
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
   residency_cost / inbound_runtime
   ```

   with ties broken by larger `inbound_runtime`, earlier deadline, smaller size,
   then object id. Add that object to `protected_initial`, subtract its size
   from remaining headroom, and recompute inbound misses. The resulting set is one
   candidate protected set.

7. Build **inbound-work frontier sets**. A frontier set is a prefix of an ordered
   source-job list whose cumulative protected inbound runtime reaches a
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
   first_work  = sum(inbound_runtime(job) for job in first_group)
   total_work  = sum(inbound_runtime(job) for job in ordered jobs)
   horizon     = ceil(sqrt(first_work * total_work))
   ```

   The group key is `deadline` for cut-demand order and `first_use` for tail
   orders.

   The policy records the first urgency group and the immediately following
   urgency group. If the first group contains a single job, it also records
   prefixes at successive doubled inbound-work targets:

   ```text
   first_work, 2 * first_work, 4 * first_work, ...
   ```

   and stops after recording the first prefix whose cumulative inbound work reaches
   `horizon`. This is a logarithmic transfer-work frontier. The scale comes from
   transfer time and the ordered job list, not from a fixed number of objects.

8. For each nonempty deduplicated protected set, run candidate specs with the
   all-host-source seed, that `protected_initial` set, and inbound lead-time
   extension. The first protected set also tries packed FIFO inbound scheduling.
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

The analytic model does not fully simulate the inbound and outbound FIFO queues. In the
real simulator:

- an inbound destination consumes device bytes when the transfer starts;
- an outbound source keeps consuming device bytes until the transfer completes;
- a queued transfer can block behind capacity at the queue head.

If the simulator reports a capacity contradiction, PressureFit translates the
error into extra pressure at the relevant boundary and reruns the same pressure
reduction pass.
This repair loop is bounded. It is for feasibility, not general makespan search.
