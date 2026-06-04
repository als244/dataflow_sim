# Workload recipe — how to ingest a custom workload

This is the API surface guide for taking *your* compute graph (a transformer
forward pass, a CNN, a custom kernel chain) and getting it simulating under
this discrete-event memory-traffic model. If you want a full real example,
see `docs/transformer-recipe.md`; this document is the reference.

## The five primitives

Everything ingested into the simulator is built out of five frozen
dataclasses defined in `simulator/src/dataflow_sim/schema.py`.

### `Object` — a piece of data present at t=0

| field      | type                                                   | units | semantics |
|------------|--------------------------------------------------------|-------|-----------|
| `id`       | `str`                                                  | —     | Globally unique id. Used by tasks to reference it. |
| `size`     | `int`                                                  | bytes | Counted against `device_capacity` / `host_capacity`. |
| `location` | `"host" \| "device"`                                   | —     | Where the object initially lives. Default `"device"`. |
| `type`     | `"weight" \| "activation" \| "gradient" \| "optimizer" \| "other"` | — | Semantic tag for the UI / policies. Default `"other"`. |

`Object` appears only in `TaskChain.initial_memory`.

### `Task` — a compute step

| field             | type                       | units  | semantics |
|-------------------|----------------------------|--------|-----------|
| `id`              | `str`                      | —      | Globally unique id. |
| `inputs`          | `list[str]`                | —      | Object ids that must be **live on device** when the task starts. Read-only unless listed in `mutates_inputs`. |
| `outputs`         | `list[OutputAlloc]`        | —      | Fresh allocations produced by this task. Output ids must not collide with anything else in the pool. |
| `runtime`         | `int`                      | cycles | Wall-time for the compute step on the (single) compute stream. |
| `releases_after`  | `list[str]`                | —      | Object ids freed from the device immediately after `task_end`. Must be `live`. |
| `offload_after`   | `list[TransferTrigger]`    | —      | Per-task D→H transfers enqueued at `task_end`. |
| `prefetch_after`  | `list[TransferTrigger]`    | —      | Per-task H→D transfers enqueued at `task_end`. |
| `mutates_inputs`  | `list[str]`                | —      | Subset of `inputs` whose bytes this task modifies in place. Policies must preserve the updated bytes for any later device use. Whether the final bytes must end on host is controlled by `TaskChain.final_locations`. |

### `OutputAlloc` — what a task produces

| field      | type                  | units | semantics |
|------------|-----------------------|-------|-----------|
| `id`       | `str`                 | —     | Fresh id; can't collide with any existing pool entry at `(id, location)`. |
| `size`     | `int`                 | bytes | Reserved at task-start, becomes `live` at task-end. |
| `location` | `"host" \| "device"`  | —     | Default `"device"`. Host outputs have no stall mechanism — must fit at task-start or the simulator raises. |
| `type`     | `ObjectType`          | —     | Same set as `Object.type`. |

### `TransferTrigger` — a single transfer plan fired by a task

| field    | type        | units  | semantics |
|----------|-------------|--------|-----------|
| `obj_id` | `str`       | —      | Which object to move. |
| `runtime`| `int \| None` | cycles | Override the bandwidth-derived runtime. `None` uses `ceil(size / bandwidth_{h2d,d2h})`. |

The trigger's direction is implicit from which task list it lives in:
`offload_after` triggers go on the D→H stream, `prefetch_after` on H→D.

### `TaskChain` — the whole workload

| field             | type                  | units            | semantics |
|-------------------|-----------------------|------------------|-----------|
| `initial_memory`  | `list[Object]`        | —                | t=0 pool state. |
| `tasks`           | `list[Task]`          | —                | Executed strictly in order. |
| `final_locations` | `dict[str, "host" \| "device"]` | —       | Optional terminal placement constraints. Omitted objects are disposable after their final use. |
| `device_capacity` | `int \| None`         | bytes            | `None` = unlimited. Hard ceiling; over-commits raise. |
| `host_capacity`   | `int \| None`         | bytes            | Same. |
| `bandwidth_h2d`   | `int \| None`         | bytes per cycle  | Default runtime for H→D triggers. Required unless every H→D trigger has an explicit `runtime`. |
| `bandwidth_d2h`   | `int \| None`         | bytes per cycle  | Same for D→H. |

`TaskChain.from_dict` and `TaskChain.load(path)` parse a JSON form with the
same field names.

## Bare vs annotated chain

A **bare chain** is just topology + sizes + runtimes: `initial_memory`,
`tasks` with `inputs`/`outputs`/`runtime`, and the capacity / bandwidth
constants. It says *what the workload does* but is silent on memory
management — every `releases_after`, `offload_after`, `prefetch_after` list
is empty, and the only data on the device at t=0 is whatever the workload
genuinely starts with there.

An **annotated chain** is a bare chain plus a memory plan: object
placements may have been moved to host in `initial_memory`, and every task
carries the `releases_after` / `offload_after` / `prefetch_after` triggers
that make the workload actually fit under `device_capacity`.

Policies in `dataflow_sim.policy.*` are functions `bare → annotated`. You
hand them the bare chain plus a budget and they return a `TaskChain` with
the annotations filled in.

## Building a bare chain

1. Enumerate every object that exists at t=0 (model weights, persistent
   buffers). For each, decide its initial location and write an `Object`.
2. Enumerate every compute step in execution order. For each, list its
   input ids, build `OutputAlloc`s for everything it produces, set a
   `runtime`. Leave `releases_after` / `offload_after` / `prefetch_after`
   empty.
3. If the latest bytes of any object must be present in a particular location
   after the chain ends, add it to `final_locations`, for example
   `{"W_0": "host"}`. Leave objects out when they are disposable after their
   final use.
4. Pick `device_capacity`, `host_capacity`, `bandwidth_h2d`,
   `bandwidth_d2h`. Use units that are internally consistent — the
   simulator never multiplies by a wall-clock conversion.
5. Construct the `TaskChain`. That's it — you can already pass it to a
   policy.

Tiny worked example:

```python
from dataflow_sim.schema import Object, Task, OutputAlloc, TaskChain

bare = TaskChain(
    initial_memory=[
        Object(id="W", size=8, location="device", type="weight"),
    ],
    tasks=[
        Task(id="fwd", inputs=["W"],
             outputs=[OutputAlloc("act", 4, "device", "activation")],
             runtime=10),
        Task(id="bwd", inputs=["W", "act"],
             outputs=[OutputAlloc("grad", 8, "device", "gradient")],
             runtime=20),
    ],
    device_capacity=24, host_capacity=64,
    bandwidth_h2d=2, bandwidth_d2h=2,
)
```

This runs as-is (everything fits under `device_capacity=24`). The
interesting case is when it doesn't — then you need a policy.

## Picking a policy

| Policy                          | Idea                                                                                   |
|---------------------------------|----------------------------------------------------------------------------------------|
| `belady_reactive`               | Reactive eviction by farthest next-use, with prefetches scheduled to hide latency.     |
| `min_grow`                      | Grow the working set as slowly as possible; only offload when capacity demands.        |
| `max_reduce`                    | Eagerly offload as soon as inputs are consumed.                                        |
| `sliding_window`                | Fixed-size sliding window over the reference stream.                                   |
| `roundtrip_planner`             | Plans full offload+prefetch round-trips against a shadow simulation.                   |
| `pressurefit`                  | Fast pressure-fit interval planning with deadline-aware H2D scheduling.|

Each is `apply_<name>_policy(bare, **kwargs) -> TaskChain`. See
`docs/policy/README.md` for the trade-offs and per-policy knobs.

## Running the simulator

```python
from dataflow_sim.simulator import run

events = run(annotated_chain)   # -> EventLog
```

`run()` normally does a two-pass simulation: the first pass discovers each
task's *actual* scheduled start time (accounting for transfer stalls); the
second re-emits snapshots with reference-stream timestamps based on those
actual starts. The return value is an `EventLog` with:

- `task_intervals: list[TaskInterval]` — one per compute task and one per
  fired transfer, with `track` in `{"compute", "h2d", "d2h"}`. This is
  what the timeline UI renders.
- `events: list[Event]` — every state transition, each carrying a
  `Snapshot` of the memory pool, the active task, and the remaining
  reference stream.
- `peak_device_bytes: int` — maximum device-pool bytes observed during
  simulation.
- `memory_trace: list[MemoryTracePoint]` — optional compact GPU-memory plot
  samples when `run(..., memory_trace=True)`.

For policy scoring or sweeps that only need makespan/peak and runtime
validation, use:

```python
score_log = run(annotated_chain, snapshots=False)
```

This returns the same `task_intervals` and `peak_device_bytes`, but leaves
`events` empty and skips snapshot/reference-stream construction.

For large UI-style runs that still need a GPU memory plot, use:

```python
trace_log = run(annotated_chain, snapshots=False, memory_trace=True)
```

This keeps `events` empty but fills `memory_trace` with aggregate device bytes
by band, without object-level memory contents or reference streams.

Key `Event.kind` values:

| kind                 | when |
|----------------------|------|
| `task_start`         | Compute task begins (all inputs live, output reservations made). |
| `task_end`           | Compute task finishes; outputs flip to `live`. |
| `release`            | `releases_after` ids freed from device. |
| `transfer_enqueue`   | A trigger fired and the transfer entered its stream's queue. |
| `transfer_start`     | A queued transfer began on its stream (destination allocated here). |
| `transfer_end`       | Transfer completed; for D→H the device source is freed. |
| `transfer_deferred`  | A prefetch fired while its host source was still being written by a D→H — it'll auto-enqueue when that D→H completes. |

`EventLog.dump(path)` writes the whole log as JSON for the dashboard.

## Common pitfalls

- **Units.** `size` is bytes, `runtime` is cycles, `bandwidth_*` is bytes
  per cycle. The simulator never converts. Keep all four in a single
  consistent system or your runtimes will be off by orders of magnitude.
- **Triggers fire at `task_end`, not `task_start`.** A prefetch you wrote
  in `task[i].prefetch_after` is only enqueued *after* task `i` ends, so it
  cannot help cover task `i`'s own input latency — it's there to cover
  task `i+k`'s.
- **Outputs can't reuse input ids.** Mutation in place is expressed by
  listing the id in `mutates_inputs`; never by reusing an input id as an
  output id.
- **Mutation is not a final-placement request.** A mutated object can be
  released after its final use if no later task needs it and it is not listed
  in `final_locations`. Add `final_locations[obj_id] = "host"` when the
  host must receive the latest bytes at chain end.
- **`device_capacity` over-commit raises.** If, after reserving a task's
  outputs, the device pool would exceed `device_capacity`, `run()` raises
  with a "over-committed device" error. That's a policy bug — your plan
  needs more aggressive offloads. The simulator does not silently spill.
- **Host has no stall.** If a task's `host`-located output won't fit at
  task-start time, `run()` raises. There's no D→H queue waiting room for
  outputs — you need to ensure host has space ahead of time.
- **Initial pool conflicts.** Duplicate `(id, location)` keys in
  `initial_memory` raise immediately.
- **Stranded queues at end.** If `run()` finishes with anything still in
  the H→D or D→H queues, it raises — that means a queued transfer's
  destination never had room. Almost always a budget mistake.

## A worked example (5-line snippet)

```python
from dataflow_sim.schema import Object, Task, OutputAlloc, TaskChain
from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.simulator import run

bare = TaskChain(
    initial_memory=[Object(id="x", size=4, location="host", type="weight")],
    tasks=[Task(id="t0", inputs=["x"], outputs=[OutputAlloc("y", 4, "device", "activation")], runtime=10)],
    device_capacity=16, host_capacity=64, bandwidth_h2d=2, bandwidth_d2h=2,
)
annotated = apply_belady_reactive_policy(bare)
events = run(annotated)
```

The Belady policy notices `x` lives on host and inserts a prefetch so `t0`
can run; with `device_capacity=16` and one 4-byte input + one 4-byte
output, nothing has to be offloaded.

## See also

- `docs/transformer-recipe.md` — full real example end-to-end.
- `docs/policy/README.md` — choosing among the built-in policies.
- `docs/problem.md` — the underlying memory-traffic problem statement.
