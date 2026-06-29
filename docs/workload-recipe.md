# Workload Recipe

Most users should author `DataflowProgram v1`, documented in
`src/dataflow_sim/workloads/README.md`. It is hardware-free, uploadable by the
webapp, and supports compute blocks, sub-op breakdowns, and optional throughput
metrics.

Use low-level `TaskChain` directly only when you are implementing a compiler, a
memory planner, simulator tests, or a benchmark that needs exact control over
release/offload/prefetch annotations.

## Layer Contracts

The stack is intentionally split:

| Layer | Input | Output | Responsibility |
|---|---|---|---|
| Workload schema | User/model authoring | `DataflowProgram` | Hardware-free ordered compute, objects, compute blocks, metrics. |
| Compiler | `DataflowProgram + HardwareSpec` | Bare `TaskChain + metadata` | Resolve costs, sizes, transfer bandwidths, and block summaries. |
| Planner | Bare `TaskChain + settings` | Annotated `TaskChain` | Decide releases, offloads, and prefetches under a fast-memory budget. |
| Simulator | Annotated `TaskChain` | `EventLog + summary inputs` | Execute the plan exactly and report intervals, stalls, transfers, and peak memory. |

Domain helpers, such as model-training builders, sit above the workload schema.
They should emit `DataflowProgram`, not planner annotations.

## Choosing The Right Interface

Use `DataflowProgram` when:

- the workload should be imported/exported through the webapp,
- hardware should be swappable after workload creation,
- you want compute block and sub-op summaries,
- you are writing model/layer-style workload builders.

Use `TaskChain` when:

- you are testing a planner or simulator rule,
- you need to hand-write an annotated plan,
- you are implementing a new schema compiler,
- you need intentionally invalid chains for validation/runtime tests.

The rest of this document describes the low-level `TaskChain` contract.

## The five primitives

Everything ingested into the simulator is built out of five frozen
dataclasses defined in `src/dataflow_sim/core/schema.py`.

### `Object` — a piece of data present at t=0

| field      | type                                                   | units | semantics |
|------------|--------------------------------------------------------|-------|-----------|
| `id`       | `str`                                                  | —     | Globally unique id. Used by tasks to reference it. |
| `size`     | `int`                                                  | bytes | Counted against `fast_memory_capacity` / `backing_memory_capacity`. |
| `location` | `"backing" \| "fast"`                                   | —     | Where the object initially lives. Default `"fast"`. |
| `type`     | `"weight" \| "activation" \| "gradient" \| "optimizer" \| "other"` | — | Semantic tag for the UI / policies. Default `"other"`. |

`Object` appears only in `TaskChain.initial_memory`.

### `Task` — a compute step

| field             | type                       | units  | semantics |
|-------------------|----------------------------|--------|-----------|
| `id`              | `str`                      | —      | Globally unique id. |
| `inputs`          | `list[str]`                | —      | Object ids that must be **live in fast memory** when the task starts. Read-only unless listed in `mutates_inputs`. |
| `outputs`         | `list[OutputAlloc]`        | —      | Fresh allocations produced by this task. Output ids must not collide with anything else in the pool. |
| `runtime`         | `int`                      | cycles | Wall-time for the compute step on the (single) compute stream. |
| `releases_after`  | `list[str]`                | —      | Object ids freed from fast memory immediately after `task_end`. Must be `live`. |
| `offload_after`   | `list[TransferTrigger]`    | —      | Per-task to-slow transfers enqueued at `task_end`. |
| `prefetch_after`  | `list[TransferTrigger]`    | —      | Per-task from-slow transfers enqueued at `task_end`. |
| `mutates_inputs`  | `list[str]`                | —      | Subset of `inputs` whose bytes this task modifies in place. Policies must preserve the updated bytes for any later compute use. Whether the final bytes must end on backing is controlled by `TaskChain.final_locations`. |

### `OutputAlloc` — what a task produces

| field      | type                  | units | semantics |
|------------|-----------------------|-------|-----------|
| `id`       | `str`                 | —     | Fresh id; can't collide with any existing pool entry at `(id, location)`. |
| `size`     | `int`                 | bytes | Reserved at task-start, becomes `live` at task-end. |
| `location` | `"backing" \| "fast"`  | —     | Default `"fast"`. Backing outputs have no stall mechanism — must fit at task-start or the simulator raises. |
| `type`     | `ObjectType`          | —     | Same set as `Object.type`. |

### `TransferTrigger` — a single transfer plan fired by a task

| field    | type        | units  | semantics |
|----------|-------------|--------|-----------|
| `obj_id` | `str`       | —      | Which object to move. |
| `runtime`| `int \| None` | cycles | Override the bandwidth-derived runtime. `None` uses `ceil(size / bandwidth_{from_slow,to_slow})`. |

The trigger's direction is implicit from which task list it lives in:
`offload_after` triggers go on the to-slow stream, `prefetch_after` on from-slow.

### `TaskChain` — the whole workload

| field             | type                  | units            | semantics |
|-------------------|-----------------------|------------------|-----------|
| `initial_memory`  | `list[Object]`        | —                | t=0 pool state. |
| `tasks`           | `list[Task]`          | —                | Executed strictly in order. |
| `final_locations` | `dict[str, "backing" \| "fast"]` | —       | Optional terminal placement constraints. Omitted objects are disposable after their final use. |
| `fast_memory_capacity` | `int \| None`         | bytes            | `None` = unlimited. Hard ceiling; over-commits raise. |
| `backing_memory_capacity`   | `int \| None`         | bytes            | Same. |
| `bandwidth_from_slow`   | `int \| None`         | bytes per cycle  | Default runtime for from-slow triggers. Required unless every from-slow trigger has an explicit `runtime`. |
| `bandwidth_to_slow`   | `int \| None`         | bytes per cycle  | Same for to-slow. |

`TaskChain.from_dict` and `TaskChain.load(path)` parse a JSON form with the
same field names.

## Bare vs annotated chain

A **bare chain** is just topology + sizes + runtimes: `initial_memory`,
`tasks` with `inputs`/`outputs`/`runtime`, and the capacity / bandwidth
constants. It says *what the workload does* but is silent on memory
management — every `releases_after`, `offload_after`, `prefetch_after` list
is empty, and the only data on the compute at t=0 is whatever the workload
genuinely starts with there.

An **annotated chain** is a bare chain plus a memory plan: object
placements may have been moved to backing in `initial_memory`, and every task
carries the `releases_after` / `offload_after` / `prefetch_after` triggers
that make the workload actually fit under `fast_memory_capacity`.

Policies in `dataflow_sim.policies.*` are functions `bare → annotated`. You
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
   `{"W_0": "backing"}`. Leave objects out when they are disposable after their
   final use.
4. Pick `fast_memory_capacity`, `backing_memory_capacity`, `bandwidth_from_slow`,
   `bandwidth_to_slow`. Use units that are internally consistent — the
   simulator never multiplies by a wall-clock conversion.
5. Construct the `TaskChain`. That's it — you can already pass it to a
   policy.

Tiny worked example:

```python
from dataflow_sim.core.schema import Object, Task, OutputAlloc, TaskChain

bare = TaskChain(
    initial_memory=[
        Object(id="W", size=8, location="fast", type="weight"),
    ],
    tasks=[
        Task(id="fwd", inputs=["W"],
             outputs=[OutputAlloc("act", 4, "fast", "activation")],
             runtime=10),
        Task(id="bwd", inputs=["W", "act"],
             outputs=[OutputAlloc("grad", 8, "fast", "gradient")],
             runtime=20),
    ],
    fast_memory_capacity=24, backing_memory_capacity=64,
    bandwidth_from_slow=2, bandwidth_to_slow=2,
)
```

This runs as-is (everything fits under `fast_memory_capacity=24`). The
interesting case is when it doesn't — then you need a policy.

## Picking a policy

| Policy                          | Idea                                                                                   |
|---------------------------------|----------------------------------------------------------------------------------------|
| `belady_reactive`               | Reactive eviction by farthest next-use, with prefetches scheduled to hide latency.     |
| `min_grow`                      | Grow the working set as slowly as possible; only offload when capacity demands.        |
| `max_reduce`                    | Eagerly offload as soon as inputs are consumed.                                        |
| `sliding_window`                | Fixed-size sliding window over the reference stream.                                   |
| `roundtrip_planner`             | Plans full offload+prefetch round-trips against a shadow simulation.                   |
| `pressurefit`                  | Fast pressure-fit interval planning with deadline-aware from-slow scheduling.|

Each is `apply_<name>_policy(bare, **kwargs) -> TaskChain`. See
`docs/policy/README.md` for the trade-offs and per-policy knobs.

## Running the simulator

```python
from dataflow_sim.engine.simulator import run

events = run(annotated_chain)   # -> EventLog
```

`run()` normally does a two-pass simulation: the first pass discovers each
task's *actual* scheduled start time (accounting for transfer stalls); the
second re-emits snapshots with reference-stream timestamps based on those
actual starts. The return value is an `EventLog` with:

- `task_intervals: list[TaskInterval]` — one per compute task and one per
  fired transfer, with `track` in `{"compute", "from_slow", "to_slow"}`. This is
  what the timeline UI renders.
- `events: list[Event]` — every state transition, each carrying a
  `Snapshot` of the memory pool, the active task, and the remaining
  reference stream.
- `peak_fast_memory_bytes: int` — maximum compute-pool bytes observed during
  simulation.
- `memory_trace: list[MemoryTracePoint]` — optional compact fast-memory plot
  samples when `run(..., memory_trace=True)`.

For policy scoring or sweeps that only need makespan/peak and runtime
validation, use:

```python
score_log = run(annotated_chain, snapshots=False)
```

This returns the same `task_intervals` and `peak_fast_memory_bytes`, but leaves
`events` empty and skips snapshot/reference-stream construction.

For large UI-style runs that still need a fast-memory plot, use:

```python
trace_log = run(annotated_chain, snapshots=False, memory_trace=True)
```

This keeps `events` empty but fills `memory_trace` with aggregate compute bytes
by band, without object-level memory contents or reference streams.

Key `Event.kind` values:

| kind                 | when |
|----------------------|------|
| `task_start`         | Compute task begins (all inputs live, output reservations made). |
| `task_end`           | Compute task finishes; outputs flip to `live`. |
| `release`            | `releases_after` ids freed from fast memory. |
| `transfer_enqueue`   | A trigger fired and the transfer entered its stream's queue. |
| `transfer_start`     | A queued transfer began on its stream (destination allocated here). |
| `transfer_end`       | Transfer completed; for to-slow the compute source is freed. |
| `transfer_deferred`  | A prefetch fired while its backing source was still being written by a to-slow — it'll auto-enqueue when that to-slow completes. |

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
  in `final_locations`. Add `final_locations[obj_id] = "backing"` when the
  backing must receive the latest bytes at chain end.
- **`fast_memory_capacity` over-commit raises.** If, after reserving a task's
  outputs, the compute pool would exceed `fast_memory_capacity`, `run()` raises
  with a "over-committed compute" error. That's a policy bug — your plan
  needs more aggressive offloads. The simulator does not silently spill.
- **Backing has no stall.** If a task's `backing`-located output won't fit at
  task-start time, `run()` raises. There's no to-slow queue waiting room for
  outputs — you need to ensure backing has space ahead of time.
- **Initial pool conflicts.** Duplicate `(id, location)` keys in
  `initial_memory` raise immediately.
- **Stranded queues at end.** If `run()` finishes with anything still in
  the from-slow or to-slow queues, it raises — that means a queued transfer's
  destination never had room. Almost always a budget mistake.

## A worked example (5-line snippet)

```python
from dataflow_sim.core.schema import Object, Task, OutputAlloc, TaskChain
from dataflow_sim.policies.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.engine.simulator import run

bare = TaskChain(
    initial_memory=[Object(id="x", size=4, location="backing", type="weight")],
    tasks=[Task(id="t0", inputs=["x"], outputs=[OutputAlloc("y", 4, "fast", "activation")], runtime=10)],
    fast_memory_capacity=16, backing_memory_capacity=64, bandwidth_from_slow=2, bandwidth_to_slow=2,
)
annotated = apply_belady_reactive_policy(bare)
events = run(annotated)
```

The Belady policy notices `x` lives on backing and inserts a prefetch so `t0`
can run; with `fast_memory_capacity=16` and one 4-byte input + one 4-byte
output, nothing has to be offloaded.

## See also

- `src/dataflow_sim/workloads/MODEL_TRAINING.md` — model-training authoring end-to-end.
- `docs/policy/README.md` — choosing among the built-in policies.
- `docs/problem.md` — the underlying memory-traffic problem statement.
