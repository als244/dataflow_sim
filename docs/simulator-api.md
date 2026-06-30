# `dataflow_sim` — API reference

The package's public surface is what `dataflow_sim/__init__.py` re-exports plus the workload builders in `dataflow_sim.workloads`, the schema dataclasses in `dataflow_sim.core.schema`, and the policy entry points in `dataflow_sim.policies`. Everything else is internal unless a doc here names it.

This doc covers the **simulator surface** (workload schema, run, validate, core IR). For policies see [docs/policy/](../docs/policy/) and [docs/workload-recipe.md](../docs/workload-recipe.md).

---

## Abstraction Contracts

The simulator stack has four contracts.

### Workload Schema Contract

`DataflowProgram v1` is hardware-free. It describes objects, ordered compute
tasks, reusable compute blocks, optional metrics, and final placement
constraints. It does not contain offload, prefetch, or release decisions.

Key fields:

- `objects[]`: initial objects with `id`, `size_bytes`, `initial_location`, and
  free-form `role`.
- `compute_blocks[]`: reusable block definitions with `key`, `name`,
  `category`, `subops`, and `metadata`.
- `tasks[]`: unique timeline instances with `id`, `label`, `group`,
  `compute_block_key`, `inputs`, `outputs`, and `mutates`. Inline `cost` is a
  convenience fallback and is normalized into a one-off compute block.
- `metrics`: optional primary throughput contract. Model-training builders set
  `primary_unit="tokens"` and `primary_count=<total tokens>`.
- `final_locations`: optional terminal placement requirements.

Cost models:

- `fixed`: measured `runtime_us`.
- `roofline`: `flops` and `memory_bytes` resolved with the selected
  `HardwareSpec`.
- `sum`: multiple fixed/roofline terms kept inside one simulator task while
  exposing sub-op breakdowns.

### Compiler Contract

`realize_dataflow_program(program, hw) -> Workload` takes a normalized
`DataflowProgram` plus `HardwareSpec` and returns:

- a bare `TaskChain`,
- normalized schema metadata,
- workload preview stats,
- compute block summaries,
- resolved per-sub-op timings,
- optional metrics metadata.

The bare chain contains concrete task runtimes and transfer bandwidths. It has
no planner annotations.

### Planner Contract

Planner entry points take `bare TaskChain + planner settings` and return an
annotated `TaskChain`. The planner is the only layer that decides
`releases_after`, `offload_after`, and `prefetch_after`.

### Simulator Contract

`run(annotated_chain) -> EventLog` executes an annotated chain and returns the
event log, task intervals, peak fast memory, and optional compact memory
trace. It does not change the plan.

### Summary Contract

`compute_workload_summary(workload, log) -> dict` in
`dataflow_sim.workloads.summary` turns a realized `Workload` plus simulator
`EventLog` into the same top-level KPI payload returned by the web API:
`makespan_us`, `tokens_per_second`, `primary_rate_per_second`,
`effective_tflops`, `hardware_tflops`, `peak_fast_memory_gb`, memory-stream
utilization, recompute percentage, and aggregate FLOP counts.

Use this when running the simulator from Python instead of recomputing UI
metrics locally:

```python
from dataflow_sim.engine.simulator import run
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.summary import compute_workload_summary

chain = apply_pressurefit_policy(workload.chain, fast_memory_capacity=cap_bytes)
log = run(chain, snapshots=False)
summary = compute_workload_summary(workload, log)
```

## Web API

The FastAPI app uses split payloads:

```json
{
  "workload": {
    "source": "schema",
    "schema": {
      "schema_version": "dataflow/v1",
      "name": "custom",
      "objects": [
        {"id": "x", "size_bytes": 1024, "initial_location": "fast", "role": "input"}
      ],
      "tasks": [
        {
          "id": "op0",
          "inputs": ["x"],
          "outputs": [{"id": "y", "size_bytes": 1024, "role": "output"}],
          "cost": {"kind": "fixed", "runtime_us": 10}
        }
      ]
    }
  },
  "hardware": {
    "preset": "custom",
    "peak_tflops_bf16": 100,
    "peak_tflops_fp8": 200,
    "peak_tflops_fp4": 400,
    "fast_memory_bw_gbs": 1000,
    "from_slow_bw_gbs": 100,
    "to_slow_bw_gbs": 100,
    "matmul_eff_bf16": 0.8,
    "matmul_eff_fp8": 0.8,
    "matmul_eff_fp4": 0.8,
    "attn_fwd_eff": 0.8,
    "attn_bwd_eff": 0.8,
    "mem_eff": 0.9
  },
  "planner": {
    "policy": "pressurefit",
    "window_size": 2,
    "fast_memory_capacity_gb": 1,
    "recompute": false
  }
}
```

Endpoints:

- `GET /api/presets`: workload and hardware presets.
- `POST /api/workloads/preview`: validates and realizes a workload against
  hardware without applying a planner.
- `POST /api/simulate`: realizes the workload, applies the planner, runs the
  simulator, and returns event/result data.

### `GET /api/presets`

Returns:

```json
{
  "workloads": {
    "llama3_8B": {"source": "model_training", "...": "..."},
    "qwen3_moe_30B-3B": {"source": "model_training", "...": "..."},
    "olmoe_7B-1B": {"source": "model_training", "...": "..."}
  },
  "hardware": {
    "H100": {
      "peak_tflops_bf16": 989,
      "peak_tflops_fp8": 1978,
      "peak_tflops_fp4": null,
      "matmul_eff_fp4": null,
      "...": "..."
    }
  }
}
```

### `POST /api/workloads/preview`

Request:

```json
{
  "workload": {"source": "schema", "schema": {"schema_version": "dataflow/v1"}},
  "hardware": {
    "preset": "custom",
    "peak_tflops_bf16": 100,
    "peak_tflops_fp8": 200,
    "peak_tflops_fp4": 400,
    "fast_memory_bw_gbs": 1000,
    "from_slow_bw_gbs": 100,
    "to_slow_bw_gbs": 100,
    "matmul_eff_bf16": 0.8,
    "matmul_eff_fp8": 0.8,
    "matmul_eff_fp4": 0.8,
    "attn_fwd_eff": 0.8,
    "attn_bwd_eff": 0.8,
    "mem_eff": 0.9
  }
}
```

Response:

```json
{
  "schema": {"schema_version": "dataflow/v1"},
  "preview": {
    "name": "custom",
    "task_count": 1,
    "object_count": 2,
    "compute_block_count": 1,
    "aggregate_task_runtime_us": 10,
    "metrics": null
  },
  "chain": {"initial_memory": [], "tasks": []},
  "breakdown": {"compute_blocks": []},
  "compute_blocks": [],
  "task_summaries": []
}
```

`chain` is the unannotated/bare chain. Every task's release/offload/prefetch
lists are empty.

### `POST /api/simulate`

Request:

```json
{
  "workload": {"source": "model_training", "...": "..."},
  "hardware": {"preset": "H100", "...": "..."},
  "planner": {
    "policy": "pressurefit",
    "window_size": 2,
    "fast_memory_capacity_gb": 40,
    "recompute": true
  }
}
```

Response:

```json
{
  "log": {"events": [], "task_intervals": []},
  "breakdown": {"compute_blocks": []},
  "summary": {
    "makespan_us": 0,
    "primary_unit": "tokens",
    "primary_rate_per_second": 0,
    "tokens_per_second": 0,
    "effective_tflops": 0,
    "hardware_tflops": 0,
    "peak_fast_memory_gb": 0
  },
  "chain": {"tasks": []},
  "workload_preview": {"task_count": 0},
  "compute_blocks": [],
  "policy_diagnostics": null
}
```

Summary metric behavior:

- Workloads with `metrics` get `primary_unit`, `primary_count`, and
  `primary_rate_per_second`.
- If `primary_unit == "tokens"`, `tokens_per_second` mirrors the primary rate.
- Generic workloads without metrics still report makespan, FLOPs, memory, and
  utilization, with no primary throughput metric.

## `dataflow_sim.engine.simulator`

### `run(chain: TaskChain, *, validate: bool = True, snapshots: bool = True, memory_trace: bool = False) -> EventLog`

Execute a fully-annotated `TaskChain` and return a complete event log.

**Arguments**
- `chain` (`TaskChain`) — annotated chain (all `releases_after` / `offload_after` / `prefetch_after` triggers populated; a policy normally builds this from a bare chain). Must have `fast_memory_capacity`, `backing_memory_capacity`, `bandwidth_from_slow`, `bandwidth_to_slow` set (or `None` for unbounded).
- `validate` (`bool`, keyword-only, default `True`) — if `True`, run [`validate_chain(chain)`](#validate_chainchain-taskchain---none) as a static prepass before stepping. Set `False` to bypass (useful when deliberately exercising runtime error paths or for raw-speed benchmarks).
- `snapshots` (`bool`, keyword-only, default `True`) — if `True`, return the full UI-grade event timeline with per-event snapshots and reference streams. If `False`, run a lightweight scoring simulation: runtime behavior is still validated and `task_intervals` / `peak_fast_memory_bytes` are populated, but `events` is empty and no snapshots/reference streams are materialized.
- `memory_trace` (`bool`, keyword-only, default `False`) — if `True`, also record compact fast-memory plot samples in `EventLog.memory_trace`. These samples contain aggregate compute bytes by band, not object-level snapshots or reference streams.

**Returns**
- `EventLog` — see [schema](#dataflow_simschema).
  - `EventLog.task_intervals` — start/end times for every compute task and every transfer (each gets its own `TaskInterval` with `track` ∈ {`compute`, `from_slow`, `to_slow`}).
  - `EventLog.events` — when `snapshots=True`, full timeline (`task_start`, `task_end`, `release`, `transfer_enqueue`, `transfer_start`, `transfer_end`, `transfer_deferred`), each with a `Snapshot` of pool state at that moment. Empty when `snapshots=False`.
  - `EventLog.peak_fast_memory_bytes` — maximum compute-pool bytes observed during the run. Available in both full and snapshot-free modes.
  - `EventLog.memory_trace` — compact aggregate fast-memory samples when `memory_trace=True`; otherwise empty.

**Raises**
- `ValidationError` — if `validate=True` and the chain fails any static rule.
- `ValueError` / `RuntimeError` — runtime contract violations the validator can't catch (transit bytes exceeding cap, deadlock, etc.). See [docs/policy/principles.md](../docs/policy/principles.md) §1.

**Mechanics (brief)**
With `snapshots=True`, the simulator runs two passes: pass 1 discovers actual stalled start times; pass 2 re-snapshots `reference_stream.next_t` from those actual starts so the UI sees realistic next-use timestamps. With `snapshots=False`, it runs one pass and skips event/snapshot construction. `memory_trace=True` is still one pass in snapshot-free mode and is meant for large UI runs that need a memory plot without object-level browsing. Makespan = `max(iv.end for iv in task_intervals)`.

**Example**
```python
from dataflow_sim.engine.simulator import run
from dataflow_sim.policies.belady_reactive import apply_belady_reactive_policy

annotated = apply_belady_reactive_policy(bare_chain)
log = run(annotated)
makespan = max(iv.end for iv in log.task_intervals)

# Faster scoring path for policies or sweeps that do not need event snapshots.
score_log = run(annotated, snapshots=False)
score = max(iv.end for iv in score_log.task_intervals)
peak = score_log.peak_fast_memory_bytes

# Large UI path: exact intervals + peak plus compact fast-memory plot data.
trace_log = run(annotated, snapshots=False, memory_trace=True)
points = trace_log.memory_trace
```

---

## `dataflow_sim.core.validate`

### `validate_chain(chain: TaskChain) -> None`

Static validation of a `TaskChain`. Raises `ValidationError` on the first violation found.

Catches every statically-computable invariant from [docs/policy/principles.md](../docs/policy/principles.md) §1, **before the simulator steps**:

- **ID resolution** — every `obj_id` in `inputs` / `releases_after` / `offload_after` / `prefetch_after` / `mutates_inputs` resolves to either `initial_memory` or a prior task's output; output ids are fresh; no `(id, location)` collision with existing pool entries.
- **Trigger validity** — prefetch only on objects statically not-on-compute; offload only on objects statically on-compute; no duplicate prefetches/offloads on the same anchor; no `prefetch + offload` of the same object on the same task.
- **Release, mutation, and final placement** — bare release is forbidden if the object has a later use and is dirty (mutated since last offload) or lacks a backing copy; objects listed in `final_locations` must end in the requested location with latest bytes.
- **Capacity** — `initial_memory` compute/backing sums ≤ cap; forced footprint at every task boundary (inputs + outputs, which must coexist) ≤ `fast_memory_capacity`.
- **Topology** — every input resolves to some producer; no self-cycles; duplicate input ids in the same task forbidden.

Does **not** check runtime properties: transit-byte residency timing, stream FIFO contention, stall amounts, makespan. "Bad-but-runnable" chains pass — they reveal themselves as idle time during simulation, not as validation failures.

**Arguments**
- `chain` (`TaskChain`) — the chain to validate (typically the output of a policy).

**Returns** — `None` (raises on failure).

**Raises** — `ValidationError` with a message of the form `<rule-token>: <task id> <description> <obj ids>`. The rule token is a kebab-case identifier (e.g. `release-of-dirty-with-later-use`) that test bench parametrization keys off of.

### `class ValidationError(ValueError)`

Subclass of `ValueError` (for backwards compat with callers catching `ValueError`). Use `except ValidationError:` to distinguish prepass failures from runtime sim raises.

---

## `dataflow_sim.core.schema`

Schema dataclasses. All `@dataclass(frozen=True)` — chains and event logs are immutable.

### `TaskChain`

The simulator's input. Bare TaskChain (no triggers) → policy → annotated TaskChain (triggers filled) → `run()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `initial_memory` | `list[Object]` | required | Objects present at t=0. `location="backing"` = backing-init (typical for weights); `location="fast"` = pre-placed in fast memory (set by policy). |
| `tasks` | `list[Task]` | required | Compute tasks in execution order. |
| `final_locations` | `dict[str, "backing" \| "fast"]` | `{}` | Optional terminal placement constraints. Omitted objects are disposable after their final use. |
| `fast_memory_capacity` | `int \| None` | `None` | Hard byte cap on fast-memory pool. `None` = unlimited. |
| `backing_memory_capacity` | `int \| None` | `None` | Hard byte cap on backing pool. `None` = unlimited. |
| `bandwidth_from_slow` | `int \| None` | `None` | Bytes per tick on from-slow stream. Required if any prefetch trigger relies on bandwidth-derived runtime. |
| `bandwidth_to_slow` | `int \| None` | `None` | Bytes per tick on to-slow stream. Required for offload triggers. |

Class methods: `TaskChain.from_dict(d)`, `TaskChain.load(path)` — round-trip with the JSON dump from `EventLog.dump()`.

### `Task`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Unique within the chain. |
| `inputs` | `list[str]` | required | Object ids read by this task. Must all be `live` in fast memory at task start. |
| `outputs` | `list[OutputAlloc]` | required | Fresh object ids produced by this task. |
| `runtime` | `int` | required | Deterministic compute time (ticks). |
| `releases_after` | `list[str]` | `[]` | Object ids the simulator releases (bare drop) at this task's end. Filled by policy. |
| `offload_after` | `list[TransferTrigger]` | `[]` | to-slow transfers enqueued at this task's end. Filled by policy. |
| `prefetch_after` | `list[TransferTrigger]` | `[]` | from-slow transfers enqueued at this task's end. Filled by policy. |
| `mutates_inputs` | `list[str]` | `[]` | Subset of `inputs` that this task modifies in place (read-modify-write). Policies must preserve updated bytes for later uses. Terminal backing/fast-memory requirements are expressed with `TaskChain.final_locations`. |

### `Object`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Unique. |
| `size` | `int` | required | Bytes. |
| `location` | `"backing" \| "fast"` | `"fast"` | Initial location at t=0. |
| `type` | `"weight" \| "activation" \| "gradient" \| "optimizer" \| "other"` | `"other"` | Semantic tag (for display + analytics; simulator doesn't branch on type). |

### `OutputAlloc`

Same fields as `Object`. Describes an output the simulator will create at task start. `id` must be fresh (not present in any prior `initial_memory` or task output).

### `TransferTrigger`

Per-task trigger that enqueues a transfer when the task ends.

| Field | Type | Default | Description |
|---|---|---|---|
| `obj_id` | `str` | required | The object to transfer. |
| `runtime` | `int \| None` | `None` | Override the bandwidth-derived runtime for this one transfer. |

### `EventLog`

`run()`'s return value.

| Field | Type | Description |
|---|---|---|
| `task_intervals` | `list[TaskInterval]` | Start/end per compute task and per transfer. `iv.track` ∈ {`compute`, `from_slow`, `to_slow`}. |
| `events` | `list[Event]` | Full timeline of events when `run(..., snapshots=True)`; empty when `snapshots=False`. Each event carries a `Snapshot` of pool state. |
| `peak_fast_memory_bytes` | `int` | Maximum fast-memory bytes observed during simulation. Populated even when `snapshots=False`. |
| `memory_trace` | `list[MemoryTracePoint]` | Compact fast-memory plot samples when `run(..., memory_trace=True)`; empty otherwise. |

Methods: `to_dict()` for JSON-safe export; `dump(path)` writes formatted JSON.

### `Event`

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Time of event. |
| `kind` | `EventKind` | One of: `task_start`, `task_end`, `release`, `transfer_enqueue`, `transfer_start`, `transfer_end`, `transfer_deferred`. |
| `snapshot` | `Snapshot` | Pool state at `t`. |
| `task_id` | `str \| None` | For task events. |
| `object_ids` | `list[str]` | For release/task events. |
| `transfer_obj` | `str \| None` | For transfer events. |
| `transfer_direction` | `"from_slow" \| "to_slow" \| None` | For transfer events. |

### `Snapshot`

State at one moment. `memory: list[MemoryEntry]`, `total_size: int`, `active_task: ActiveTask | None`, `reference_stream: list[Reference]`.

### `MemoryEntry`

A single object's row in a snapshot. Includes `state` ∈ {`live`, `reserved`, `pending_inbound`, `inbound`, `pending_outbound`, `outbound`} and `next_ref_t` (next time the object will appear as an input, or `None` if never).

### `MemoryTracePoint`

A compact aggregate sample for memory plots. Fields:

| Field | Type | Description |
|---|---|---|
| `t` | `int` | Time of sample. |
| `fast_bytes_by_band` | `dict[str, int]` | Compute bytes by display band. Keys are object types (`weight`, `activation`, `gradient`, `optimizer`, `other`) plus transfer-state bands (`inbound`, `outbound`, `pending_outbound`). |

---

## `dataflow_sim.policies`

### `get_all_policies() -> list[tuple[str, PolicyFn]]`

Canonical list of every selectable policy. Each entry is `(name, fn)`:

| Name | Stem | Description |
|---|---|---|
| `sliding_window` | hand-crafted | Fixed-width window over weights / gradients / activations |
| `belady_reactive` | auto | Shadow-simulator + farthest-next-use eviction |
| `roundtrip_planner` | auto | Constructive offload/prefetch round-trip packing |
| `max_reduce` | auto | Analytic top-down: start at MAX residency, evict under cap pressure |
| `min_grow` | auto | MIN-seeded over-shrink + beam search using the simulator as cost oracle |
| `pressurefit` | auto | Pressure-fit interval planning; fastest of four verified inbound schedules |

Each `fn` accepts a bare `TaskChain` (with `fast_memory_capacity` already set) and returns the annotated chain. Adapters in `get_all_policies()` paper over per-policy kwarg differences. Use this when iterating across all policies — adding a new policy means only updating this function.

Individual policy entry points (each module's `apply_<stem>_policy`) are also re-exported from `dataflow_sim.policies` for direct use with custom kwargs (e.g. `window_size`, `time_budget_s`).

PressureFit also exposes `plan_pressurefit_policy(...) -> (TaskChain, PressureFitDiagnostics)`.
Use `apply_pressurefit_policy(...)` when you only need the annotated chain; use
`plan_pressurefit_policy(...)` when you want per-schedule timings and which of
the four inbound schedules was selected.

---

## See also

- [docs/problem.md](../docs/problem.md) — the formal scheduling problem.
- [docs/workload-recipe.md](../docs/workload-recipe.md) — how to build your own bare chain.
- [workloads/MODEL_TRAINING.md](../src/dataflow_sim/workloads/MODEL_TRAINING.md) — model-training workload authoring and lowering.
- [docs/policy/principles.md](../docs/policy/principles.md) — invariants every chain must satisfy.
- [docs/policy/README.md](../docs/policy/README.md) — which policy to use when.
