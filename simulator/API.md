# `dataflow_sim` — API reference

The package's public surface is what `dataflow_sim/__init__.py` re-exports plus the schema dataclasses in `dataflow_sim.schema` and the policy entry points in `dataflow_sim.policy`. Everything else is internal.

This doc covers the **simulator surface** (run, validate, schema). For policies see [docs/policy/](../docs/policy/) and [docs/workload-recipe.md](../docs/workload-recipe.md).

---

## `dataflow_sim.simulator`

### `run(chain: TaskChain, *, validate: bool = True, snapshots: bool = True) -> EventLog`

Execute a fully-annotated `TaskChain` and return a complete event log.

**Arguments**
- `chain` (`TaskChain`) — annotated chain (all `releases_after` / `offload_after` / `prefetch_after` triggers populated; a policy normally builds this from a bare chain). Must have `device_capacity`, `host_capacity`, `bandwidth_h2d`, `bandwidth_d2h` set (or `None` for unbounded).
- `validate` (`bool`, keyword-only, default `True`) — if `True`, run [`validate_chain(chain)`](#validate_chainchain-taskchain---none) as a static prepass before stepping. Set `False` to bypass (useful when deliberately exercising runtime error paths or for raw-speed benchmarks).
- `snapshots` (`bool`, keyword-only, default `True`) — if `True`, return the full UI-grade event timeline with per-event snapshots and reference streams. If `False`, run a lightweight scoring simulation: runtime behavior is still validated and `task_intervals` / `peak_device_bytes` are populated, but `events` is empty and no snapshots/reference streams are materialized.

**Returns**
- `EventLog` — see [schema](#dataflow_simschema).
  - `EventLog.task_intervals` — start/end times for every compute task and every transfer (each gets its own `TaskInterval` with `track` ∈ {`compute`, `h2d`, `d2h`}).
  - `EventLog.events` — when `snapshots=True`, full timeline (`task_start`, `task_end`, `release`, `transfer_enqueue`, `transfer_start`, `transfer_end`, `transfer_deferred`), each with a `Snapshot` of pool state at that moment. Empty when `snapshots=False`.
  - `EventLog.peak_device_bytes` — maximum device-pool bytes observed during the run. Available in both full and snapshot-free modes.

**Raises**
- `ValidationError` — if `validate=True` and the chain fails any static rule.
- `ValueError` / `RuntimeError` — runtime contract violations the validator can't catch (transit bytes exceeding cap, deadlock, etc.). See [docs/policy/principles.md](../docs/policy/principles.md) §1.

**Mechanics (brief)**
With `snapshots=True`, the simulator runs two passes: pass 1 discovers actual stalled start times; pass 2 re-snapshots `reference_stream.next_t` from those actual starts so the UI sees realistic next-use timestamps. With `snapshots=False`, it runs one pass and skips event/snapshot construction. Makespan = `max(iv.end for iv in task_intervals)`.

**Example**
```python
from dataflow_sim.simulator import run
from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy

annotated = apply_belady_reactive_policy(bare_chain)
log = run(annotated)
makespan = max(iv.end for iv in log.task_intervals)

# Faster scoring path for policies or sweeps that do not need event snapshots.
score_log = run(annotated, snapshots=False)
score = max(iv.end for iv in score_log.task_intervals)
peak = score_log.peak_device_bytes
```

---

## `dataflow_sim.validate`

### `validate_chain(chain: TaskChain) -> None`

Static validation of a `TaskChain`. Raises `ValidationError` on the first violation found.

Catches every statically-computable invariant from [docs/policy/principles.md](../docs/policy/principles.md) §1, **before the simulator steps**:

- **ID resolution** — every `obj_id` in `inputs` / `releases_after` / `offload_after` / `prefetch_after` / `mutates_inputs` resolves to either `initial_memory` or a prior task's output; output ids are fresh; no `(id, location)` collision with existing pool entries.
- **Trigger validity** — prefetch only on objects statically not-on-device; offload only on objects statically on-device; no duplicate prefetches/offloads on the same anchor; no `prefetch + offload` of the same object on the same task.
- **Release, mutation, and final placement** — bare release is forbidden if the object has a later use and is dirty (mutated since last offload) or lacks a host copy; objects listed in `final_locations` must end in the requested location with latest bytes.
- **Capacity** — `initial_memory` device/host sums ≤ cap; forced footprint at every task boundary (inputs + outputs, which must coexist) ≤ `device_capacity`.
- **Topology** — every input resolves to some producer; no self-cycles; duplicate input ids in the same task forbidden.

Does **not** check runtime properties: transit-byte residency timing, stream FIFO contention, stall amounts, makespan. "Bad-but-runnable" chains pass — they reveal themselves as idle time during simulation, not as validation failures.

**Arguments**
- `chain` (`TaskChain`) — the chain to validate (typically the output of a policy).

**Returns** — `None` (raises on failure).

**Raises** — `ValidationError` with a message of the form `<rule-token>: <task id> <description> <obj ids>`. The rule token is a kebab-case identifier (e.g. `release-of-dirty-with-later-use`) that test bench parametrization keys off of.

### `class ValidationError(ValueError)`

Subclass of `ValueError` (for backwards compat with callers catching `ValueError`). Use `except ValidationError:` to distinguish prepass failures from runtime sim raises.

---

## `dataflow_sim.schema`

Schema dataclasses. All `@dataclass(frozen=True)` — chains and event logs are immutable.

### `TaskChain`

The simulator's input. Bare TaskChain (no triggers) → policy → annotated TaskChain (triggers filled) → `run()`.

| Field | Type | Default | Description |
|---|---|---|---|
| `initial_memory` | `list[Object]` | required | Objects present at t=0. `location="host"` = host-init (typical for weights); `location="device"` = pre-placed on device (set by policy). |
| `tasks` | `list[Task]` | required | Compute tasks in execution order. |
| `final_locations` | `dict[str, "host" \| "device"]` | `{}` | Optional terminal placement constraints. Omitted objects are disposable after their final use. |
| `device_capacity` | `int \| None` | `None` | Hard byte cap on device pool. `None` = unlimited. |
| `host_capacity` | `int \| None` | `None` | Hard byte cap on host pool. `None` = unlimited. |
| `bandwidth_h2d` | `int \| None` | `None` | Bytes per tick on H2D stream. Required if any prefetch trigger relies on bandwidth-derived runtime. |
| `bandwidth_d2h` | `int \| None` | `None` | Bytes per tick on D2H stream. Required for offload triggers. |

Class methods: `TaskChain.from_dict(d)`, `TaskChain.load(path)` — round-trip with the JSON dump from `EventLog.dump()`.

### `Task`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Unique within the chain. |
| `inputs` | `list[str]` | required | Object ids read by this task. Must all be `live` on device at task start. |
| `outputs` | `list[OutputAlloc]` | required | Fresh object ids produced by this task. |
| `runtime` | `int` | required | Deterministic compute time (ticks). |
| `releases_after` | `list[str]` | `[]` | Object ids the simulator releases (bare drop) at this task's end. Filled by policy. |
| `offload_after` | `list[TransferTrigger]` | `[]` | D2H transfers enqueued at this task's end. Filled by policy. |
| `prefetch_after` | `list[TransferTrigger]` | `[]` | H2D transfers enqueued at this task's end. Filled by policy. |
| `mutates_inputs` | `list[str]` | `[]` | Subset of `inputs` that this task modifies in place (read-modify-write). Policies must preserve updated bytes for later uses. Terminal host/device requirements are expressed with `TaskChain.final_locations`. |

### `Object`

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Unique. |
| `size` | `int` | required | Bytes. |
| `location` | `"host" \| "device"` | `"device"` | Initial location at t=0. |
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
| `task_intervals` | `list[TaskInterval]` | Start/end per compute task and per transfer. `iv.track` ∈ {`compute`, `h2d`, `d2h`}. |
| `events` | `list[Event]` | Full timeline of events when `run(..., snapshots=True)`; empty when `snapshots=False`. Each event carries a `Snapshot` of pool state. |
| `peak_device_bytes` | `int` | Maximum device-pool bytes observed during simulation. Populated even when `snapshots=False`. |

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
| `transfer_direction` | `"h2d" \| "d2h" \| None` | For transfer events. |

### `Snapshot`

State at one moment. `memory: list[MemoryEntry]`, `total_size: int`, `active_task: ActiveTask | None`, `reference_stream: list[Reference]`.

### `MemoryEntry`

A single object's row in a snapshot. Includes `state` ∈ {`live`, `reserved`, `pending_inbound`, `inbound`, `pending_outbound`, `outbound`} and `next_ref_t` (next time the object will appear as an input, or `None` if never).

---

## `dataflow_sim.policy`

### `get_all_policies() -> list[tuple[str, PolicyFn]]`

Canonical list of every selectable policy. Each entry is `(name, fn)`:

| Name | Stem | Description |
|---|---|---|
| `sliding_window` | hand-crafted | Fixed-width window over weights / gradients / activations |
| `belady_reactive` | auto | Shadow-simulator + farthest-next-use eviction |
| `roundtrip_planner` | auto | Constructive offload/prefetch round-trip packing |
| `max_reduce` | auto | Analytic top-down: start at MAX residency, evict under cap pressure |
| `min_grow` | auto | MIN-seeded over-shrink + beam search using the simulator as cost oracle |
| `pressurefit` | auto | Pressure-fit interval planning with bounded candidate specs |

Each `fn` accepts a bare `TaskChain` (with `device_capacity` already set) and returns the annotated chain. Adapters in `get_all_policies()` paper over per-policy kwarg differences. Use this when iterating across all policies — adding a new policy means only updating this function.

Individual policy entry points (each module's `apply_<stem>_policy`) are also re-exported from `dataflow_sim.policy` for direct use with custom kwargs (e.g. `window_size`, `time_budget_s`).

---

## See also

- [docs/problem.md](../docs/problem.md) — the formal scheduling problem.
- [docs/workload-recipe.md](../docs/workload-recipe.md) — how to build your own bare chain.
- [docs/transformer-recipe.md](../docs/transformer-recipe.md) — how the example app maps transformer training onto the API.
- [docs/policy/principles.md](../docs/policy/principles.md) — invariants every chain must satisfy.
- [docs/policy/README.md](../docs/policy/README.md) — which policy to use when.
