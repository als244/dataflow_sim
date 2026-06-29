from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

Time = float

Location = Literal["backing", "fast"]
ObjectType = Literal["weight", "activation", "gradient", "optimizer", "other"]
MemoryState = Literal[
    "live",              # readable / consumable
    "reserved",          # output of active compute task; memory accounted for but not yet written
    "pending_inbound",   # destination of a queued (not yet started) inbound transfer
    "inbound",           # destination of an in-progress inbound transfer
    "pending_outbound",  # source of a queued (not yet started) outbound transfer
    "outbound",          # source of an in-progress outbound transfer (will be freed on completion)
]


@dataclass(frozen=True)
class Object:
    id: str
    size: int
    location: Location = "fast"
    type: ObjectType = "other"


@dataclass(frozen=True)
class OutputAlloc:
    id: str
    size: int
    location: Location = "fast"
    type: ObjectType = "other"


@dataclass(frozen=True)
class TransferTrigger:
    """Per-task trigger that enqueues a transfer when the task ends.
    `runtime` overrides the bandwidth-derived default for this one transfer.
    """
    obj_id: str
    runtime: Time | None = None


@dataclass(frozen=True)
class Task:
    """A compute task.

    `inputs` are read-only by default — a task does not modify the byte
    contents of an input UNLESS that input's id also appears in
    `mutates_inputs`. Mutated inputs need a write-back (offload to backing)
    after the mutation so the backing copy reflects the update; planners
    must NOT release a mutated input via a bare release (which would
    discard the update) — they must offload it.

    Outputs always introduce fresh object ids — a task can't reuse an
    existing input id as an output id. Mutation is the general workload
    primitive (e.g., for transformer training, b_i mutates dW_i and head
    mutates dW_head; planners don't need to know about "gradients" by
    type-name, just about which inputs are listed in `mutates_inputs`).
    """
    id: str
    inputs: list[str]
    outputs: list[OutputAlloc]
    runtime: Time
    releases_after: list[str] = field(default_factory=list)
    offload_after: list[TransferTrigger] = field(default_factory=list)
    prefetch_after: list[TransferTrigger] = field(default_factory=list)
    mutates_inputs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskChain:
    initial_memory: list[Object]
    tasks: list[Task]
    # Optional terminal placement constraints, keyed by object id.
    # Example: {"W_0": "backing"} means the latest bytes for W_0 must be on
    # backing when the chain finishes. Objects omitted here are disposable after
    # their final use unless later tasks require them.
    final_locations: dict[str, Location] = field(default_factory=dict)
    # Optional per-location capacity ceilings. None = unlimited.
    fast_memory_capacity: int | None = None
    backing_memory_capacity: int | None = None
    # Bytes per time unit on each transfer stream. Per-trigger `runtime` overrides.
    # Required if any trigger relies on bandwidth-derived runtime.
    bandwidth_from_slow: int | None = None
    bandwidth_to_slow: int | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TaskChain":
        def _obj(o: dict[str, Any]) -> Object:
            return Object(
                id=o["id"],
                size=int(o["size"]),
                location=o.get("location", "fast"),
                type=o.get("type", "other"),
            )

        def _out(o: dict[str, Any]) -> OutputAlloc:
            return OutputAlloc(
                id=o["id"],
                size=int(o["size"]),
                location=o.get("location", "fast"),
                type=o.get("type", "other"),
            )

        def _trig(x: Any) -> TransferTrigger:
            if isinstance(x, str):
                return TransferTrigger(obj_id=x)
            return TransferTrigger(obj_id=x["id"], runtime=x.get("runtime"))

        return TaskChain(
            initial_memory=[_obj(o) for o in d.get("initial_memory", [])],
            tasks=[
                Task(
                    id=t["id"],
                    inputs=list(t.get("inputs", [])),
                    outputs=[_out(o) for o in t.get("outputs", [])],
                    runtime=float(t["runtime"]),
                    releases_after=list(t.get("releases_after", [])),
                    offload_after=[_trig(x) for x in t.get("offload_after", [])],
                    prefetch_after=[_trig(x) for x in t.get("prefetch_after", [])],
                    mutates_inputs=list(t.get("mutates_inputs", [])),
                )
                for t in d.get("tasks", [])
            ],
            final_locations=dict(d.get("final_locations", {})),
            fast_memory_capacity=d.get("fast_memory_capacity"),
            backing_memory_capacity=d.get("backing_memory_capacity"),
            bandwidth_from_slow=d.get("bandwidth_from_slow"),
            bandwidth_to_slow=d.get("bandwidth_to_slow"),
        )

    @staticmethod
    def load(path: str | Path) -> "TaskChain":
        with open(path) as f:
            return TaskChain.from_dict(json.load(f))


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    size: int
    location: Location
    type: ObjectType
    state: MemoryState
    # next time this object appears as an input in the remaining chain (None if never)
    next_ref_t: Time | None


@dataclass(frozen=True)
class ActiveTask:
    id: str
    ends_at: Time


@dataclass(frozen=True)
class Reference:
    obj_id: str
    ref_t: Time
    ref_task: str


@dataclass(frozen=True)
class Snapshot:
    memory: list[MemoryEntry]
    total_size: int
    active_task: ActiveTask | None
    reference_stream: list[Reference]


EventKind = Literal[
    "task_start",
    "task_end",
    "release",
    "transfer_enqueue",
    "transfer_start",
    "transfer_end",
    # A trigger fired but couldn't enqueue immediately because its source
    # is still being produced (e.g. a prefetch whose backing source is still
    # being offloaded). The transfer will be auto-enqueued when the source
    # becomes live; a subsequent `transfer_enqueue` event marks that moment.
    "transfer_deferred",
]

TransferDirection = Literal["from_slow", "to_slow"]


@dataclass(frozen=True)
class Event:
    t: Time
    kind: EventKind
    snapshot: Snapshot
    task_id: str | None = None
    object_ids: list[str] = field(default_factory=list)
    # Populated for transfer_* events
    transfer_obj: str | None = None
    transfer_direction: TransferDirection | None = None


@dataclass(frozen=True)
class TaskInterval:
    task_id: str
    start: Time
    end: Time
    track: str = "compute"


@dataclass(frozen=True)
class MemoryTracePoint:
    """Compact fast-memory sample for UI plotting.

    The keys in `fast_bytes_by_band` match the memory timeline bands:
    object types for live/reserved bytes plus inbound/outbound transfer states.
    It intentionally omits object ids and reference-stream data.
    """
    t: Time
    fast_bytes_by_band: dict[str, int]


@dataclass(frozen=True)
class EventLog:
    task_intervals: list[TaskInterval]
    events: list[Event]
    peak_fast_memory_bytes: int = 0
    memory_trace: list[MemoryTracePoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
