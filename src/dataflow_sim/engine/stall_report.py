"""Post-process an EventLog into stall and backlog evidence.

The report answers, from the simulator's ground truth, where a plan's time
went: how much compute stalled, whether each stall waited on an inbound
transfer (and for which object) or on compute capacity, how busy each
transfer stream was, and when each stream had enqueued work waiting behind
it (backlog). Layered planners — e.g. recompute selection — consume this
instead of re-deriving pressure from any analytic model.

Works on snapshot-free logs: only `task_intervals` plus the annotated chain
are required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dataflow_sim.core.schema import EventLog, TaskChain


@dataclass(frozen=True)
class StallReport:
    makespan_us: float
    compute_busy_us: float
    stall_us: float                      # total compute idle between tasks
    input_wait_us: float                 # stalls ending exactly at an input's from_slow arrival
    capacity_wait_us: float              # stalls ending exactly at a to_slow completion
    other_wait_us: float                 # stalls with no attributable transfer event
    stall_by_object: dict[str, float]    # input-wait blame per object id
    stream_busy_us: dict[str, float]     # {"from_slow": ..., "to_slow": ...}
    backlog_us: dict[str, float]         # time each stream had waiting work
    backlog_windows: dict[str, list[tuple[float, float]]]
    transfer_backlog_overlap: dict[str, float] = field(default_factory=dict)
    # per object: its own transfer time spent inside its stream's backlog
    # windows (time during which removing this traffic would unblock others)


def _transfer_obj(task_id: str) -> str:
    # "from_slow:OBJ" or "from_slow:OBJ#2" -> "OBJ"
    obj = task_id.split(":", 1)[1]
    return obj.split("#", 1)[0]


def _merge_windows(points: list[tuple[float, int]]) -> list[tuple[float, float]]:
    """Sweep (+1 at enqueue, -1 at start) deltas into waiting>0 windows."""
    events = sorted(points)
    windows: list[tuple[float, float]] = []
    depth = 0
    open_t: float | None = None
    for t, delta in events:
        prev = depth
        depth += delta
        if prev == 0 and depth > 0:
            open_t = t
        elif prev > 0 and depth == 0 and open_t is not None:
            if t > open_t:
                windows.append((open_t, t))
            open_t = None
    return windows


def _overlap(a0: float, a1: float, windows: list[tuple[float, float]]) -> float:
    total = 0.0
    for w0, w1 in windows:
        lo, hi = max(a0, w0), min(a1, w1)
        if hi > lo:
            total += hi - lo
    return total


def build_stall_report(chain: TaskChain, log: EventLog) -> StallReport:
    compute = [iv for iv in log.task_intervals if iv.track == "compute"]
    transfers = {
        "from_slow": [iv for iv in log.task_intervals if iv.track == "from_slow"],
        "to_slow": [iv for iv in log.task_intervals if iv.track == "to_slow"],
    }
    makespan = max((iv.end for iv in log.task_intervals), default=0)
    compute_busy = sum(iv.end - iv.start for iv in compute)

    # Transfer arrival times by direction: (end_time -> object ids ending then)
    ends_at: dict[str, dict[float, set[str]]] = {"from_slow": {}, "to_slow": {}}
    for direction, ivs in transfers.items():
        for iv in ivs:
            ends_at[direction].setdefault(iv.end, set()).add(_transfer_obj(iv.task_id))

    tasks_by_id = {t.id: t for t in chain.tasks}

    stall_us = input_wait = capacity_wait = other_wait = 0.0
    stall_by_object: dict[str, float] = {}
    prev_end = 0.0
    for iv in compute:
        gap = iv.start - prev_end
        prev_end = iv.end
        if gap <= 0:
            continue
        stall_us += gap
        task = tasks_by_id.get(iv.task_id)
        arrived = ends_at["from_slow"].get(iv.start, set())
        blocking = sorted(arrived & set(task.inputs)) if task else []
        if blocking:
            input_wait += gap
            share = gap / len(blocking)
            for oid in blocking:
                stall_by_object[oid] = stall_by_object.get(oid, 0) + share
        elif iv.start in ends_at["to_slow"]:
            capacity_wait += gap
        else:
            other_wait += gap

    # Backlog: a transfer is waiting from its trigger task's compute end (or
    # for the first transfers, t=0 has no trigger -> use its own start) until
    # it begins on the stream. Triggers are matched to transfer instances in
    # order per (direction, object).
    trigger_times: dict[str, dict[str, list[float]]] = {"from_slow": {}, "to_slow": {}}
    compute_end_by_task = {iv.task_id: iv.end for iv in compute}
    for t in chain.tasks:
        t_end = compute_end_by_task.get(t.id)
        if t_end is None:
            continue
        for trig in t.prefetch_after:
            trigger_times["from_slow"].setdefault(trig.obj_id, []).append(t_end)
        for trig in t.offload_after:
            trigger_times["to_slow"].setdefault(trig.obj_id, []).append(t_end)

    backlog_windows: dict[str, list[tuple[float, float]]] = {}
    backlog_us: dict[str, float] = {}
    stream_busy: dict[str, float] = {}
    for direction, ivs in transfers.items():
        deltas: list[tuple[float, int]] = []
        seen: dict[str, int] = {}
        for iv in sorted(ivs, key=lambda x: x.start):
            obj = _transfer_obj(iv.task_id)
            idx = seen.get(obj, 0)
            seen[obj] = idx + 1
            fires = trigger_times[direction].get(obj, [])
            enqueue = fires[idx] if idx < len(fires) else iv.start
            enqueue = min(enqueue, iv.start)
            if enqueue < iv.start:
                deltas.append((enqueue, 1))
                deltas.append((iv.start, -1))
        windows = _merge_windows(deltas)
        backlog_windows[direction] = windows
        backlog_us[direction] = sum(b - a for a, b in windows)
        stream_busy[direction] = sum(iv.end - iv.start for iv in ivs)

    transfer_backlog_overlap: dict[str, float] = {}
    for direction, ivs in transfers.items():
        windows = backlog_windows[direction]
        if not windows:
            continue
        for iv in ivs:
            obj = _transfer_obj(iv.task_id)
            ov = _overlap(iv.start, iv.end, windows)
            if ov:
                transfer_backlog_overlap[obj] = (
                    transfer_backlog_overlap.get(obj, 0) + ov
                )

    return StallReport(
        makespan_us=makespan,
        compute_busy_us=compute_busy,
        stall_us=stall_us,
        input_wait_us=input_wait,
        capacity_wait_us=capacity_wait,
        other_wait_us=other_wait,
        stall_by_object=stall_by_object,
        stream_busy_us=stream_busy,
        backlog_us=backlog_us,
        backlog_windows=backlog_windows,
        transfer_backlog_overlap=transfer_backlog_overlap,
    )
