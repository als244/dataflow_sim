"""Shared helpers for the auto-policy planners (belady_reactive reactive Belady and roundtrip_planner
proactive round-trip planner).

Contains:
  * Pure structural helpers that read the bare chain: object sizes/types,
    last-use indices, ideal-start times, and per-object use sequences.
  * The trigger-kind alias and `_PendingTrigger` dataclass used by both
    planners' trigger emission.
  * Final-stage utilities used by both planners' entry points: annotation
    assembly into a `TaskChain`, final-placement writeback insertion, simulator
    verification with iterative refinement, and a best-effort makespan probe.
"""
from __future__ import annotations

import bisect
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

from dataflow_sim.core.schema import Object, Task, TaskChain, TransferTrigger

TriggerKind = Literal["release", "offload", "prefetch"]


@dataclass
class _PendingTrigger:
    kind: TriggerKind
    obj_id: str
    deadline: int  # by which time (simulator t) the trigger's effect must be in place


# ---------- structural helpers ----------

def _compute_ideal_starts(bare: TaskChain) -> dict[str, int]:
    """Ideal cumulative start times assuming zero stalls."""
    starts: dict[str, int] = {}
    t = 0
    for task in bare.tasks:
        starts[task.id] = t
        t += task.runtime
    return starts


def _object_sizes(bare: TaskChain) -> dict[str, int]:
    """All objects' sizes, from initial memory and from task outputs."""
    sizes: dict[str, int] = {}
    for obj in bare.initial_memory:
        sizes[obj.id] = obj.size
    for task in bare.tasks:
        for out in task.outputs:
            sizes.setdefault(out.id, out.size)
    return sizes


def _object_types(bare: TaskChain) -> dict[str, str]:
    types: dict[str, str] = {}
    for obj in bare.initial_memory:
        types[obj.id] = obj.type
    for task in bare.tasks:
        for out in task.outputs:
            types.setdefault(out.id, out.type)
    return types


def _compute_uses(bare: TaskChain, ideal_starts: dict[str, int]) -> dict[str, list[int]]:
    """For each object id, the sorted list of ideal start times at which it is
    referenced as an input by some task."""
    uses: dict[str, list[int]] = defaultdict(list)
    for task in bare.tasks:
        t = ideal_starts[task.id]
        for inp in task.inputs:
            uses[inp].append(t)
    return dict(uses)


def _next_use_after(uses: dict[str, list[int]], obj: str, t: int) -> float:
    """Earliest ideal start time >= t at which obj is used as input. infinity if never."""
    lst = uses.get(obj, [])
    idx = bisect.bisect_left(lst, t)
    if idx < len(lst):
        return lst[idx]
    return math.inf


def _last_use_task_idx(bare: TaskChain) -> dict[str, int]:
    """For each object id, the largest task index that uses it as input.
    Used for GC: an object is structurally dead AFTER task i iff
    `last_use_task_idx[obj] <= i`. This is time-independent so it's
    robust to actual-start drift from ideal-start.
    """
    last: dict[str, int] = {}
    for i, task in enumerate(bare.tasks):
        for inp in task.inputs:
            last[inp] = i
    return last


@dataclass
class _UseEvent:
    """A single use of an object as an input by some task."""
    task_idx: int
    ideal_start: int  # = ideal_starts[task.id]


def _object_uses_by_task_idx(
    bare: TaskChain, ideal_starts: dict[str, int]
) -> dict[str, list[_UseEvent]]:
    """For each object id, return the sorted list of `_UseEvent` records — one
    per (task_idx, ideal_start) at which the object is referenced as input.
    Multiple uses by the same task collapse to one event (a task either uses
    an input or it doesn't; duplicated entries in `inputs` would be a bug).
    """
    seen: dict[str, set[int]] = defaultdict(set)
    by_obj: dict[str, list[_UseEvent]] = defaultdict(list)
    for i, task in enumerate(bare.tasks):
        t = ideal_starts[task.id]
        for inp in task.inputs:
            if i in seen[inp]:
                continue
            seen[inp].add(i)
            by_obj[inp].append(_UseEvent(task_idx=i, ideal_start=t))
    # Each obj's list is naturally sorted by task_idx (and thus ideal_start).
    return dict(by_obj)


# ---------- annotation assembly ----------

def _apply_annotations(
    bare: TaskChain,
    initial_compute: set[str],
    annotations: dict[int, dict[TriggerKind, list[str]]],
    fast_memory_capacity: int | None,
) -> TaskChain:
    """Construct the final annotated TaskChain."""
    backing_objs = {o.id: o for o in bare.initial_memory if o.location == "backing"}

    new_initial: list[Object] = list(bare.initial_memory)
    for oid in initial_compute:
        src = backing_objs[oid]
        new_initial.append(
            Object(id=src.id, size=src.size, location="fast", type=src.type)
        )

    new_tasks: list[Task] = []
    for i, task in enumerate(bare.tasks):
        ann = annotations.get(i, {"release": [], "offload": [], "prefetch": []})
        # Deduplicate while preserving order
        rel = list(dict.fromkeys(ann["release"]))
        off = list(dict.fromkeys(ann["offload"]))
        pre = list(dict.fromkeys(ann["prefetch"]))
        new_tasks.append(
            Task(
                id=task.id,
                inputs=task.inputs,
                outputs=task.outputs,
                runtime=task.runtime,
                releases_after=rel,
                offload_after=[TransferTrigger(obj_id=o) for o in off],
                prefetch_after=[TransferTrigger(obj_id=o) for o in pre],
                mutates_inputs=task.mutates_inputs,
            )
        )

    return TaskChain(
        initial_memory=new_initial,
        tasks=new_tasks,
        bandwidth_from_slow=bare.bandwidth_from_slow,
        bandwidth_to_slow=bare.bandwidth_to_slow,
        final_locations=bare.final_locations,
        fast_memory_capacity=fast_memory_capacity,
        backing_memory_capacity=bare.backing_memory_capacity,
    )


def _add_gradient_writebacks(chain: TaskChain) -> TaskChain:
    """Guarantee requested final backing placements.

    Historical callers use this helper name, but the behavior is now keyed by
    the general ``TaskChain.final_locations`` contract. For every object whose
    final location is ``"backing"``, replace the terminal departure at its final
    use/production anchor with an offload so backing receives the latest bytes.
    Earlier evictions and prefetches are left intact because they may be
    necessary for capacity; triggers after the terminal anchor are removed.
    """
    protected_objs = {
        oid for oid, loc in chain.final_locations.items() if loc == "backing"
    }

    if not protected_objs:
        return chain
    last_anchor: dict[str, int] = {}
    for i, task in enumerate(chain.tasks):
        for out in task.outputs:
            if out.id in protected_objs:
                last_anchor[out.id] = i
        for inp in task.inputs:
            if inp in protected_objs:
                last_anchor[inp] = i

    writeback_at: dict[int, list[str]] = defaultdict(list)
    for obj_id, idx in last_anchor.items():
        writeback_at[idx].append(obj_id)

    new_tasks: list[Task] = []
    for i, task in enumerate(chain.tasks):
        terminal_oids = {
            oid for oid, anchor in last_anchor.items() if i >= anchor
        }
        rel = [r for r in task.releases_after if r not in terminal_oids]
        off = [tr for tr in task.offload_after if tr.obj_id not in terminal_oids]
        pre = [tr for tr in task.prefetch_after if tr.obj_id not in terminal_oids]
        if i in writeback_at:
            off = off + [TransferTrigger(obj_id=oid) for oid in writeback_at[i]]
        new_tasks.append(
            replace(task, releases_after=rel, offload_after=off, prefetch_after=pre)
        )
    return replace(chain, tasks=new_tasks)


# ---------- verification + iterative refinement ----------

_PREFETCH_CAP_ERR = re.compile(
    r"task '(\S+)' cannot prefetch '(\S+)': insufficient compute capacity"
)
_OFFLOAD_IN_FLIGHT_ERR = re.compile(
    r"input '(\S+)' is being offloaded \(state=(\S+)\)"
)


def _verify_and_refine(chain: TaskChain, *, max_iters: int = 20) -> TaskChain:
    """Verify the chain runs in the simulator; refine on common failure modes.

    Handled errors:
    - "cannot prefetch X: insufficient compute capacity" -> shift prefetch
      one task earlier (or remove it if it's an over-eager re-prefetch).
    - "input X is being offloaded" -> the planner over-evicted X. Remove
      the offload trigger for X; if X also has a paired prefetch shortly
      after, remove that too.

    Loops up to `max_iters` times before giving up."""
    from dataflow_sim.engine.simulator import run  # local import to avoid cycle
    for _ in range(max_iters):
        try:
            run(chain)
            return chain
        except ValueError as e:
            msg = str(e)
            m_pre = _PREFETCH_CAP_ERR.search(msg)
            m_off = _OFFLOAD_IN_FLIGHT_ERR.search(msg)
            if m_pre:
                task_id, obj_id = m_pre.group(1), m_pre.group(2)
                shifted = _shift_prefetch_earlier(chain, task_id, obj_id)
                if shifted is None:
                    raise
                chain = shifted
            elif m_off:
                obj_id = m_off.group(1)
                shifted = _remove_offload(chain, obj_id)
                if shifted is None:
                    raise
                chain = shifted
            else:
                raise
    run(chain)
    return chain


def _remove_offload(chain: TaskChain, obj_id: str) -> TaskChain | None:
    """Remove the most recent offload trigger for obj_id (and the paired
    prefetch, if one follows). Returns None if no offload exists."""
    new_tasks = list(chain.tasks)
    # Find and remove the latest offload of obj_id
    off_idx = -1
    for i in range(len(new_tasks) - 1, -1, -1):
        if any(tr.obj_id == obj_id for tr in new_tasks[i].offload_after):
            off_idx = i
            break
    if off_idx < 0:
        return None
    cur = new_tasks[off_idx]
    new_off = [tr for tr in cur.offload_after if tr.obj_id != obj_id]
    new_tasks[off_idx] = replace(cur, offload_after=new_off)
    # Also drop any subsequent prefetch of obj_id (would race with the
    # now-removed offload).
    for j in range(off_idx + 1, len(new_tasks)):
        t = new_tasks[j]
        if any(tr.obj_id == obj_id for tr in t.prefetch_after):
            new_pre = [tr for tr in t.prefetch_after if tr.obj_id != obj_id]
            new_tasks[j] = replace(t, prefetch_after=new_pre)
            break
    return replace(chain, tasks=new_tasks)


def _shift_prefetch_earlier(
    chain: TaskChain, task_id: str, obj_id: str
) -> TaskChain | None:
    """Move the prefetch of obj_id from task `task_id` to its predecessor.
    Returns None if no earlier task exists or no such trigger is found."""
    task_idx = next((i for i, t in enumerate(chain.tasks) if t.id == task_id), -1)
    if task_idx <= 0:
        return None

    new_tasks = list(chain.tasks)
    # Remove from current task
    cur = new_tasks[task_idx]
    new_pre = [p for p in cur.prefetch_after if p.obj_id != obj_id]
    if len(new_pre) == len(cur.prefetch_after):
        return None  # trigger not found
    new_tasks[task_idx] = replace(cur, prefetch_after=new_pre)
    # Add to predecessor
    prev = new_tasks[task_idx - 1]
    new_tasks[task_idx - 1] = replace(
        prev, prefetch_after=list(prev.prefetch_after) + [TransferTrigger(obj_id=obj_id)]
    )
    return replace(chain, tasks=new_tasks)


def _try_makespan(chain: TaskChain, refinement_iters: int) -> int | None:
    """Best-effort: return the verified-runnable chain's makespan, or None
    if it raises during verification/simulation."""
    from dataflow_sim.engine.simulator import run as _sim_run
    try:
        refined = _verify_and_refine(chain, max_iters=refinement_iters)
        log = _sim_run(refined)
        return max(iv.end for iv in log.task_intervals)
    except Exception:
        return None
