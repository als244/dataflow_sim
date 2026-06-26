"""ShadowSimulator: state-machine mirror of `dataflow_sim.engine.simulator` used by the
auto-policy to make timing-aware planning decisions.

Mirrors the simulator's state semantics:
- Pool keyed by (obj_id, location), each entry has a state.
- Two FIFO transfer streams (from_slow, to_slow) with per-direction bandwidth.
- Compute lane (serial).

But operates in "decide-and-record" mode: instead of executing triggers it
finds in the task chain, the shadow exposes query primitives (predicted state
at time t, predicted compute usage at time t, etc.) and mutators (issue release
/ offload / prefetch at a given task boundary). Each mutator updates the
shadow state immediately so subsequent queries reflect the consequence.

The planner is responsible for walking the chain in order, using the shadow
to make decisions before each task. At the end, `annotations` holds the per-
task-boundary trigger lists that get applied to the bare chain.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from dataflow_sim.core.schema import Location, MemoryState, TaskChain

INF = math.inf


@dataclass
class _Entry:
    """A shadow pool entry. The state evolves over time as scheduled transfers
    complete; `state_now` is the snapshot at `shadow.current_time`."""
    obj_id: str
    location: Location
    size: int
    state: MemoryState
    type: str
    # When this entry first appeared in this location (for first-availability).
    appeared_at: int = 0
    # Task index that produces this entry's content. Used by the planner to
    # reject trigger placement at a past boundary k < producer_task_idx.
    # -1 for initial-pool entries (available from boundary 0 onward).
    producer_task_idx: int = -1


@dataclass
class _Transfer:
    """A scheduled transfer on one of the streams."""
    obj_id: str
    direction: Literal["from_slow", "to_slow"]
    src_size: int   # bytes (drives transfer time)
    runtime: int    # tau = ceil(src_size / bandwidth)
    enqueue_at: int # absolute time, == end-of-the-task-this-trigger-fires-on
    start_at: int   # absolute time, the transfer actually starts on the stream
    end_at: int     # = start_at + runtime


class ShadowSimulator:
    """A state-machine planner that mirrors the real simulator's semantics.

    Usage pattern (driven by the auto-policy):

        shadow = ShadowSimulator(bare_chain)
        for i, task in enumerate(bare_chain.tasks):
            # Inspect / decide
            target_start = shadow.earliest_task_start(task, ideal_starts[task.id])
            # Issue prefetches / evictions as needed (updates shadow state)
            # ...
            # Run the task in shadow time:
            shadow.advance_to(target_start)
            shadow.run_task(i, task, start_t=target_start)

        annotated = shadow.to_annotated_chain(fast_memory_capacity, backing_memory_capacity)
    """

    def __init__(self, bare: TaskChain) -> None:
        self.chain = bare
        self.bw_from_slow = bare.bandwidth_from_slow
        self.bw_to_slow = bare.bandwidth_to_slow
        self.fast_memory_capacity = bare.fast_memory_capacity
        self.backing_memory_capacity = bare.backing_memory_capacity

        # Object metadata (collected from initial memory + outputs)
        self.sizes: dict[str, int] = {}
        self.types: dict[str, str] = {}
        for obj in bare.initial_memory:
            self.sizes[obj.id] = obj.size
            self.types[obj.id] = obj.type
        for task in bare.tasks:
            for out in task.outputs:
                self.sizes.setdefault(out.id, out.size)
                self.types.setdefault(out.id, out.type)

        # Pool (keyed by (obj_id, location))
        self.pool: dict[tuple[str, Location], _Entry] = {}
        for obj in bare.initial_memory:
            key = (obj.id, obj.location)
            self.pool[key] = _Entry(
                obj_id=obj.id, location=obj.location, size=obj.size,
                state="live", type=obj.type, appeared_at=0,
            )

        # Stream schedules — list of pending+in-flight transfers, sorted by enqueue_at (FIFO)
        self.sched_from_slow: list[_Transfer] = []
        self.sched_to_slow: list[_Transfer] = []

        # Time bookkeeping
        self.compute_busy_until = 0
        self.current_time = 0

        # Annotations to be applied to the bare chain
        self.annotations: dict[int, dict[str, list[str]]] = defaultdict(
            lambda: {"release": [], "offload": [], "prefetch": []}
        )

        # Augmented initial-pool decisions (objects to add to fast memory initially)
        self.added_initial_compute: set[str] = set()

        # Per-boundary compute-pool snapshots. boundary_pool_size[k] = predicted
        # compute-pool size in bytes immediately AFTER task k's end-of-task
        # triggers have all fired (release, offload, prefetch). Only valid for
        # boundaries 0..last_snapshotted_iter (those that run_task has snapped).
        # Later iters' retroactive trigger placements at boundary q' <= last_snap
        # adjust bps[q'..last_snap]; future indices stay garbage until snapped.
        n = len(bare.tasks)
        initial_dev = sum(
            obj.size for obj in bare.initial_memory if obj.location == "fast"
        )
        self.boundary_pool_size: list[int] = [initial_dev] * max(n, 1)
        self.last_snapshotted_iter: int = -1
        # Actual boundary end times (= actual_start[k] + runtime[k]), set by
        # run_task. Used in issue_offload's bps adjustment so completion-vs-
        # boundary comparisons reflect real timing instead of ideal.
        self.actual_boundary_end: list[int] = [0] * max(n, 1)

    # ---------- query primitives ----------

    def task_end(self, task_idx: int, ideal_starts: dict[str, int]) -> int:
        """End time of task task_idx given ideal cumulative starts (input)."""
        task = self.chain.tasks[task_idx]
        return ideal_starts[task.id] + task.runtime

    def current_compute_usage(self) -> int:
        return sum(e.size for (_, loc), e in self.pool.items() if loc == "fast")

    def current_backing_usage(self) -> int:
        return sum(e.size for (_, loc), e in self.pool.items() if loc == "backing")

    def predicted_compute_usage_at(self, t: int) -> int:
        """Compute bytes used at time t, after processing all transfer
        completions with end_at <= t."""
        usage = self.current_compute_usage()
        for tx in self.sched_to_slow:
            if tx.start_at <= t < tx.end_at:
                continue  # outbound — still on compute until end_at
            if tx.end_at <= t:
                # Transfer completed; source compute entry removed
                # (But only count if source is currently in pool — it is, since
                # we don't remove until completion. The shadow.pool reflects
                # the *current* state; transfers in sched_to_slow are scheduled
                # but not yet completed. So usage minus completing transfers.)
                usage -= tx.src_size
        # Also account for from_slow transfers: their dest entries are already in
        # pool (state pending_inbound or inbound) and counted in current_compute_usage.
        # Completion doesn't change usage (just flips state to live).
        return usage

    def predicted_backing_usage_at(self, t: int) -> int:
        usage = self.current_backing_usage()
        # to_slow transfers: their dest backing entries are already in pool (pending_inbound)
        # — usage already includes them. Completion doesn't change usage.
        # from_slow transfers: don't add backing entries.
        return usage

    def predicted_object_ready_t(self, obj_id: str, location: Location) -> int:
        """Earliest time obj_id is `live` at `location`. INF if never."""
        entry = self.pool.get((obj_id, location))
        if entry is None:
            return INF
        if entry.state == "live":
            return entry.appeared_at
        # In some transit state — find the relevant completion
        relevant = self.sched_from_slow if location == "fast" else self.sched_to_slow
        for tx in relevant:
            if tx.obj_id == obj_id:
                # For from_slow: dest is compute, completion makes it live
                # For to_slow: dest is backing, completion makes it live
                # Both apply.
                if (location == "fast" and tx.direction == "from_slow") or \
                   (location == "backing" and tx.direction == "to_slow"):
                    return tx.end_at
        # No scheduled transfer; state is some non-live transient with no completion
        return INF

    def predicted_input_ready_t(self, obj_id: str) -> int:
        """Earliest time obj_id is live on compute. INF if it'll never get there
        without further policy action."""
        return self.predicted_object_ready_t(obj_id, "fast")

    def stream_busy_until(self, direction: Literal["from_slow", "to_slow"]) -> int:
        """End time of the last scheduled transfer on this stream (or current_time)."""
        sched = self.sched_from_slow if direction == "from_slow" else self.sched_to_slow
        if not sched:
            return self.current_time
        return max(tx.end_at for tx in sched)

    def is_dead(self, obj_id: str, remaining_uses: list[int]) -> bool:
        """True if obj_id has no future uses (no more entries in its use timeline)."""
        return len(remaining_uses) == 0

    # ---------- mutators ----------

    def add_to_initial_compute(self, obj_id: str) -> None:
        """Promote a backing-resident object to also be on compute from t=0."""
        if (obj_id, "fast") in self.pool:
            return
        backing_entry = self.pool.get((obj_id, "backing"))
        if backing_entry is None:
            raise ValueError(f"cannot add {obj_id!r} to initial compute: no backing entry")
        self.pool[(obj_id, "fast")] = _Entry(
            obj_id=obj_id, location="fast", size=backing_entry.size,
            state="live", type=backing_entry.type, appeared_at=0,
        )
        self.added_initial_compute.add(obj_id)

    def issue_release(self, obj_id: str, at_boundary_idx: int, boundary_end_t: int) -> None:
        """Record a release trigger; updates shadow pool immediately (the
        compute entry is removed at boundary_end_t)."""
        key = (obj_id, "fast")
        if key not in self.pool:
            return  # already gone
        size = self.pool[key].size
        # In the simulator, releases fire after task_end. We model this as
        # removing the entry at boundary_end_t.
        del self.pool[key]
        self.annotations[at_boundary_idx]["release"].append(obj_id)
        # Decrement bps for snapshotted boundaries at/after this one.
        end = min(self.last_snapshotted_iter + 1, len(self.boundary_pool_size))
        for k in range(at_boundary_idx, end):
            self.boundary_pool_size[k] -= size

    def issue_offload(self, obj_id: str, at_boundary_idx: int, boundary_end_t: int) -> int:
        """Record an offload trigger. Returns the scheduled completion time."""
        if self.bw_to_slow is None:
            raise ValueError("bandwidth_to_slow required for offload triggers")
        dev_entry = self.pool.get((obj_id, "fast"))
        if dev_entry is None or dev_entry.state != "live":
            raise ValueError(
                f"cannot offload {obj_id!r}: compute state is "
                f"{dev_entry.state if dev_entry else 'absent'}"
            )
        size = dev_entry.size
        tau = max(1, math.ceil(size / self.bw_to_slow))
        # Insert into to_slow schedule maintaining FIFO order by enqueue_at
        self._insert_transfer(self.sched_to_slow, _Transfer(
            obj_id=obj_id, direction="to_slow", src_size=size, runtime=tau,
            enqueue_at=boundary_end_t, start_at=0, end_at=0,  # recomputed below
        ))
        # Update source state: pending_outbound (occupies compute until transfer end)
        dev_entry.state = "pending_outbound"
        # Update destination on backing: overwrite mode if exists, else create
        backing_entry = self.pool.get((obj_id, "backing"))
        if backing_entry is None:
            self.pool[(obj_id, "backing")] = _Entry(
                obj_id=obj_id, location="backing", size=size,
                state="pending_inbound", type=dev_entry.type, appeared_at=boundary_end_t,
            )
        else:
            # Overwrite: existing backing entry becomes pending_inbound (hidden until completion)
            backing_entry.state = "pending_inbound"
        self.annotations[at_boundary_idx]["offload"].append(obj_id)
        # Find the just-inserted transfer's ideal-time completion (in shadow's
        # FIFO accounting). Real exec may complete later due to actual-vs-ideal
        # drift; we account for this conservatively by also subtracting a
        # `drift_at_offload` from actual boundary times in the comparison.
        completion_t = boundary_end_t + tau
        for tx in self.sched_to_slow:
            if tx.obj_id == obj_id and tx.enqueue_at == boundary_end_t:
                completion_t = tx.end_at
                break
        # Drift at the offload's enqueue boundary: how much real time has
        # already slipped past ideal. Add this to the predicted ideal
        # completion to estimate REAL completion.
        if at_boundary_idx <= self.last_snapshotted_iter:
            ideal_end_q = sum(t.runtime for t in self.chain.tasks[:at_boundary_idx + 1])
            drift = max(0, self.actual_boundary_end[at_boundary_idx] - ideal_end_q)
        else:
            drift = 0
        real_completion = completion_t + drift
        # For each snapshotted downstream boundary k, subtract if its ACTUAL
        # end time is past the real-completion estimate.
        end = min(self.last_snapshotted_iter + 1, len(self.boundary_pool_size))
        for k in range(at_boundary_idx + 1, end):
            if self.actual_boundary_end[k] >= real_completion:
                self.boundary_pool_size[k] -= size
        return completion_t

    def issue_prefetch(self, obj_id: str, at_boundary_idx: int, boundary_end_t: int) -> int:
        """Record a prefetch trigger. Returns the scheduled completion time."""
        if self.bw_from_slow is None:
            raise ValueError("bandwidth_from_slow required for prefetch triggers")
        backing_entry = self.pool.get((obj_id, "backing"))
        if backing_entry is None or backing_entry.state != "live":
            raise ValueError(
                f"cannot prefetch {obj_id!r}: backing state is "
                f"{backing_entry.state if backing_entry else 'absent'}"
            )
        if (obj_id, "fast") in self.pool:
            raise ValueError(f"cannot prefetch {obj_id!r}: compute copy already exists")
        size = backing_entry.size
        tau = max(1, math.ceil(size / self.bw_from_slow))
        self._insert_transfer(self.sched_from_slow, _Transfer(
            obj_id=obj_id, direction="from_slow", src_size=size, runtime=tau,
            enqueue_at=boundary_end_t, start_at=0, end_at=0,
        ))
        # Reserve compute entry as pending_inbound
        self.pool[(obj_id, "fast")] = _Entry(
            obj_id=obj_id, location="fast", size=size,
            state="pending_inbound", type=backing_entry.type, appeared_at=boundary_end_t,
        )
        self.annotations[at_boundary_idx]["prefetch"].append(obj_id)
        # Increment bps for snapshotted boundaries at/after this one
        end = min(self.last_snapshotted_iter + 1, len(self.boundary_pool_size))
        for k in range(at_boundary_idx, end):
            self.boundary_pool_size[k] += size
        for tx in self.sched_from_slow:
            if tx.obj_id == obj_id and tx.enqueue_at == boundary_end_t:
                return tx.end_at
        return boundary_end_t + tau

    def _insert_transfer(self, schedule: list[_Transfer], tx: _Transfer) -> None:
        """Insert tx into schedule maintaining FIFO order by enqueue_at; recompute
        start_at/end_at for all items downstream."""
        # Find insertion point
        idx = 0
        for i, existing in enumerate(schedule):
            if existing.enqueue_at > tx.enqueue_at:
                break
            idx = i + 1
        schedule.insert(idx, tx)
        # Recompute start_at/end_at: each transfer starts at max(its enqueue_at,
        # prev transfer's end_at)
        prev_end = 0
        for t in schedule:
            t.start_at = max(t.enqueue_at, prev_end)
            t.end_at = t.start_at + t.runtime
            prev_end = t.end_at

    def advance_to(self, target_t: float) -> None:
        """Process all scheduled transfer completions with end_at <= target_t.
        Updates pool state accordingly."""
        while True:
            next_h = self.sched_from_slow[0].end_at if self.sched_from_slow else INF
            next_d = self.sched_to_slow[0].end_at if self.sched_to_slow else INF
            next_t = min(next_h, next_d)
            if math.isinf(next_t) or next_t > target_t:
                break
            if next_h <= next_d:
                tx = self.sched_from_slow.pop(0)
                # from_slow completion: compute entry → live
                key = (tx.obj_id, "fast")
                if key in self.pool:
                    self.pool[key].state = "live"
            else:
                tx = self.sched_to_slow.pop(0)
                # to_slow completion: source compute entry removed; backing entry → live
                src_key = (tx.obj_id, "fast")
                if src_key in self.pool:
                    del self.pool[src_key]
                dst_key = (tx.obj_id, "backing")
                if dst_key in self.pool:
                    self.pool[dst_key].state = "live"
        self.current_time = target_t

    def run_task(self, task_idx: int, task, start_t: int) -> None:
        """Mark task as running: reserve outputs, advance time through it,
        mark outputs live."""
        # Reserve outputs (added to pool as `reserved`)
        for out in task.outputs:
            key = (out.id, out.location)
            self.pool[key] = _Entry(
                obj_id=out.id, location=out.location, size=out.size,
                state="reserved", type=out.type, appeared_at=start_t,
                producer_task_idx=task_idx,
            )
        end_t = start_t + task.runtime
        # Advance through compute (transfers may complete during this window)
        self.advance_to(end_t)
        # Outputs become live
        for out in task.outputs:
            entry = self.pool.get((out.id, out.location))
            if entry is not None:
                entry.state = "live"
        self.compute_busy_until = end_t
        self.current_time = end_t
        # Snapshot compute pool size at this boundary. Mark snapshotted so
        # subsequent trigger placements can adjust this bps slot.
        if task_idx < len(self.boundary_pool_size):
            self.boundary_pool_size[task_idx] = self.current_compute_usage()
            self.actual_boundary_end[task_idx] = end_t
            self.last_snapshotted_iter = max(self.last_snapshotted_iter, task_idx)

    # ---------- annotation export ----------

    def to_annotated_chain(self) -> TaskChain:
        """Build the final annotated TaskChain from the recorded annotations
        plus initial-compute augmentations."""
        from dataflow_sim.core.schema import Object, Task, TransferTrigger

        # Augmented initial memory
        backing_objs = {o.id: o for o in self.chain.initial_memory if o.location == "backing"}
        new_initial = list(self.chain.initial_memory)
        for oid in self.added_initial_compute:
            src = backing_objs[oid]
            new_initial.append(Object(
                id=src.id, size=src.size, location="fast", type=src.type,
            ))

        new_tasks: list[Task] = []
        for i, task in enumerate(self.chain.tasks):
            ann = self.annotations.get(i, {"release": [], "offload": [], "prefetch": []})
            rel = list(dict.fromkeys(ann["release"]))
            off = list(dict.fromkeys(ann["offload"]))
            pre = list(dict.fromkeys(ann["prefetch"]))
            new_tasks.append(Task(
                id=task.id, inputs=task.inputs, outputs=task.outputs,
                runtime=task.runtime,
                releases_after=rel,
                offload_after=[TransferTrigger(obj_id=o) for o in off],
                prefetch_after=[TransferTrigger(obj_id=o) for o in pre],
                mutates_inputs=task.mutates_inputs,
            ))

        return TaskChain(
            initial_memory=new_initial,
            tasks=new_tasks,
            bandwidth_from_slow=self.chain.bandwidth_from_slow,
            bandwidth_to_slow=self.chain.bandwidth_to_slow,
            final_locations=self.chain.final_locations,
            fast_memory_capacity=self.chain.fast_memory_capacity,
            backing_memory_capacity=self.chain.backing_memory_capacity,
        )
