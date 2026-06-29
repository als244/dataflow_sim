from __future__ import annotations

from collections.abc import Iterable

from dataflow_sim.core.schema import Reference, Task


def compute_reference_stream(
    remaining_tasks: Iterable[Task],
    task_start_times: dict[str, float],
) -> list[Reference]:
    """Forward-looking 'reference tape' over the remaining task chain.

    For every distinct object that the remaining tasks will *touch* — either by
    reading it as an input or by allocating it as an output — record the
    timestamp of the first task that touches it. Outputs count because
    reserving pool memory for them at task start is itself an access (and
    something the scheduler needs to plan for).

    Returned sorted by (ref_t, obj_id).
    """
    first_ref: dict[str, Reference] = {}
    for task in remaining_tasks:
        t = task_start_times[task.id]
        for obj_id in task.inputs:
            if obj_id not in first_ref:
                first_ref[obj_id] = Reference(obj_id=obj_id, ref_t=t, ref_task=task.id)
        for out in task.outputs:
            if out.id not in first_ref:
                first_ref[out.id] = Reference(obj_id=out.id, ref_t=t, ref_task=task.id)
    return sorted(first_ref.values(), key=lambda r: (r.ref_t, r.obj_id))


def next_ref_time(
    obj_id: str,
    remaining_tasks: Iterable[Task],
    task_start_times: dict[str, float],
) -> float | None:
    """First time `obj_id` is touched (input or output) in the remaining chain."""
    for task in remaining_tasks:
        if obj_id in task.inputs or any(o.id == obj_id for o in task.outputs):
            return task_start_times[task.id]
    return None
