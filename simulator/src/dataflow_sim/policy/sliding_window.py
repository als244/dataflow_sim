"""Hand-crafted sliding-window weight/grad-buffer policy for training chains.

Given a bare training chain (from `build_bare_training_chain`), annotate it
with the sliding-window trigger pattern documented in
`core/workloads/training.py` (pre-Step-1 docstring) and reproduced in
RESEARCH.md §1. Workload-specific: keys off the `f_i`, `b_i`, `W_i`, `dW_i`,
`A_i` id conventions of the training builder.

Pattern (per layer index i, w = window_size, L inferred from the chain):

    after f_i, i ≤ L-3:                offload A_i
    after f_i, i + w ≤ L-1:            release W_i; prefetch W_{i+w}
    after f_{L-2}:                     prefetch dW_{L-1}
    after f_{L-1}:                     prefetch dW_{L-2}
    after b_i:                         release W_i; offload dW_i (write-back)
    after b_i, i - w ≥ 0:              prefetch W_{i-w}
    after b_i, i - 2 ≥ 0:              prefetch dW_{i-2}
    after b_i, i - 2 ∈ offloaded_A:    prefetch A_{i-2}

Initial device residency added: W_0..W_{w-1}, W_head, dW_head; plus any dW_j
not brought in by the forward dW preamble or the backward dW cascade (covers
edge cases like L=1).
"""
from __future__ import annotations

from dataflow_sim.schema import Object, OutputAlloc, Task, TaskChain, TransferTrigger


def apply_sliding_window_policy(
    bare: TaskChain,
    *,
    window_size: int = 2,
    device_capacity: int | None = None,
) -> TaskChain:
    """Annotate a bare training chain with the sliding-window pattern.

    Infers `L` from the number of `f_i` tasks in `bare`. Returns a fresh
    `TaskChain` with augmented `initial_memory` and per-task triggers; the
    bare input is left untouched.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    L = sum(1 for t in bare.tasks if t.id.startswith("f_"))
    if L < 1:
        raise ValueError("could not infer L (no f_i tasks in bare chain)")

    # Look up object sizes/types from bare initial memory (all model state lives
    # there as host-resident entries).
    host_objs: dict[str, Object] = {
        obj.id: obj for obj in bare.initial_memory if obj.location == "host"
    }

    # --- augmented initial memory ---
    new_initial: list[Object] = list(bare.initial_memory)

    # First `window_size` weights on device (so forward can start).
    for i in range(min(window_size, L)):
        src = host_objs[f"W_{i}"]
        new_initial.append(Object(id=src.id, size=src.size, location="device", type=src.type))
    # Head weights kept permanently device-resident.
    for hid in ("W_head", "dW_head"):
        src = host_objs[hid]
        new_initial.append(Object(id=src.id, size=src.size, location="device", type=src.type))

    # Pre-place any dW that no forward task / backward cascade will prefetch.
    prefetched_dws: set[int] = set()
    if L >= 2:
        prefetched_dws.add(L - 1)  # by f_{L-2}
        prefetched_dws.add(L - 2)  # by f_{L-1}
    for i in range(L - 1, 1, -1):
        if i - 2 >= 0:
            prefetched_dws.add(i - 2)
    for j in range(L):
        if j not in prefetched_dws:
            src = host_objs[f"dW_{j}"]
            new_initial.append(
                Object(id=src.id, size=src.size, location="device", type=src.type)
            )

    # --- annotated tasks ---
    offloaded_acts = set(range(L - 2))  # A_0..A_{L-3} (empty for L<3)

    new_tasks: list[Task] = []
    for task in bare.tasks:
        if task.id.startswith("f_"):
            new_tasks.append(_annotate_forward(task, L, window_size))
        elif task.id == "head":
            new_tasks.append(_annotate_head(task))
        elif task.id.startswith("r_"):
            new_tasks.append(task)  # unchanged
        elif task.id.startswith("b_"):
            new_tasks.append(_annotate_backward(task, L, window_size, offloaded_acts))
        else:
            new_tasks.append(task)

    return TaskChain(
        initial_memory=new_initial,
        tasks=new_tasks,
        bandwidth_h2d=bare.bandwidth_h2d,
        bandwidth_d2h=bare.bandwidth_d2h,
        device_capacity=device_capacity,
        host_capacity=bare.host_capacity,
    )


def _annotate_forward(task: Task, L: int, w: int) -> Task:
    i = int(task.id.split("_")[1])
    in_act = task.inputs[0]  # "input" or "y_{i-1}"

    releases: list[str] = [in_act]
    offloads: list[TransferTrigger] = []
    prefetches: list[TransferTrigger] = []

    # Activation offload: A_i for i in [0, L-3]
    if i <= L - 3:
        offloads.append(TransferTrigger(obj_id=f"A_{i}"))

    # Forward window slide: release W_i, prefetch W_{i+w}
    if i + w <= L - 1:
        releases.append(f"W_{i}")
        prefetches.append(TransferTrigger(obj_id=f"W_{i + w}"))

    # Forward dW preamble
    if i == L - 2:
        prefetches.append(TransferTrigger(obj_id=f"dW_{L - 1}"))
    if i == L - 1 and L >= 2:
        prefetches.append(TransferTrigger(obj_id=f"dW_{L - 2}"))

    return Task(
        id=task.id,
        inputs=task.inputs,
        outputs=task.outputs,
        runtime=task.runtime,
        releases_after=releases,
        offload_after=offloads,
        prefetch_after=prefetches,
    )


def _annotate_head(task: Task) -> Task:
    # Just release y_{L-1} (already the only release; matches bare task's
    # natural release for in-act). Head weights kept on device → no transfers.
    return Task(
        id=task.id,
        inputs=task.inputs,
        outputs=task.outputs,
        runtime=task.runtime,
        releases_after=[task.inputs[0]],  # y_{L-1}
    )


def _annotate_backward(task: Task, L: int, w: int, offloaded_acts: set[int]) -> Task:
    i = int(task.id.split("_")[1])
    upstream = task.inputs[0]  # dy_head or dy_{i+1}

    releases: list[str] = [upstream, f"A_{i}", f"W_{i}"]
    offloads: list[TransferTrigger] = [TransferTrigger(obj_id=f"dW_{i}")]  # write-back
    prefetches: list[TransferTrigger] = []

    # Backward window slide
    if i - w >= 0:
        prefetches.append(TransferTrigger(obj_id=f"W_{i - w}"))

    # dW cascade
    if i - 2 >= 0:
        prefetches.append(TransferTrigger(obj_id=f"dW_{i - 2}"))

    # Activation prefetch for offloaded A_{i-2}
    if i - 2 in offloaded_acts:
        prefetches.append(TransferTrigger(obj_id=f"A_{i - 2}"))

    return Task(
        id=task.id,
        inputs=task.inputs,
        outputs=task.outputs,
        runtime=task.runtime,
        releases_after=releases,
        offload_after=offloads,
        prefetch_after=prefetches,
    )
