"""Hand-crafted sliding-window policy for simple training chains.

Given a bare legacy training chain, annotate it
with the sliding-window trigger pattern (see
`docs/policy/other_policies/sliding-window.md`). Workload-specific: keys off
the training builder's task/object id conventions.

Pattern (per layer index i, w = window_size, L inferred from the chain):

    after f_i, i ≤ L-3:                offload A_i
    after f_i, i + w ≤ L-1:            release W_i; prefetch W_{i+w}
    after f_{L-2}:                     prefetch dW_{L-1}
    after f_{L-1}:                     prefetch dW_{L-2}
    after b_i:                         release W_i; offload dW_i (write-back)
    after b_i, i - w ≥ 0:              prefetch W_{i-w}
    after b_i, i - 2 ≥ 0:              prefetch dW_{i-2}
    after b_i, i - 2 ∈ offloaded_A:    prefetch A_{i-2}

Initial compute residency added: W_0..W_{w-1}, W_head, dW_head; plus any dW_j
not brought in by the forward dW preamble or the backward dW cascade (covers
edge cases like L=1).
"""
from __future__ import annotations

from typing import Literal

from dataflow_sim.core.schema import Object, Task, TaskChain, TransferTrigger

_Mode = Literal["legacy", "step_aware"]


def apply_sliding_window_policy(
    bare: TaskChain,
    *,
    window_size: int = 2,
    fast_memory_capacity: int | None = None,
) -> TaskChain:
    """Annotate a bare training chain with the sliding-window pattern.

    Infers `L` from the number of `f_i` tasks in `bare`. Returns a fresh
    `TaskChain` with augmented `initial_memory` and per-task triggers; the
    bare input is left untouched.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if any(t.id.startswith("step_") for t in bare.tasks):
        raise ValueError(
            "sliding_window does not support optimizer-step tasks; use a general policy"
        )
    mode, L = _infer_mode_and_layers(bare)
    if L < 1:
        raise ValueError("could not infer L (no f_i tasks in bare chain)")

    # Look up object sizes/types from bare initial memory (all model state lives
    # there as backing-resident entries).
    backing_objs: dict[str, Object] = {
        obj.id: obj for obj in bare.initial_memory if obj.location == "backing"
    }

    # --- augmented initial memory ---
    new_initial: list[Object] = list(bare.initial_memory)

    # First `window_size` weights on compute (so forward can start).
    for i in range(min(window_size, L)):
        src = backing_objs[f"W_{i}"]
        new_initial.append(Object(id=src.id, size=src.size, location="fast", type=src.type))
    # Head weights kept permanently compute-resident.
    head_ids = ("W_head", "dW_head") if mode == "legacy" else ("W_head",)
    for hid in head_ids:
        src = backing_objs[hid]
        new_initial.append(Object(id=src.id, size=src.size, location="fast", type=src.type))

    if mode == "legacy":
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
                src = backing_objs[f"dW_{j}"]
                new_initial.append(
                    Object(id=src.id, size=src.size, location="fast", type=src.type)
                )

    # --- annotated tasks ---
    offloaded_acts = set(range(L - 2))  # A_0..A_{L-3} (empty for L<3)

    new_tasks: list[Task] = []
    for task in bare.tasks:
        if task.id.startswith("f_"):
            new_tasks.append(_annotate_forward(task, L, window_size, mode))
        elif task.id == _head_id(mode):
            new_tasks.append(_annotate_head(task, mode))
        elif task.id.startswith("r_"):
            new_tasks.append(task)  # unchanged
        elif task.id.startswith("b_"):
            new_tasks.append(_annotate_backward(task, L, window_size, offloaded_acts, mode))
        else:
            new_tasks.append(task)

    return TaskChain(
        initial_memory=new_initial,
        tasks=new_tasks,
        bandwidth_from_slow=bare.bandwidth_from_slow,
        bandwidth_to_slow=bare.bandwidth_to_slow,
        final_locations=bare.final_locations,
        fast_memory_capacity=fast_memory_capacity,
        backing_memory_capacity=bare.backing_memory_capacity,
    )


def _infer_mode_and_layers(bare: TaskChain) -> tuple[_Mode, int]:
    forward_ids = [t.id for t in bare.tasks if t.id.startswith("f_")]
    if not forward_ids:
        raise ValueError("could not infer L (no f_i tasks in bare chain)")

    split_ids = [tid.split("_") for tid in forward_ids]
    if all(len(parts) == 2 for parts in split_ids):
        return "legacy", len(forward_ids)

    if all(len(parts) == 4 for parts in split_ids):
        triples = [(int(parts[1]), int(parts[2]), int(parts[3])) for parts in split_ids]
        steps = {k for k, _, _ in triples}
        accums = {j for _, j, _ in triples}
        layers = sorted(i for _, _, i in triples)
        if steps == {0} and accums == {0} and layers == list(range(len(layers))):
            return "step_aware", len(layers)
        raise ValueError(
            "sliding_window supports only one training step and one accumulation "
            "round; use a general policy for multi-step or grad-accum chains"
        )

    raise ValueError(
        "sliding_window does not recognize this training-chain ID convention; "
        "use a general policy"
    )


def _layer_idx(task: Task, mode: _Mode) -> int:
    return int(task.id.split("_")[1] if mode == "legacy" else task.id.split("_")[3])


def _head_id(mode: _Mode) -> str:
    return "head" if mode == "legacy" else "head_0_0"


def _act_id(mode: _Mode, i: int) -> str:
    return f"A_{i}" if mode == "legacy" else f"A_0_0_{i}"


def _annotate_forward(task: Task, L: int, w: int, mode: _Mode) -> Task:
    i = _layer_idx(task, mode)
    in_act = task.inputs[0]  # "input" or "y_{i-1}"

    releases: list[str] = [in_act]
    offloads: list[TransferTrigger] = []
    prefetches: list[TransferTrigger] = []

    # Activation offload: A_i for i in [0, L-3]
    if i <= L - 3:
        offloads.append(TransferTrigger(obj_id=_act_id(mode, i)))

    # Forward window slide: release W_i, prefetch W_{i+w}
    if i + w <= L - 1:
        releases.append(f"W_{i}")
        prefetches.append(TransferTrigger(obj_id=f"W_{i + w}"))

    if mode == "legacy":
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


def _annotate_head(task: Task, mode: _Mode) -> Task:
    # Just release y_{L-1} (already the only release; matches bare task's
    # natural release for in-act). Head weights kept on compute → no transfers.
    releases = [task.inputs[0]]
    if mode == "step_aware":
        releases.extend(out.id for out in task.outputs if out.id.startswith("dW_head_"))
    return Task(
        id=task.id,
        inputs=task.inputs,
        outputs=task.outputs,
        runtime=task.runtime,
        releases_after=releases,
    )


def _annotate_backward(
    task: Task, L: int, w: int, offloaded_acts: set[int], mode: _Mode
) -> Task:
    i = _layer_idx(task, mode)
    upstream = task.inputs[0]  # dy_head or dy_{i+1}

    releases: list[str] = [upstream, _act_id(mode, i), f"W_{i}"]
    offloads: list[TransferTrigger] = []
    if mode == "legacy":
        offloads.append(TransferTrigger(obj_id=f"dW_{i}"))  # write-back
    else:
        releases.extend(out.id for out in task.outputs if out.id.startswith("dW_"))
    prefetches: list[TransferTrigger] = []

    # Backward window slide
    if i - w >= 0:
        prefetches.append(TransferTrigger(obj_id=f"W_{i - w}"))

    if mode == "legacy":
        # dW cascade
        if i - 2 >= 0:
            prefetches.append(TransferTrigger(obj_id=f"dW_{i - 2}"))

    # Activation prefetch for offloaded A_{i-2}
    if i - 2 in offloaded_acts:
        prefetches.append(TransferTrigger(obj_id=_act_id(mode, i - 2)))

    return Task(
        id=task.id,
        inputs=task.inputs,
        outputs=task.outputs,
        runtime=task.runtime,
        releases_after=releases,
        offload_after=offloads,
        prefetch_after=prefetches,
    )
