"""Static prepass validator for TaskChain.

Catches every correctness invariant from docs/policy/principles.md §1 that
can be decided WITHOUT stepping the simulator clock. Raises
:class:`ValidationError` (a :class:`ValueError` subclass for backward compat
with callers that already catch ``ValueError``) on the first violation found.

The error message format is::

    <rule-token>: task '<task_id>' <human-readable reason> (obj '<obj_id>'...)

The leading kebab-case ``<rule-token>`` lets tests match violations without
depending on the prose. Tokens used:

    duplicate-initial-memory       unknown-input
    duplicate-input-id             duplicate-output-id
    duplicate-output-shadows-initial
    releases_after-not-in-inputs   mutates_inputs-not-in-inputs
    offload-unknown-obj            prefetch-unknown-obj
    self-cycle
    prefetch-already-on-device     offload-not-on-device
    duplicate-prefetch             duplicate-offload
    conflict-prefetch-and-offload-same-object
    final-location-unknown-obj     final-location-invalid
    final-location-not-on-host     final-location-dirty-on-host
    final-location-not-on-device
    release-of-dirty-with-later-use
    release-no-host-copy-with-later-use
    released-then-referenced
    release-and-offload-same-object
    initial_memory-overflow
    forced-footprint-exceeds-device_capacity

Runtime-only properties (FIFO contention timing, makespan, transit-byte
residency, stall counts) are NOT checked here — bad-but-runnable chains
pass validation and surface as idle time during simulation.
"""
from __future__ import annotations

from dataflow_sim.core.schema import TaskChain


class ValidationError(ValueError):
    """Raised by :func:`validate_chain` on any static invariant violation."""


# ---------------------------------------------------------------------------
# Category checks
# ---------------------------------------------------------------------------


def validate_id_resolution(chain: TaskChain) -> None:
    """Every referenced id resolves; output ids are fresh; duplicate input ids
    rejected; releases_after / mutates_inputs are subsets of inputs.
    """
    # All ids known so far (ever introduced, regardless of current residency).
    known: set[str] = set()
    # Per-location initial residency (for shadowing detection).
    initial_ids: set[str] = {o.id for o in chain.initial_memory}

    # Detect duplicate (id, location) entries in initial_memory.
    seen_initial: set[tuple[str, str]] = set()
    for obj in chain.initial_memory:
        key = (obj.id, obj.location)
        if key in seen_initial:
            raise ValidationError(
                f"duplicate-initial-memory: ({obj.id!r}, {obj.location!r}) "
                f"appears twice in initial_memory"
            )
        seen_initial.add(key)
        known.add(obj.id)

    for task in chain.tasks:
        # Duplicate ids within this task's inputs list.
        seen_in: set[str] = set()
        for inp in task.inputs:
            if inp in seen_in:
                raise ValidationError(
                    f"duplicate-input-id: task {task.id!r} lists input {inp!r} "
                    f"more than once"
                )
            seen_in.add(inp)

        # Output-id freshness checks FIRST (before self-cycle), so an output
        # that shadows initial_memory reports as a duplicate rather than as
        # a confusing self-cycle on the input that points to the initial obj.
        own_outputs: set[str] = set()
        for out in task.outputs:
            if out.id in initial_ids:
                raise ValidationError(
                    f"duplicate-output-shadows-initial: task {task.id!r} output {out.id!r} "
                    f"collides with an id already present in initial_memory"
                )
            if out.id in known:
                raise ValidationError(
                    f"duplicate-output-id: task {task.id!r} output {out.id!r} "
                    f"collides with an output id already introduced earlier in the chain"
                )
            if out.id in own_outputs:
                raise ValidationError(
                    f"duplicate-output-id: task {task.id!r} declares output {out.id!r} "
                    f"more than once"
                )
            own_outputs.add(out.id)

        # Self-cycle: task references its own output as input.
        for inp in task.inputs:
            if inp in own_outputs:
                raise ValidationError(
                    f"self-cycle: task {task.id!r} consumes its own output {inp!r}"
                )

        # Every input must already be known (initial OR a prior task's output).
        for inp in task.inputs:
            if inp not in known:
                raise ValidationError(
                    f"unknown-input: task {task.id!r} references unknown input {inp!r} "
                    f"which is not in initial_memory and not produced by any prior task"
                )

        # releases_after must reference some statically-known id. The principle
        # ("releases must name the obj as input") is STRICTER than the runtime
        # contract ("release any live-on-device object"); the current auto-
        # policies emit GC-style releases of objects this task didn't consume,
        # which the runtime accepts. We mirror the runtime contract here and
        # leave the principle-strict check (release-by-non-consumer) as a
        # future tightening an open design question.
        for rid in task.releases_after:
            if rid not in known and rid not in own_outputs:
                raise ValidationError(
                    f"releases_after-unknown-obj: task {task.id!r} releases {rid!r} "
                    f"which is not a known object at this point in the chain"
                )

        # mutates_inputs ⊆ inputs (mutation only applies to inputs).
        for mid in task.mutates_inputs:
            if mid not in seen_in:
                raise ValidationError(
                    f"mutates_inputs-not-in-inputs: task {task.id!r} mutates {mid!r} "
                    f"which is not in its inputs"
                )

        # Trigger obj_ids must resolve to a known id at this point in the chain.
        for trig in task.offload_after:
            if trig.obj_id not in known and trig.obj_id not in own_outputs:
                raise ValidationError(
                    f"offload-unknown-obj: task {task.id!r} offloads {trig.obj_id!r} "
                    f"which is not a known object at this point in the chain"
                )
        for trig in task.prefetch_after:
            if trig.obj_id not in known and trig.obj_id not in own_outputs:
                raise ValidationError(
                    f"prefetch-unknown-obj: task {task.id!r} prefetches {trig.obj_id!r} "
                    f"which is not a known object at this point in the chain"
                )

        # Commit this task's outputs into the known set for downstream tasks.
        known.update(own_outputs)

    for oid, loc in chain.final_locations.items():
        if oid not in known:
            raise ValidationError(
                f"final-location-unknown-obj: final_locations references {oid!r}, "
                "which is not in initial_memory and not produced by any task"
            )
        if loc not in ("host", "device"):
            raise ValidationError(
                f"final-location-invalid: final_locations[{oid!r}]={loc!r} is not "
                "'host' or 'device'"
            )


def validate_triggers(chain: TaskChain) -> None:
    """Static state-tracking of device/host residency across the chain.

    Rejects prefetches targeting an already-on-device object, offloads of
    not-on-device objects, duplicate prefetches/offloads on the same anchor,
    and conflicting prefetch+offload of the same object on the same task.
    """
    on_device: set[str] = set()
    on_host: set[str] = set()
    for obj in chain.initial_memory:
        if obj.location == "device":
            on_device.add(obj.id)
        else:
            on_host.add(obj.id)

    for task in chain.tasks:
        # Same-task conflict: same obj in BOTH offload_after AND prefetch_after.
        off_ids = {t.obj_id for t in task.offload_after}
        pf_ids = {t.obj_id for t in task.prefetch_after}
        conflict = off_ids & pf_ids
        if conflict:
            oid = next(iter(conflict))
            raise ValidationError(
                f"conflict-prefetch-and-offload-same-object: task {task.id!r} schedules "
                f"both a prefetch and an offload for {oid!r}"
            )

        # Trigger fire AFTER the task completes, so this task's outputs are
        # already live on their declared location when we evaluate triggers.
        # Compute the post-output device set up front so trigger checks see it.
        device_at_triggers = set(on_device)
        host_at_triggers = set(on_host)
        for out in task.outputs:
            if out.location == "device":
                device_at_triggers.add(out.id)
            else:
                host_at_triggers.add(out.id)

        # Duplicate offload triggers (same obj listed twice in this task).
        seen_off: set[str] = set()
        for trig in task.offload_after:
            if trig.obj_id in seen_off:
                raise ValidationError(
                    f"duplicate-offload: task {task.id!r} schedules offload of "
                    f"{trig.obj_id!r} more than once"
                )
            seen_off.add(trig.obj_id)
            if trig.obj_id not in device_at_triggers:
                raise ValidationError(
                    f"offload-not-on-device: task {task.id!r} offloads {trig.obj_id!r} "
                    f"which is not statically resident on device at this point"
                )

        # Duplicate prefetch triggers.
        seen_pf: set[str] = set()
        for trig in task.prefetch_after:
            if trig.obj_id in seen_pf:
                raise ValidationError(
                    f"duplicate-prefetch: task {task.id!r} schedules prefetch of "
                    f"{trig.obj_id!r} more than once"
                )
            seen_pf.add(trig.obj_id)
            if trig.obj_id in device_at_triggers:
                raise ValidationError(
                    f"prefetch-already-on-device: task {task.id!r} prefetches "
                    f"{trig.obj_id!r} which is already on device"
                )

        # Commit end-of-step state changes for the NEXT task:
        # 1. Outputs landed (already reflected in device_at_triggers).
        on_device = device_at_triggers
        on_host = host_at_triggers
        # 2. Offloads add a host copy AND remove from device (post-completion).
        for trig in task.offload_after:
            on_host.add(trig.obj_id)
            on_device.discard(trig.obj_id)
        # 3. Prefetches make the object device-resident.
        for trig in task.prefetch_after:
            on_device.add(trig.obj_id)
        # 4. Bare releases drop from device.
        for rid in task.releases_after:
            on_device.discard(rid)


def validate_releases_mutation(chain: TaskChain) -> None:
    """Dirty-tracking / host-copy / lifetime invariants for releases & mutations.

    Assumes id-resolution + triggers passed (so inputs are well-formed and
    obj_ids resolve). The on-device set here is a duplicate of the triggers
    walk; we re-do it locally to avoid coupling.
    """
    on_device: set[str] = set()
    on_host: set[str] = set()
    dirty: set[str] = set()  # mutated since last offload — host copy is stale
    # For each released id: (task_id, was_dirty_at_release, had_host_copy_at_release).
    released_meta: dict[str, tuple[str, bool, bool]] = {}

    for obj in chain.initial_memory:
        if obj.location == "device":
            on_device.add(obj.id)
        else:
            on_host.add(obj.id)

    for i, task in enumerate(chain.tasks):
        # Same-task release + offload of the same id — semantically ambiguous.
        rel_set = set(task.releases_after)
        off_set = {t.obj_id for t in task.offload_after}
        both = rel_set & off_set
        if both:
            oid = next(iter(both))
            raise ValidationError(
                f"release-and-offload-same-object: task {task.id!r} lists {oid!r} "
                f"in both releases_after and offload_after"
            )

        # Inputs at this task must be on device. If absent, either it was
        # released earlier (rich message keyed off `released_meta`) or it
        # was never device-resident at all.
        for inp in task.inputs:
            if inp not in on_device:
                if inp in released_meta:
                    rel_task, was_dirty, had_host = released_meta[inp]
                    if was_dirty:
                        raise ValidationError(
                            f"release-of-dirty-with-later-use: input {inp!r} was "
                            f"bare-released at task {rel_task!r} while dirty (mutated "
                            f"without write-back), then consumed later by task {task.id!r}"
                        )
                    if not had_host:
                        raise ValidationError(
                            f"release-no-host-copy-with-later-use: input {inp!r} was "
                            f"released at task {rel_task!r} with no host copy, then "
                            f"consumed later by task {task.id!r}"
                        )
                    raise ValidationError(
                        f"released-then-referenced: input {inp!r} was released at task "
                        f"{rel_task!r} and not re-prefetched before task {task.id!r}"
                    )
                # Not released — must be a static "never-on-device" case.
                # Topology/id_resolution should have caught truly unknown ids;
                # this branch covers host-only-without-prefetch references.
                raise ValidationError(
                    f"released-then-referenced: task {task.id!r} input {inp!r} is not "
                    f"on device at task start and was never prefetched"
                )

        # Commit task end-of-step state changes in this order:
        # 1. Outputs land on their declared location.
        for out in task.outputs:
            if out.location == "device":
                on_device.add(out.id)
            else:
                on_host.add(out.id)
        # 2. Mutations dirty the device copy.
        for mid in task.mutates_inputs:
            dirty.add(mid)
        # 3. Offloads clear dirty + add host copy + drop device residency.
        for trig in task.offload_after:
            dirty.discard(trig.obj_id)
            on_host.add(trig.obj_id)
            on_device.discard(trig.obj_id)
        # 4. Prefetches re-add device residency.
        for trig in task.prefetch_after:
            on_device.add(trig.obj_id)
            dirty.discard(trig.obj_id)
            # Re-prefetching also clears any prior "released" marker.
            released_meta.pop(trig.obj_id, None)
        # 5. Bare releases drop device residency and snapshot release-time
        #    metadata for the later-reference check above.
        for rid in task.releases_after:
            released_meta[rid] = (
                task.id,
                rid in dirty,
                rid in on_host,
            )
            on_device.discard(rid)
            dirty.discard(rid)

    for oid, loc in chain.final_locations.items():
        if loc == "host":
            if oid not in on_host:
                raise ValidationError(
                    f"final-location-not-on-host: object {oid!r} is required on host "
                    "at chain end but no host copy is available"
                )
            if oid in dirty:
                raise ValidationError(
                    f"final-location-dirty-on-host: object {oid!r} is required on "
                    "host at chain end but its latest device bytes were not offloaded"
                )
        elif loc == "device" and oid not in on_device:
            raise ValidationError(
                f"final-location-not-on-device: object {oid!r} is required on device "
                "at chain end but is not device-resident"
            )


def validate_capacity(chain: TaskChain) -> None:
    """Forced-footprint capacity checks that no policy can hide."""
    # Initial-memory sums per location (mirrors simulator._check_initial_capacity).
    alloc = {"host": 0, "device": 0}
    for obj in chain.initial_memory:
        alloc[obj.location] += obj.size
    for loc, cap in (("device", chain.device_capacity), ("host", chain.host_capacity)):
        if cap is None:
            continue
        if alloc[loc] > cap:
            raise ValidationError(
                f"initial_memory-overflow: <initial_memory> cannot allocate "
                f"{alloc[loc]} on {loc}: capacity={cap}"
            )

    if chain.device_capacity is None:
        return

    # Per-task forced footprint: sum of input sizes (all live on device at
    # dispatch) + sum of device-located output sizes (reserved at dispatch).
    # We need a size lookup that follows the same flow as id_resolution.
    size_of: dict[str, int] = {o.id: o.size for o in chain.initial_memory}
    for task in chain.tasks:
        # Inputs must already be in size_of (id_resolution guaranteed it).
        input_bytes = sum(size_of.get(i, 0) for i in task.inputs)
        out_device_bytes = sum(o.size for o in task.outputs if o.location == "device")
        forced = input_bytes + out_device_bytes
        if forced > chain.device_capacity:
            raise ValidationError(
                f"forced-footprint-exceeds-device_capacity: task {task.id!r} requires "
                f"{input_bytes} bytes of inputs + {out_device_bytes} bytes of device "
                f"outputs = {forced} > device_capacity={chain.device_capacity}; "
                f"cannot satisfy device memory need under any policy"
            )
        for out in task.outputs:
            size_of[out.id] = out.size


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_chain(chain: TaskChain) -> None:
    """Static validation of a TaskChain. Raises ValidationError on first violation.

    Catches all statically-computable correctness invariants from
    docs/policy/principles.md §1, BEFORE the simulator starts stepping.

    Does NOT check runtime properties: transit-byte residency, stall amounts,
    FIFO contention timing, makespan. Bad-but-runnable chains pass this check
    and reveal themselves as idle time during simulation.
    """
    # Order matters: later checks assume earlier ones passed.
    validate_id_resolution(chain)
    validate_triggers(chain)
    validate_releases_mutation(chain)
    validate_capacity(chain)
