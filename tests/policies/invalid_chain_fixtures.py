"""Invalid TaskChain fixtures for the ID-RESOLUTION violation category.

Each fixture in this file builds the MINIMAL TaskChain that violates exactly
one ID-resolution invariant from docs/policy/principles.md (§1 Correctness).
The simulator (or an upcoming static prepass validator) should reject each
chain with an error message whose text contains the corresponding keyword
from `EXPECTED_ERROR_KEYWORDS`.

This file is consumed by tests/test_validate_chain.py (phase 4).
"""

from dataflow_sim.core.schema import Object, OutputAlloc, Task, TaskChain, TransferTrigger


def make_invalid_id_resolution_unknown_input() -> TaskChain:
    """task.inputs references an id that is neither in initial_memory nor any
    prior task's outputs. Violates §1: every input id must resolve to a live
    object produced upstream. Expected validator error includes: 'unknown input'.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W", "GHOST"],  # GHOST never defined anywhere
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_id_resolution_release_not_in_inputs() -> TaskChain:
    """task.releases_after lists an id that is not known anywhere in the chain.

    Note: per docs/policy/principles.md §1, a task may only release ids it
    consumed as inputs. The current validator relaxes this to "release any
    statically-known id" (matching the runtime's contract; the principle-
    strict check is a future tightening an open design question). This
    fixture therefore exercises the relaxed check: releasing a truly unknown
    id should still fail.

    Expected validator error includes: 'releases_after'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
                releases_after=["GHOST"],  # GHOST is unknown
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_id_resolution_offload_unknown_obj() -> TaskChain:
    """task.offload_after.obj_id references an id that doesn't resolve to any
    object known at this point in the chain. Violates §1: transfer triggers
    must name a real object. Expected validator error includes: 'offload'.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
                offload_after=[TransferTrigger(obj_id="NOPE")],  # NOPE doesn't exist
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_id_resolution_prefetch_unknown_obj() -> TaskChain:
    """task.prefetch_after.obj_id references an id that doesn't resolve to any
    object known at this point in the chain. Violates §1: transfer triggers
    must name a real object. Expected validator error includes: 'prefetch'.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
                prefetch_after=[TransferTrigger(obj_id="MISSING")],  # MISSING doesn't exist
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_id_resolution_output_shadows_initial() -> TaskChain:
    """A task output uses an id already present in initial_memory.
    Violates §1: outputs must introduce fresh ids; reusing an existing id
    creates ambiguity about which object the id refers to.
    Expected validator error includes: 'duplicate'.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="W", size=1)],  # collides with initial_memory id
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_id_resolution_output_id_collision() -> TaskChain:
    """Two tasks produce outputs with the SAME id.
    Violates §1: every object id introduced into the chain must be unique.
    Expected validator error includes: 'duplicate'.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="A", size=1)],
                runtime=1,
            ),
            Task(
                id="t1",
                inputs=["W"],
                outputs=[OutputAlloc(id="A", size=1)],  # collides with t0's output
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_id_resolution_mutates_not_in_inputs() -> TaskChain:
    """task.mutates_inputs lists an id NOT in task.inputs.
    Violates §1: a task can only mutate objects it actually reads as inputs.
    Expected validator error includes: 'mutates_inputs'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="G", size=1, location="fast", type="gradient"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
                mutates_inputs=["G"],  # G is not in inputs
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


EXPECTED_ERROR_KEYWORDS: dict[str, str | list[str]] = {
    "make_invalid_id_resolution_unknown_input": "unknown input",
    "make_invalid_id_resolution_release_not_in_inputs": "releases_after",
    "make_invalid_id_resolution_offload_unknown_obj": "offload",
    "make_invalid_id_resolution_prefetch_unknown_obj": "prefetch",
    "make_invalid_id_resolution_output_shadows_initial": "duplicate",
    "make_invalid_id_resolution_output_id_collision": "duplicate",
    "make_invalid_id_resolution_mutates_not_in_inputs": "mutates_inputs",
}


# ---------------------------------------------------------------------------
# Category: TRIGGER VALIDITY
#
# Principles cited (docs/policy/principles.md §1):
#   * "Prefetch of X is valid only when X's compute entry is absent or in-flight
#      outbound; offload of X is valid only when compute entry is `live` and any
#      backing entry is `live` with matching size."
#
# All five fixtures below violate this trigger-validity invariant in distinct
# ways. The simulator should reject each (today: at runtime; eventually: via
# the static prepass we are building toward).
# ---------------------------------------------------------------------------


def make_invalid_trigger_validity_prefetch_already_on_compute() -> TaskChain:
    """Prefetch of X when X is statically already on compute.

    Rule: "Prefetch of X is valid only when X's compute entry is absent or
    in-flight outbound." Here W starts in compute-resident initial_memory and
    t0 schedules a prefetch_after of W with no intervening offload — the
    prefetch targets a slot that is already `live` on compute.

    Expected validator error includes: 'prefetch'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="W_backing", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                prefetch_after=[TransferTrigger(obj_id="W")],
            ),
            Task(id="t1", inputs=["W"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_trigger_validity_offload_not_on_compute() -> TaskChain:
    """Offload of X when X is statically not on compute.

    Rule: "Offload of X is valid only when compute entry is `live`." Here W
    only exists on backing (initial_memory location="backing") and is never
    prefetched. t0 attempts to offload_after W — there is no compute entry to
    move.

    Expected validator error includes: 'offload'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="A", size=1, location="fast", type="weight"),
            Object(id="W", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["A"],
                outputs=[],
                runtime=1,
                offload_after=[TransferTrigger(obj_id="W")],
            ),
            Task(id="t1", inputs=["A"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_trigger_validity_duplicate_prefetch_same_task() -> TaskChain:
    """Two prefetches of the SAME object attached to one task's prefetch_after.

    Rule: "Prefetch of X is valid only when X's compute entry is absent or
    in-flight outbound." The first prefetch makes X pending_inbound/inbound;
    the second targets a slot that is no longer absent — duplicate enqueue
    against the same destination.

    Expected validator error includes: 'duplicate'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="A", size=1, location="fast", type="weight"),
            Object(id="W", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["A"],
                outputs=[],
                runtime=1,
                prefetch_after=[
                    TransferTrigger(obj_id="W"),
                    TransferTrigger(obj_id="W"),
                ],
            ),
            Task(id="t1", inputs=["A", "W"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_trigger_validity_duplicate_offload_same_task() -> TaskChain:
    """Two offloads of the SAME object attached to one task's offload_after.

    Rule: "Offload of X is valid only when compute entry is `live`." The first
    offload flips the compute entry to pending_outbound/outbound; the second
    finds the compute entry no longer `live` (or attempts to overwrite an
    in-flight transfer).

    Expected validator error includes: 'duplicate'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="W_backing", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                offload_after=[
                    TransferTrigger(obj_id="W"),
                    TransferTrigger(obj_id="W"),
                ],
            ),
            Task(id="t1", inputs=[], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_trigger_validity_prefetch_and_offload_same_object_same_task() -> TaskChain:
    """Both a prefetch_after AND offload_after on the SAME object on one task.

    Rule: "Prefetch of X is valid only when X's compute entry is absent or
    in-flight outbound; offload of X is valid only when compute entry is
    `live`." These two conditions are mutually exclusive at the same anchor
    — one of the triggers necessarily targets a state the other forbids
    (and the result is a no-op round-trip even if the simulator tolerated
    the race).

    Expected validator error includes: 'conflict'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="W_backing", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                offload_after=[TransferTrigger(obj_id="W")],
                prefetch_after=[TransferTrigger(obj_id="W")],
            ),
            Task(id="t1", inputs=[], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


EXPECTED_ERROR_KEYWORDS.update(
    {
        "make_invalid_trigger_validity_prefetch_already_on_compute": "prefetch",
        "make_invalid_trigger_validity_offload_not_on_compute": "offload",
        "make_invalid_trigger_validity_duplicate_prefetch_same_task": "duplicate",
        "make_invalid_trigger_validity_duplicate_offload_same_task": "duplicate",
        "make_invalid_trigger_validity_prefetch_and_offload_same_object_same_task": "conflict",
    }
)


# ---------------------------------------------------------------------------
# Category: RELEASE & MUTATION
#
# Principles cited (docs/policy/principles.md §1):
#   * "An object may be released only by a task that names it as input, and
#      only when its compute entry is `live`."
#   * "An object cannot be released if it has another use AND (backing lacks a
#      copy OR object is dirty)."
#   * "A mutated input must be offloaded after its last mutation, before any
#      bare release of the same object."
#
# Each fixture below violates one of these rules in isolation. The bare
# release of a dirty / compute-only object is the user's seed bug; the others
# pin down adjacent corners of the same invariant family.
# ---------------------------------------------------------------------------


def make_invalid_release_mutation_dirty_with_later_use() -> TaskChain:
    """Bare release of a DIRTY object that has a later use.

    Rule: 'An object cannot be released if it has another use AND ... object
    is dirty.' Also: 'A mutated input must be offloaded after its last
    mutation, before any bare release of the same object.'

    W is mutated by t0, then bare-released by t0; t2 later consumes W. A
    bare release of a dirty object with a later use silently discards the
    mutation (any backing copy is stale).

    Expected validator error includes: 'dirty'.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                mutates_inputs=["W"],
                releases_after=["W"],  # BAD: dirty W has a later use at t2
            ),
            Task(id="t1", inputs=[], outputs=[], runtime=1),
            Task(id="t2", inputs=["W"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_release_mutation_no_backing_copy_with_later_use() -> TaskChain:
    """Bare release of a DEVICE-ONLY object that has a later use.

    Rule: 'An object cannot be released if it has another use AND (backing
    lacks a copy ...).'

    A is produced compute-only by t0 (never offloaded) and bare-released by
    t0, but t2 re-references A — there is no backing copy to re-prefetch from,
    so the bare release strands the only copy.

    Expected validator error includes: 'backing'.
    """
    return TaskChain(
        initial_memory=[],
        tasks=[
            Task(
                id="t0",
                inputs=[],
                outputs=[
                    OutputAlloc(id="A", size=1, location="fast", type="activation"),
                ],
                runtime=1,
                releases_after=["A"],  # BAD: A is compute-only and has a later use
            ),
            Task(id="t1", inputs=[], outputs=[], runtime=1),
            Task(id="t2", inputs=["A"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_release_mutation_release_not_in_inputs() -> TaskChain:
    """A release whose target is NOT in this task's `inputs`.

    PER PRINCIPLE: 'An object may be released only by a task that names it
    as input.' This is the strictest reading.

    CURRENT VALIDATOR: the principle-strict check is RELAXED — the auto
    policies (belady_reactive/roundtrip_planner/max_reduce) emit GC-style releases on tasks that didn't consume
    the object (the runtime accepts this; tightening would require a policy
    refactor an open design question). So this fixture currently
    represents a chain the validator accepts. EXPECTED_ERROR_KEYWORDS value
    is None to pin that behavior; flip to 'input' once the policies are
    fixed and the validator re-tightens.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(id="t0", inputs=["W"], outputs=[], runtime=1),
            Task(
                id="t1",
                inputs=[],  # W absent here
                outputs=[],
                runtime=1,
                releases_after=["W"],  # principle-violation; currently accepted
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_release_mutation_mutation_never_offloaded() -> TaskChain:
    """A mutated input with final backing placement that is never offloaded.

    Rule: if final_locations[obj] == "backing", backing must receive the latest
    bytes by chain end.

    W starts compute-resident. t0 mutates W in place, then the chain ends with
    no offload of W. The terminal backing placement cannot be satisfied.

    Expected validator error includes: 'backing'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="W_backing", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                mutates_inputs=["W"],  # BAD: mutated, never offloaded, never released
            ),
        ],
        final_locations={"W": "backing"},
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_release_mutation_release_then_later_reference() -> TaskChain:
    """A bare release followed by a later reference with no re-prefetch.

    Rule: lifetime tracking — a bare-released object cannot be referenced
    by a later task without an intervening re-prefetch (else the runtime
    raises 'unknown input' mid-execution; the static prepass should catch
    it up front).

    W is clean and backing-backed, so t0's bare release is structurally legal
    in isolation. The bug is that t2 references W with no intervening
    prefetch_after — lifetime tracking should flag that W is no longer in
    fast memory at t2's start.

    Expected validator error includes: 'released'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="W_backing", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                releases_after=["W"],  # released; no re-prefetch scheduled
            ),
            Task(id="t1", inputs=[], outputs=[], runtime=1),
            Task(id="t2", inputs=["W"], outputs=[], runtime=1),  # BAD: W is gone
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_release_mutation_release_and_offload_same_object() -> TaskChain:
    """Both releases_after AND offload_after on the SAME object in one task.

    Rule: well-formedness / §1 trigger composition — a single task may not
    both release and offload the same object in the same trigger group; the
    semantics are ambiguous (does the offload's source still exist after
    release? does the release fire before or after to-slow starts?).

    t0 lists W in BOTH releases_after AND offload_after.

    Expected validator error includes: 'both'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W", size=1, location="fast", type="weight"),
            Object(id="W_backing", size=1, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[],
                runtime=1,
                mutates_inputs=["W"],
                releases_after=["W"],                          # BAD: same object ...
                offload_after=[TransferTrigger(obj_id="W")],   # ... as offload
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


EXPECTED_ERROR_KEYWORDS.update(
    {
        "make_invalid_release_mutation_dirty_with_later_use": "dirty",
        "make_invalid_release_mutation_no_backing_copy_with_later_use": "backing",
        "make_invalid_release_mutation_release_not_in_inputs": None,
        "make_invalid_release_mutation_mutation_never_offloaded": "backing",
        "make_invalid_release_mutation_release_then_later_reference": "released",
        "make_invalid_release_mutation_release_and_offload_same_object": "both",
    }
)


# ---------------------------------------------------------------------------
# Category: TOPOLOGICAL / DEADLOCK
#
# Principles cited (docs/policy/principles.md §1):
#   * "Every task input must be `live` on compute by task start — either
#      resident or via a prefetch whose from-slow (plus any blocking to-slow)
#      completes before the earliest start." (missing-input deadlock raise)
#   * "Plans must terminate. No policy may produce a chain that deadlocks
#      with empty queues and missing inputs."
#   * Implicit DAG/ordering invariant: the chain executes in list order,
#      so every input id must resolve to initial_memory OR to an output of
#      an EARLIER task — no self-edges, no forward references.
#
# The simulator catches most of these mid-run via:
#   * simulator.py L396 — `input {inp!r} is not present in pool`
#   * simulator.py L494-497 — `task {id!r} deadlocked at t=...`
# A static prepass should reject them up front with a producer/consumer
# graph reason. The keyword sets below accept either the runtime message
# or the (future) prepass message.
# ---------------------------------------------------------------------------


def make_invalid_topological_unproduced_input() -> TaskChain:
    """A task consumes an id that NO upstream task produces and that is not
    in initial_memory.

    Rule: 'Every task input must be `live` on compute by task start.' If no
    task produces the id and it is not pre-placed, the input can never
    become live — guaranteed deadlock at this task's dispatch.

    Expected validator error includes: 'PHANTOM' (or 'not produced' /
    'unproduced' / 'not present').
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W"],
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
            ),
            Task(
                id="t1",
                inputs=["A0", "PHANTOM"],  # PHANTOM is never produced anywhere
                outputs=[OutputAlloc(id="A1", size=1)],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_topological_self_cycle() -> TaskChain:
    """A task lists its OWN output id in its inputs.

    Rule: the producer-consumer graph must be acyclic; a self-edge is the
    minimal cycle. Outputs only become live at task_end, so the task
    cannot consume an id it has not yet produced.

    Expected validator error includes: 'self' (or 'cycle' / 'own output').
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W", "SELF"],  # SELF is t0's own output, not yet live
                outputs=[OutputAlloc(id="SELF", size=1)],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_topological_forward_reference() -> TaskChain:
    """Task t0 consumes id 'B' which is produced by a LATER task t1, AND
    t1 consumes t0's output 'A' — a length-2 cycle in the producer-consumer
    graph. Even setting the cycle aside, the chain runs in list order, so
    t0's forward reference to B fails at dispatch.

    Rule: every input must be live by task start; for a sequential chain
    this means 'produced by an earlier-indexed task or in initial_memory.'

    Expected validator error includes: 'B' (or 'forward reference' /
    'produced by later' / 'cycle' / 'not present').
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W", "B"],  # B is produced by t1 (later) — forward ref
                outputs=[OutputAlloc(id="A", size=1)],
                runtime=1,
            ),
            Task(
                id="t1",
                inputs=["A"],
                outputs=[OutputAlloc(id="B", size=1)],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_topological_empty_chain() -> TaskChain:
    """A chain with zero tasks.

    EXPECTED BEHAVIOR (documented; not yet decided): the simulator
    currently treats this as a NO-OP — the main loop body never executes
    and an empty EventLog is returned. A static prepass MAY choose to
    reject it as an authoring error (no work scheduled = caller bug), in
    which case the message should include 'empty'. This fixture exists so
    the test bench can pin current behavior (no-op) and flip the
    expectation later.

    Expected validator behavior: NO ERROR today (EXPECTED_ERROR_KEYWORDS
    value is None). Flip to 'empty' when/if a prepass rejects it.
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_topological_duplicate_input_id() -> TaskChain:
    """A task lists the same id twice in its `inputs` list.

    Rule (implicit, well-formedness): a task's input set must be a set —
    duplicates have no defined semantics (does the id count toward the
    input footprint twice? get released twice if also in releases_after?).
    The simulator iterates over `inputs` as a list and silently
    double-counts in some paths, which can mask the bug until much later.

    Expected validator error includes: 'duplicate' (and ideally the
    repeated id 'W').
    """
    return TaskChain(
        initial_memory=[Object(id="W", size=1, location="fast", type="weight")],
        tasks=[
            Task(
                id="t0",
                inputs=["W", "W"],  # duplicate id in inputs list
                outputs=[OutputAlloc(id="A0", size=1)],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


# Skipped: "compute task with backing-located output."
# Verified against schema.py (OutputAlloc.location accepts "backing") and
# simulator.py L501-507 (the main loop explicitly handles backing-located
# outputs via backing_outputs_size capacity check + backing pool allocation).
# This is a valid construct — a compute task whose result is written
# directly to backing — not a violation. No fixture written.


# Topological/deadlock keywords. Some entries are lists: the test bench
# should accept the error if ANY one of the keywords is a (case-
# insensitive) substring of the raised message, since runtime messages
# (today) and prepass messages (tomorrow) phrase the same violation
# differently. A None value means no error is expected (current no-op
# behavior pinned for later flip).
EXPECTED_ERROR_KEYWORDS.update(
    {
        "make_invalid_topological_unproduced_input": [
            "PHANTOM", "not produced", "unproduced", "not present",
        ],
        "make_invalid_topological_self_cycle": ["self", "cycle", "own output"],
        "make_invalid_topological_forward_reference": [
            "forward reference", "produced by later", "cycle", "not present", "'B'",
        ],
        "make_invalid_topological_empty_chain": None,
        "make_invalid_topological_duplicate_input_id": ["duplicate", "'W'"],
    }
)


# ---------------------------------------------------------------------------
# Category: CAPACITY FEASIBILITY
#
# Principles cited (docs/policy/principles.md §1):
#   * "Free compute bytes + scheduled-to-slow reclaim must cover the task's
#      compute-located output footprint at dispatch."
#   * "Backing pool + task's backing-located outputs must fit `backing_memory_capacity` at
#      task start (no backing stall mechanism exists)."
#   * (implicit at t=0) initial_memory at each location must fit its cap.
#   * "Every task input must be `live` on compute by task start."
#
# Each fixture below violates exactly one capacity invariant. These are
# forced-footprint failures: NO policy could schedule the chain successfully
# under the given capacity, because at some moment the minimum required byte
# count exceeds the cap regardless of eviction strategy. Where the simulator
# already raises with a clear message (initial overflow, output reservation
# exceeds cap), the expected keyword captures that message; the input-only
# overflow surfaces as a generic deadlock today and is a candidate for a
# clearer static prepass check.
# ---------------------------------------------------------------------------


def make_invalid_capacity_initial_compute_overflow() -> TaskChain:
    """Initial compute-located objects sum > fast_memory_capacity.

    Rule: capacity feasibility at t=0 — the chain hasn't started a task yet
    and the compute pool already exceeds the cap. Simulator's
    `_check_initial_capacity` raises immediately with a message naming
    `<initial_memory>` and the offending location.

    Expected validator error includes: 'initial_memory'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W0", size=60, location="fast", type="weight"),
            Object(id="W1", size=60, location="fast", type="weight"),
        ],
        tasks=[
            Task(id="t0", inputs=["W0", "W1"], outputs=[], runtime=1),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=200,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_capacity_initial_backing_overflow() -> TaskChain:
    """Initial backing-located objects sum > backing_memory_capacity.

    Rule: capacity feasibility for the HOST pool at t=0. Same
    `_check_initial_capacity` path as the compute variant, but the overflow
    is on the backing side.

    Expected validator error includes: 'initial_memory'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="W0", size=60, location="backing", type="weight"),
            Object(id="W1", size=60, location="backing", type="weight"),
        ],
        tasks=[
            Task(id="t0", inputs=[], outputs=[], runtime=1),
        ],
        fast_memory_capacity=200,
        backing_memory_capacity=100,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_capacity_forced_footprint_exceeds_cap() -> TaskChain:
    """Forced (inputs + outputs) footprint at task t exceeds fast_memory_capacity.

    Rule: "Free compute bytes + scheduled-to-slow reclaim must cover the task's
    compute-located output footprint at dispatch." Task t0 requires A
    (size 50) AND B (size 50) live on compute AND must reserve C (size 50)
    — total 150 bytes against a cap of 100. Inputs cannot be evicted while
    a task is running, so no policy can hide this. With A and B placed on
    compute at t=0 (already 100/100 used), `compute_outputs_ready_t` cannot
    satisfy the 50-byte output reservation and raises.

    Expected validator error includes: 'cannot satisfy fast memory need'.
    """
    return TaskChain(
        initial_memory=[
            Object(id="A", size=50, location="fast", type="weight"),
            Object(id="B", size=50, location="fast", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["A", "B"],
                outputs=[OutputAlloc(id="C", size=50, location="fast", type="activation")],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=200,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_capacity_input_footprint_exceeds_cap() -> TaskChain:
    """A single task's INPUT-only footprint exceeds fast_memory_capacity.

    Rule: "Every task input must be `live` on compute by task start." Task
    t0 needs A (size 60) AND B (size 60) both live on compute
    simultaneously; cap=100, so 120 > 100 is unreachable for any policy.
    Both start on backing with no prefetch triggers attached upstream (there
    is no upstream task here), so the simulator surfaces this as a
    deadlock when `input_ready_t` cannot make both inputs live within the
    cap.

    Expected validator error includes: 'deadlock' (or, if a static
    prepass is added, 'input footprint exceeds fast_memory_capacity').
    """
    return TaskChain(
        initial_memory=[
            Object(id="A", size=60, location="backing", type="weight"),
            Object(id="B", size=60, location="backing", type="weight"),
        ],
        tasks=[
            Task(
                id="t0",
                inputs=["A", "B"],
                outputs=[],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=200,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


def make_invalid_capacity_output_footprint_exceeds_cap() -> TaskChain:
    """A single task's OUTPUT-only footprint exceeds fast_memory_capacity.

    Rule: "Free compute bytes + scheduled-to-slow reclaim must cover the task's
    compute-located output footprint at dispatch." Task t0 must reserve a
    single compute-located output of size 150 against a cap of 100 — no
    eviction scheme can free more than 100 bytes of compute space.
    `compute_outputs_ready_t` walks all scheduled to_slow offloads, finds none,
    and raises.

    Expected validator error includes: 'cannot satisfy fast memory need'.
    """
    return TaskChain(
        initial_memory=[],
        tasks=[
            Task(
                id="t0",
                inputs=[],
                outputs=[OutputAlloc(id="C", size=150, location="fast", type="activation")],
                runtime=1,
            ),
        ],
        fast_memory_capacity=100,
        backing_memory_capacity=200,
        bandwidth_from_slow=1,
        bandwidth_to_slow=1,
    )


EXPECTED_ERROR_KEYWORDS.update(
    {
        "make_invalid_capacity_initial_compute_overflow": "initial_memory",
        "make_invalid_capacity_initial_backing_overflow": "initial_memory",
        "make_invalid_capacity_forced_footprint_exceeds_cap": "cannot satisfy fast memory need",
        "make_invalid_capacity_input_footprint_exceeds_cap": ["deadlock", "input"],
        "make_invalid_capacity_output_footprint_exceeds_cap": "cannot satisfy fast memory need",
    }
)
