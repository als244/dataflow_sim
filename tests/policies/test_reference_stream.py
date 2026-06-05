from dataflow_sim.core.reference_stream import compute_reference_stream, next_ref_time
from dataflow_sim.core.schema import OutputAlloc, Task


def _task(id_: str, inputs, runtime=1, outputs=None, releases=None):
    return Task(
        id=id_,
        inputs=list(inputs),
        outputs=[OutputAlloc(**o) for o in (outputs or [])],
        runtime=runtime,
        releases_after=list(releases or []),
    )


def test_next_ref_finds_first_appearance():
    tasks = [
        _task("a", ["x"]),
        _task("b", ["y"]),
        _task("c", ["x"]),
    ]
    starts = {"a": 0, "b": 1, "c": 2}
    assert next_ref_time("x", tasks, starts) == 0
    assert next_ref_time("y", tasks, starts) == 1
    assert next_ref_time("z", tasks, starts) is None


def test_compute_reference_stream_includes_every_input_first_use():
    tasks = [
        _task("a", ["x"], runtime=10),
        _task("b", ["y", "x"], runtime=5),
        _task("c", ["z"], runtime=2),
    ]
    starts = {"a": 0, "b": 10, "c": 15}
    refs = compute_reference_stream(tasks, starts)
    assert [(r.obj_id, r.ref_t, r.ref_task) for r in refs] == [
        ("x", 0, "a"),
        ("y", 10, "b"),
        ("z", 15, "c"),
    ]


def test_compute_reference_stream_includes_outputs_at_producer_time():
    """An object's first reference is when it's first allocated (as an output)
    or first read (as an input), whichever comes first."""
    tasks = [
        _task("creator", [], runtime=5, outputs=[{"id": "future_obj", "size": 1}]),
        _task("consumer", ["future_obj"], runtime=3),
    ]
    starts = {"creator": 0, "consumer": 5}
    refs = compute_reference_stream(tasks, starts)
    # future_obj first appears as an output of `creator` at t=0
    r = next(r for r in refs if r.obj_id == "future_obj")
    assert r.ref_t == 0
    assert r.ref_task == "creator"


def test_compute_reference_stream_terminal_output_appears():
    """An output that is never consumed downstream should still appear in the
    stream (allocated by its producer)."""
    tasks = [
        _task("producer", [], runtime=3, outputs=[{"id": "dead_output", "size": 1}]),
    ]
    starts = {"producer": 0}
    refs = compute_reference_stream(tasks, starts)
    assert any(r.obj_id == "dead_output" and r.ref_t == 0 for r in refs)


def test_compute_reference_stream_returns_only_first_use():
    tasks = [
        _task("a", ["x"]),
        _task("b", ["x"]),
    ]
    starts = {"a": 0, "b": 1}
    refs = compute_reference_stream(tasks, starts)
    assert len(refs) == 1
    assert refs[0].ref_task == "a"
