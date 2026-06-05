import pytest

from dataflow_sim.engine.simulator import run
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.training.transformer import build_layerwise_training_chain


def _pressurefit_chain(*args, **kwargs):
    return apply_pressurefit_policy(build_layerwise_training_chain(*args, **kwargs))


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_training_chain_runs_clean(L):
    chain = _pressurefit_chain(L, fwd_runtime=10)
    log = run(chain)
    compute_intervals = [iv for iv in log.task_intervals if iv.track == "compute"]
    assert len(compute_intervals) == 3 * L + 1


def test_bare_chain_uses_step_accum_layer_task_ids():
    bare = build_layerwise_training_chain(L=2, grad_accum_rounds=2, num_steps=2)
    ids = [t.id for t in bare.tasks]
    assert ids[:7] == [
        "f_0_0_0", "f_0_0_1", "head_0_0",
        "r_0_0_1", "b_0_0_1", "r_0_0_0", "b_0_0_0",
    ]
    assert ids[-7:] == [
        "f_1_1_0", "f_1_1_1", "head_1_1",
        "r_1_1_1", "b_1_1_1", "r_1_1_0", "b_1_1_0",
    ]


def test_inputs_are_per_step_and_accumulation_round():
    bare = build_layerwise_training_chain(L=2, grad_accum_rounds=2, num_steps=2)
    initial = {o.id: o for o in bare.initial_memory}
    assert initial["input_0_0"].location == "device"
    assert initial["input_0_1"].location == "host"
    assert initial["input_1_0"].location == "host"
    assert initial["input_1_1"].location == "host"


def test_step_gradients_are_produced_then_mutated_within_step():
    bare = build_layerwise_training_chain(L=2, grad_accum_rounds=2, num_steps=2)

    first_accum = next(t for t in bare.tasks if t.id == "b_1_0_1")
    second_accum = next(t for t in bare.tasks if t.id == "b_1_1_1")
    assert first_accum.inputs == ["dy_head_1_0", "A_1_0_1", "W_1"]
    assert {out.id for out in first_accum.outputs} == {"dy_1_0_1", "dW_1_1"}
    assert first_accum.mutates_inputs == []

    assert second_accum.inputs == ["dy_head_1_1", "A_1_1_1", "W_1", "dW_1_1"]
    assert [out.id for out in second_accum.outputs] == ["dy_1_1_1"]
    assert second_accum.mutates_inputs == ["dW_1_1"]


def test_optimizer_steps_are_per_training_step_and_mutate_persistent_state():
    bare = build_layerwise_training_chain(
        L=2,
        grad_accum_rounds=2,
        num_steps=2,
        optimizer_state_size=128,
        optimizer_runtime=7,
    )

    assert [t.id for t in bare.tasks[-2:]] == ["step_1_0", "step_1_1"]
    step = next(t for t in bare.tasks if t.id == "step_1_1")
    assert step.inputs == ["dW_1_1", "W_1", "O_1"]
    assert step.outputs == []
    assert step.runtime == 7
    assert step.mutates_inputs == ["W_1", "O_1"]

    assert bare.final_locations == {}
    finalized = build_layerwise_training_chain(
        L=2,
        grad_accum_rounds=2,
        num_steps=2,
        optimizer_state_size=128,
        optimizer_runtime=7,
        final_model_state_on_host=True,
    )
    assert finalized.final_locations == {
        "W_0": "host", "O_0": "host",
        "W_1": "host", "O_1": "host",
    }


def test_backward_runtime_is_2x_forward_and_r_is_zero():
    chain = _pressurefit_chain(L=2, fwd_runtime=10)
    log = run(chain)
    durations = {
        iv.task_id: iv.end - iv.start
        for iv in log.task_intervals
        if iv.track == "compute"
    }
    for i in range(2):
        assert durations[f"f_0_0_{i}"] == 10
        assert durations[f"b_0_0_{i}"] == 20
        assert durations[f"r_0_0_{i}"] == 0


def test_r_i_immediately_precedes_b_i():
    chain = _pressurefit_chain(L=3)
    log = run(chain)
    order = [iv.task_id for iv in log.task_intervals if iv.track == "compute"]
    for i in range(3):
        ri = order.index(f"r_0_0_{i}")
        bi = order.index(f"b_0_0_{i}")
        assert bi == ri + 1, f"r_0_0_{i} should immediately precede b_0_0_{i}"


def test_activations_released_from_device_after_their_backward():
    chain = _pressurefit_chain(L=3)
    log = run(chain)
    final_device_ids = {m.id for m in log.events[-1].snapshot.memory if m.location == "device"}
    assert final_device_ids.isdisjoint({"A_0_0_0", "A_0_0_1", "A_0_0_2"})
