import pytest
from pydantic import ValidationError

from dataflow_sim.core.validate import validate_chain
from dataflow_sim.workloads.common.hardware import HardwareSpec
from dataflow_sim.workloads.dataflow import (
    DataflowProgram,
    normalize_dataflow_program,
    preview_dataflow_program,
    realize_dataflow_program,
)


def _hw() -> HardwareSpec:
    return HardwareSpec(
        peak_tflops_bf16=100,
        peak_tflops_fp8=200,
        peak_tflops_fp4=400,
        fast_memory_bw_gbs=1000,
        from_slow_bw_gbs=50,
        to_slow_bw_gbs=40,
        matmul_eff_bf16=0.8,
        matmul_eff_fp8=0.8,
        matmul_eff_fp4=0.8,
        attn_fwd_eff=0.7,
        attn_bwd_eff=0.6,
        mem_eff=0.9,
    )


def _program(**overrides):
    body = {
        "schema_version": "dataflow/v1",
        "name": "generic-test",
        "description": "",
        "metadata": {},
        "objects": [
            {
                "id": "x",
                "size_bytes": 1024,
                "initial_location": "fast",
                "role": "input",
            }
        ],
        "tasks": [
            {
                "id": "op0",
                "label": "op0",
                "group": "stage0",
                "inputs": ["x"],
                "outputs": [
                    {"id": "y", "size_bytes": 2048, "role": "activation"}
                ],
                "cost": {
                    "kind": "sum",
                    "terms": [
                        {"kind": "fixed", "name": "measured", "runtime_us": 3},
                        {
                            "kind": "roofline",
                            "name": "matmul",
                            "flops": 1_000_000,
                            "memory_bytes": 128_000,
                            "efficiency": "matmul",
                        },
                    ],
                },
            }
        ],
        "final_locations": {},
    }
    body.update(overrides)
    return body


def test_generic_program_realizes_to_valid_task_chain():
    program = DataflowProgram.model_validate(_program())
    workload = realize_dataflow_program(program, _hw())

    chain = workload.chain
    assert chain.bandwidth_from_slow == 50_000
    assert chain.bandwidth_to_slow == 40_000
    assert chain.initial_memory[0].type == "activation"
    assert chain.tasks[0].id == "op0"
    assert chain.tasks[0].runtime > 3
    validate_chain(chain)


def test_preview_counts_roles_and_groups():
    preview = preview_dataflow_program(DataflowProgram.model_validate(_program()))

    assert preview["name"] == "generic-test"
    assert preview["task_count"] == 1
    assert preview["group_counts"] == {"stage0": 1}
    assert preview["role_bytes"]["input"] == 1024
    assert preview["role_bytes"]["activation"] == 2048
    assert preview["compute_block_count"] == 1
    assert preview["compute_block_instance_counts"] == {"inline:op0": 1}


def test_inline_cost_normalizes_to_one_off_compute_block():
    program = normalize_dataflow_program(DataflowProgram.model_validate(_program()))

    assert program.tasks[0].cost is None
    assert program.tasks[0].compute_block_key == "inline:op0"
    assert program.compute_blocks[0].key == "inline:op0"
    assert program.compute_blocks[0].subops[0].name == "measured"


def test_compute_block_summary_reports_total_effective_tflops():
    workload = realize_dataflow_program(DataflowProgram.model_validate(_program()), _hw())
    block = workload.metadata["compute_blocks"][0]

    expected = (
        block["total_effective_flops"]
        / (block["total_runtime_us"] * 1e-6)
        / 1e12
    )
    assert block["effective_tflops"] == pytest.approx(expected)


def test_unsupported_fp4_matmul_hardware_fails_clearly():
    body = _program()
    body["tasks"][0]["cost"]["terms"][1]["efficiency"] = "matmul_fp4"
    hw = HardwareSpec(
        peak_tflops_bf16=100,
        peak_tflops_fp8=200,
        peak_tflops_fp4=None,
        fast_memory_bw_gbs=1000,
        from_slow_bw_gbs=50,
        to_slow_bw_gbs=40,
        matmul_eff_bf16=0.8,
        matmul_eff_fp8=0.8,
        matmul_eff_fp4=None,
        attn_fwd_eff=0.7,
        attn_bwd_eff=0.6,
        mem_eff=0.9,
    )

    with pytest.raises(ValueError, match="FP4"):
        realize_dataflow_program(DataflowProgram.model_validate(body), hw)


def test_repeated_tasks_share_compute_block_summary():
    program = DataflowProgram.model_validate(
        _program(
            compute_blocks=[
                {
                    "key": "shared_block",
                    "name": "Shared Block",
                    "category": "stage",
                    "subops": [{"kind": "fixed", "name": "measured", "runtime_us": 5}],
                }
            ],
            tasks=[
                {
                    "id": "op0",
                    "label": "First Op",
                    "group": "stage",
                    "compute_block_key": "shared_block",
                    "inputs": ["x"],
                    "outputs": [{"id": "y", "size_bytes": 2048, "role": "activation"}],
                },
                {
                    "id": "op1",
                    "label": "Second Op",
                    "group": "stage",
                    "compute_block_key": "shared_block",
                    "inputs": ["y"],
                    "outputs": [{"id": "z", "size_bytes": 2048, "role": "activation"}],
                },
            ],
        )
    )

    workload = realize_dataflow_program(program, _hw())
    blocks = workload.metadata["compute_blocks"]
    assert len(blocks) == 1
    assert blocks[0]["key"] == "shared_block"
    assert blocks[0]["instance_count"] == 2
    assert blocks[0]["task_ids"] == ["op0", "op1"]
    assert [task.id for task in workload.chain.tasks] == ["op0", "op1"]


def test_metrics_preview_and_summary_metadata():
    body = _program(
        metrics={
            "primary_unit": "items",
            "primary_count": 12,
            "metadata": {"batch": 3},
        }
    )
    program = DataflowProgram.model_validate(body)
    preview = preview_dataflow_program(program)
    workload = realize_dataflow_program(program, _hw())

    assert preview["metrics"]["primary_unit"] == "items"
    assert workload.metadata["summary"]["metrics"]["primary_count"] == 12


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"objects": [
            {"id": "x", "size_bytes": 1, "initial_location": "fast", "role": "input"},
            {"id": "x", "size_bytes": 1, "initial_location": "fast", "role": "input"},
        ]}, "duplicate object id"),
        ({"compute_blocks": [
            {
                "key": "dup",
                "name": "Dup",
                "category": "stage",
                "subops": [{"kind": "fixed", "runtime_us": 1}],
            },
            {
                "key": "dup",
                "name": "Dup Again",
                "category": "stage",
                "subops": [{"kind": "fixed", "runtime_us": 1}],
            },
        ]}, "duplicate compute block key"),
        ({"tasks": [
            {
                "id": "bad",
                "label": "Bad",
                "inputs": ["x"],
                "outputs": [],
                "compute_block_key": "missing_block",
            }
        ]}, "unknown compute block"),
        ({"tasks": [
            {
                "id": "bad",
                "label": "Bad",
                "inputs": ["x"],
                "outputs": [],
            }
        ]}, "exactly one of compute_block_key or cost"),
        ({"tasks": [
            {
                "id": "dup",
                "label": "First",
                "inputs": ["x"],
                "outputs": [{"id": "y", "size_bytes": 1}],
                "cost": {"kind": "fixed", "runtime_us": 1},
            },
            {
                "id": "dup",
                "label": "Second",
                "inputs": ["y"],
                "outputs": [{"id": "z", "size_bytes": 1}],
                "cost": {"kind": "fixed", "runtime_us": 1},
            },
        ]}, "duplicate task id"),
        ({"tasks": [
            {
                "id": "a",
                "label": "Dup Label",
                "inputs": ["x"],
                "outputs": [{"id": "y", "size_bytes": 1}],
                "cost": {"kind": "fixed", "runtime_us": 1},
            },
            {
                "id": "b",
                "label": "Dup Label",
                "inputs": ["y"],
                "outputs": [{"id": "z", "size_bytes": 1}],
                "cost": {"kind": "fixed", "runtime_us": 1},
            },
        ]}, "duplicate task label"),
        ({"tasks": [
            {
                "id": "bad",
                "inputs": ["missing"],
                "outputs": [],
                "cost": {"kind": "fixed", "runtime_us": 1},
            }
        ]}, "unknown input"),
        ({"tasks": [
            {
                "id": "bad",
                "inputs": ["x"],
                "outputs": [],
                "mutates": ["y"],
                "cost": {"kind": "fixed", "runtime_us": 1},
            }
        ]}, "mutates"),
        ({"tasks": [
            {
                "id": "bad",
                "inputs": ["x"],
                "outputs": [{"id": "y", "size_bytes": 1}, {"id": "y", "size_bytes": 1}],
                "cost": {"kind": "fixed", "runtime_us": 1},
            }
        ]}, "declares output"),
        ({"tasks": [
            {
                "id": "bad",
                "inputs": ["x"],
                "outputs": [],
                "cost": {"kind": "roofline", "efficiency": "matmul"},
            }
        ]}, "requires flops or memory_bytes"),
    ],
)
def test_program_validation_errors_are_specific(patch, message):
    with pytest.raises(ValidationError, match=message):
        DataflowProgram.model_validate(_program(**patch))
