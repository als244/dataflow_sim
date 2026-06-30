"""Generic workload schema and compiler.

`DataflowProgram` is the hardware-free public workload format.  It describes
ordered compute tasks over named memory objects.  The simulator still runs the
lower-level `TaskChain`; this module is the bridge from the portable schema to
that concrete IR once hardware is selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataflow_sim.core.schema import (
    Location,
    Object,
    ObjectType,
    OutputAlloc,
    Task,
    TaskChain,
)
from dataflow_sim.workloads.common.hardware import (
    HardwareSpec,
    gbs_to_bytes_per_microsecond,
)
from dataflow_sim.workloads.common.workload import Workload


CostKind = Literal["fixed", "roofline", "sum"]
EfficiencyName = Literal[
    "matmul",
    "matmul_bf16",
    "matmul_fp8",
    "matmul_fp4",
    "attention",
    "attention_fwd",
    "attention_bwd",
    "memory",
    "scale_up",
    "custom",
]


class DataflowMetrics(BaseModel):
    """Optional throughput contract for summaries.

    A generic dataflow program may omit metrics.  Training builders set
    ``primary_unit="tokens"`` with the total token count so the webapp can show
    token throughput without knowing anything model-family-specific.
    """

    model_config = ConfigDict(extra="forbid")

    primary_unit: str = Field(min_length=1)
    primary_count: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataflowCost(BaseModel):
    """Hardware-free cost model for one task or sub-term.

    `fixed` is for measured runtimes.  `roofline` resolves from flops/bytes
    against a selected hardware profile.  `sum` composes several fixed or
    roofline terms into one simulator task, which lets authoring layers expose
    op-level breakdowns without changing the compute stream granularity.
    """

    model_config = ConfigDict(extra="forbid")

    kind: CostKind
    name: str | None = None
    runtime_us: float | None = Field(default=None, ge=0)
    flops: int = Field(default=0, ge=0)
    memory_bytes: int = Field(default=0, ge=0)
    efficiency: EfficiencyName = "memory"
    count: int = Field(default=1, ge=1)
    effective_flops: int | None = Field(default=None, ge=0)
    compute_eff: float | None = Field(default=None, gt=0, le=1)
    mem_eff: float | None = Field(default=None, gt=0, le=1)
    terms: list["DataflowCost"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_kind(self) -> "DataflowCost":
        if self.kind == "fixed":
            if self.runtime_us is None:
                raise ValueError("fixed cost requires runtime_us")
            if self.terms:
                raise ValueError("fixed cost cannot contain terms")
        elif self.kind == "roofline":
            if self.runtime_us is not None:
                raise ValueError("roofline cost cannot set runtime_us")
            if self.terms:
                raise ValueError("roofline cost cannot contain terms")
            if self.flops == 0 and self.memory_bytes == 0:
                raise ValueError("roofline cost requires flops or memory_bytes")
            if self.efficiency == "custom" and self.compute_eff is None and self.flops > 0:
                raise ValueError("custom compute roofline cost requires compute_eff")
        elif self.kind == "sum":
            if self.runtime_us is not None:
                raise ValueError("sum cost cannot set runtime_us")
            if not self.terms:
                raise ValueError("sum cost requires at least one term")
        return self


class ComputeBlock(BaseModel):
    """Reusable structural compute block.

    ``key`` is the stable identity used by tasks.  ``name`` is display text.
    ``subops`` are the fixed/roofline terms that resolve to one simulator task
    per task instance.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = Field(default="compute", min_length=1)
    subops: list[DataflowCost] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataflowObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    initial_location: Location = "backing"
    role: str = Field(default="other", min_length=1)


class DataflowOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    location: Location = "fast"
    role: str = Field(default="other", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataflowTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str | None = None
    group: str = Field(default="compute", min_length=1)
    compute_block_key: str | None = Field(default=None, min_length=1)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[DataflowOutput] = Field(default_factory=list)
    mutates: list[str] = Field(default_factory=list)
    cost: DataflowCost | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_task(self) -> "DataflowTask":
        if (self.compute_block_key is None) == (self.cost is None):
            raise ValueError(
                f"task {self.id!r} must set exactly one of compute_block_key or cost"
            )
        seen_inputs: set[str] = set()
        for obj_id in self.inputs:
            if obj_id in seen_inputs:
                raise ValueError(f"task {self.id!r} lists input {obj_id!r} more than once")
            seen_inputs.add(obj_id)
        for obj_id in self.mutates:
            if obj_id not in seen_inputs:
                raise ValueError(
                    f"task {self.id!r} mutates {obj_id!r}, which is not an input"
                )
        seen_outputs: set[str] = set()
        for out in self.outputs:
            if out.id in seen_outputs:
                raise ValueError(
                    f"task {self.id!r} declares output {out.id!r} more than once"
                )
            seen_outputs.add(out.id)
        return self


class DataflowProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dataflow/v1"] = "dataflow/v1"
    name: str = Field(min_length=1)
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: DataflowMetrics | None = None
    objects: list[DataflowObject] = Field(default_factory=list)
    compute_blocks: list[ComputeBlock] = Field(default_factory=list)
    tasks: list[DataflowTask] = Field(min_length=1)
    final_locations: dict[str, Location] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_references(self) -> "DataflowProgram":
        block_keys: set[str] = set()
        for block in self.compute_blocks:
            if block.key in block_keys:
                raise ValueError(f"duplicate compute block key {block.key!r}")
            block_keys.add(block.key)

        known: set[str] = set()
        for obj in self.objects:
            if obj.id in known:
                raise ValueError(f"duplicate object id {obj.id!r}")
            known.add(obj.id)

        task_ids: set[str] = set()
        task_labels: set[str] = set()
        for task in self.tasks:
            if task.id in task_ids:
                raise ValueError(f"duplicate task id {task.id!r}")
            task_ids.add(task.id)
            if task.label is not None:
                if task.label in task_labels:
                    raise ValueError(f"duplicate task label {task.label!r}")
                task_labels.add(task.label)
            if task.compute_block_key is not None and task.compute_block_key not in block_keys:
                raise ValueError(
                    f"task {task.id!r} references unknown compute block "
                    f"{task.compute_block_key!r}"
                )
            for obj_id in task.inputs:
                if obj_id not in known:
                    raise ValueError(
                        f"task {task.id!r} references unknown input {obj_id!r}"
                    )
            output_ids = {out.id for out in task.outputs}
            for obj_id in task.inputs:
                if obj_id in output_ids:
                    raise ValueError(
                        f"task {task.id!r} consumes its own output {obj_id!r}"
                    )
            for out in task.outputs:
                if out.id in known:
                    raise ValueError(
                        f"task {task.id!r} output {out.id!r} collides with an existing object"
                    )
            known.update(output_ids)

        for obj_id in self.final_locations:
            if obj_id not in known:
                raise ValueError(
                    f"final_locations references unknown object {obj_id!r}"
                )
        return self


DataflowCost.model_rebuild()
ComputeBlock.model_rebuild()


@dataclass(frozen=True)
class ResolvedCost:
    runtime_us: float
    rows: list[dict[str, Any]]


def _cost_terms(cost: DataflowCost) -> list[DataflowCost]:
    if cost.kind == "sum":
        return list(cost.terms)
    return [cost]


def _block_cost(block: ComputeBlock) -> DataflowCost:
    return DataflowCost(kind="sum", name=block.name, terms=list(block.subops))


def normalize_dataflow_program(program: DataflowProgram) -> DataflowProgram:
    """Return a schema where every task references a compute block.

    This is the compiler boundary: authoring may use inline ``task.cost`` for
    convenience, but preview/simulation metadata is always block-based.
    """
    blocks = list(program.compute_blocks)
    block_keys = {block.key for block in blocks}
    tasks: list[DataflowTask] = []
    for task in program.tasks:
        if task.compute_block_key is not None:
            tasks.append(task)
            continue
        assert task.cost is not None
        key = f"inline:{task.id}"
        if key in block_keys:
            raise ValueError(f"generated inline compute block key {key!r} collides")
        block_keys.add(key)
        blocks.append(
            ComputeBlock(
                key=key,
                name=task.label or task.id,
                category=task.group,
                subops=_cost_terms(task.cost),
                metadata={"generated_from_task": task.id},
            )
        )
        tasks.append(
            task.model_copy(update={"compute_block_key": key, "cost": None})
        )

    return program.model_copy(update={"compute_blocks": blocks, "tasks": tasks})


def role_to_object_type(role: str) -> ObjectType:
    key = role.strip().lower().replace("-", "_")
    if key in {"weight", "weights", "parameter", "parameters", "param"}:
        return "weight"
    if key in {"activation", "activations", "input", "output", "tensor"}:
        return "activation"
    if key in {"gradient", "grad", "grads"}:
        return "gradient"
    if key in {"optimizer", "optimizer_state", "optimizer_states"}:
        return "optimizer"
    return "other"


def _eff_value(cost: DataflowCost, hw: HardwareSpec) -> float:
    if cost.compute_eff is not None:
        return cost.compute_eff
    if cost.efficiency in {"matmul", "matmul_bf16"}:
        return hw.matmul_eff_bf16
    if cost.efficiency == "matmul_fp8":
        return hw.matmul_eff_fp8
    if cost.efficiency == "matmul_fp4":
        if hw.matmul_eff_fp4 is None:
            raise ValueError("hardware does not define FP4 matmul efficiency")
        return hw.matmul_eff_fp4
    if cost.efficiency in {"attention", "attention_fwd"}:
        return hw.attn_fwd_eff
    if cost.efficiency == "attention_bwd":
        return hw.attn_bwd_eff
    if cost.efficiency in {"memory", "scale_up"}:
        return 1.0
    raise ValueError("custom roofline cost requires compute_eff")


def _peak_tflops(cost: DataflowCost, hw: HardwareSpec) -> float:
    if cost.efficiency == "matmul_bf16":
        return hw.peak_tflops_bf16
    if cost.efficiency == "matmul_fp8":
        return hw.peak_tflops_fp8
    if cost.efficiency == "matmul_fp4":
        if hw.peak_tflops_fp4 is None:
            raise ValueError("hardware does not define FP4 peak TFLOP/s")
        return hw.peak_tflops_fp4
    return hw.peak_tflops_bf16


def _timing_row(
    *,
    name: str,
    kind: Literal["compute", "memory"],
    flops: int,
    effective_flops: int,
    bytes_: int,
    count: int,
    math_us: float | None,
    mem_us: float,
    per_call_us: float,
    per_call_us_exact: float,
    total_us: float,
    bound_by: Literal["compute", "memory"],
    effective_tflops: float | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "flops": flops,
        "effective_flops": effective_flops,
        "bytes": bytes_,
        "count": count,
        "math_us": math_us,
        "mem_us": mem_us,
        "per_call_us": per_call_us,
        "per_call_us_exact": per_call_us_exact,
        "total_us": total_us,
        "bound_by": bound_by,
        "effective_tflops": effective_tflops,
    }


def resolve_cost(cost: DataflowCost, hw: HardwareSpec, *, default_name: str) -> ResolvedCost:
    if cost.kind == "sum":
        rows: list[dict[str, Any]] = []
        total_us = 0
        for idx, term in enumerate(cost.terms):
            name = term.name or f"{default_name}.{idx}"
            resolved = resolve_cost(term, hw, default_name=name)
            total_us += resolved.runtime_us
            rows.extend(resolved.rows)
        return ResolvedCost(runtime_us=total_us, rows=rows)

    name = cost.name or default_name
    if cost.kind == "fixed":
        assert cost.runtime_us is not None
        per_call_us = cost.runtime_us
        total_us = per_call_us * cost.count
        effective_flops = cost.effective_flops if cost.effective_flops is not None else cost.flops
        return ResolvedCost(
            runtime_us=total_us,
            rows=[
                _timing_row(
                    name=name,
                    kind="compute" if cost.flops > 0 else "memory",
                    flops=cost.flops,
                    effective_flops=effective_flops,
                    bytes_=cost.memory_bytes,
                    count=cost.count,
                    math_us=per_call_us if cost.flops > 0 else None,
                    mem_us=per_call_us,
                    per_call_us=per_call_us,
                    per_call_us_exact=float(per_call_us),
                    total_us=total_us,
                    bound_by="compute" if cost.flops > 0 else "memory",
                    effective_tflops=None,
                )
            ],
        )

    eff_flops = cost.effective_flops if cost.effective_flops is not None else cost.flops
    mem_eff = cost.mem_eff if cost.mem_eff is not None else hw.mem_eff
    memory_bw_gbs = (
        hw.scale_up_bw_gbs if cost.efficiency == "scale_up" else hw.fast_memory_bw_gbs
    )
    lane_mem_eff = 1.0 if cost.efficiency == "scale_up" else mem_eff
    if cost.memory_bytes > 0 and memory_bw_gbs > 0 and lane_mem_eff > 0:
        mem_seconds = cost.memory_bytes / (memory_bw_gbs * 1e9 * lane_mem_eff)
        mem_us_exact = mem_seconds * 1e6
        mem_us = mem_us_exact
    else:
        mem_seconds = 0.0
        mem_us_exact = 0.0
        mem_us = 0

    if cost.flops <= 0:
        per_call_us = mem_us
        total_us = per_call_us * cost.count
        return ResolvedCost(
            runtime_us=total_us,
            rows=[
                _timing_row(
                    name=name,
                    kind="memory",
                    flops=0,
                    effective_flops=0,
                    bytes_=cost.memory_bytes,
                    count=cost.count,
                    math_us=None,
                    mem_us=mem_us,
                    per_call_us=per_call_us,
                    per_call_us_exact=mem_us_exact,
                    total_us=total_us,
                    bound_by="memory",
                    effective_tflops=None,
                )
            ],
        )

    eff = _eff_value(cost, hw)
    math_seconds = cost.flops / (_peak_tflops(cost, hw) * 1e12 * eff)
    math_us_exact = math_seconds * 1e6
    math_us = math_us_exact
    per_call_us = max(math_us, mem_us)
    per_call_us_exact = max(math_us_exact, mem_us_exact)
    total_us = per_call_us * cost.count
    bound_by: Literal["compute", "memory"] = (
        "memory" if mem_us_exact > math_us_exact else "compute"
    )
    binding_seconds = max(math_seconds, mem_seconds)
    effective_tflops = (
        eff_flops / (binding_seconds * 1e12) if binding_seconds > 0 else 0.0
    )
    return ResolvedCost(
        runtime_us=total_us,
        rows=[
            _timing_row(
                name=name,
                kind="compute",
                flops=cost.flops,
                effective_flops=eff_flops,
                bytes_=cost.memory_bytes,
                count=cost.count,
                math_us=math_us,
                mem_us=mem_us,
                per_call_us=per_call_us,
                per_call_us_exact=per_call_us_exact,
                total_us=total_us,
                bound_by=bound_by,
                effective_tflops=effective_tflops,
            )
        ],
    )


def _generic_breakdown(
    block_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    total_us = sum(b["total_runtime_us"] for b in block_summaries)
    return {
        "compute_blocks": block_summaries,
        # Transitional section fields retained for callers that have not yet
        # moved to compute_blocks. New UI/API consumers should use
        # breakdown["compute_blocks"].
        "fwd": block_summaries[0]["subops"] if block_summaries else [],
        "bwd": [],
        "head": [],
        "optimizer": [],
        "totals_us": {
            "layer_fwd": total_us,
            "layer_bwd": 0,
            "head": 0,
            "optimizer_step": 0,
            "layer_recompute": 0,
        },
    }


def _sum_row_field(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row[key]) * int(row.get("count", 1)) for row in rows)


def _block_bound_by(rows: list[dict[str, Any]]) -> str:
    compute_us = sum(row["total_us"] for row in rows if row["bound_by"] == "compute")
    memory_us = sum(row["total_us"] for row in rows if row["bound_by"] == "memory")
    if compute_us == 0 and memory_us == 0:
        return "none"
    return "compute" if compute_us >= memory_us else "memory"


def _tflops(flops: int, runtime_us: float) -> float | None:
    if flops <= 0 or runtime_us <= 0:
        return None
    return flops / (runtime_us * 1e-6) / 1e12


def _build_block_summaries(
    program: DataflowProgram,
    resolved_by_key: dict[str, ResolvedCost],
    tasks_by_key: dict[str, list[DataflowTask]],
) -> list[dict[str, Any]]:
    block_by_key = {block.key: block for block in program.compute_blocks}
    summaries: list[dict[str, Any]] = []
    for block in program.compute_blocks:
        tasks = tasks_by_key.get(block.key, [])
        if not tasks:
            continue
        resolved = resolved_by_key[block.key]
        rows = resolved.rows
        instance_count = len(tasks)
        per_flops = _sum_row_field(rows, "flops")
        per_eff_flops = _sum_row_field(rows, "effective_flops")
        per_bytes = _sum_row_field(rows, "bytes")
        total_runtime_us = resolved.runtime_us * instance_count
        total_flops = per_flops * instance_count
        total_eff_flops = per_eff_flops * instance_count
        summaries.append(
            {
                "key": block.key,
                "name": block.name,
                "category": block.category,
                "instance_count": instance_count,
                "per_instance_runtime_us": resolved.runtime_us,
                "total_runtime_us": total_runtime_us,
                "per_instance_flops": per_flops,
                "total_flops": total_flops,
                "per_instance_effective_flops": per_eff_flops,
                "total_effective_flops": total_eff_flops,
                "per_instance_bytes": per_bytes,
                "total_bytes": per_bytes * instance_count,
                "hardware_tflops": _tflops(total_flops, total_runtime_us),
                "effective_tflops": _tflops(total_eff_flops, total_runtime_us),
                "bound_by": _block_bound_by(rows),
                "subops": rows,
                "task_ids": [task.id for task in tasks],
                "task_labels": [task.label or task.id for task in tasks],
                "metadata": block_by_key[block.key].metadata,
            }
        )
    return summaries


def preview_dataflow_program(program: DataflowProgram) -> dict[str, Any]:
    program = normalize_dataflow_program(program)
    role_bytes: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    block_counts: dict[str, int] = {}
    initial_fast_bytes = 0
    initial_backing_bytes = 0
    output_bytes = 0

    for obj in program.objects:
        role_bytes[obj.role] = role_bytes.get(obj.role, 0) + obj.size_bytes
        if obj.initial_location == "fast":
            initial_fast_bytes += obj.size_bytes
        else:
            initial_backing_bytes += obj.size_bytes

    for task in program.tasks:
        group_counts[task.group] = group_counts.get(task.group, 0) + 1
        assert task.compute_block_key is not None
        block_counts[task.compute_block_key] = block_counts.get(task.compute_block_key, 0) + 1
        for out in task.outputs:
            role_bytes[out.role] = role_bytes.get(out.role, 0) + out.size_bytes
            output_bytes += out.size_bytes

    return {
        "name": program.name,
        "description": program.description,
        "schema_version": program.schema_version,
        "object_count": len(program.objects) + sum(len(s.outputs) for s in program.tasks),
        "initial_object_count": len(program.objects),
        "task_count": len(program.tasks),
        "initial_fast_bytes": initial_fast_bytes,
        "initial_backing_bytes": initial_backing_bytes,
        "output_bytes": output_bytes,
        "role_bytes": role_bytes,
        "group_counts": group_counts,
        "compute_block_count": len(program.compute_blocks),
        "compute_block_instance_counts": block_counts,
        "metrics": (
            program.metrics.model_dump(mode="json")
            if program.metrics is not None
            else None
        ),
    }


def realize_dataflow_program(program: DataflowProgram, hw: HardwareSpec) -> Workload:
    program = normalize_dataflow_program(program)
    initial = [
        Object(
            id=obj.id,
            size=obj.size_bytes,
            location=obj.initial_location,
            type=role_to_object_type(obj.role),
        )
        for obj in program.objects
    ]

    chain_tasks: list[Task] = []
    task_summaries: list[dict[str, Any]] = []
    block_by_key = {block.key: block for block in program.compute_blocks}
    resolved_by_key = {
        block.key: resolve_cost(_block_cost(block), hw, default_name=block.name)
        for block in program.compute_blocks
    }
    tasks_by_key: dict[str, list[DataflowTask]] = {
        block.key: [] for block in program.compute_blocks
    }
    for task in program.tasks:
        assert task.compute_block_key is not None
        block = block_by_key[task.compute_block_key]
        resolved = resolved_by_key[task.compute_block_key]
        tasks_by_key[task.compute_block_key].append(task)
        chain_tasks.append(
            Task(
                id=task.id,
                inputs=list(task.inputs),
                outputs=[
                    OutputAlloc(
                        id=out.id,
                        size=out.size_bytes,
                        location=out.location,
                        type=role_to_object_type(out.role),
                    )
                    for out in task.outputs
                ],
                runtime=resolved.runtime_us,
                mutates_inputs=list(task.mutates),
            )
        )
        task_summaries.append(
            {
                "id": task.id,
                "label": task.label or task.id,
                "group": task.group,
                "compute_block_key": task.compute_block_key,
                "compute_block_name": block.name,
                "runtime_us": resolved.runtime_us,
                "inputs": len(task.inputs),
                "outputs": len(task.outputs),
            }
        )

    chain = TaskChain(
        initial_memory=initial,
        tasks=chain_tasks,
        final_locations=dict(program.final_locations),
        bandwidth_from_slow=gbs_to_bytes_per_microsecond(hw.from_slow_bw_gbs),
        bandwidth_to_slow=gbs_to_bytes_per_microsecond(hw.to_slow_bw_gbs),
        fast_memory_capacity=None,
    )
    block_summaries = _build_block_summaries(program, resolved_by_key, tasks_by_key)
    breakdown = _generic_breakdown(block_summaries)
    metrics = (
        program.metrics.model_dump(mode="json")
        if program.metrics is not None
        else None
    )
    preview = preview_dataflow_program(program)
    preview["aggregate_task_runtime_us"] = sum(
        task["runtime_us"] for task in task_summaries
    )
    return Workload(
        chain=chain,
        metadata={
            "program": program.model_dump(mode="json"),
            "preview": preview,
            "breakdown": breakdown,
            "compute_blocks": block_summaries,
            "task_summaries": task_summaries,
            "metrics": metrics,
            "summary": {
                "kind": "dataflow",
                "metrics": metrics,
            },
        },
    )
