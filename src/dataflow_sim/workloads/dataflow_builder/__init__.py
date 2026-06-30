"""Symbolic builder utilities for producing ``DataflowProgram`` objects."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal

from dataflow_sim.workloads.dataflow import (
    ComputeBlock,
    DataflowCost,
    DataflowMetrics,
    DataflowObject,
    DataflowOutput,
    DataflowProgram,
    DataflowTask,
)


DType = Literal["fp4", "fp8", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"]
ComputePrecision = Literal["bf16", "fp8", "fp4"]
TensorRole = Literal["input", "activation", "parameter", "gradient", "optimizer_state", "output", "other"]
OptimizerMode = Literal["none", "adamw", "muon", "sgd"]


_DTYPE_BYTES = {
    "fp4": 0.5,
    "fp8": 1,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}


def normalize_dtype(dtype: str) -> str:
    key = dtype.strip().lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if key == "bfloat16":
        return "bf16"
    if key == "float16":
        return "fp16"
    if key == "float32":
        return "fp32"
    return key


def dtype_nbytes(dtype: str) -> float:
    return _DTYPE_BYTES[normalize_dtype(dtype)]


def normalize_compute_precision(precision: str) -> ComputePrecision:
    key = normalize_dtype(precision)
    if key not in {"bf16", "fp8", "fp4"}:
        raise ValueError(f"unsupported compute precision {precision!r}")
    return key  # type: ignore[return-value]


@dataclass(frozen=True)
class DTypePolicy:
    param: str = "bf16"
    activation: str = "bf16"
    expert_dispatch: str = "bf16"
    gradient: str = "bf16"
    optimizer_state: str = "bf16"
    compute: str = "bf16"
    expert_param: str = "bf16"
    expert_compute: str = "bf16"

    def dtype_for_role(self, role: str) -> str:
        if role in {"parameter", "param", "weight"}:
            return normalize_dtype(self.param)
        if role in {"gradient", "grad"}:
            return normalize_dtype(self.gradient)
        if role in {"optimizer_state", "optimizer"}:
            return normalize_dtype(self.optimizer_state)
        return normalize_dtype(self.activation)

    @property
    def compute_precision(self) -> ComputePrecision:
        return normalize_compute_precision(self.compute)

    @property
    def expert_compute_precision(self) -> ComputePrecision:
        return normalize_compute_precision(self.expert_compute)


@dataclass(frozen=True)
class OpDTypePolicy:
    activation_bpe: float = 2
    expert_dispatch_bpe: float = 2
    weight_bpe: float = 2
    gradient_bpe: float = 2
    optimizer_state_bpe: float = 2
    expert_weight_bpe: float = 2
    compute_precision: ComputePrecision = "bf16"
    expert_compute_precision: ComputePrecision = "bf16"

    @classmethod
    def from_dtype_policy(cls, policy: DTypePolicy) -> "OpDTypePolicy":
        return cls(
            activation_bpe=dtype_nbytes(policy.activation),
            expert_dispatch_bpe=dtype_nbytes(policy.expert_dispatch),
            weight_bpe=dtype_nbytes(policy.param),
            gradient_bpe=dtype_nbytes(policy.gradient),
            optimizer_state_bpe=dtype_nbytes(policy.optimizer_state),
            expert_weight_bpe=dtype_nbytes(policy.expert_param),
            compute_precision=policy.compute_precision,
            expert_compute_precision=policy.expert_compute_precision,
        )

    @classmethod
    def from_single_bpe(cls, bytes_per_element: int) -> "OpDTypePolicy":
        return cls(
            activation_bpe=bytes_per_element,
            expert_dispatch_bpe=bytes_per_element,
            weight_bpe=bytes_per_element,
            gradient_bpe=bytes_per_element,
            optimizer_state_bpe=bytes_per_element,
            expert_weight_bpe=bytes_per_element,
        )

    def __float__(self) -> float:
        return self.activation_bpe

    def __mul__(self, other: float) -> float:
        return self.activation_bpe * other

    def __rmul__(self, other: float) -> float:
        return other * self.activation_bpe


@dataclass(frozen=True)
class TensorRef:
    id: str
    shape: tuple[int, int]
    dtype: str = "bf16"
    role: TensorRole = "activation"
    metadata: dict[str, Any] = field(default_factory=dict)
    size_bytes_override: int | None = None

    @property
    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def size_bytes(self) -> int:
        if self.size_bytes_override is not None:
            return self.size_bytes_override
        bytes_per_element = dtype_nbytes(self.dtype)
        return math.ceil(self.numel * bytes_per_element)


@dataclass
class ModuleCall:
    module: Any
    scope: str
    phase: str
    inputs: list[TensorRef] = field(default_factory=list)
    outputs: list[TensorRef] = field(default_factory=list)
    params: list[TensorRef] = field(default_factory=list)
    saved_tensors: dict[str, TensorRef] = field(default_factory=dict)
    op_specs: list[DataflowCost] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    seqlen: int
    num_seqs: int
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer: OptimizerMode = "none"
    final_model_state_on_backing: bool = False

    @property
    def tokens(self) -> int:
        return self.seqlen * self.num_seqs


class TraceContext:
    """Accumulates symbolic model execution and lowers it to schema objects."""

    def __init__(
        self,
        *,
        name: str,
        dtype_policy: DTypePolicy | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.dtype_policy = dtype_policy or DTypePolicy()
        self.objects: list[DataflowObject] = []
        self.compute_blocks: dict[str, ComputeBlock] = {}
        self.tasks: list[DataflowTask] = []
        self.calls: list[ModuleCall] = []
        self._call_stack: list[ModuleCall] = []
        self.tensors: dict[str, TensorRef] = {}
        self._object_ids: set[str] = set()
        self._task_ids: set[str] = set()

    def tensor(
        self,
        id: str,
        shape: tuple[int, int],
        *,
        role: TensorRole = "activation",
        dtype: str | None = None,
        metadata: dict[str, Any] | None = None,
        size_bytes: int | None = None,
    ) -> TensorRef:
        tensor = TensorRef(
            id=id,
            shape=shape,
            dtype=normalize_dtype(dtype or self.dtype_policy.dtype_for_role(role)),
            role=role,
            metadata=dict(metadata or {}),
            size_bytes_override=size_bytes,
        )
        self.tensors[id] = tensor
        return tensor

    def initial_tensor(
        self,
        id: str,
        shape: tuple[int, int],
        *,
        role: TensorRole,
        initial_location: Literal["fast", "backing"] = "backing",
        dtype: str | None = None,
        metadata: dict[str, Any] | None = None,
        size_bytes: int | None = None,
    ) -> TensorRef:
        if id in self._object_ids:
            raise ValueError(f"duplicate initial tensor id {id!r}")
        tensor = self.tensor(
            id,
            shape,
            role=role,
            dtype=dtype,
            metadata=metadata,
            size_bytes=size_bytes,
        )
        self.objects.append(
            DataflowObject(
                id=tensor.id,
                size_bytes=tensor.size_bytes,
                initial_location=initial_location,
                role=role,
            )
        )
        self._object_ids.add(id)
        return tensor

    def dataflow_output(
        self,
        tensor: TensorRef,
        *,
        location: Literal["fast", "backing"] = "fast",
    ) -> DataflowOutput:
        return DataflowOutput(
            id=tensor.id,
            size_bytes=tensor.size_bytes,
            location=location,
            role=tensor.role,
            metadata={"dtype": tensor.dtype, **tensor.metadata},
        )

    def register_block(
        self,
        key: str,
        *,
        name: str,
        category: str,
        subops: list[DataflowCost],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        block = ComputeBlock(
            key=key,
            name=name,
            category=category,
            subops=subops,
            metadata=dict(metadata or {}),
        )
        existing = self.compute_blocks.get(key)
        if existing is None:
            self.compute_blocks[key] = block
            return
        if existing.model_dump(mode="json") != block.model_dump(mode="json"):
            raise ValueError(f"conflicting compute block definition for {key!r}")

    @property
    def current_call(self) -> ModuleCall | None:
        return self._call_stack[-1] if self._call_stack else None

    def save_tensor(self, name: str, tensor: TensorRef) -> None:
        call = self.current_call
        if call is None:
            raise RuntimeError("save_tensor requires an active DataflowModule call")
        call.saved_tensors[name] = tensor

    def add_op_specs(self, subops: list[DataflowCost]) -> None:
        call = self.current_call
        if call is None:
            raise RuntimeError("add_op_specs requires an active DataflowModule call")
        call.op_specs.extend(subops)

    def emit_task(
        self,
        *,
        id: str,
        label: str,
        group: str,
        block_key: str,
        block_name: str,
        subops: list[DataflowCost],
        inputs: list[TensorRef],
        outputs: list[TensorRef] | None = None,
        mutates: list[TensorRef] | None = None,
        block_metadata: dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> DataflowTask:
        if id in self._task_ids:
            raise ValueError(f"duplicate task id {id!r}")
        self.register_block(
            block_key,
            name=block_name,
            category=group,
            subops=subops,
            metadata=block_metadata,
        )
        task = DataflowTask(
            id=id,
            label=label,
            group=group,
            compute_block_key=block_key,
            inputs=[tensor.id for tensor in inputs],
            outputs=[self.dataflow_output(tensor) for tensor in outputs or []],
            mutates=[tensor.id for tensor in mutates or []],
            metadata=dict(task_metadata or {}),
        )
        self.tasks.append(task)
        self._task_ids.add(id)
        return task

    def program(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        metrics: DataflowMetrics | None = None,
        final_locations: dict[str, Literal["fast", "backing"]] | None = None,
    ) -> DataflowProgram:
        return DataflowProgram(
            name=self.name,
            description=self.description,
            metadata=dict(metadata or {}),
            metrics=metrics,
            objects=self.objects,
            compute_blocks=list(self.compute_blocks.values()),
            tasks=self.tasks,
            final_locations=dict(final_locations or {}),
        )


class DataflowModule:
    """Base class for symbolic modules with explicit backward rules."""

    def __init__(self, *, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def __call__(self, ctx: TraceContext, *inputs: TensorRef, **kwargs: Any) -> Any:
        call = ModuleCall(
            module=self,
            scope=self.name,
            phase="forward",
            inputs=list(inputs),
        )
        ctx._call_stack.append(call)
        try:
            outputs = self.forward(ctx, *inputs, **kwargs)
        finally:
            ctx._call_stack.pop()
        call.outputs = _tensor_refs(outputs)
        ctx.calls.append(call)
        return outputs

    def forward(self, ctx: TraceContext, *inputs: TensorRef, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backward(
        self,
        ctx: TraceContext,
        call: ModuleCall,
        *grad_outputs: TensorRef,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not define backward")

    def recompute(
        self,
        ctx: TraceContext,
        call: ModuleCall,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not define recompute")


def trace_dataflow_model(
    model: DataflowModule,
    input_shape: tuple[int, int],
    *,
    name: str,
    input_name: str = "input",
    dtype_policy: DTypePolicy | None = None,
) -> DataflowProgram:
    ctx = TraceContext(name=name, dtype_policy=dtype_policy)
    x = ctx.initial_tensor(input_name, input_shape, role="input", initial_location="fast")
    model(ctx, x)
    return ctx.program(metadata={"kind": "dataflow_builder.forward"})


def trace_training_model(
    model: Any,
    input_shape: tuple[int, int],
    training: TrainingConfig,
    *,
    name: str | None = None,
    dtype_policy: DTypePolicy | None = None,
) -> DataflowProgram:
    if not hasattr(model, "build_training_program"):
        raise TypeError("trace_training_model requires a model with build_training_program")
    return model.build_training_program(
        training,
        input_shape=input_shape,
        name=name,
        dtype_policy=dtype_policy,
    )


def _tensor_refs(value: Any) -> list[TensorRef]:
    if isinstance(value, TensorRef):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[TensorRef] = []
        for item in value:
            out.extend(_tensor_refs(item))
        return out
    if isinstance(value, dict):
        out: list[TensorRef] = []
        for item in value.values():
            out.extend(_tensor_refs(item))
        return out
    return []
