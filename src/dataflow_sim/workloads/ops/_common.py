"""Shared helpers for symbolic workload ops."""
from __future__ import annotations

import math
from typing import Literal

from dataflow_sim.workloads.dataflow import DataflowCost


Efficiency = Literal[
    "matmul",
    "matmul_bf16",
    "matmul_fp8",
    "matmul_fp4",
    "attention_fwd",
    "attention_bwd",
    "memory",
    "scale_up",
    "custom",
]


def matmul_efficiency(compute_precision: str = "bf16") -> Efficiency:
    key = compute_precision.strip().lower()
    if key in {"bf16", "bfloat16"}:
        return "matmul_bf16"
    if key == "fp8":
        return "matmul_fp8"
    if key == "fp4":
        return "matmul_fp4"
    raise ValueError(f"unsupported matmul compute precision {compute_precision!r}")


def roofline(
    name: str,
    *,
    flops: int = 0,
    memory_bytes: float = 0,
    efficiency: Efficiency = "memory",
    count: int = 1,
    effective_flops: int | None = None,
    compute_eff: float | None = None,
    mem_eff: float | None = None,
) -> DataflowCost:
    """Return one hardware-free sub-op cost term."""
    return DataflowCost(
        kind="roofline",
        name=name,
        flops=flops,
        effective_flops=flops if effective_flops is None else effective_flops,
        memory_bytes=math.ceil(memory_bytes),
        efficiency=efficiency,
        count=count,
        compute_eff=compute_eff,
        mem_eff=mem_eff,
    )


def memory_op(
    name: str,
    memory_bytes: float,
    *,
    count: int = 1,
    efficiency: Efficiency = "memory",
) -> DataflowCost:
    return roofline(
        name,
        flops=0,
        effective_flops=0,
        memory_bytes=memory_bytes,
        efficiency=efficiency,
        count=count,
    )


def fixed(name: str, runtime_us: float) -> DataflowCost:
    return DataflowCost(kind="fixed", name=name, runtime_us=runtime_us)


def attention_score_terms(
    tokens: int,
    *,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
) -> int:
    """Return sum(sequence_length ** 2) for attention cost formulas."""
    if sequence_lengths is not None:
        if sum(sequence_lengths) != tokens:
            raise ValueError("sequence_lengths must sum to tokens")
        return sum(length * length for length in sequence_lengths)
    if seqlen is None:
        return tokens * tokens
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    if tokens % seqlen != 0:
        raise ValueError("tokens must be divisible by seqlen when sequence_lengths is absent")
    return (tokens // seqlen) * seqlen * seqlen


def sliding_attention_score_terms(
    tokens: int,
    *,
    window_size: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
) -> int:
    """Return sum(sequence_length * min(sequence_length, window_size))."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if sequence_lengths is not None:
        if sum(sequence_lengths) != tokens:
            raise ValueError("sequence_lengths must sum to tokens")
        return sum(length * min(length, window_size) for length in sequence_lengths)
    if seqlen is None:
        return tokens * min(tokens, window_size)
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    if tokens % seqlen != 0:
        raise ValueError("tokens must be divisible by seqlen when sequence_lengths is absent")
    return (tokens // seqlen) * seqlen * min(seqlen, window_size)


def topk_attention_score_terms(
    tokens: int,
    *,
    top_k: int,
    seqlen: int | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | None = None,
) -> int:
    """Return sum(sequence_length * min(sequence_length, top_k))."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if sequence_lengths is not None:
        if sum(sequence_lengths) != tokens:
            raise ValueError("sequence_lengths must sum to tokens")
        return sum(length * min(length, top_k) for length in sequence_lengths)
    if seqlen is None:
        return tokens * min(tokens, top_k)
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")
    if tokens % seqlen != 0:
        raise ValueError("tokens must be divisible by seqlen when sequence_lengths is absent")
    return (tokens // seqlen) * seqlen * min(seqlen, top_k)
