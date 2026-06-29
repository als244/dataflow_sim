"""Shared helpers for symbolic workload ops."""
from __future__ import annotations

from typing import Literal

from dataflow_sim.workloads.dataflow import DataflowCost


Efficiency = Literal["matmul", "attention_fwd", "attention_bwd", "memory", "custom"]


def roofline(
    name: str,
    *,
    flops: int = 0,
    memory_bytes: int = 0,
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
        memory_bytes=memory_bytes,
        efficiency=efficiency,
        count=count,
        compute_eff=compute_eff,
        mem_eff=mem_eff,
    )


def memory_op(name: str, memory_bytes: int, *, count: int = 1) -> DataflowCost:
    return roofline(
        name,
        flops=0,
        effective_flops=0,
        memory_bytes=memory_bytes,
        efficiency="memory",
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
