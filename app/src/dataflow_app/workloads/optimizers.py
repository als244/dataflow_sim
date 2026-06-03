"""Optimizer-state sizing and optimizer-step cost models.

This module is intentionally model-agnostic. A workload supplies the logical
weight matrices for one layer, and the optimizer formulas turn those shapes
into state bytes, memory traffic, and flops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


OptimizerMode = Literal["none", "adamw", "muon"]


@dataclass(frozen=True)
class OptimizerMatrix:
    """One logical matrix consumed by a matrix-shaped optimizer."""

    name: str
    rows: int
    cols: int
    count: int = 1


def optimizer_state_bytes(weight_bytes: int, optimizer: OptimizerMode) -> int:
    """Bytes in a per-layer optimizer-state object.

    AdamW carries two state tensors; Muon carries one momentum tensor. The
    caller chooses the element size by passing the already-sized `weight_bytes`.
    """
    if optimizer == "none":
        return 0
    factor = 2 if optimizer == "adamw" else 1
    return factor * weight_bytes


def adamw_step_bytes(weight_bytes: int) -> int:
    """Persistent-object traffic for one layer's AdamW step.

    Reads: gradient, weight, optimizer state. Writes: weight, optimizer state.
    Since AdamW state is `2 * weight_bytes`, total traffic is
    `W + W + 2W + W + 2W = 7W`.
    """
    return 7 * weight_bytes


def muon_matrix_flops_bytes(
    rows: int,
    cols: int,
    *,
    ns_iters: int = 5,
    bytes_per_element: int = 2,
) -> tuple[int, int]:
    """Approximate Muon work for one matrix using Newton-Schulz iterations.

    The step transposes tall matrices so the loop operates on `X` with shape
    `(n, m)` and `n <= m`. Per iteration:

    - `A = X @ X.T`: `2 n^2 m` flops.
    - `B = bA + c(A @ A)`: `2 n^3 + 3 n^2` flops.
    - `X = aX + B @ X`: `2 n^2 m + 2 n m` flops.

    Traffic counts bf16 elements for persistent updates plus minimum scratch
    traffic for the three iteration stages: `5 n m + 6 n^2` elements per
    iteration.
    """
    n = min(rows, cols)
    m = max(rows, cols)
    elems = n * m
    flops = 10 * elems + ns_iters * (
        4 * n * n * m
        + 2 * n * n * n
        + 2 * elems
        + 3 * n * n
    )
    bytes_total = bytes_per_element * (
        12 * elems
        + ns_iters * (5 * elems + 6 * n * n)
    )
    return flops, bytes_total


def muon_step_flops_bytes(
    matrices: Iterable[OptimizerMatrix],
    *,
    ns_iters: int = 5,
    bytes_per_element: int = 2,
) -> tuple[int, int]:
    """Aggregate Muon work across a layer's logical weight matrices."""
    total_flops = 0
    total_bytes = 0
    for mat in matrices:
        flops, bytes_total = muon_matrix_flops_bytes(
            mat.rows,
            mat.cols,
            ns_iters=ns_iters,
            bytes_per_element=bytes_per_element,
        )
        total_flops += flops * mat.count
        total_bytes += bytes_total * mat.count
    return total_flops, total_bytes
