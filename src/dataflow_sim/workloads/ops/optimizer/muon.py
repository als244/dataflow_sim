"""Muon optimizer op helpers."""
from __future__ import annotations

from typing import Iterable

from dataflow_sim.workloads.dataflow import DataflowCost
from dataflow_sim.workloads.ops._common import roofline
from dataflow_sim.workloads.ops.optimizer.state import OptimizerMatrix


def muon_matrix_flops_bytes(
    rows: int,
    cols: int,
    *,
    ns_iters: int = 5,
    bytes_per_element: int = 2,
) -> tuple[int, int]:
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
    total_flops = 0
    total_bytes = 0
    for matrix in matrices:
        flops, bytes_total = muon_matrix_flops_bytes(
            matrix.rows,
            matrix.cols,
            ns_iters=ns_iters,
            bytes_per_element=bytes_per_element,
        )
        total_flops += flops * matrix.count
        total_bytes += bytes_total * matrix.count
    return total_flops, total_bytes


def muon_matrix_step(
    name: str,
    *,
    matrix: OptimizerMatrix,
    bytes_per_element: int = 2,
    ns_iters: int = 5,
) -> DataflowCost:
    flops, bytes_total = muon_matrix_flops_bytes(
        matrix.rows,
        matrix.cols,
        ns_iters=ns_iters,
        bytes_per_element=bytes_per_element,
    )
    return roofline(
        name,
        flops=flops,
        effective_flops=flops,
        memory_bytes=bytes_total,
        efficiency="matmul",
        count=matrix.count,
    )


def muon_step(
    name: str,
    *,
    matrices: Iterable[OptimizerMatrix],
    bytes_per_element: int = 2,
    ns_iters: int = 5,
) -> DataflowCost:
    flops, bytes_total = muon_step_flops_bytes(
        matrices,
        ns_iters=ns_iters,
        bytes_per_element=bytes_per_element,
    )
    return roofline(
        name,
        flops=flops,
        effective_flops=flops,
        memory_bytes=bytes_total,
        efficiency="matmul",
    )
