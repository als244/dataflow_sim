"""Reusable workload/simulation summary metrics.

The simulator returns an `EventLog`; workloads carry model/runtime metadata.
This module combines the two into the same top-level KPIs exposed by the web
API, without requiring callers to import the FastAPI server.
"""
from __future__ import annotations

from typing import Any

from dataflow_sim.core.schema import EventLog
from dataflow_sim.workloads.common.workload import Workload


def log_makespan_us(log: EventLog) -> float:
    return max((interval.end for interval in log.task_intervals), default=0)


def peak_fast_memory_gb(log: EventLog) -> float:
    if getattr(log, "peak_fast_memory_bytes", 0):
        return log.peak_fast_memory_bytes / (1024**3)
    peak_bytes = 0
    for event in log.events:
        bytes_ = sum(
            memory.size
            for memory in event.snapshot.memory
            if memory.location == "fast"
        )
        if bytes_ > peak_bytes:
            peak_bytes = bytes_
    return peak_bytes / (1024**3)


def interval_busy_us(
    log: EventLog,
    *,
    track: str | None = None,
    task_prefix: str | None = None,
) -> float:
    total = 0.0
    for interval in log.task_intervals:
        if track is not None and interval.track != track:
            continue
        if task_prefix is not None and not interval.task_id.startswith(task_prefix):
            continue
        total += interval.end - interval.start
    return total


def compute_workload_summary(workload: Workload, log: EventLog) -> dict[str, Any]:
    """Return top-level KPIs for a realized workload and simulator log."""
    return compute_summary(
        log,
        workload.metadata.get("breakdown", {}),
        workload.metadata.get("summary", {}),
    )


def compute_summary(
    log: EventLog,
    breakdown: dict[str, Any],
    summary_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate top-level KPIs from simulator output and workload metadata."""
    makespan = log_makespan_us(log)
    peak_gb = peak_fast_memory_gb(log)
    summary_meta = summary_meta or {}
    n_layers = int(summary_meta.get("n_layers", 1))
    metrics = summary_meta.get("metrics") or {}
    primary_unit = metrics.get("primary_unit")
    primary_count = float(metrics.get("primary_count", 0) or 0)
    total_tokens = int(
        primary_count
        if primary_unit == "tokens"
        else summary_meta.get("total_tokens", 0)
    )
    grad_accum_rounds = int(summary_meta.get("grad_accum_rounds", 1))
    num_steps = int(summary_meta.get("num_steps", 1))

    if makespan <= 0:
        return {
            "makespan_us": 0,
            "tokens_per_second": 0.0,
            "primary_unit": primary_unit,
            "primary_count": primary_count,
            "primary_rate_per_second": 0.0,
            "effective_tflops": 0.0,
            "hardware_tflops": 0.0,
            "peak_fast_memory_gb": peak_gb,
            "idle_pct": 0.0,
            "recompute_pct": 0.0,
            "from_slow_util_pct": 0.0,
            "to_slow_util_pct": 0.0,
            "total_flops": 0,
            "total_effective_flops": 0,
        }

    compute_busy = interval_busy_us(log, track="compute")
    from_slow_busy = interval_busy_us(log, track="from_slow")
    to_slow_busy = interval_busy_us(log, track="to_slow")
    recompute_busy = interval_busy_us(log, track="compute", task_prefix="r_")

    def subop_recompute_us(subop: dict[str, Any]) -> float:
        if subop["kind"] != "compute" or subop["flops"] <= 0:
            return 0.0
        if subop["effective_flops"] >= subop["flops"]:
            return 0.0
        discount = (subop["flops"] - subop["effective_flops"]) / subop["flops"]
        return float(subop["total_us"]) * discount

    block_rows = breakdown.get("compute_blocks") or []
    if block_rows:
        total_flops = sum(int(block.get("total_flops", 0)) for block in block_rows)
        total_eff_flops = sum(
            int(block.get("total_effective_flops", 0))
            for block in block_rows
        )
        recompute_busy += sum(
            subop_recompute_us(subop) * int(block.get("instance_count", 1))
            for block in block_rows
            if block.get("category") != "recompute"
            for subop in block.get("subops", [])
        )
        primary_rate = (
            primary_count / (makespan * 1e-6) if primary_count > 0 else 0.0
        )
        return {
            "makespan_us": makespan,
            "total_flops": total_flops,
            "total_effective_flops": total_eff_flops,
            "tokens_per_second": (
                primary_rate if primary_unit == "tokens" else 0.0
            ),
            "primary_unit": primary_unit,
            "primary_count": primary_count,
            "primary_rate_per_second": primary_rate,
            "effective_tflops": total_eff_flops / (makespan * 1e-6) / 1e12,
            "hardware_tflops": total_flops / (makespan * 1e-6) / 1e12,
            "peak_fast_memory_gb": peak_gb,
            "idle_pct": (makespan - compute_busy) / makespan * 100.0,
            "recompute_pct": recompute_busy / makespan * 100.0,
            "from_slow_util_pct": from_slow_busy / makespan * 100.0,
            "to_slow_util_pct": to_slow_busy / makespan * 100.0,
        }

    fwd_rows = breakdown.get("fwd", [])
    bwd_rows = breakdown.get("bwd", [])
    head_rows = breakdown.get("head", [])
    optimizer_rows = breakdown.get("optimizer", [])

    per_layer_subop_recompute_us = sum(
        subop_recompute_us(subop) for subop in fwd_rows + bwd_rows
    )
    head_subop_recompute_us = sum(subop_recompute_us(subop) for subop in head_rows)
    recompute_busy += (
        (per_layer_subop_recompute_us * n_layers + head_subop_recompute_us)
        * grad_accum_rounds
        * num_steps
    )
    per_layer_fwd_flops = sum(subop["flops"] * subop["count"] for subop in fwd_rows)
    per_layer_bwd_flops = sum(subop["flops"] * subop["count"] for subop in bwd_rows)
    head_flops_total = sum(subop["flops"] * subop["count"] for subop in head_rows)
    optimizer_flops_per_step = (
        sum(subop["flops"] * subop["count"] for subop in optimizer_rows) * n_layers
    )
    per_layer_fwd_eff = sum(
        subop["effective_flops"] * subop["count"] for subop in fwd_rows
    )
    per_layer_bwd_eff = sum(
        subop["effective_flops"] * subop["count"] for subop in bwd_rows
    )
    head_eff_total = sum(
        subop["effective_flops"] * subop["count"] for subop in head_rows
    )
    optimizer_eff_per_step = (
        sum(subop["effective_flops"] * subop["count"] for subop in optimizer_rows)
        * n_layers
    )

    per_round_flops = (
        (per_layer_fwd_flops + per_layer_bwd_flops) * n_layers
        + head_flops_total
    )
    per_round_eff_flops = (
        (per_layer_fwd_eff + per_layer_bwd_eff) * n_layers
        + head_eff_total
    )
    total_flops = (
        per_round_flops * grad_accum_rounds * num_steps
        + optimizer_flops_per_step * num_steps
    )
    total_eff_flops = (
        per_round_eff_flops * grad_accum_rounds * num_steps
        + optimizer_eff_per_step * num_steps
    )
    primary_rate = (
        primary_count / (makespan * 1e-6) if primary_count > 0 else 0.0
    )
    return {
        "makespan_us": makespan,
        "total_flops": total_flops,
        "total_effective_flops": total_eff_flops,
        "tokens_per_second": (
            primary_rate
            if primary_unit == "tokens"
            else (total_tokens / (makespan * 1e-6) if total_tokens > 0 else 0.0)
        ),
        "primary_unit": primary_unit,
        "primary_count": primary_count,
        "primary_rate_per_second": primary_rate,
        "effective_tflops": total_eff_flops / (makespan * 1e-6) / 1e12,
        "hardware_tflops": total_flops / (makespan * 1e-6) / 1e12,
        "peak_fast_memory_gb": peak_gb,
        "idle_pct": (makespan - compute_busy) / makespan * 100.0,
        "recompute_pct": recompute_busy / makespan * 100.0,
        "from_slow_util_pct": from_slow_busy / makespan * 100.0,
        "to_slow_util_pct": to_slow_busy / makespan * 100.0,
    }
