"""Recompute rewrite declarations shared between workloads and planners.

A workload that supports activation recomputation declares, per saved
activation object, the discrete options available. Level 0 always means
"save the full activation, no recompute work". Higher levels save fewer
bytes and add runtime through a recompute task in the rebuilt chain. Binary
recomputation is the two-option special case; partial recomputation (save
part of the activation, recompute the rest) adds intermediate levels
without changing this contract.

Planners never interpret model semantics: they only see object ids, byte
counts, microseconds, and the compute-block keys that generated the rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecomputeOption:
    level: int            # 0 = save full activation, no recompute work
    saved_bytes: int      # activation bytes forward still saves at this level
    recompute_us: float   # runtime added by recompute at this level
    label: str = ""


@dataclass(frozen=True)
class RecomputeRewrite:
    """Discrete recompute choices for one saved-activation object.

    The decision is instance-specific (`object_id`) because memory pressure is
    instance-specific, but the available choices are defined by compute blocks.
    `f_compute_block_key` points to the block that normally saves the object,
    and `r_compute_block_key` points to the block used if recompute wins.
    """
    object_id: str                          # e.g. "A_0_0_5"
    f_task_id: str                          # producer when level == 0
    r_task_id: str                          # recompute producer when level > 0
    options: tuple[RecomputeOption, ...]    # ascending level; options[0].level == 0
    f_compute_block_key: str = ""
    r_compute_block_key: str = ""
    group_key: str = ""
