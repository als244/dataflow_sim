"""Shared PressureFit types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ScheduleSpec:
    """One inbound-schedule variant of the shared planning pipeline.

    All variants plan residency with the same seed and the same pressure
    reduction; they differ only in how interval entries become prefetch
    triggers (and whether entries are extended earlier first).
    """
    name: str
    pack_inbound: bool = False
    extend_inbound: bool = False
    respect_interval_start: bool = False
    # Pressure-clamped packing: a packed job may not fire earlier than the
    # strict boundary model allows (see _pressure_clamped_fire).
    clamp_inbound: bool = False


_IntervalSet = dict[str, list[tuple[int, int]]]
