from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from dataflow_sim.core.schema import TaskChain


@dataclass(frozen=True)
class Workload:
    """A bare task chain plus optional workload-specific metadata."""

    chain: TaskChain
    metadata: Mapping[str, Any] = field(default_factory=dict)

