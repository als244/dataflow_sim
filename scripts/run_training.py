#!/usr/bin/env python
"""Build an L-layer training chain, run the simulator, and write the event log
into the UI. Also dumps the generated chain JSON to examples/training_L{L}.json.

Usage:
    python scripts/run_training.py [L=3] [--bw-from-slow N] [--bw-to-slow N]

By default both bandwidths are 8 (bytes/time-unit), which keeps the L=3 demo
stall-free. Lowering --bw-from-slow (e.g. to 1) is the easy way to show compute stalls
on the timeline: prefetches arrive too late and b_i waits for A_i to land.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.engine.simulator import run
from workload_helpers import build_tiny_training_chain


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("L", nargs="?", type=int, default=3)
    p.add_argument("--bw-from-slow", type=int, default=8, dest="bw_from_slow")
    p.add_argument("--bw-to-slow", type=int, default=8, dest="bw_to_slow")
    args = p.parse_args()

    bare = build_tiny_training_chain(
        layers=args.L,
        bandwidth_from_slow=args.bw_from_slow,
        bandwidth_to_slow=args.bw_to_slow,
    )
    chain = apply_pressurefit_policy(bare)
    chain_path = REPO / f"examples/training_L{args.L}.json"
    with open(chain_path, "w") as f:
        json.dump(asdict(chain), f, indent=2)

    log = run(chain)
    log_path = REPO / "ui/src/event_log.json"
    log.dump(log_path)

    # Detect compute stalls by checking for gaps between consecutive compute intervals
    compute_intervals = sorted(
        [iv for iv in log.task_intervals if iv.track == "compute"],
        key=lambda iv: iv.start,
    )
    stalls = []
    for prev, cur in zip(compute_intervals, compute_intervals[1:]):
        if cur.start > prev.end:
            stalls.append((prev.task_id, cur.task_id, cur.start - prev.end))

    print(f"L:        {args.L}")
    print(f"bw from-slow:   {args.bw_from_slow}")
    print(f"bw to-slow: {args.bw_to_slow}")
    print(f"tasks:    {len(chain.tasks)}")
    print(f"events:   {len(log.events)}")
    duration = max((iv.end for iv in log.task_intervals), default=0)
    print(f"duration: {duration}")
    if stalls:
        print(f"stalls:   {len(stalls)}")
        for prev_id, cur_id, gap in stalls:
            print(f"          {cur_id} stalled {gap} units after {prev_id}")
    else:
        print("stalls:   none")
    print(f"chain:    {chain_path}")
    print(f"log:      {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
