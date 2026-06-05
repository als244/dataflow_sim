#!/usr/bin/env python
"""Head-to-head comparison of auto-policy vs. sliding-window baseline.

Usage:
    python scripts/compare_policies.py [--L 3] [--cap 800] [--window 2]
                                       [--bw-h2d 8] [--bw-d2h 8]
    python scripts/compare_policies.py --sweep

The single-config mode runs both policies on one parameter combination and
prints a side-by-side metrics table. `--sweep` runs a grid and produces a
markdown table of (L, cap, window) cells.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from dataflow_sim.policies.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policies.max_reduce import apply_max_reduce_policy
from dataflow_sim.policies.min_grow import apply_min_grow_policy
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.policies.sliding_window import apply_sliding_window_policy
from dataflow_sim.engine.simulator import run
from dataflow_sim.workloads.training.transformer import build_layerwise_training_chain


@dataclass
class Metrics:
    ok: bool
    error: str | None
    makespan: int | None
    compute_stall: int | None
    peak_device: int | None
    bytes_h2d: int | None
    bytes_d2h: int | None
    n_releases: int | None
    n_offloads: int | None
    n_prefetches: int | None


def measure(annotated) -> Metrics:
    try:
        log = run(annotated)
    except Exception as e:
        return Metrics(False, f"{type(e).__name__}: {e}", *([None] * 8))

    compute = sorted(
        [iv for iv in log.task_intervals if iv.track == "compute"],
        key=lambda iv: iv.start,
    )
    makespan = max(iv.end for iv in log.task_intervals)
    stall = sum(max(0, cur.start - prev.end) for prev, cur in zip(compute, compute[1:]))

    # Peak device usage: walk through snapshots, sum device sizes
    peak = 0
    for ev in log.events:
        dev = sum(m.size for m in ev.snapshot.memory if m.location == "device")
        peak = max(peak, dev)

    # Transfer bytes: sum sizes for each transfer_start event
    bytes_h2d = bytes_d2h = 0
    for ev in log.events:
        if ev.kind == "transfer_start":
            if ev.transfer_direction == "h2d":
                for m in ev.snapshot.memory:
                    if m.id == ev.transfer_obj and m.location == "device":
                        bytes_h2d += m.size
                        break
            elif ev.transfer_direction == "d2h":
                for m in ev.snapshot.memory:
                    if m.id == ev.transfer_obj and m.location == "device":
                        bytes_d2h += m.size
                        break

    n_rel = sum(len(t.releases_after) for t in annotated.tasks)
    n_off = sum(len(t.offload_after) for t in annotated.tasks)
    n_pre = sum(len(t.prefetch_after) for t in annotated.tasks)
    return Metrics(True, None, makespan, stall, peak, bytes_h2d, bytes_d2h, n_rel, n_off, n_pre)


def _safe_apply(fn, *args, **kwargs) -> Metrics | None:
    """Returns None on policy-side failure (caller surfaces a FAIL row)."""
    try:
        chain = fn(*args, **kwargs)
    except Exception as e:
        return Metrics(False, f"policy: {type(e).__name__}: {e}", *([None] * 8))
    return measure(chain)


def run_one(L: int, cap: int | None, window: int, bw_h2d: int, bw_d2h: int) -> dict[str, Metrics]:
    bare = build_layerwise_training_chain(L=L, bandwidth_h2d=bw_h2d, bandwidth_d2h=bw_d2h)
    from dataclasses import replace
    bare_with_cap = replace(bare, device_capacity=cap)
    return {
        "sliding": _safe_apply(apply_sliding_window_policy, bare, window_size=window, device_capacity=cap),
        "auto": _safe_apply(apply_belady_reactive_policy, bare, device_capacity=cap),
        "v4": _safe_apply(apply_max_reduce_policy, bare_with_cap),
        "v5": _safe_apply(apply_min_grow_policy, bare_with_cap, time_budget_s=10.0),
        "pressurefit": _safe_apply(apply_pressurefit_policy, bare, device_capacity=cap),
    }


def _fmt(v):
    return "—" if v is None else str(v)


def print_one(L: int, cap: int | None, window: int, bw_h2d: int, bw_d2h: int) -> None:
    print(f"\n=== L={L}, cap={cap}, window={window}, bw_h2d={bw_h2d}, bw_d2h={bw_d2h} ===\n")
    res = run_one(L, cap, window, bw_h2d, bw_d2h)
    cols = ["sliding", "auto", "v4", "v5", "pressurefit"]
    rows = [("status", *("OK" if res[c].ok else "FAIL" for c in cols))]
    for metric in ("makespan", "compute_stall", "peak_device", "bytes_h2d", "bytes_d2h",
                   "n_releases", "n_offloads", "n_prefetches"):
        rows.append((metric, *(_fmt(getattr(res[c], metric)) for c in cols)))
    header = f"  {'metric':<16}" + "".join(f" {c:>10}" for c in cols)
    print(header)
    print(f"  {'-' * 16}" + "".join(f" {'-' * 10}" for _ in cols))
    for row in rows:
        print(f"  {row[0]:<16}" + "".join(f" {str(v):>10}" for v in row[1:]))
    for c in cols:
        if not res[c].ok:
            print(f"\n  {c} error: {res[c].error}")


def sweep(bw_h2d: int = 8, bw_d2h: int = 8) -> None:
    Ls = [3, 5, 10]
    windows = [2]   # window matters only for sliding; reduce noise by fixing it
    rows: list[dict] = []
    for L in Ls:
        caps = [None, 1500, 1200, 1000, 800, 600]
        for cap in caps:
            for w in windows:
                res = run_one(L, cap, w, bw_h2d, bw_d2h)
                rows.append({
                    "L": L, "cap": cap, "w": w,
                    "sliding": res["sliding"], "auto": res["auto"],
                    "v4": res["v4"], "v5": res["v5"],
                    "pressurefit": res["pressurefit"],
                })

    print(f"\n# Sweep (bw_h2d={bw_h2d}, bw_d2h={bw_d2h})\n")
    print("| L | cap | sliding | v2 | v4 | v5 | pressurefit | best | pressurefit vs best |")
    print("|---|-----|---------|-----|-----|-----|----------|------|------------------|")
    n_v5_wins = n_v5_ties = n_v5_loses = 0
    for r in rows:
        cells = {}
        for name, key in (("sw", "sliding"), ("v2", "auto"), ("v4", "v4"), ("v5", "v5"), ("pressurefit", "pressurefit")):
            m = r[key]
            cells[name] = str(m.makespan) if m.ok else "FAIL"
        valid_others = [m.makespan for m in (r["sliding"], r["auto"], r["v4"], r["v5"]) if m.ok]
        best = min(valid_others) if valid_others else None
        pressurefit = r["pressurefit"].makespan if r["pressurefit"].ok else None
        if pressurefit is None or best is None:
            cmp = "—"
        elif pressurefit < best:
            cmp = f"WIN -{best - pressurefit}"
            n_v5_wins += 1
        elif pressurefit == best:
            cmp = "tie"
            n_v5_ties += 1
        else:
            cmp = f"LOSE +{pressurefit - best}"
            n_v5_loses += 1
        cap_s = "∞" if r["cap"] is None else str(r["cap"])
        print(f"| {r['L']} | {cap_s} | {cells['sw']} | {cells['v2']} | {cells['v4']} | {cells['v5']} | {cells['pressurefit']} | {best} | {cmp} |")
    print(f"\nPressureFit summary: {n_v5_wins} wins, {n_v5_ties} ties, {n_v5_loses} losses vs best-of-prior.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--cap", type=int, default=None)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--bw-h2d", type=int, default=8, dest="bw_h2d")
    p.add_argument("--bw-d2h", type=int, default=8, dest="bw_d2h")
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    if args.sweep:
        sweep(args.bw_h2d, args.bw_d2h)
    else:
        print_one(args.L, args.cap, args.window, args.bw_h2d, args.bw_d2h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
