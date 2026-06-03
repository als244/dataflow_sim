#!/usr/bin/env python
"""Transformer policy sweep — llama3_8B across seqlen x num_seqs x cap.

Compares belady_reactive, max_reduce, min_grow, pressurefit, and
sliding_window on a grid of real transformer configs. Prints a markdown
table with min_grow vs best-of-prior comparison.

Usage:
    python app/scripts/sweep_transformer.py [--model llama3_8B] [--budget 15]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

from dataflow_sim.policy.belady_reactive import apply_belady_reactive_policy
from dataflow_sim.policy.max_reduce import apply_max_reduce_policy
from dataflow_sim.policy.min_grow import apply_min_grow_policy
from dataflow_sim.policy.pressurefit import apply_pressurefit_policy
from dataflow_sim.policy.sliding_window import apply_sliding_window_policy
from dataflow_sim.simulator import run
from dataflow_app.workloads.presets import HARDWARE_PRESETS, load_model_presets
from dataflow_app.workloads.training import build_transformer_bare_chain
from dataflow_app.workloads.transformer import TrainingConfig


def _safe(name, fn, *args, **kw):
    t0 = time.monotonic()
    try:
        ann = fn(*args, **kw)
        log = run(ann)
        return max(iv.end for iv in log.task_intervals), time.monotonic() - t0, None
    except Exception as e:
        return None, time.monotonic() - t0, f"{type(e).__name__}: {str(e)[:60]}"


def sweep(model: str, budget: float):
    models = load_model_presets()
    spec = models[model]
    hw = HARDWARE_PRESETS["H100"]

    seqlens = [2048, 4096, 8192, 16384, 32768, 65536]
    num_seqs_list = [1, 2, 4]
    caps_gb = [20, 30, 40, 80]

    print(f"# Sweep: model={model} (L={spec.n_layers}) hw=H100 v5_budget={budget}s\n")
    print("| sql | M | cap GB | sliding | v2 | v4 | v5 | pressurefit | best | pressurefit vs best | pressurefit time |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    n_v5_wins = n_v5_ties = n_v5_loses = 0
    for sql in seqlens:
        for M in num_seqs_list:
            cfg = TrainingConfig(seqlen=sql, num_seqs=M)
            bare, _ = build_transformer_bare_chain(spec, hw, cfg)
            for cap_gb in caps_gb:
                cap = int(cap_gb * 1e9)
                bare_cap = replace(bare, device_capacity=cap)
                sw_ms, _, sw_err = _safe("sliding", apply_sliding_window_policy, bare, window_size=2, device_capacity=cap)
                v2_ms, _, v2_err = _safe("v2", apply_belady_reactive_policy, bare_cap)
                v4_ms, _, v4_err = _safe("v4", apply_max_reduce_policy, bare_cap)
                v5_ms, v5_t, v5_err = _safe("v5", apply_min_grow_policy, bare_cap, time_budget_s=budget)
                pressurefit_ms, pressurefit_t, pressurefit_err = _safe(
                    "pressurefit", apply_pressurefit_policy, bare, device_capacity=cap,
                )

                cells = {
                    "sw": str(sw_ms) if sw_ms is not None else "ERR",
                    "v2": str(v2_ms) if v2_ms is not None else "ERR",
                    "v4": str(v4_ms) if v4_ms is not None else "ERR",
                    "v5": str(v5_ms) if v5_ms is not None else "ERR",
                    "pressurefit": str(pressurefit_ms) if pressurefit_ms is not None else "ERR",
                }
                valid_others = [x for x in (sw_ms, v2_ms, v4_ms, v5_ms) if x is not None]
                best = min(valid_others) if valid_others else None
                if pressurefit_ms is None or best is None:
                    cmp = "—"
                elif pressurefit_ms < best:
                    cmp = f"WIN -{best - pressurefit_ms}"
                    n_v5_wins += 1
                elif pressurefit_ms == best:
                    cmp = "tie"
                    n_v5_ties += 1
                else:
                    cmp = f"LOSE +{pressurefit_ms - best}"
                    n_v5_loses += 1
                print(
                    f"| {sql} | {M} | {cap_gb} | {cells['sw']} | {cells['v2']} | {cells['v4']} | "
                    f"{cells['v5']} | {cells['pressurefit']} | {best if best is not None else '—'} | "
                    f"{cmp} | {pressurefit_t:.1f}s |",
                    flush=True,
                )

    print(f"\nPressureFit summary: {n_v5_wins} wins, {n_v5_ties} ties, {n_v5_loses} losses vs best-of-prior.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3_8B")
    p.add_argument("--budget", type=float, default=15.0, help="min_grow search time budget per config (s)")
    args = p.parse_args()
    sweep(args.model, args.budget)


if __name__ == "__main__":
    raise SystemExit(main())
