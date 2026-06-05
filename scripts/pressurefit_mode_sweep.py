"""Compare PressureFit portfolio modes across transformer configs.

Examples:
    python scripts/pressurefit_mode_sweep.py --quick
    python scripts/pressurefit_mode_sweep.py --compact --optimizer adamw --num-steps 2
    python scripts/pressurefit_mode_sweep.py --canonical --out /tmp/pf_modes.csv

The CSV has one row per (config, mode). The printed summary compares fast/auto
against full where both modes succeed, and reports selected-candidate counts.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.transformer import build_transformer_training_workload
from dataflow_sim.workloads.training.transformer import TrainingConfig
from dataflow_sim.policies.pressurefit import PressureFitPortfolioMode, plan_pressurefit_policy
from dataflow_sim.engine.simulator import run as sim_run

MODES: tuple[PressureFitPortfolioMode, ...] = ("auto", "fast", "full")

CANONICAL_HARDWARE_CAPS_GB = {
    "H100": [20, 30, 40, 80, None],
    "RTX_5090": [8, 16, 24, 32],
}
CANONICAL_NUM_SEQS = [1, 2, 4, 8]
CANONICAL_SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

COMPACT_HARDWARE_CAPS_GB = {
    "H100": [20, 30, 40, 80],
    "RTX_5090": [8, 16, 32],
}
COMPACT_NUM_SEQS = [1, 4, 8]
COMPACT_SEQLENS = [1024, 4096, 8192, 16384]

QUICK_HARDWARE_CAPS_GB = {"H100": [30, 40], "RTX_5090": [8, 16]}
QUICK_NUM_SEQS = [1, 4]
QUICK_SEQLENS = [2048, 8192]


def _axes(args):
    if args.canonical:
        return CANONICAL_HARDWARE_CAPS_GB, CANONICAL_NUM_SEQS, CANONICAL_SEQLENS
    if args.quick:
        return QUICK_HARDWARE_CAPS_GB, QUICK_NUM_SEQS, QUICK_SEQLENS
    return COMPACT_HARDWARE_CAPS_GB, COMPACT_NUM_SEQS, COMPACT_SEQLENS


def _configs(args):
    hw_caps, num_seqs, seqlens = _axes(args)
    for hw_name, caps in hw_caps.items():
        for cap_gb, M, S in product(caps, num_seqs, seqlens):
            yield hw_name, cap_gb, M, S


def _cap_bytes(cap_gb: int | None) -> int | None:
    if cap_gb is None:
        return None
    return int(cap_gb * (1024 ** 3))


def _run_one(args, hw_name: str, cap_gb: int | None, M: int, S: int, mode: PressureFitPortfolioMode) -> dict:
    models = load_model_presets()
    spec = models["llama3_8B"]
    hw = HARDWARE_PRESETS[hw_name]
    cfg = TrainingConfig(
        seqlen=S,
        num_seqs=M,
        grad_accum_rounds=args.grad_accum_rounds,
        num_steps=args.num_steps,
        optimizer=args.optimizer,
        final_model_state_on_host=args.final_model_state_on_host,
    )
    cap_bytes = _cap_bytes(cap_gb)
    t0 = time.perf_counter()
    try:
        bare = build_transformer_training_workload(spec, hw, cfg).chain
        bare = replace(bare, device_capacity=cap_bytes)
        chain, diagnostics = plan_pressurefit_policy(
            bare,
            device_capacity=cap_bytes,
            portfolio_mode=mode,
        )
        log = sim_run(chain, snapshots=False)
        makespan = max(iv.end for iv in log.task_intervals)
        return {
            "hardware": hw_name,
            "cap_GB": "unlimited" if cap_gb is None else cap_gb,
            "num_seqs": M,
            "seqlen": S,
            "grad_accum_rounds": args.grad_accum_rounds,
            "num_steps": args.num_steps,
            "optimizer": args.optimizer,
            "final_model_state_on_host": args.final_model_state_on_host,
            "mode": mode,
            "effective_mode": diagnostics.effective_portfolio_mode,
            "ok": True,
            "error": "",
            "makespan_us": makespan,
            "peak_device_bytes": log.peak_device_bytes,
            "policy_wall_s": time.perf_counter() - t0,
            "planning_wall_s": diagnostics.planning_time_s,
            "candidate_count": diagnostics.candidate_count,
            "valid_candidate_count": diagnostics.valid_candidate_count,
            "selected_candidate": diagnostics.selected_candidate,
        }
    except Exception as e:
        return {
            "hardware": hw_name,
            "cap_GB": "unlimited" if cap_gb is None else cap_gb,
            "num_seqs": M,
            "seqlen": S,
            "grad_accum_rounds": args.grad_accum_rounds,
            "num_steps": args.num_steps,
            "optimizer": args.optimizer,
            "final_model_state_on_host": args.final_model_state_on_host,
            "mode": mode,
            "effective_mode": "",
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "makespan_us": "",
            "peak_device_bytes": "",
            "policy_wall_s": time.perf_counter() - t0,
            "planning_wall_s": "",
            "candidate_count": "",
            "valid_candidate_count": "",
            "selected_candidate": "",
        }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "hardware",
        "cap_GB",
        "num_seqs",
        "seqlen",
        "grad_accum_rounds",
        "num_steps",
        "optimizer",
        "final_model_state_on_host",
        "mode",
        "effective_mode",
        "ok",
        "error",
        "makespan_us",
        "peak_device_bytes",
        "policy_wall_s",
        "planning_wall_s",
        "candidate_count",
        "valid_candidate_count",
        "selected_candidate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> None:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (
            row["hardware"],
            row["cap_GB"],
            row["num_seqs"],
            row["seqlen"],
            row["grad_accum_rounds"],
            row["num_steps"],
            row["optimizer"],
            row["final_model_state_on_host"],
        )
        grouped.setdefault(key, {})[row["mode"]] = row

    print("\nMode comparison against full, where both modes succeeded:")
    for mode in ("auto", "fast"):
        gaps_pct: list[float] = []
        gaps_us: list[int] = []
        speedups: list[float] = []
        exact = 0
        both_ok = 0
        for per_mode in grouped.values():
            full = per_mode.get("full")
            other = per_mode.get(mode)
            if not full or not other or not full["ok"] or not other["ok"]:
                continue
            both_ok += 1
            full_ms = int(full["makespan_us"])
            other_ms = int(other["makespan_us"])
            gap = other_ms - full_ms
            if gap == 0:
                exact += 1
            gaps_us.append(gap)
            gaps_pct.append(gap / full_ms * 100.0 if full_ms else 0.0)
            full_wall = float(full["planning_wall_s"] or 0.0)
            other_wall = float(other["planning_wall_s"] or 0.0)
            if other_wall > 0:
                speedups.append(full_wall / other_wall)
        if not both_ok:
            print(f"  {mode}: no shared successful rows")
            continue
        print(
            f"  {mode}: {exact}/{both_ok} exact, "
            f"max gap {max(gaps_us)} us ({max(gaps_pct):.2f}%), "
            f"mean gap {statistics.mean(gaps_us):.1f} us "
            f"({statistics.mean(gaps_pct):.3f}%), "
            f"mean full/other planning speedup "
            f"{statistics.mean(speedups):.2f}x"
        )

    print("\nSelected candidate counts:")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if not row["ok"]:
            continue
        key = (row["mode"], row["selected_candidate"])
        counts[key] = counts.get(key, 0) + 1
    for (mode, candidate), count in sorted(counts.items()):
        print(f"  {mode:<4} {candidate:<32} {count}")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="small smoke grid")
    group.add_argument("--compact", action="store_true", help="moderate default grid")
    group.add_argument("--canonical", action="store_true", help="full canonical grid")
    parser.add_argument("--out", default="/tmp/pressurefit_mode_sweep.csv")
    parser.add_argument("--grad-accum-rounds", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--optimizer", choices=["none", "adamw", "muon"], default="none")
    parser.add_argument("--final-model-state-on-host", action="store_true")
    args = parser.parse_args()

    configs = list(_configs(args))
    print(f"{len(configs)} configs x {len(MODES)} modes = {len(configs) * len(MODES)} runs")
    rows: list[dict] = []
    t0 = time.perf_counter()
    for idx, (hw_name, cap_gb, M, S) in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] {hw_name} cap={cap_gb} "
            f"M={M} S={S}",
            flush=True,
        )
        for mode in MODES:
            row = _run_one(args, hw_name, cap_gb, M, S, mode)
            rows.append(row)
            if row["ok"]:
                print(
                    f"  {mode:<4} {row['effective_mode']:<4} "
                    f"ms={row['makespan_us']} "
                    f"plan={float(row['planning_wall_s']):.3f}s "
                    f"selected={row['selected_candidate']}",
                    flush=True,
                )
            else:
                print(f"  {mode:<4} ERR {row['error']}", flush=True)

    out = Path(args.out)
    _write_csv(out, rows)
    print(f"\nWrote {out}")
    print(f"Total wall: {time.perf_counter() - t0:.1f}s")
    _summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
