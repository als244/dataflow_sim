"""Recompute-selection sweep: dense + MoE models across hardware, capacity,
microbatch, grad-accum, and sequence length.

For each config, compares fixed recompute choices (none / all / every-other
layer) against the evidence-directed greedy loop in
`dataflow_sim.planning.recompute`, all planned with PressureFit and scored by
the simulator. Each row also carries the infinite-cap ideal makespan, the gap
to it, and a stall/stream diagnosis of where the remaining gap lives.

Run:
    python scripts/recompute_sweep.py --quick
    python scripts/recompute_sweep.py            # full grid
"""
from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from dataclasses import replace

from dataflow_sim.engine.simulator import run as sim_run
from dataflow_sim.engine.stall_report import build_stall_report
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_transformer_training_workload,
)

MODELS = ["llama3_8B", "sparse_16Bx3B", "qwen3_30Bx3B"]
HW_CAPS = {"H100": [10, 20, 40, 80], "RTX_5090": [8, 16, 24]}
SEQLENS = [4096, 16384]
NUM_SEQS = [1, 4, 8]
GA_ROUNDS = [1, 8]

QUICK_HW_CAPS = {"H100": [20, 40]}
QUICK_NUM_SEQS = [2]
QUICK_GA_ROUNDS = [1]

# --comprehensive: the pinned baseline grid (docs/internal/measurements/).
COMPREHENSIVE_MODELS = [
    "nanogpt_124M", "llama3_8B", "dense_15B", "qwen3_32B",
    "olmoe_7Bx1B", "sparse_16Bx3B", "qwen3_30Bx3B",
    "qwen3_moe_shallow", "mini_deepseek", "small_deepseek",
]
COMPREHENSIVE_HW_CAPS = {"H100": [10, 20, 40, 80], "RTX_5090": [8, 16, 24, 32]}
COMPREHENSIVE_SEQLENS = [2048, 8192, 32768]
COMPREHENSIVE_NUM_SEQS = [1, 4, 8]
COMPREHENSIVE_GA_ROUNDS = [1, 8, 32]

FIELDS = [
    "model", "hardware", "cap_GB", "seqlen", "num_seqs", "ga", "n_layers",
    "tasks", "ideal", "none", "all", "half", "greedy", "best",
    "gap_vs_ideal_pct", "greedy_recomputed", "greedy_evals", "greedy_wall_s",
    "stall_pct", "input_wait_pct", "capacity_wait_pct",
    "h2d_busy_pct", "d2h_busy_pct", "w_blame_pct", "a_blame_pct", "errors",
]


def _makespan(chain) -> int:
    log = sim_run(chain, snapshots=False)
    return max(iv.end for iv in log.task_intervals)


def _ideal_one(key):
    """Phase 1: the infinite-cap ideal, shared by every cap row of a config."""
    model_name, hw_name, S, M, ga = key
    spec = load_model_presets()[model_name]
    hw = HARDWARE_PRESETS[hw_name]
    cfg = TrainingConfig(seqlen=S, num_seqs=M, grad_accum_rounds=ga)
    try:
        wl = build_transformer_training_workload(spec, hw, cfg)
        ideal = _makespan(apply_pressurefit_policy(wl.chain))
        print(f"ideal {key} = {ideal}", file=sys.stderr, flush=True)
        return key, ideal
    except Exception as e:
        print(f"ideal {key} ERR {type(e).__name__}", file=sys.stderr, flush=True)
        return key, f"ideal:{type(e).__name__}"


def _run_one(payload):
    model_name, hw_name, cap_gb, S, M, ga, ideal = payload
    spec = load_model_presets()[model_name]
    hw = HARDWARE_PRESETS[hw_name]
    cfg = TrainingConfig(seqlen=S, num_seqs=M, grad_accum_rounds=ga)
    cap = int(cap_gb * 1e9)
    row = {
        "model": model_name, "hardware": hw_name, "cap_GB": cap_gb,
        "seqlen": S, "num_seqs": M, "ga": ga, "n_layers": spec.n_layers,
    }
    errors = []
    try:
        wl = build_transformer_training_workload(spec, hw, cfg)
    except Exception as e:
        row["errors"] = f"build:{type(e).__name__}"
        return row
    rewrites = wl.metadata["recompute_rewrites"]
    row["tasks"] = len(wl.chain.tasks)

    if isinstance(ideal, int):
        row["ideal"] = ideal
    else:
        row["ideal"] = ""
        errors.append(str(ideal))

    def build(levels, _cap=cap):
        w = build_transformer_training_workload(spec, hw, cfg, recompute=levels)
        return replace(w.chain, device_capacity=_cap)

    t0 = time.perf_counter()
    try:
        result = plan_with_recompute(
            build, rewrites, lambda b: apply_pressurefit_policy(b),
            max_wall_s=600,
        )
        # none/all/half come from the loop's own baseline + seed evaluations
        # (identical deterministic pipeline) instead of separate re-plans.
        row["none"] = result.baseline_makespan_us
        seed_mk = {
            step.converted[0]: step.makespan_us
            for step in result.history
            if step.converted and step.converted[0].startswith("<seed:")
        }
        for name in ("all", "half"):
            mk = seed_mk.get(f"<seed:{name}>")
            if mk is None:
                row[name] = ""
                errors.append(f"{name}:seed-failed")
            else:
                row[name] = mk
        rep = result.report
        mk = result.makespan_us
        row["greedy"] = mk
        row["greedy_recomputed"] = sum(1 for v in result.levels.values() if v >= 1)
        row["greedy_evals"] = len(result.history) + 1
        iw = rep.input_wait_us
        w_blame = sum(v for o, v in rep.stall_by_object.items() if o.startswith("W"))
        a_blame = sum(v for o, v in rep.stall_by_object.items() if o.startswith("A"))
        row["stall_pct"] = round(rep.stall_us / mk * 100, 1)
        row["input_wait_pct"] = round(iw / mk * 100, 1)
        row["capacity_wait_pct"] = round(rep.capacity_wait_us / mk * 100, 1)
        row["h2d_busy_pct"] = round(rep.stream_busy_us["h2d"] / mk * 100, 1)
        row["d2h_busy_pct"] = round(rep.stream_busy_us["d2h"] / mk * 100, 1)
        row["w_blame_pct"] = round(w_blame / max(1, iw) * 100, 1)
        row["a_blame_pct"] = round(a_blame / max(1, iw) * 100, 1)
        if row.get("ideal"):
            row["gap_vs_ideal_pct"] = round((mk - row["ideal"]) / row["ideal"] * 100, 1)
    except Exception as e:
        errors.append(f"greedy:{type(e).__name__}")
    row["greedy_wall_s"] = round(time.perf_counter() - t0, 2)

    valid = {
        k: row[k] for k in ("none", "all", "half", "greedy")
        if row.get(k) not in ("", None)
    }
    row["best"] = min(valid, key=valid.get) if valid else "ERR"
    row["errors"] = ";".join(errors)
    print(
        f"done {model_name} {hw_name} cap={cap_gb} S={S} M={M} ga={ga} "
        f"best={row['best']} gap={row.get('gap_vs_ideal_pct', '?')}%",
        file=sys.stderr, flush=True,
    )
    return row


def _config_key(row_or_payload) -> tuple:
    if isinstance(row_or_payload, dict):
        r = row_or_payload
        return (r["model"], r["hardware"], str(r["cap_GB"]), str(r["seqlen"]),
                str(r["num_seqs"]), str(r["ga"]))
    m, hw_name, cap, S, M, ga = row_or_payload[:6]
    return (m, hw_name, str(cap), str(S), str(M), str(ga))


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument(
        "--comprehensive", action="store_true",
        help="the pinned baseline grid (all models, full axes)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="/tmp/recompute_sweep.csv")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip configs already present in --out and append to it",
    )
    args = parser.parse_args()

    if args.comprehensive:
        models = COMPREHENSIVE_MODELS
        hw_caps, num_seqs, ga_rounds = (
            COMPREHENSIVE_HW_CAPS, COMPREHENSIVE_NUM_SEQS, COMPREHENSIVE_GA_ROUNDS,
        )
        seqlens = COMPREHENSIVE_SEQLENS
    else:
        models = MODELS
        hw_caps = QUICK_HW_CAPS if args.quick else HW_CAPS
        num_seqs = QUICK_NUM_SEQS if args.quick else NUM_SEQS
        ga_rounds = QUICK_GA_ROUNDS if args.quick else GA_ROUNDS
        seqlens = SEQLENS

    payloads = [
        (m, hw_name, cap, S, M, ga)
        for m, (hw_name, caps), S, M, ga in itertools.product(
            models, hw_caps.items(), seqlens, num_seqs, ga_rounds,
        )
        for cap in caps
    ]
    # Heaviest first so stragglers overlap instead of trailing the run.
    payloads.sort(key=lambda p: (p[5], p[3], p[4]), reverse=True)

    done_keys: set = set()
    import os
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done_keys = {_config_key(r) for r in csv.DictReader(f)}
        payloads = [p for p in payloads if _config_key(p) not in done_keys]
        print(f"resume: skipping {len(done_keys)} completed configs", flush=True)

    t0 = time.monotonic()
    # Phase 1: infinite-cap ideals, shared by every cap row of a config.
    ideal_keys = sorted({(p[0], p[1], p[3], p[4], p[5]) for p in payloads}, reverse=True)
    print(f"phase 1: {len(ideal_keys)} unique ideals, {args.workers} workers", flush=True)
    with mp.Pool(args.workers) as pool:
        ideals = dict(pool.imap_unordered(_ideal_one, ideal_keys, chunksize=1))
    print(f"phase 1 done in {time.monotonic()-t0:.0f}s", flush=True)

    payloads = [
        (*p, ideals[(p[0], p[1], p[3], p[4], p[5])]) for p in payloads
    ]
    print(f"phase 2: {len(payloads)} configs, {args.workers} workers", flush=True)
    mode = "a" if (args.resume and done_keys) else "w"
    n = 0
    # imap_unordered with chunksize=1: heavy configs can't pile up in one
    # worker's pre-assigned chunk. Rows are appended (and flushed) as they
    # complete so a long run is resumable.
    with open(args.out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        with mp.Pool(args.workers) as pool:
            for row in pool.imap_unordered(_run_one, payloads, chunksize=1):
                writer.writerow(row)
                f.flush()
                n += 1
    print(f"Wrote {args.out} ({n} new configs) in {time.monotonic()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
