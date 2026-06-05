"""Canonical llama3-8B sweep across hardware, capacity, batch, sequence length,
and every policy registered via `get_all_policies()`.

Run:
    python scripts/canonical_sweep.py            # default: full sweep
    python scripts/canonical_sweep.py --quick              # tiny smoke subset

Output: CSV at /tmp/canonical_sweep.csv (one row per config, one column per
policy with the makespan in microseconds).
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from itertools import product

from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.transformer import build_transformer_training_workload
from dataflow_sim.workloads.training.transformer import TrainingConfig
from dataflow_sim.policies import get_all_policies
from dataflow_sim.engine.simulator import run as sim_run


# Sweep axes
HARDWARE_CAPS_GB = {
    "H100": [20, 30, 40, 80, None],   # None = unlimited
    "RTX_5090": [8, 16, 24, 32],
}
NUM_SEQS = [1, 2, 4, 8]
SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

QUICK_HARDWARE_CAPS_GB = {"H100": [40, None]}
QUICK_NUM_SEQS = [1, 4]
QUICK_SEQLENS = [2048, 8192]


def all_configs(quick: bool = False):
    if quick:
        hw_caps, ms, sls = QUICK_HARDWARE_CAPS_GB, QUICK_NUM_SEQS, QUICK_SEQLENS
    else:
        hw_caps, ms, sls = HARDWARE_CAPS_GB, NUM_SEQS, SEQLENS
    for hw_name, caps in hw_caps.items():
        for cap_gb, M, S in product(caps, ms, sls):
            yield (hw_name, cap_gb, M, S)


def _run_one(args):
    hw_name, cap_gb, M, S, policy_name = args
    tag = f"hw={hw_name} cap={cap_gb} M={M} S={S} pol={policy_name}"
    print(f"START {tag}", file=sys.stderr, flush=True)
    models = load_model_presets()
    spec = models["llama3_8B"]
    hw = HARDWARE_PRESETS[hw_name]
    cfg = TrainingConfig(seqlen=S, num_seqs=M)
    cap_bytes = int(cap_gb * 1e9) if cap_gb is not None else None
    t0 = time.monotonic()
    try:
        bare = build_transformer_training_workload(spec, hw, cfg).chain
        if cap_bytes is not None:
            bare = replace(bare, device_capacity=cap_bytes)
        policies = dict(get_all_policies())
        annotated = policies[policy_name](bare)
        ms_value = max(iv.end for iv in sim_run(annotated).task_intervals)
        dur = time.monotonic() - t0
        print(f"DONE  {tag}  ms={ms_value}  ({dur:.2f}s)", file=sys.stderr, flush=True)
        return (hw_name, cap_gb, M, S, policy_name, ms_value, None, dur)
    except Exception as e:
        dur = time.monotonic() - t0
        print(f"ERR   {tag}  {type(e).__name__}  ({dur:.2f}s)", file=sys.stderr, flush=True)
        return (hw_name, cap_gb, M, S, policy_name, None, type(e).__name__, dur)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny subset for smoke")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="/tmp/canonical_sweep.csv")
    args = parser.parse_args()

    policy_names = [name for name, _ in get_all_policies()]
    configs = list(all_configs(quick=args.quick))
    pairs = [(*c, p) for c in configs for p in policy_names]
    print(f"{len(configs)} configs × {len(policy_names)} policies = {len(pairs)} pairs", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    t_start = time.monotonic()
    results = []
    # mp.Pool.imap_unordered has a reproducible tail-batch hang on this workload
    # (last ~14 results stuck in IPC queue, parent never drains). ProcessPoolExecutor
    # is also fragile (BrokenProcessPool on worker death is unrecoverable).
    # Workaround: apply_async with callbacks + a stall watchdog. The callback drains
    # results as workers complete; the watchdog gives up after STALL_GRACE_S of no
    # progress and writes partial results.
    STALL_GRACE_S = 120
    last_progress_t = time.monotonic()
    last_count = 0

    def _on_result(r):
        results.append(r)

    def _on_error(e):
        print(f"  [worker error: {type(e).__name__}: {str(e)[:80]}]", flush=True)

    with mp.Pool(args.workers) as pool:
        for p in pairs:
            pool.apply_async(_run_one, (p,), callback=_on_result, error_callback=_on_error)
        pool.close()
        # Poll for completion or stall
        while len(results) < len(pairs):
            time.sleep(2)
            now = time.monotonic()
            if len(results) > last_count:
                last_count = len(results)
                last_progress_t = now
                elapsed = now - t_start
                print(
                    f"[{len(results)}/{len(pairs)}] {elapsed:.0f}s elapsed; "
                    f"avg {elapsed/max(1,len(results)):.2f}s/pair; "
                    f"ETA {elapsed * (len(pairs) - len(results)) / max(1,len(results)):.0f}s",
                    flush=True,
                )
            elif now - last_progress_t > STALL_GRACE_S:
                print(
                    f"⚠ stall watchdog: {len(results)}/{len(pairs)} after "
                    f"{now-last_progress_t:.0f}s of no progress. Writing partial CSV.",
                    flush=True,
                )
                pool.terminate()
                break
        pool.terminate()  # cleanup whether finished or stalled

    # Pivot: one row per (hw, cap, M, S); columns per policy
    table: dict[tuple, dict[str, object]] = {}
    err_table: dict[tuple, dict[str, str]] = {}
    for hw_name, cap_gb, M, S, policy_name, ms_value, err, _wall in results:
        key = (hw_name, cap_gb, M, S)
        table.setdefault(key, {})[policy_name] = ms_value
        if err:
            err_table.setdefault(key, {})[policy_name] = err

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hardware", "cap_GB", "num_seqs", "seqlen", *policy_names, "best_policy"])
        for key in sorted(table.keys(), key=lambda k: (k[0], k[1] if k[1] is not None else 1e9, k[2], k[3])):
            row = table[key]
            values = {p: row.get(p) for p in policy_names}
            valid = {p: v for p, v in values.items() if v is not None}
            best = min(valid, key=valid.get) if valid else ""
            cap_str = "unlimited" if key[1] is None else str(key[1])
            w.writerow([key[0], cap_str, key[2], key[3], *(values[p] if values[p] is not None else "ERR" for p in policy_names), best])

    print(f"Wrote {args.out} ({len(table)} configs)", flush=True)
    print(f"Total wall: {time.monotonic() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
