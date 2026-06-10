"""Canonical llama3-8B sweep across hardware, capacity, batch, sequence length,
optimizer, gradient-accumulation rounds, and policy.

Run:
    python scripts/canonical_sweep.py --quick
    python scripts/canonical_sweep.py --canonical
    python scripts/canonical_sweep.py --canonical \
        --optimizers adamw,muon \
        --grad-accum-rounds 1,2,4,8,32 \
        --pressurefit-mode full \
        --pressurefit-candidates-out /tmp/pressurefit_candidates.csv

The policy CSV has one row per config and one makespan column per policy.
When requested, the PressureFit candidate CSV has one row per full-mode
candidate per config, including skipped/error candidates.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from dataflow_sim.core.schema import TaskChain
from dataflow_sim.engine.simulator import run as sim_run
from dataflow_sim.policies import get_all_policies
from dataflow_sim.policies.pressurefit import (
    PressureFitDiagnostics,
    PressureFitPortfolioMode,
    apply_pressurefit_policy,
    plan_pressurefit_policy,
)
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.optimizers import OptimizerMode
from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_transformer_training_workload,
)


HARDWARE_CAPS_GB = {
    "H100": [20, 30, 40, 80, None],
    "RTX_5090": [8, 16, 24, 32],
}
NUM_SEQS = [1, 2, 4, 8]
SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

QUICK_HARDWARE_CAPS_GB = {"H100": [40, None]}
QUICK_NUM_SEQS = [1, 4]
QUICK_SEQLENS = [2048, 8192]

OPTIMIZER_CHOICES: tuple[OptimizerMode, ...] = ("none", "adamw", "muon")
DEFAULT_OPTIMIZERS: tuple[OptimizerMode, ...] = ("none",)
DEFAULT_GRAD_ACCUM_ROUNDS = (1,)
ANALYSIS_OPTIMIZERS: tuple[OptimizerMode, ...] = ("adamw", "muon")
ANALYSIS_GRAD_ACCUM_ROUNDS = (1, 2, 4, 8, 32)
CONFIG_KEY_FIELDS = (
    "hardware",
    "cap_GB",
    "num_seqs",
    "seqlen",
    "optimizer",
    "grad_accum_rounds",
    "num_steps",
    "final_model_state_on_host",
)

SweepRow = tuple[
    dict[str, Any],
    dict[str, int | None],
    dict[str, str],
    list[dict[str, Any]],
    float,
]


def _parse_optimizer_list(raw: str) -> tuple[OptimizerMode, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("optimizer list cannot be empty")
    invalid = [value for value in values if value not in OPTIMIZER_CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown optimizer(s): {', '.join(invalid)}"
        )
    return values  # type: ignore[return-value]


def _parse_int_list(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    if not values:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("values must be >= 1")
    return values


def _parse_string_list(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def _axes(args: argparse.Namespace):
    if args.quick:
        return QUICK_HARDWARE_CAPS_GB, QUICK_NUM_SEQS, QUICK_SEQLENS
    return HARDWARE_CAPS_GB, NUM_SEQS, SEQLENS


def all_configs(args: argparse.Namespace):
    hw_caps, num_seqs, seqlens = _axes(args)
    for hw_name, caps in hw_caps.items():
        for cap_gb, M, S, optimizer, grad_accum_rounds in product(
            caps,
            num_seqs,
            seqlens,
            args.optimizers,
            args.grad_accum_rounds,
        ):
            yield {
                "hardware": hw_name,
                "cap_GB": "unlimited" if cap_gb is None else cap_gb,
                "cap_gb_raw": cap_gb,
                "num_seqs": M,
                "seqlen": S,
                "optimizer": optimizer,
                "grad_accum_rounds": grad_accum_rounds,
                "num_steps": args.num_steps,
                "final_model_state_on_host": args.final_model_state_on_host,
            }


def _cap_bytes(cap_gb: int | None) -> int | None:
    if cap_gb is None:
        return None
    # Keep the historical canonical-sweep convention: labels are GB, not GiB.
    return int(cap_gb * 1e9)


def _policy_fns(pressurefit_mode: PressureFitPortfolioMode):
    policies = []
    for name, fn in get_all_policies():
        if name == "pressurefit":
            policies.append((
                name,
                lambda b, mode=pressurefit_mode: apply_pressurefit_policy(
                    b,
                    device_capacity=b.device_capacity,
                    portfolio_mode=mode,
                ),
            ))
        else:
            policies.append((name, fn))
    return policies


def _build_bare(config: dict[str, Any]) -> TaskChain:
    models = load_model_presets()
    spec = models["llama3_8B"]
    hw = HARDWARE_PRESETS[config["hardware"]]
    cfg = TrainingConfig(
        seqlen=config["seqlen"],
        num_seqs=config["num_seqs"],
        grad_accum_rounds=config["grad_accum_rounds"],
        num_steps=config["num_steps"],
        optimizer=config["optimizer"],
        final_model_state_on_host=config["final_model_state_on_host"],
    )
    bare = build_transformer_training_workload(spec, hw, cfg).chain
    return replace(bare, device_capacity=_cap_bytes(config["cap_gb_raw"]))


def _makespan_us(chain: TaskChain) -> int:
    log = sim_run(chain, snapshots=False)
    return max(iv.end for iv in log.task_intervals)


def _config_public_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "hardware": config["hardware"],
        "cap_GB": config["cap_GB"],
        "num_seqs": config["num_seqs"],
        "seqlen": config["seqlen"],
        "optimizer": config["optimizer"],
        "grad_accum_rounds": config["grad_accum_rounds"],
        "num_steps": config["num_steps"],
        "final_model_state_on_host": config["final_model_state_on_host"],
    }


def _config_key(config: dict[str, Any]) -> tuple[str, ...]:
    public = _config_public_fields(config)
    return tuple(str(public[field]) for field in CONFIG_KEY_FIELDS)


def _completed_config_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        raise SystemExit(f"--resume-from file does not exist: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or ())
        missing = [field for field in CONFIG_KEY_FIELDS if field not in fieldnames]
        if missing:
            raise SystemExit(
                f"--resume-from file is missing required column(s): {', '.join(missing)}"
            )
        return {tuple(row[field] for field in CONFIG_KEY_FIELDS) for row in reader}


def _candidate_rows(
    config: dict[str, Any],
    diagnostics: PressureFitDiagnostics,
) -> list[dict[str, Any]]:
    best = diagnostics.selected_makespan_us
    rows: list[dict[str, Any]] = []
    for candidate in diagnostics.candidates:
        makespan = candidate.makespan_us
        gap_us = "" if makespan is None else makespan - best
        gap_pct = (
            ""
            if makespan is None or best == 0
            else (makespan - best) / best * 100.0
        )
        rows.append({
            **_config_public_fields(config),
            "pressurefit_mode": diagnostics.portfolio_mode,
            "effective_pressurefit_mode": diagnostics.effective_portfolio_mode,
            "candidate_name": candidate.name,
            "family": candidate.family,
            "status": candidate.status,
            "selected": candidate.selected,
            "makespan_us": "" if makespan is None else makespan,
            "best_makespan_us": best,
            "gap_vs_best_us": gap_us,
            "gap_vs_best_pct": gap_pct,
            "wall_time_s": candidate.wall_time_s,
            "seed": candidate.seed,
            "pack_inbound": candidate.pack_inbound,
            "extend_inbound": candidate.extend_inbound,
            "respect_interval_start": candidate.respect_interval_start,
            "latest_inbound": candidate.latest_inbound,
            "reserve_pressure": candidate.reserve_pressure,
            "protected_count": candidate.protected_count,
            "protected_bytes": candidate.protected_bytes,
            "error": candidate.error or "",
        })
    return rows


def _candidate_error_row(
    config: dict[str, Any],
    pressurefit_mode: PressureFitPortfolioMode,
    error: Exception,
) -> dict[str, Any]:
    return {
        **_config_public_fields(config),
        "pressurefit_mode": pressurefit_mode,
        "effective_pressurefit_mode": "",
        "candidate_name": "",
        "family": "",
        "status": "planner_error",
        "selected": False,
        "makespan_us": "",
        "best_makespan_us": "",
        "gap_vs_best_us": "",
        "gap_vs_best_pct": "",
        "wall_time_s": "",
        "seed": "",
        "pack_inbound": "",
        "extend_inbound": "",
        "respect_interval_start": "",
        "latest_inbound": "",
        "reserve_pressure": "",
        "protected_count": "",
        "protected_bytes": "",
        "error": f"{type(error).__name__}: {error}",
    }


def _run_one_config(payload):
    config, policy_names, pressurefit_mode, collect_candidates = payload
    tag = (
        f"hw={config['hardware']} cap={config['cap_GB']} "
        f"M={config['num_seqs']} S={config['seqlen']} "
        f"opt={config['optimizer']} ga={config['grad_accum_rounds']}"
    )
    print(f"START {tag}", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    policy_results: dict[str, int | None] = {}
    policy_errors: dict[str, str] = {}
    candidate_rows: list[dict[str, Any]] = []

    try:
        bare = _build_bare(config)
    except Exception as e:
        for policy_name in policy_names:
            policy_results[policy_name] = None
            policy_errors[policy_name] = f"{type(e).__name__}: {e}"
        if collect_candidates:
            candidate_rows.append(_candidate_error_row(config, pressurefit_mode, e))
        dur = time.monotonic() - t0
        print(f"ERR   {tag} build {type(e).__name__} ({dur:.2f}s)", file=sys.stderr, flush=True)
        return config, policy_results, policy_errors, candidate_rows, dur

    pressurefit_done = False
    if collect_candidates:
        try:
            _chain, diagnostics = plan_pressurefit_policy(
                bare,
                device_capacity=bare.device_capacity,
                portfolio_mode=pressurefit_mode,
            )
            candidate_rows.extend(_candidate_rows(config, diagnostics))
            policy_results["pressurefit"] = diagnostics.selected_makespan_us
            pressurefit_done = True
        except Exception as e:
            candidate_rows.append(_candidate_error_row(config, pressurefit_mode, e))
            policy_results["pressurefit"] = None
            policy_errors["pressurefit"] = f"{type(e).__name__}: {e}"

    policies_by_name = dict(_policy_fns(pressurefit_mode))
    for policy_name in policy_names:
        if policy_name == "pressurefit" and pressurefit_done:
            continue
        try:
            annotated = policies_by_name[policy_name](bare)
            policy_results[policy_name] = _makespan_us(annotated)
        except Exception as e:
            policy_results[policy_name] = None
            policy_errors[policy_name] = f"{type(e).__name__}: {e}"

    dur = time.monotonic() - t0
    valid = {name: value for name, value in policy_results.items() if value is not None}
    best = min(valid, key=valid.get) if valid else "ERR"
    print(f"DONE  {tag} best={best} ({dur:.2f}s)", file=sys.stderr, flush=True)
    return config, policy_results, policy_errors, candidate_rows, dur


def _write_policy_csv(
    path: Path,
    rows: list[SweepRow],
    policy_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "hardware",
            "cap_GB",
            "num_seqs",
            "seqlen",
            "optimizer",
            "grad_accum_rounds",
            "num_steps",
            "final_model_state_on_host",
            *policy_names,
            "best_policy",
        ])
        for config, policy_results, _policy_errors, _candidate_rows, _dur in sorted(
            rows,
            key=lambda row: (
                row[0]["hardware"],
                row[0]["cap_gb_raw"] if row[0]["cap_gb_raw"] is not None else 1e18,
                row[0]["num_seqs"],
                row[0]["seqlen"],
                row[0]["optimizer"],
                row[0]["grad_accum_rounds"],
            ),
        ):
            values = {name: policy_results.get(name) for name in policy_names}
            valid = {name: value for name, value in values.items() if value is not None}
            best = min(valid, key=valid.get) if valid else ""
            public = _config_public_fields(config)
            writer.writerow([
                public["hardware"],
                public["cap_GB"],
                public["num_seqs"],
                public["seqlen"],
                public["optimizer"],
                public["grad_accum_rounds"],
                public["num_steps"],
                public["final_model_state_on_host"],
                *(
                    values[name] if values[name] is not None else "ERR"
                    for name in policy_names
                ),
                best,
            ])


def _write_policy_errors(
    path: Path,
    rows: list[SweepRow],
    policy_names: list[str],
) -> bool:
    error_rows: list[dict[str, Any]] = []
    for config, _policy_results, policy_errors, _candidate_rows, _dur in rows:
        for policy_name in policy_names:
            error = policy_errors.get(policy_name)
            if error:
                error_rows.append({
                    **_config_public_fields(config),
                    "policy": policy_name,
                    "error": error,
                })
    if not error_rows:
        if path.exists():
            path.unlink()
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "hardware",
            "cap_GB",
            "num_seqs",
            "seqlen",
            "optimizer",
            "grad_accum_rounds",
            "num_steps",
            "final_model_state_on_host",
            "policy",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)
    return True


def _write_candidate_csv(path: Path, rows: list[SweepRow]) -> int:
    candidate_rows: list[dict[str, Any]] = []
    for _config, _policy_results, _policy_errors, per_config, _dur in rows:
        candidate_rows.extend(per_config)
    if not candidate_rows:
        return 0

    fieldnames = [
        "hardware",
        "cap_GB",
        "num_seqs",
        "seqlen",
        "optimizer",
        "grad_accum_rounds",
        "num_steps",
        "final_model_state_on_host",
        "pressurefit_mode",
        "effective_pressurefit_mode",
        "candidate_name",
        "family",
        "status",
        "selected",
        "makespan_us",
        "best_makespan_us",
        "gap_vs_best_us",
        "gap_vs_best_pct",
        "wall_time_s",
        "seed",
        "pack_inbound",
        "extend_inbound",
        "respect_interval_start",
        "latest_inbound",
        "reserve_pressure",
        "protected_count",
        "protected_bytes",
        "error",
    ]
    candidate_rows.sort(key=lambda row: (
        row["hardware"],
        row["cap_GB"] if row["cap_GB"] != "unlimited" else 1e18,
        row["num_seqs"],
        row["seqlen"],
        row["optimizer"],
        row["grad_accum_rounds"],
        row["candidate_name"],
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    return len(candidate_rows)


def _write_outputs(
    *,
    policy_out: Path,
    errors_out: Path | None,
    candidate_out: Path | None,
    rows: list[SweepRow],
    policy_names: list[str],
) -> tuple[int, bool]:
    _write_policy_csv(policy_out, rows, policy_names)
    wrote_errors = False
    if errors_out is not None:
        wrote_errors = _write_policy_errors(errors_out, rows, policy_names)
    candidate_count = 0
    if candidate_out is not None:
        candidate_count = _write_candidate_csv(candidate_out, rows)
    return candidate_count, wrote_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="tiny subset for smoke")
    group.add_argument("--canonical", action="store_true", help="full canonical grid")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="/tmp/canonical_sweep.csv")
    parser.add_argument("--errors-out", default="", help="optional policy-error CSV")
    parser.add_argument(
        "--pressurefit-candidates-out",
        default="",
        help="optional CSV with one row per PressureFit candidate",
    )
    parser.add_argument(
        "--pressurefit-mode",
        choices=["auto", "fast", "full"],
        default="auto",
        help="PressureFit portfolio mode for the pressurefit policy/candidates",
    )
    parser.add_argument(
        "--optimizers",
        type=_parse_optimizer_list,
        default=DEFAULT_OPTIMIZERS,
        help="comma-separated optimizer modes",
    )
    parser.add_argument(
        "--grad-accum-rounds",
        type=_parse_int_list,
        default=DEFAULT_GRAD_ACCUM_ROUNDS,
        help="comma-separated gradient-accumulation rounds",
    )
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--final-model-state-on-host", action="store_true")
    parser.add_argument(
        "--policies",
        type=_parse_string_list,
        default=("all",),
        help="comma-separated policies to run, or all",
    )
    parser.add_argument(
        "--optimizer-grad-analysis",
        action="store_true",
        help="shortcut for --optimizers adamw,muon --grad-accum-rounds 1,2,4,8,32",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="rewrite partial CSV outputs after this many completed configs; 0 disables",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="policy CSV whose completed config rows should be skipped",
    )
    parser.add_argument("--stall-grace-s", type=int, default=600)
    args = parser.parse_args()

    if args.optimizer_grad_analysis:
        args.optimizers = ANALYSIS_OPTIMIZERS
        args.grad_accum_rounds = ANALYSIS_GRAD_ACCUM_ROUNDS
        if args.pressurefit_mode == "auto":
            args.pressurefit_mode = "full"

    all_policy_names = [name for name, _fn in get_all_policies()]
    if args.policies == ("all",):
        policy_names = all_policy_names
    else:
        unknown = [name for name in args.policies if name not in all_policy_names]
        if unknown:
            raise SystemExit(f"unknown policy name(s): {', '.join(unknown)}")
        policy_names = list(args.policies)
    if args.resume_from:
        resume_from = Path(args.resume_from)
        if resume_from.expanduser().resolve() == Path(args.out).expanduser().resolve():
            raise SystemExit("--resume-from and --out must be different files")
    configs = list(all_configs(args))
    if args.resume_from:
        completed_keys = _completed_config_keys(Path(args.resume_from))
        original_count = len(configs)
        configs = [config for config in configs if _config_key(config) not in completed_keys]
        print(
            f"Resume: skipping {original_count - len(configs)} completed configs "
            f"from {args.resume_from}",
            flush=True,
        )
    collect_candidates = bool(args.pressurefit_candidates_out)
    total_policy_runs = len(configs) * len(policy_names)
    print(f"{len(configs)} configs x {len(policy_names)} policies = {total_policy_runs} policy runs")
    if collect_candidates:
        print(f"PressureFit candidate diagnostics: mode={args.pressurefit_mode}")
    print(f"Workers: {args.workers}", flush=True)

    t_start = time.monotonic()
    rows: list[SweepRow] = []
    last_progress_t = time.monotonic()
    last_count = 0
    last_checkpoint_count = 0
    policy_out = Path(args.out)
    errors_out = Path(args.errors_out) if args.errors_out else None
    candidate_out = (
        Path(args.pressurefit_candidates_out)
        if args.pressurefit_candidates_out
        else None
    )

    payloads = [
        (config, policy_names, args.pressurefit_mode, collect_candidates)
        for config in configs
    ]

    def _on_result(result):
        rows.append(result)

    def _on_error(error):
        print(
            f"  [worker error: {type(error).__name__}: {str(error)[:120]}]",
            flush=True,
        )

    with mp.Pool(args.workers) as pool:
        for payload in payloads:
            pool.apply_async(
                _run_one_config,
                (payload,),
                callback=_on_result,
                error_callback=_on_error,
            )
        pool.close()
        while len(rows) < len(payloads):
            time.sleep(2)
            now = time.monotonic()
            if len(rows) > last_count:
                last_count = len(rows)
                last_progress_t = now
                elapsed = now - t_start
                eta = elapsed * (len(payloads) - len(rows)) / max(1, len(rows))
                print(
                    f"[{len(rows)}/{len(payloads)}] {elapsed:.0f}s elapsed; "
                    f"avg {elapsed / max(1, len(rows)):.2f}s/config; "
                    f"ETA {eta:.0f}s",
                    flush=True,
                )
                if (
                    args.checkpoint_every > 0
                    and len(rows) - last_checkpoint_count >= args.checkpoint_every
                ):
                    candidate_count, _wrote_errors = _write_outputs(
                        policy_out=policy_out,
                        errors_out=errors_out,
                        candidate_out=candidate_out,
                        rows=rows,
                        policy_names=policy_names,
                    )
                    last_checkpoint_count = len(rows)
                    print(
                        f"checkpoint: wrote {len(rows)} configs"
                        + (
                            f", {candidate_count} candidate rows"
                            if candidate_out is not None
                            else ""
                        ),
                        flush=True,
                    )
            elif now - last_progress_t > args.stall_grace_s:
                print(
                    f"stall watchdog: {len(rows)}/{len(payloads)} after "
                    f"{now - last_progress_t:.0f}s of no progress. Writing partial CSV.",
                    flush=True,
                )
                pool.terminate()
                break
        pool.terminate()

    candidate_count, wrote_errors = _write_outputs(
        policy_out=policy_out,
        errors_out=errors_out,
        candidate_out=candidate_out,
        rows=rows,
        policy_names=policy_names,
    )
    print(f"Wrote {policy_out} ({len(rows)} configs)")
    if errors_out is not None and wrote_errors:
        print(f"Wrote {errors_out}")
    if candidate_out is not None:
        print(f"Wrote {candidate_out} ({candidate_count} candidate rows)")

    print(f"Total wall: {time.monotonic() - t_start:.1f}s", flush=True)
    return 0 if len(rows) == len(payloads) else 2


if __name__ == "__main__":
    raise SystemExit(main())
