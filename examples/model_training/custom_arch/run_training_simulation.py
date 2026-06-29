"""Run the custom TinyMixer model through planner + simulator APIs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dataflow_sim.engine.simulator import run as simulator_run
from dataflow_sim.planning.recompute import plan_with_recompute
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.dataflow_builder import DTypePolicy, TrainingConfig
from dataflow_sim.workloads.summary import compute_workload_summary

from custom_model import TinyMixerConfig, TinyMixerForTraining


def build_inputs(args: argparse.Namespace):
    config = TinyMixerConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        classes=args.classes,
        wide_every=args.wide_every,
    )
    training = TrainingConfig(
        seqlen=args.seqlen,
        num_seqs=args.num_seqs,
        grad_accum_rounds=args.grad_accum_rounds,
        num_steps=args.num_steps,
        optimizer=args.optimizer,
        final_model_state_on_backing=args.final_model_state_on_backing,
    )
    dtype_policy = DTypePolicy(
        param=args.dtype,
        activation=args.dtype,
        gradient=args.dtype,
        optimizer_state=args.dtype,
    )
    return config, training, dtype_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--classes", type=int, default=32_000)
    parser.add_argument(
        "--wide-every",
        type=int,
        default=0,
        help="Every Nth layer uses 2x hidden dim; 0 keeps all layers identical.",
    )
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--grad-accum-rounds", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=["none", "sgd", "adamw"],
        default="adamw",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--hardware", choices=sorted(HARDWARE_PRESETS), default="H100")
    parser.add_argument("--fast-memory-gb", type=float, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--recompute-iters", type=int, default=4)
    parser.add_argument("--final-model-state-on-backing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config, training, dtype_policy = build_inputs(args)
    model = TinyMixerForTraining(config)
    hw = HARDWARE_PRESETS[args.hardware]
    cap_bytes = (
        int(round(args.fast_memory_gb * (1024**3)))
        if args.fast_memory_gb is not None
        else None
    )

    def build_workload(levels=None):
        return model.build_training_workload(
            training,
            hw,
            input_shape=(training.tokens, config.d_model),
            dtype_policy=dtype_policy,
            recompute=levels,
        )

    def apply_policy(chain):
        return apply_pressurefit_policy(chain, fast_memory_capacity=cap_bytes)

    workload = build_workload()
    recompute_levels = {}
    if args.recompute:
        result = plan_with_recompute(
            lambda levels: build_workload(levels).chain,
            workload.metadata["recompute_rewrites"],
            apply_policy,
            max_iters=args.recompute_iters,
            max_wall_s=10,
        )
        recompute_levels = dict(result.levels)
        workload = build_workload(recompute_levels)
        annotated_plan = result.chain
    else:
        annotated_plan = apply_policy(workload.chain)

    log = simulator_run(annotated_plan, snapshots=False)
    summary = compute_workload_summary(workload, log)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        program_json = json.dumps(workload.metadata["program"], indent=2) + "\n"
        (args.out_dir / "webapp_upload.dataflow.json").write_text(program_json)
        (args.out_dir / "program.dataflow.json").write_text(
            program_json
        )
        (args.out_dir / "unannotated_plan.json").write_text(
            json.dumps(asdict(workload.chain), indent=2) + "\n"
        )
        (args.out_dir / "annotated_plan.json").write_text(
            json.dumps(asdict(annotated_plan), indent=2) + "\n"
        )
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    print(json.dumps({
        "summary": summary,
        "task_count": len(annotated_plan.tasks),
        "recompute_selected": sum(
            1 for level in recompute_levels.values() if level >= 1
        ),
        "outputs": (
            {
                "directory": str(args.out_dir),
                "webapp_upload": "webapp_upload.dataflow.json",
                "program": "program.dataflow.json",
                "unannotated_plan": "unannotated_plan.json",
                "annotated_plan": "annotated_plan.json",
                "summary": "summary.json",
            }
            if args.out_dir is not None
            else None
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
