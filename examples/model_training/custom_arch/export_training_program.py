"""Export the custom TinyMixer training workload as DataflowProgram JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataflow_sim.workloads.dataflow_builder import DTypePolicy, TrainingConfig

from custom_model import TinyMixerConfig, TinyMixerForTraining


def build_program(args: argparse.Namespace):
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
    )
    dtype_policy = DTypePolicy(
        param=args.dtype,
        activation=args.dtype,
        gradient=args.dtype,
        optimizer_state=args.dtype,
    )
    return TinyMixerForTraining(config).build_training_program(
        training,
        input_shape=(training.tokens, config.d_model),
        dtype_policy=dtype_policy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--classes", type=int, default=32_000)
    parser.add_argument("--wide-every", type=int, default=0)
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tiny_mixer_training.dataflow.json"),
    )
    args = parser.parse_args()

    program = build_program(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(program.model_dump(mode="json"), indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
