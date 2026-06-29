"""Export a transformer training workload with the modular dataflow builder.

This example uses the new side-by-side workload stack:

    ops -> modules -> model family -> training_builder -> DataflowProgram JSON

No PyTorch dependency is involved. The model is traced symbolically from a
token-major input shape ``[num_tokens, d_model]``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataflow_sim.workloads.dataflow_builder import (
    DTypePolicy,
    TrainingConfig,
    trace_training_model,
)
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining
from dataflow_sim.workloads.models.olmoe import OLMoEConfig, OLMoEForTraining
from dataflow_sim.workloads.models.qwen3 import Qwen3Config, Qwen3ForTraining
from dataflow_sim.workloads.models.qwen3_moe import Qwen3MoEConfig, Qwen3MoEForTraining


MODEL_FACTORIES = {
    "llama3": (Llama3Config, Llama3ForTraining, "8B"),
    "qwen3": (Qwen3Config, Qwen3ForTraining, "8B"),
    "qwen3_moe": (Qwen3MoEConfig, Qwen3MoEForTraining, "30B-3B"),
    "olmoe": (OLMoEConfig, OLMoEForTraining, "7B-1B"),
}


DIMENSION_ARGS = {
    "vocab_size",
    "n_layers",
    "d_model",
    "head_dim",
    "n_heads",
    "n_kv_heads",
    "expert_dim",
    "num_shared_experts",
    "num_routed_experts",
    "top_k",
    "qk_norm",
}


def _dimension_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in DIMENSION_ARGS:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def build_program(args: argparse.Namespace):
    config_cls, model_cls, default_scale = MODEL_FACTORIES[args.model]
    scale = args.scale or default_scale
    config = config_cls.preset(scale, **_dimension_overrides(args))
    training = TrainingConfig(
        seqlen=args.seqlen,
        num_seqs=args.num_seqs,
        grad_accum_rounds=args.grad_accum_rounds,
        num_steps=args.num_steps,
        optimizer=args.optimizer,
        final_model_state_on_backing=args.final_model_state_on_backing,
    )
    dtype_policy = DTypePolicy(
        param=args.param_dtype or args.dtype,
        activation=args.activation_dtype or args.dtype,
        gradient=args.gradient_dtype or args.dtype,
        optimizer_state=args.optimizer_state_dtype or args.dtype,
    )
    model = model_cls(config)
    return trace_training_model(
        model,
        (training.tokens, config.d_model),
        training,
        name=args.name or f"{args.model}-{scale}-training",
        dtype_policy=dtype_policy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORIES),
        default="llama3",
        help="Model family to instantiate.",
    )
    parser.add_argument(
        "--scale",
        default=None,
        help="Family scale preset, e.g. 8B, 70B, 405B, 32B, 30B-3B, or 7B-1B.",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--vocab-size", dest="vocab_size", type=int, default=None)
    parser.add_argument("--n-layers", dest="n_layers", type=int, default=None)
    parser.add_argument("--d-model", dest="d_model", type=int, default=None)
    parser.add_argument("--head-dim", dest="head_dim", type=int, default=None)
    parser.add_argument("--n-heads", dest="n_heads", type=int, default=None)
    parser.add_argument("--n-kv-heads", dest="n_kv_heads", type=int, default=None)
    parser.add_argument("--expert-dim", dest="expert_dim", type=int, default=None)
    parser.add_argument("--num-shared-experts", dest="num_shared_experts", type=int, default=None)
    parser.add_argument("--num-routed-experts", dest="num_routed_experts", type=int, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--qk-norm", dest="qk_norm", action="store_true", default=None)
    parser.add_argument("--no-qk-norm", dest="qk_norm", action="store_false")
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--grad-accum-rounds", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=["none", "adamw", "muon", "sgd"],
        default="adamw",
    )
    parser.add_argument("--final-model-state-on-backing", action="store_true")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--param-dtype", default=None)
    parser.add_argument("--activation-dtype", default=None)
    parser.add_argument("--gradient-dtype", default=None)
    parser.add_argument("--optimizer-state-dtype", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("transformer_training.dataflow.json"),
        help="Path to write the exported DataflowProgram JSON.",
    )
    args = parser.parse_args()

    program = build_program(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(program.model_dump(mode="json"), indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
