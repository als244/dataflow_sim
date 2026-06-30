"""Export a built-in model-family training workload.

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
from dataflow_sim.workloads.models.registry import MODEL_FAMILIES


MODEL_FACTORIES = {
    key: (entry.config_cls, entry.builder_cls, entry.presets[0])
    for key, entry in MODEL_FAMILIES.items()
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
    "intermediate_size",
    "full_attention_interval",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "gdn_chunk_size",
    "router_aux_loss_coef",
    "mtp_num_hidden_layers",
    "first_k_dense_replace",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "index_topk_freq",
    "index_skip_topk_offset",
    "train_indexer",
    "routed_scaling_factor",
    "scoring_func",
    "shared_expert_dim",
    "mamba_num_heads",
    "mamba_head_dim",
    "ssm_state_size",
    "conv_kernel",
    "mamba_chunk_size",
    "n_groups",
    "hybrid_override_pattern",
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
        expert_dispatch=args.expert_dispatch_dtype or args.dtype,
        gradient=args.gradient_dtype or args.dtype,
        optimizer_state=args.optimizer_state_dtype or args.dtype,
        compute=args.compute_precision or args.dtype,
        expert_param=args.expert_weight_dtype or args.dtype,
        expert_compute=args.expert_compute_precision or args.dtype,
        indexer_activation=args.indexer_activation_dtype or "fp8",
        indexer_compute=args.indexer_compute_precision or "fp8",
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
        help="Family scale preset, e.g. 8B, 30B-3B, qwen3_5_27B, or 1T-32B.",
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
    parser.add_argument("--intermediate-size", dest="intermediate_size", type=int, default=None)
    parser.add_argument("--full-attention-interval", dest="full_attention_interval", type=int, default=None)
    parser.add_argument("--linear-num-key-heads", dest="linear_num_key_heads", type=int, default=None)
    parser.add_argument("--linear-key-head-dim", dest="linear_key_head_dim", type=int, default=None)
    parser.add_argument("--linear-num-value-heads", dest="linear_num_value_heads", type=int, default=None)
    parser.add_argument("--linear-value-head-dim", dest="linear_value_head_dim", type=int, default=None)
    parser.add_argument("--linear-conv-kernel-dim", dest="linear_conv_kernel_dim", type=int, default=None)
    parser.add_argument("--gdn-chunk-size", dest="gdn_chunk_size", type=int, default=None)
    parser.add_argument("--router-aux-loss-coef", dest="router_aux_loss_coef", type=float, default=None)
    parser.add_argument("--mtp-num-hidden-layers", dest="mtp_num_hidden_layers", type=int, default=None)
    parser.add_argument("--first-k-dense-replace", dest="first_k_dense_replace", type=int, default=None)
    parser.add_argument("--q-lora-rank", dest="q_lora_rank", type=int, default=None)
    parser.add_argument("--kv-lora-rank", dest="kv_lora_rank", type=int, default=None)
    parser.add_argument("--qk-nope-head-dim", dest="qk_nope_head_dim", type=int, default=None)
    parser.add_argument("--qk-rope-head-dim", dest="qk_rope_head_dim", type=int, default=None)
    parser.add_argument("--v-head-dim", dest="v_head_dim", type=int, default=None)
    parser.add_argument("--index-n-heads", dest="index_n_heads", type=int, default=None)
    parser.add_argument("--index-head-dim", dest="index_head_dim", type=int, default=None)
    parser.add_argument("--index-topk", dest="index_topk", type=int, default=None)
    parser.add_argument("--index-topk-freq", dest="index_topk_freq", type=int, default=None)
    parser.add_argument("--index-skip-topk-offset", dest="index_skip_topk_offset", type=int, default=None)
    parser.add_argument("--train-indexer", dest="train_indexer", action="store_true", default=None)
    parser.add_argument("--no-train-indexer", dest="train_indexer", action="store_false")
    parser.add_argument("--routed-scaling-factor", dest="routed_scaling_factor", type=float, default=None)
    parser.add_argument("--scoring-func", dest="scoring_func", default=None)
    parser.add_argument("--shared-expert-dim", dest="shared_expert_dim", type=int, default=None)
    parser.add_argument("--mamba-num-heads", dest="mamba_num_heads", type=int, default=None)
    parser.add_argument("--mamba-head-dim", dest="mamba_head_dim", type=int, default=None)
    parser.add_argument("--ssm-state-size", dest="ssm_state_size", type=int, default=None)
    parser.add_argument("--conv-kernel", dest="conv_kernel", type=int, default=None)
    parser.add_argument("--mamba-chunk-size", dest="mamba_chunk_size", type=int, default=None)
    parser.add_argument("--n-groups", dest="n_groups", type=int, default=None)
    parser.add_argument("--hybrid-override-pattern", dest="hybrid_override_pattern", default=None)
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
    parser.add_argument("--expert-dispatch-dtype", default=None)
    parser.add_argument("--gradient-dtype", default=None)
    parser.add_argument("--optimizer-state-dtype", default=None)
    parser.add_argument("--compute-precision", default=None)
    parser.add_argument("--expert-weight-dtype", default=None)
    parser.add_argument("--expert-compute-precision", default=None)
    parser.add_argument("--indexer-activation-dtype", default=None)
    parser.add_argument("--indexer-compute-precision", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("model_training.dataflow.json"),
        help="Path to write the exported DataflowProgram JSON.",
    )
    args = parser.parse_args()

    program = build_program(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(program.model_dump(mode="json"), indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
