"""Export a custom transformer-training workload as DataflowProgram JSON.

Users define model/layer structure here. The training builder lowers that model
definition to the same generic schema accepted by the webapp's Custom Schema tab.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Literal

from dataflow_sim.workloads.models.transformer import TransformerSpec
from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_heterogeneous_transformer_training_program,
)


LayerKind = Literal["dense", "moe"]


@dataclass(frozen=True)
class LayerDef:
    kind: LayerKind
    expert_dim: int
    num_routed_experts: int = 0
    top_k: int = 0


def model_definition() -> list[LayerDef]:
    """A tiny heterogeneous model definition.

    A real model exporter could build this list from a config file, PyTorch
    module inventory, or another model registry.
    """
    return [
        LayerDef("dense", expert_dim=2048),
        LayerDef("dense", expert_dim=2048),
        LayerDef("moe", expert_dim=1536, num_routed_experts=8, top_k=2),
        LayerDef("dense", expert_dim=2048),
    ]


def to_transformer_spec(base: TransformerSpec, layer: LayerDef) -> TransformerSpec:
    if layer.kind == "dense":
        return replace(
            base,
            n_layers=1,
            expert_dim=layer.expert_dim,
            num_shared_experts=1,
            num_routed_experts=0,
            top_k=0,
        )
    return replace(
        base,
        n_layers=1,
        expert_dim=layer.expert_dim,
        num_shared_experts=1,
        num_routed_experts=layer.num_routed_experts,
        top_k=layer.top_k,
    )


def build_program():
    base = TransformerSpec(
        vocab_size=32_000,
        n_layers=1,
        d_model=512,
        head_dim=64,
        n_heads=8,
        n_kv_heads=8,
        expert_dim=2048,
        num_shared_experts=1,
        num_routed_experts=0,
        top_k=0,
        qk_norm=True,
    )
    layer_specs = [to_transformer_spec(base, layer) for layer in model_definition()]
    training = TrainingConfig(
        seqlen=128,
        num_seqs=1,
        grad_accum_rounds=1,
        num_steps=1,
        optimizer="adamw",
        final_model_state_on_backing=True,
    )
    return build_heterogeneous_transformer_training_program(
        layer_specs,
        training,
        name="custom-heterogeneous-transformer-training",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("heterogeneous_transformer_training.dataflow.json"),
        help="Path to write the exported DataflowProgram JSON.",
    )
    args = parser.parse_args()

    program = build_program()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(program.model_dump(mode="json"), indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
