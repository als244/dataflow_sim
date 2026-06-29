"""Shared script helpers for the modular model-training workload stack."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from dataflow_sim.core.schema import TaskChain
from dataflow_sim.workloads.common.hardware import HardwareSpec
from dataflow_sim.workloads.common.workload import Workload
from dataflow_sim.workloads.dataflow_builder import TrainingConfig
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining
from dataflow_sim.workloads.models.olmoe import OLMoEConfig, OLMoEForTraining
from dataflow_sim.workloads.models.qwen3 import Qwen3Config, Qwen3ForTraining
from dataflow_sim.workloads.models.qwen3_moe import Qwen3MoEConfig, Qwen3MoEForTraining


PUBLIC_MODEL_PRESETS: dict[str, str] = {
    "llama3_8B": "llama3",
    "llama3_70B": "llama3",
    "llama3_405B": "llama3",
    "qwen3_4B": "qwen3",
    "qwen3_8B": "qwen3",
    "qwen3_32B": "qwen3",
    "qwen3_moe_30B-3B": "qwen3_moe",
    "olmoe_7B-1B": "olmoe",
}

_MODEL_BUILDERS = {
    "llama3": (Llama3Config, Llama3ForTraining),
    "qwen3": (Qwen3Config, Qwen3ForTraining),
    "qwen3_moe": (Qwen3MoEConfig, Qwen3MoEForTraining),
    "olmoe": (OLMoEConfig, OLMoEForTraining),
}


def model_config(model_name: str, **overrides: Any):
    family = PUBLIC_MODEL_PRESETS.get(model_name)
    if family is None:
        known = ", ".join(PUBLIC_MODEL_PRESETS)
        raise ValueError(f"unknown model preset {model_name!r}; known presets: {known}")
    config_cls, _ = _MODEL_BUILDERS[family]
    return config_cls.from_model_dims(model_name, **overrides)


def build_training_workload(
    model_name: str,
    hw: HardwareSpec,
    training: TrainingConfig,
    *,
    recompute: dict[str, int] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Workload:
    config = model_config(model_name, **dict(overrides or {}))
    family = PUBLIC_MODEL_PRESETS[model_name]
    _, model_cls = _MODEL_BUILDERS[family]
    return model_cls(config).build_training_workload(
        training,
        hw,
        input_shape=(training.tokens, config.d_model),
        recompute=recompute,
    )


def build_tiny_training_chain(
    *,
    layers: int,
    bandwidth_from_slow: int = 8,
    bandwidth_to_slow: int = 8,
    fast_memory_capacity: int | None = None,
) -> TaskChain:
    """Return a tiny modular Llama-style chain for policy demos.

    Bandwidth arguments are already in the simulator's bytes-per-microsecond
    units to preserve the old scripts' CLI behavior.
    """
    hw = HardwareSpec(
        peak_tflops=100,
        fast_memory_bw_gbs=1000,
        from_slow_bw_gbs=max(bandwidth_from_slow / 1000, 0.001),
        to_slow_bw_gbs=max(bandwidth_to_slow / 1000, 0.001),
        matmul_eff=0.8,
        attn_fwd_eff=0.8,
        attn_bwd_eff=0.8,
        mem_eff=0.9,
    )
    training = TrainingConfig(seqlen=8, num_seqs=1, optimizer="none")
    workload = build_training_workload(
        "llama3_8B",
        hw,
        training,
        overrides={
            "n_layers": layers,
            "d_model": 16,
            "head_dim": 8,
            "n_heads": 2,
            "n_kv_heads": 1,
            "expert_dim": 32,
            "vocab_size": 64,
            "qk_norm": False,
        },
    )
    return replace(
        workload.chain,
        fast_memory_capacity=fast_memory_capacity,
        bandwidth_from_slow=bandwidth_from_slow,
        bandwidth_to_slow=bandwidth_to_slow,
    )
