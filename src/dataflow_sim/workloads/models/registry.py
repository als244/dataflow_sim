"""Model-family registry for built-in modular training workloads."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models.deepseek_v3 import DeepSeekV3Config, DeepSeekV3ForTraining
from dataflow_sim.workloads.models.deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForTraining
from dataflow_sim.workloads.models.glm5_2 import GLM52Config, GLM52ForTraining
from dataflow_sim.workloads.models.gpt_oss import GPTOSSConfig, GPTOSSForTraining
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining
from dataflow_sim.workloads.models.nemotron_h import NemotronHConfig, NemotronHForTraining
from dataflow_sim.workloads.models.olmoe import OLMoEConfig, OLMoEForTraining
from dataflow_sim.workloads.models.qwen3 import Qwen3Config, Qwen3ForTraining
from dataflow_sim.workloads.models.qwen3_hybrid_dense import (
    QwenHybridDenseConfig,
    QwenHybridDenseForTraining,
)
from dataflow_sim.workloads.models.qwen3_hybrid_moe import (
    QwenHybridMoEConfig,
    QwenHybridMoEForTraining,
)
from dataflow_sim.workloads.models.qwen3_moe import Qwen3MoEConfig, Qwen3MoEForTraining


@dataclass(frozen=True)
class ModelFieldDescriptor:
    key: str
    label: str
    kind: str = "number"
    min: float | None = 0
    step: float | None = 1
    advanced: bool = False

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class ModelFamilyRegistryEntry:
    key: str
    label: str
    config_cls: type
    builder_cls: type
    presets: tuple[str, ...]
    fields: tuple[ModelFieldDescriptor, ...]
    has_moe: bool = False
    has_indexer: bool = False

    @property
    def config_field_names(self) -> tuple[str, ...]:
        return tuple(field.key for field in self.fields)

    def preset_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "presets": list(self.presets),
            "fields": [field.payload() for field in self.fields],
            "capabilities": {
                "has_moe": self.has_moe,
                "has_indexer": self.has_indexer,
            },
        }


BASE_FIELDS = (
    ModelFieldDescriptor("vocab_size", "Vocabulary Size", min=1),
    ModelFieldDescriptor("n_layers", "Layers", min=1),
    ModelFieldDescriptor("d_model", "Model Width", min=1),
    ModelFieldDescriptor("head_dim", "Head Dim", min=1),
    ModelFieldDescriptor("n_heads", "Attention Heads", min=1),
    ModelFieldDescriptor("n_kv_heads", "KV Heads", min=1),
    ModelFieldDescriptor("expert_dim", "Expert Dim", min=0),
    ModelFieldDescriptor("num_shared_experts", "Shared Experts", min=0),
    ModelFieldDescriptor("num_routed_experts", "Routed Experts", min=0),
    ModelFieldDescriptor("top_k", "Top K", min=0),
    ModelFieldDescriptor("qk_norm", "QK Norm", kind="boolean", min=None, step=None),
)

QWEN_HYBRID_FIELDS = BASE_FIELDS + (
    ModelFieldDescriptor("intermediate_size", "Dense Intermediate", min=0, advanced=True),
    ModelFieldDescriptor("full_attention_interval", "Full Attention Interval", min=1, advanced=True),
    ModelFieldDescriptor("linear_num_key_heads", "Linear Key Heads", min=1, advanced=True),
    ModelFieldDescriptor("linear_key_head_dim", "Linear Key Head Dim", min=1, advanced=True),
    ModelFieldDescriptor("linear_num_value_heads", "Linear Value Heads", min=1, advanced=True),
    ModelFieldDescriptor("linear_value_head_dim", "Linear Value Head Dim", min=1, advanced=True),
    ModelFieldDescriptor("linear_conv_kernel_dim", "Linear Conv Kernel", min=1, advanced=True),
    ModelFieldDescriptor("gdn_chunk_size", "GDN Chunk Size", min=1, advanced=True),
)

DEEPSEEK_FIELDS = BASE_FIELDS + (
    ModelFieldDescriptor("intermediate_size", "Dense Intermediate", min=1, advanced=True),
    ModelFieldDescriptor("first_k_dense_replace", "Dense Prefix Layers", min=0, advanced=True),
    ModelFieldDescriptor("q_lora_rank", "Q LoRA Rank", min=0, advanced=True),
    ModelFieldDescriptor("kv_lora_rank", "KV LoRA Rank", min=1, advanced=True),
    ModelFieldDescriptor("qk_nope_head_dim", "QK NoPE Head Dim", min=1, advanced=True),
    ModelFieldDescriptor("qk_rope_head_dim", "QK RoPE Head Dim", min=1, advanced=True),
    ModelFieldDescriptor("v_head_dim", "Value Head Dim", min=1, advanced=True),
)

DEEPSEEK_V32_FIELDS = DEEPSEEK_FIELDS + (
    ModelFieldDescriptor("index_n_heads", "Indexer Heads", min=1, advanced=True),
    ModelFieldDescriptor("index_head_dim", "Indexer Head Dim", min=1, advanced=True),
    ModelFieldDescriptor("index_topk", "Indexer Top K", min=1, advanced=True),
    ModelFieldDescriptor("train_indexer", "Train Indexer", kind="boolean", min=None, step=None, advanced=True),
)

GLM52_FIELDS = DEEPSEEK_V32_FIELDS + (
    ModelFieldDescriptor("index_topk_freq", "IndexShare Frequency", min=1, advanced=True),
    ModelFieldDescriptor("index_skip_topk_offset", "IndexShare Offset", min=0, advanced=True),
)

NEMOTRON_FIELDS = BASE_FIELDS + (
    ModelFieldDescriptor("shared_expert_dim", "Shared Expert Dim", min=0),
    ModelFieldDescriptor("intermediate_size", "Dense Intermediate", min=0, advanced=True),
    ModelFieldDescriptor("mamba_num_heads", "Mamba Heads", min=1, advanced=True),
    ModelFieldDescriptor("mamba_head_dim", "Mamba Head Dim", min=1, advanced=True),
    ModelFieldDescriptor("ssm_state_size", "SSM State Size", min=1, advanced=True),
    ModelFieldDescriptor("conv_kernel", "Conv Kernel", min=1, advanced=True),
    ModelFieldDescriptor("mamba_chunk_size", "Mamba Chunk Size", min=1, advanced=True),
    ModelFieldDescriptor("n_groups", "Mamba Groups", min=1, advanced=True),
    ModelFieldDescriptor("hybrid_override_pattern", "Hybrid Pattern", kind="text", min=None, step=None, advanced=True),
)

GPT_OSS_FIELDS = BASE_FIELDS + (
    ModelFieldDescriptor("sliding_window", "Sliding Window", min=1, advanced=True),
)

MODEL_FAMILIES: dict[str, ModelFamilyRegistryEntry] = {
    "llama3": ModelFamilyRegistryEntry(
        key="llama3",
        label="Llama 3",
        config_cls=Llama3Config,
        builder_cls=Llama3ForTraining,
        presets=("llama3_8B", "llama3_70B", "llama3_405B"),
        fields=BASE_FIELDS,
    ),
    "qwen3": ModelFamilyRegistryEntry(
        key="qwen3",
        label="Qwen3 Dense",
        config_cls=Qwen3Config,
        builder_cls=Qwen3ForTraining,
        presets=("qwen3_4B", "qwen3_8B", "qwen3_32B"),
        fields=BASE_FIELDS,
    ),
    "qwen3_moe": ModelFamilyRegistryEntry(
        key="qwen3_moe",
        label="Qwen3 MoE",
        config_cls=Qwen3MoEConfig,
        builder_cls=Qwen3MoEForTraining,
        presets=("qwen3_moe_30B-3B", "qwen3_moe_235B-A22B"),
        fields=BASE_FIELDS,
        has_moe=True,
    ),
    "olmoe": ModelFamilyRegistryEntry(
        key="olmoe",
        label="OLMoE",
        config_cls=OLMoEConfig,
        builder_cls=OLMoEForTraining,
        presets=("olmoe_7B-1B",),
        fields=BASE_FIELDS,
        has_moe=True,
    ),
    "qwen3_hybrid_dense": ModelFamilyRegistryEntry(
        key="qwen3_hybrid_dense",
        label="Qwen3.5/3.6 Dense",
        config_cls=QwenHybridDenseConfig,
        builder_cls=QwenHybridDenseForTraining,
        presets=("qwen3_5_9B", "qwen3_5_27B"),
        fields=QWEN_HYBRID_FIELDS,
    ),
    "qwen3_hybrid_moe": ModelFamilyRegistryEntry(
        key="qwen3_hybrid_moe",
        label="Qwen3.5/3.6 MoE",
        config_cls=QwenHybridMoEConfig,
        builder_cls=QwenHybridMoEForTraining,
        presets=(
            "qwen3_5_35B-A3B",
            "qwen3_5_122B-A10B",
            "qwen3_5_397B-A17B",
        ),
        fields=QWEN_HYBRID_FIELDS,
        has_moe=True,
    ),
    "deepseek_v3": ModelFamilyRegistryEntry(
        key="deepseek_v3",
        label="DeepSeek-V3",
        config_cls=DeepSeekV3Config,
        builder_cls=DeepSeekV3ForTraining,
        presets=("deepseek_v3_671B-37B", "kimi_k2_1T-32B"),
        fields=DEEPSEEK_FIELDS,
        has_moe=True,
    ),
    "deepseek_v3_2": ModelFamilyRegistryEntry(
        key="deepseek_v3_2",
        label="DeepSeek-V3.2",
        config_cls=DeepSeekV32Config,
        builder_cls=DeepSeekV32ForTraining,
        presets=(
            "deepseek_v3_2_671B-37B",
            "glm_5_744B-40B",
        ),
        fields=DEEPSEEK_V32_FIELDS,
        has_moe=True,
        has_indexer=True,
    ),
    "glm_5_2": ModelFamilyRegistryEntry(
        key="glm_5_2",
        label="GLM-5.2 IndexShare",
        config_cls=GLM52Config,
        builder_cls=GLM52ForTraining,
        presets=("glm_5_2_744B-40B",),
        fields=GLM52_FIELDS,
        has_moe=True,
        has_indexer=True,
    ),
    "gpt_oss": ModelFamilyRegistryEntry(
        key="gpt_oss",
        label="GPT-OSS",
        config_cls=GPTOSSConfig,
        builder_cls=GPTOSSForTraining,
        presets=("gpt_oss_20B", "gpt_oss_120B"),
        fields=GPT_OSS_FIELDS,
        has_moe=True,
    ),
    "nemotron_h": ModelFamilyRegistryEntry(
        key="nemotron_h",
        label="NVIDIA Nemotron 3",
        config_cls=NemotronHConfig,
        builder_cls=NemotronHForTraining,
        presets=(
            "nemotron3_nano_30B-A3B",
            "nemotron3_super_120B-A12B",
            "nemotron3_ultra_550B-A55B",
        ),
        fields=NEMOTRON_FIELDS,
        has_moe=True,
    ),
}

def iter_model_presets():
    rows = [
        (preset, family.key, family.config_cls.from_model_dims(preset))
        for family in MODEL_FAMILIES.values()
        for preset in family.presets
    ]
    yield from sorted(rows, key=lambda row: row[0].lower())


def model_families_payload() -> dict[str, dict[str, Any]]:
    return {
        key: entry.preset_payload()
        for key, entry in sorted(MODEL_FAMILIES.items())
    }
