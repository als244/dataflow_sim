"""DeepSeek-V3.2 model-family definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataflow_sim.workloads.models._config import load_model_dims
from dataflow_sim.workloads.modules import (
    DeepSeekV32Block,
    DeepSeekV32Dimensions,
    LanguageModelingHead,
    head_params,
    optimizer_ops_for_matrices,
)
from dataflow_sim.workloads.modules.optimizer import (
    matrix_gradient_bytes,
    matrix_weight_bytes,
    optimizer_state_bytes_for_matrices,
)
from dataflow_sim.workloads.training_builder import (
    TrainingBuilder,
    TrainingHeadSpec,
    TrainingLayerSpec,
)


_DEEPSEEK_V32_FIELDS = {
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
    "intermediate_size",
    "first_k_dense_replace",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "train_indexer",
    "qk_norm",
}

_ALIASES = {
    "671b-37b": "deepseek_v3_2_671B-37B",
    "deepseek-v3.2": "deepseek_v3_2_671B-37B",
    "deepseek-v3_2": "deepseek_v3_2_671B-37B",
    "deepseek_v3.2": "deepseek_v3_2_671B-37B",
    "deepseek_v3_2": "deepseek_v3_2_671B-37B",
    "deepseek_v3_2_671b-37b": "deepseek_v3_2_671B-37B",
    "deepseek_v3_2_671B-37B": "deepseek_v3_2_671B-37B",
    "glm-5": "glm_5_744B-40B",
    "glm5": "glm_5_744B-40B",
    "glm_5": "glm_5_744B-40B",
    "glm_5_744b-40b": "glm_5_744B-40B",
    "glm_5_744B-40B": "glm_5_744B-40B",
    "glm-5.1": "glm_5_744B-40B",
    "glm5.1": "glm_5_744B-40B",
    "glm_5_1": "glm_5_744B-40B",
    "glm_5_1_744b-40b": "glm_5_744B-40B",
    "glm_5_1_744B-40B": "glm_5_744B-40B",
}

_FIELD_ALIASES = {
    "hidden_size": "d_model",
    "dim": "d_model",
    "num_hidden_layers": "n_layers",
    "n_dense_layers": "first_k_dense_replace",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "inter_dim": "intermediate_size",
    "moe_inter_dim": "expert_dim",
    "moe_intermediate_size": "expert_dim",
    "n_shared_experts": "num_shared_experts",
    "n_routed_experts": "num_routed_experts",
    "n_activated_experts": "top_k",
    "num_experts_per_tok": "top_k",
}


def _config_kwargs(body: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, value in body.items():
        normalized = _FIELD_ALIASES.get(key, key)
        if normalized in _DEEPSEEK_V32_FIELDS:
            kwargs[normalized] = value
    for key, value in overrides.items():
        normalized = _FIELD_ALIASES.get(key, key)
        kwargs[normalized] = value
    kwargs.setdefault("n_kv_heads", kwargs.get("n_heads"))
    if "head_dim" not in kwargs and {"qk_nope_head_dim", "qk_rope_head_dim"} <= set(kwargs):
        kwargs["head_dim"] = kwargs["qk_nope_head_dim"] + kwargs["qk_rope_head_dim"]
    kwargs.setdefault("qk_norm", True)
    kwargs.setdefault("train_indexer", True)
    return kwargs


@dataclass(frozen=True)
class DeepSeekV32Config:
    vocab_size: int
    n_layers: int
    d_model: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    expert_dim: int
    num_shared_experts: int
    num_routed_experts: int
    top_k: int
    intermediate_size: int
    first_k_dense_replace: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    qk_norm: bool = True
    train_indexer: bool = True
    preset_name: str = "custom"

    @classmethod
    def from_model_dims(cls, key: str, **overrides: Any) -> "DeepSeekV32Config":
        body = load_model_dims(key)
        kwargs = _config_kwargs(body, overrides)
        kwargs.setdefault("preset_name", key)
        return cls(**kwargs)

    @classmethod
    def preset(cls, scale: str = "671B-37B", **overrides: Any) -> "DeepSeekV32Config":
        key = _ALIASES.get(scale, _ALIASES.get(scale.lower()))
        if key is None:
            raise ValueError(
                f"unknown DeepSeek-V3.2 scale {scale!r}; use 671B-37B, GLM-5, "
                "GLM-5.1, or a deepseek_v3_2_* preset key"
            )
        return cls.from_model_dims(key, **overrides)

    def dimensions(self) -> DeepSeekV32Dimensions:
        if self.q_lora_rank <= 0:
            raise ValueError("DeepSeek-V3.2 DSA requires q_lora_rank > 0")
        if self.first_k_dense_replace > self.n_layers:
            raise ValueError("first_k_dense_replace cannot exceed n_layers")
        return DeepSeekV32Dimensions(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            d_model=self.d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            expert_dim=self.expert_dim,
            num_shared_experts=self.num_shared_experts,
            num_routed_experts=self.num_routed_experts,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            first_k_dense_replace=self.first_k_dense_replace,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            qk_norm=self.qk_norm,
            train_indexer=self.train_indexer,
        )


def _layer_spec(index: int, dims: DeepSeekV32Dimensions) -> TrainingLayerSpec:
    dense = dims.is_dense_layer(index)
    block = DeepSeekV32Block(dims, dense_ffn=dense)
    variant = "dense_prefix" if dense else "moe_suffix"
    block_key = f"deepseek_v3_2.{variant}_block"
    block_name = "DeepSeek-V3.2 Dense Prefix Block" if dense else "DeepSeek-V3.2 MoE Block"
    matrices = dims.layer_matrices(dense=dense)
    trainable_matrices = dims.layer_matrices(dense=dense, trainable_only=True)
    return TrainingLayerSpec(
        name=f"layer_{index}",
        input_dim=dims.d_model,
        output_dim=dims.d_model,
        param_count=dims.params_per_layer(dense=dense),
        gradient_count=dims.trainable_params_per_layer(dense=dense),
        saved_activation_width=dims.saved_activation_width(dense=dense),
        saved_activation_bytes=(
            lambda tokens, seqlen, policy, dims=dims, dense=dense: dims.saved_activation_bytes(
                tokens=tokens,
                seqlen=seqlen,
                dense=dense,
                policy=policy,
            )
        ),
        forward_ops=(
            lambda tokens, seqlen, bpe, block=block: block.forward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bpe,
            )
        ),
        backward_ops=(
            lambda tokens, seqlen, bpe, block=block: block.backward_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bpe,
            )
        ),
        recompute_ops=(
            lambda tokens, seqlen, bpe, block=block: block.recompute_ops(
                tokens=tokens,
                seqlen=seqlen,
                bytes_per_element=bpe,
            )
        ),
        optimizer_ops=(
            lambda optimizer, bpe, matrices=trainable_matrices: optimizer_ops_for_matrices(
                "deepseek_v3_2_optimizer",
                matrices=matrices,
                optimizer=optimizer,
                bytes_per_element=bpe,
            )
        ),
        parameter_bytes=lambda policy, parallelism, matrices=matrices: matrix_weight_bytes(
            matrices,
            policy,
            parallelism,
        ),
        gradient_bytes=(
            lambda policy, parallelism, matrices=trainable_matrices: matrix_gradient_bytes(
                matrices,
                policy,
                parallelism,
            )
        ),
        optimizer_state_bytes=(
            lambda optimizer, policy, parallelism, matrices=trainable_matrices: optimizer_state_bytes_for_matrices(
                matrices,
                optimizer,
                policy,
                parallelism,
            )
        ),
        block_key=block_key,
        block_name=block_name,
        optimizer_block_key=f"{block_key}.optimizer_step",
        metadata={
            "deepseek_v3_2": asdict(dims),
            "dense_ffn": dense,
            "variant": variant,
        },
    )


def _head_spec(dims: DeepSeekV32Dimensions) -> TrainingHeadSpec:
    head_dims = dims.ffn_dimensions(dense=False)
    head = LanguageModelingHead(head_dims)
    return TrainingHeadSpec(
        name="head",
        input_dim=dims.d_model,
        param_count=head_params(head_dims),
        forward_ops=(
            lambda tokens, bpe, head=head: head.forward_ops(
                tokens=tokens,
                bytes_per_element=bpe,
            )
        ),
        backward_ops=(
            lambda tokens, bpe, head=head: head.backward_ops(
                tokens=tokens,
                bytes_per_element=bpe,
            )
        ),
        block_key="lm_head",
        block_name="LM Head",
        metadata={"deepseek_v3_2": asdict(dims)},
    )


class DeepSeekV32ForTraining(TrainingBuilder):
    family_name = "deepseek_v3_2"
    metadata_kind = "training.deepseek_v3_2.modular"

    def __init__(self, config: DeepSeekV32Config) -> None:
        self.config = config
        self.dims = config.dimensions()
        super().__init__(
            family_name=self.family_name,
            metadata_kind=self.metadata_kind,
            preset_name=config.preset_name,
            layers=[_layer_spec(index, self.dims) for index in range(config.n_layers)],
            head=_head_spec(self.dims),
            model_metadata={"deepseek_v3_2": asdict(self.dims)},
        )
