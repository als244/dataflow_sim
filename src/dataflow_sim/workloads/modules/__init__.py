"""Composable symbolic workload modules."""
from dataflow_sim.workloads.modules.deepseek_block import DeepSeekBlock
from dataflow_sim.workloads.modules.deepseek_dimensions import DeepSeekDimensions
from dataflow_sim.workloads.modules.deepseek_v3_2_block import DeepSeekV32Block
from dataflow_sim.workloads.modules.deepseek_v3_2_dimensions import DeepSeekV32Dimensions
from dataflow_sim.workloads.modules.dense_attention import DenseAttention
from dataflow_sim.workloads.modules.dsa_sparse_attention import DSASparseAttention
from dataflow_sim.workloads.modules.dimensions import (
    TransformerDimensions,
    active_params_per_layer,
    head_params,
    layer_activation_elements_per_token,
    layer_weight_matrices,
    params_per_layer,
)
from dataflow_sim.workloads.modules.gpt_oss_attention import GPTOSSAttention
from dataflow_sim.workloads.modules.gpt_oss_block import GPTOSSBlock
from dataflow_sim.workloads.modules.gpt_oss_dimensions import GPTOSSDimensions
from dataflow_sim.workloads.modules.mla_attention import MLAAttention
from dataflow_sim.workloads.modules.mlp import SwiGLUMLP
from dataflow_sim.workloads.modules.moe import MoE
from dataflow_sim.workloads.modules.nemotron_attention import NemotronAttention
from dataflow_sim.workloads.modules.nemotron_block import NemotronBlock
from dataflow_sim.workloads.modules.nemotron_dimensions import NemotronDimensions
from dataflow_sim.workloads.modules.nemotron_mamba import NemotronMamba
from dataflow_sim.workloads.modules.optimizer import (
    OptimizerStep,
    optimizer_ops_for_matrices,
)
from dataflow_sim.workloads.modules.qwen_hybrid_block import QwenHybridBlock
from dataflow_sim.workloads.modules.qwen_hybrid_dimensions import (
    QwenHybridDimensions,
)
from dataflow_sim.workloads.modules.qwen_hybrid_full_attention import (
    QwenHybridFullAttention,
)
from dataflow_sim.workloads.modules.qwen_hybrid_linear_attention import (
    QwenHybridLinearAttention,
)
from dataflow_sim.workloads.modules.relu2_mlp import ReLU2MLP
from dataflow_sim.workloads.modules.relu2_moe import ReLU2MoE
from dataflow_sim.workloads.modules.transformer_block import (
    TransformerBlock,
)
from dataflow_sim.workloads.modules.language_modeling_head import (
    LanguageModelingHead,
)

__all__ = [
    "DeepSeekBlock",
    "DeepSeekDimensions",
    "DeepSeekV32Block",
    "DeepSeekV32Dimensions",
    "DenseAttention",
    "DSASparseAttention",
    "GPTOSSAttention",
    "GPTOSSBlock",
    "GPTOSSDimensions",
    "MLAAttention",
    "MoE",
    "NemotronAttention",
    "NemotronBlock",
    "NemotronDimensions",
    "NemotronMamba",
    "OptimizerStep",
    "optimizer_ops_for_matrices",
    "QwenHybridBlock",
    "QwenHybridDimensions",
    "QwenHybridFullAttention",
    "QwenHybridLinearAttention",
    "ReLU2MLP",
    "ReLU2MoE",
    "SwiGLUMLP",
    "TransformerBlock",
    "TransformerDimensions",
    "LanguageModelingHead",
    "active_params_per_layer",
    "head_params",
    "layer_activation_elements_per_token",
    "layer_weight_matrices",
    "params_per_layer",
]
