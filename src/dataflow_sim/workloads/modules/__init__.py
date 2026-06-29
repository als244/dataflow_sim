"""Composable symbolic workload modules."""
from dataflow_sim.workloads.modules.dense_attention import DenseAttention
from dataflow_sim.workloads.modules.dimensions import (
    TransformerDimensions,
    active_params_per_layer,
    head_params,
    layer_activation_elements_per_token,
    layer_weight_matrices,
    params_per_layer,
)
from dataflow_sim.workloads.modules.mlp import SwiGLUMLP
from dataflow_sim.workloads.modules.moe import MoE
from dataflow_sim.workloads.modules.optimizer import OptimizerStep
from dataflow_sim.workloads.modules.recompute import zero_recompute_slot
from dataflow_sim.workloads.modules.transformer_block import (
    TransformerBlock,
)
from dataflow_sim.workloads.modules.language_modeling_head import (
    LanguageModelingHead,
)

__all__ = [
    "DenseAttention",
    "MoE",
    "OptimizerStep",
    "SwiGLUMLP",
    "TransformerBlock",
    "TransformerDimensions",
    "LanguageModelingHead",
    "active_params_per_layer",
    "head_params",
    "layer_activation_elements_per_token",
    "layer_weight_matrices",
    "params_per_layer",
    "zero_recompute_slot",
]
