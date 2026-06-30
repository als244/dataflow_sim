"""Backward-phase op helpers."""
from dataflow_sim.workloads.ops.backward.activation import (
    gated_multiply_grad,
    gelu_grad,
    relu2_grad,
    silu_grad,
    swiglu_grad,
)
from dataflow_sim.workloads.ops.backward.attention import attention_grad, rope_grad
from dataflow_sim.workloads.ops.backward.convolution import depthwise_causal_conv1d_grad
from dataflow_sim.workloads.ops.backward.dsa_sparse_attention import dsa_sparse_attention_grad
from dataflow_sim.workloads.ops.backward.linear_attention import (
    gated_delta_rule_grad,
    gated_rms_norm_grad,
)
from dataflow_sim.workloads.ops.backward.lightning_indexer import lightning_index_score_grad
from dataflow_sim.workloads.ops.backward.loss import cross_entropy_grad
from dataflow_sim.workloads.ops.backward.matmul import matmul_input_grad, matmul_weight_grad
from dataflow_sim.workloads.ops.backward.mamba import (
    mamba_chunk_scan_grad,
    mamba_gated_rms_norm_grad,
)
from dataflow_sim.workloads.ops.backward.mla_attention import (
    mla_attention_grad,
    mla_rope_grad,
)
from dataflow_sim.workloads.ops.backward.movement import gather_grad, reduce_grad, scatter_grad
from dataflow_sim.workloads.ops.backward.norm import layer_norm_grad, qk_norm_grad, rms_norm_grad
from dataflow_sim.workloads.ops.backward.sliding_attention import sliding_attention_grad

__all__ = [
    "attention_grad",
    "cross_entropy_grad",
    "depthwise_causal_conv1d_grad",
    "dsa_sparse_attention_grad",
    "gather_grad",
    "gated_delta_rule_grad",
    "gated_multiply_grad",
    "gated_rms_norm_grad",
    "gelu_grad",
    "layer_norm_grad",
    "lightning_index_score_grad",
    "mamba_chunk_scan_grad",
    "mamba_gated_rms_norm_grad",
    "matmul_input_grad",
    "matmul_weight_grad",
    "mla_attention_grad",
    "mla_rope_grad",
    "qk_norm_grad",
    "reduce_grad",
    "relu2_grad",
    "rms_norm_grad",
    "rope_grad",
    "scatter_grad",
    "silu_grad",
    "sliding_attention_grad",
    "swiglu_grad",
]
