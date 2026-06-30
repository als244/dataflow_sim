"""Forward-phase op helpers."""
from dataflow_sim.workloads.ops.forward.activation import (
    gated_multiply,
    gelu,
    relu2,
    silu,
    swiglu,
)
from dataflow_sim.workloads.ops.forward.attention import attention, rope
from dataflow_sim.workloads.ops.forward.convolution import depthwise_causal_conv1d
from dataflow_sim.workloads.ops.forward.dsa_sparse_attention import dsa_sparse_attention
from dataflow_sim.workloads.ops.forward.linear_attention import (
    gated_delta_rule,
    gated_rms_norm,
)
from dataflow_sim.workloads.ops.forward.lightning_indexer import lightning_index_score
from dataflow_sim.workloads.ops.forward.loss import cross_entropy
from dataflow_sim.workloads.ops.forward.matmul import matmul
from dataflow_sim.workloads.ops.forward.mamba import (
    mamba_chunk_scan,
    mamba_gated_rms_norm,
)
from dataflow_sim.workloads.ops.forward.mla_attention import mla_attention, mla_rope
from dataflow_sim.workloads.ops.forward.movement import gather, memory, reduce, scatter
from dataflow_sim.workloads.ops.forward.norm import layer_norm, qk_norm, rms_norm
from dataflow_sim.workloads.ops.forward.sliding_attention import sliding_attention

__all__ = [
    "attention",
    "cross_entropy",
    "depthwise_causal_conv1d",
    "dsa_sparse_attention",
    "gather",
    "gated_delta_rule",
    "gated_multiply",
    "gated_rms_norm",
    "gelu",
    "layer_norm",
    "lightning_index_score",
    "matmul",
    "memory",
    "mla_attention",
    "mla_rope",
    "qk_norm",
    "mamba_chunk_scan",
    "mamba_gated_rms_norm",
    "reduce",
    "relu2",
    "rms_norm",
    "rope",
    "scatter",
    "silu",
    "sliding_attention",
    "swiglu",
]
