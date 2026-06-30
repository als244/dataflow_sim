# DeepSeek-V3.2 DSA Cost Derivations

This note records the provisional DeepSeek Sparse Attention formulas used by
the modular workload builder. They are intentionally reviewable and keep the
same major-op style as the existing model families.

## Symbols

- `T`: total tokens in the microbatch.
- `L_i`: sequence length for sequence `i`.
- `S = sum_i L_i^2`: dense query-key score domain.
- `K = index_topk`.
- `S_k = sum_i L_i * min(K, L_i)`: selected sparse attention domain.
- `H`: attention heads.
- `Hi`: Lightning Indexer heads.
- `Di`: Lightning Indexer head dimension.
- `QL`: query LoRA rank.
- `KL`: KV LoRA rank.
- `Qn`: no-RoPE QK head dimension.
- `Qr`: RoPE QK head dimension.
- `V`: value head dimension.
- `train_indexer`: whether the Lightning Indexer projection/scoring branch is
  trained. Forward sparse selection always runs either way.
- `indexer_mode`: `full` for layers that compute Lightning Indexer projections,
  scores, and selected sparse positions; `shared` for GLM-5.2 IndexShare layers
  that reuse positions from a previous full-index layer.

## Training Stage Assumptions

The emitted DeepSeek-V3.2 workload models the sparse-training stage, not the
dense indexer warm-up stage.

In sparse training, forward scoring still ranges over the dense candidate
domain `S` because the kernel must choose top-k keys for each query. The
Lightning Indexer training loss, however, is modeled only on the selected token
set, so indexer backward uses `S_k`.

A dense indexer warm-up workload would be a different mode:

```text
dense_warmup_indexer_bwd_flops           = 6 * S * Hi * Di
dense_warmup_indexer_bwd_effective_flops = 4 * S * Hi * Di
```

That mode would also save or recompute dense-score loss state differently. It is
not emitted by the current built-in DeepSeek-V3.2 training preset.

GLM-5.2 uses the same DSA sparse-attention core with IndexShare. In the public
744B-40B preset, 21 of 78 layers are `full` indexer layers and 57 are `shared`
indexer layers. The first three dense-prefix layers are full-index layers; in
the sparse suffix, the first full-index layer appears at offset 3 and then
every four layers. Shared-index layers emit the sparse DSA attention core, but
skip Lightning Indexer projection, score, score backward, indexer projection
wgrad, indexer parameter, indexer optimizer-state, and indexer saved-activation
costs. The v1 builder also omits an explicit shared selected-index dependency
or object for those layers because it is small compared with score and sparse
attention math.

## Forward

Projection matmuls are emitted as regular matmul sub-ops:

```text
q_a_proj:            2 * T * D  * QL
q_b_proj:            2 * T * QL * H * (Qn + Qr)
index_q_b_proj:      2 * T * QL * Hi * Di
index_k_proj:        2 * T * D  * Di
index_weight_proj:   2 * T * D  * Hi
kv_a_proj_with_mqa:  2 * T * D  * (KL + Qr)
kv_b_proj:           2 * T * KL * H * (Qn + V)
o_proj:              2 * T * H * V * D
```

For `indexer_mode=shared`, the three `index_*_proj` matmuls and
`lightning_index_score` are not emitted. The later MLA/DSA projection and
sparse attention terms are unchanged.

The Lightning Indexer score is modeled as the quadratic scoring matmul:

```text
lightning_index_score_flops = 2 * S * Hi * Di
```

The builder assumes this score is computed by a FlashAttention-style tiled
top-k kernel. It does not materialize the dense `S * Hi` score tensor. The
forward score sub-op charges only activation input traffic plus selected-state
traffic:

```text
lightning_index_score_bytes =
    T * Hi * Di * indexer_activation_bpe
  + T * Di * indexer_activation_bpe
  + T * Hi * indexer_activation_bpe
  + S_k * index_bytes
  + (train_indexer ? S_k * indexer_activation_bpe : 0)
```

The `S_k * index_bytes` term stores selected key positions for the sparse DSA
attention path. The selected indexer-score term is only needed when the
Lightning Indexer is trained. In implementation, q/k/w indexer lanes and the
selected score state use Indexer Activation DType.

The small scalar pieces around score weighting, masking, and top-k bookkeeping
are not emitted as separate v1 costs. A comparison upper bound for selection is:

```text
selector_compare_flops = S * ceil(log2(K))
```

For the public `K=2048`, this is `11 * S`, versus `16384 * S` for the default
`2 * Hi * Di` score term with `Hi=64` and `Di=128`.

The selected DSA sparse attention core is modeled as:

```text
dsa_sparse_attn_flops = H * (2 * KL + Qr) * S_k
```

The `kv_b_proj` transform is kept as a regular matmul, so it is not folded into
the sparse core sub-op even though public summaries may describe the combined
DSA core as:

```text
2 * T * H * KL * (Qn + V) + H * (2 * KL + Qr) * S_k
```

## Backward

The sparse attention backward core is provisional:

```text
dsa_sparse_attn_bwd_flops           = H * (5 * KL + 2 * Qr) * S_k
dsa_sparse_attn_bwd_effective_flops = H * (4 * KL + 2 * Qr) * S_k
```

Projection dgrad/wgrad terms use the same matmul-gradient helpers as other
modules. When `train_indexer=true`, the indexer score backward is modeled over
selected score paths:

```text
lightning_index_score_bwd_flops           = 6 * S_k * Hi * Di
lightning_index_score_bwd_effective_flops = 4 * S_k * Hi * Di
```

The extra `2 * S_k * Hi * Di` term is a selected-score recompute term, analogous
to the recompute surcharge in FlashAttention-style attention backward. The
useful backward work is the dQ-like and dK-like score gradient work over the
selected set.

When `train_indexer=true`, the v1 implementation emits wgrad matmuls for
`index_q_b_proj`, `index_k_proj`, and `index_weight_proj`, but omits
hidden-state dgrad from the indexer branch. That matches the current assumption
that the indexer branch is not a separate major activation-gradient path into
the backbone.

When `train_indexer=false`, forward scoring and sparse selection still run, but
the emitted training program omits:

- `lightning_index_score_bwd`,
- `index_q_b_proj_wgrad`,
- `index_k_proj_wgrad`,
- `index_weight_proj_wgrad`,
- indexer-only gradient/optimizer state bytes.

## DTypes

- Indexer projection weights use the normal Weight DType.
- Indexer projection matmuls use the normal Compute Precision.
- Lightning Indexer q/k/w lanes and selected-score storage use Indexer
  Activation DType, default `fp8`.
- Lightning Indexer score math uses Indexer Compute Precision, default `fp8`.
- DSA sparse attention core math uses attention forward/backward efficiency
  fields, not the indexer precision selector.

For saved activation object sizing, the builder adds:

```text
S_k * index_bytes + (train_indexer ? S_k * bytes(Indexer Activation DType) : 0)
```

to the normal activation-context estimate for each DeepSeek-V3.2 DSA block.
The builder never adds the dense quadratic `S * Hi` score tensor to `A_*`.
With `train_indexer=false`, selected indices remain saved for sparse attention
backward, but selected indexer scores are omitted from `A_*`.

For GLM-5.2 `indexer_mode=shared` layers, the whole selected-state addend above
is omitted in v1. This models the shared sparse positions as a small dependency
from an earlier full-index layer rather than as a repeated per-layer activation
object.
