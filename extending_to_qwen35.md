# Add Qwen3.5/Qwen3.6 Hybrid Dense and MoE Built-Ins

## Summary

Add one shared Qwen3.5/Qwen3.6 text-backbone implementation with two public
model variants: dense and MoE. These models use a hybrid 3-linear-attention /
1-full-attention pattern, so they need new exact-ish Gated DeltaNet ops rather
than being represented as ordinary Qwen3 GQA presets.

Use public config/modeling sources:

- [Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/config.json)
- [Qwen3.6-27B config](https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json)
- [Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/config.json)
- [HF Qwen3Next modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)

## Key Changes

- Add separate model files for the two architecture variants:
  - `qwen3_hybrid_dense.py`: dense FFN + hybrid token mixer.
  - `qwen3_hybrid_moe.py`: sparse MoE + shared expert + hybrid token mixer.
- Add presets as data rows, not separate architecture code:
  - Dense seeds: `qwen3_5_9B`, `qwen3_5_27B`.
  - MoE seeds: `qwen3_5_35B-A3B`, `qwen3_5_122B-A10B`,
    `qwen3_5_397B-A17B`.
  - All support preset-plus-overrides for scale changes.
- Add a typed `QwenHybridDimensions`/config path with fields for
  `layer_types`, `full_attention_interval`, `linear_*` dims,
  `intermediate_size`, `expert_dim`, `num_routed_experts`, `top_k`,
  `num_shared_experts`, `mtp_num_hidden_layers`, `rope_parameters`, and
  router metadata.
  - Keep `expert_dim` as the internal expert hidden width, matching the
    existing `TransformerDimensions` and `MoE` module.
  - Map upstream names such as `moe_intermediate_size`,
    `shared_expert_intermediate_size`, `num_experts`, and
    `num_experts_per_tok` into the internal vocabulary at preset/config load
    time.
- Add new modules/ops:
  - `GatedDeltaNetAttention` with `in_proj_qkvz`, `in_proj_ba`, depthwise
    causal conv, gated delta rule core, gated RMS norm, and `out_proj`.
  - `QwenHybridFullAttention` with gated `q_proj`, `k_proj`, `v_proj`, q/k
    norm, partial/mrope RoPE, full attention, sigmoid gate multiply, and
    accumulated `o_proj`.
  - `QwenHybridBlock` composes the new token mixers with existing
    `SwiGLUMLP` or existing `MoE`; do not add a Qwen-specific MoE module for
    v1.
- Use distinct compute-block keys for linear vs full layers and dense vs MoE
  blocks, so reusable block summaries never collapse incompatible sub-op lists.

## File Inventory

Ops:

- `src/dataflow_sim/workloads/ops/forward/convolution.py`
- `src/dataflow_sim/workloads/ops/backward/convolution.py`
- `src/dataflow_sim/workloads/ops/forward/linear_attention.py`
- `src/dataflow_sim/workloads/ops/backward/linear_attention.py`
- Extend `src/dataflow_sim/workloads/ops/forward/attention.py` and
  `src/dataflow_sim/workloads/ops/backward/attention.py` only if the current
  helpers cannot express separate Q/K/V dimensions cleanly.
- Extend `src/dataflow_sim/workloads/ops/forward/activation.py` and
  `src/dataflow_sim/workloads/ops/backward/activation.py` only for the
  non-router gated-attention multiply if existing helpers are insufficient.
- Do not add router op files in v1.

Modules:

- `src/dataflow_sim/workloads/modules/qwen_hybrid_dimensions.py`
- `src/dataflow_sim/workloads/modules/qwen_hybrid_linear_attention.py`
- `src/dataflow_sim/workloads/modules/qwen_hybrid_full_attention.py`
- `src/dataflow_sim/workloads/modules/qwen_hybrid_block.py`

Models:

- `src/dataflow_sim/workloads/models/qwen3_hybrid_dense.py`
- `src/dataflow_sim/workloads/models/qwen3_hybrid_moe.py`

## Public Interfaces

- Add UI/server family keys `qwen3_hybrid_dense` and `qwen3_hybrid_moe` with
  display labels `Qwen3.5/3.6 Dense` and `Qwen3.5/3.6 MoE`.
- Extend model params to expose advanced Qwen hybrid fields only for these
  families.
- Keep existing `qwen3` and `qwen3_moe` behavior unchanged.
- Model text-only training for now using `text_config`; preserve
  multimodal/vision presence in metadata for future vision-tower work.
- Skip MTP, explicit router logits/top-k/scoring ops, separate shared-expert
  scalar gate ops, and router auxiliary loss tasks in v1. Keep
  `mtp_num_hidden_layers`, `router_aux_loss_coef`, and upstream router metadata
  for future fidelity upgrades.

## Cost Model Defaults

- Full attention uses Qwen3Next projection shapes: `q_proj` outputs
  `2 * n_heads * head_dim`, `k/v_proj` output `n_kv_heads * head_dim`, and
  attention uses current full-attention roofline logic plus explicit
  gate/norm/mrope memory ops.
  - Let `Q = n_heads * head_dim`, `KV = n_kv_heads * head_dim`, and
    `S = sum(seq_len^2)`.
  - `q_proj`: matmul `d_model -> 2 * Q`.
  - `k_proj`: matmul `d_model -> KV`.
  - `v_proj`: matmul `d_model -> KV`.
  - `q_norm`: memory `2 * tokens * Q * bytes_per_element`.
  - `k_norm`: memory `2 * tokens * KV * bytes_per_element`.
  - RoPE/mRoPE: provisional memory
    `2 * tokens * head_dim * (n_heads + n_kv_heads) * bytes_per_element`.
  - Attention forward: provisional FLOPs `2 * n_heads * head_dim * S`.
  - Gated output multiply: provisional memory
    `3 * tokens * Q * bytes_per_element`.
  - `o_proj`: matmul `Q -> d_model` with `accumulate=True`.
- Gated DeltaNet uses config-derived shapes:
  - `key_dim = linear_num_key_heads * linear_key_head_dim`
  - `value_dim = linear_num_value_heads * linear_value_head_dim`
  - `conv_dim = 2 * key_dim + value_dim`
  - `in_proj_qkvz = d_model -> 2 * key_dim + 2 * value_dim`
  - `in_proj_ba = d_model -> 2 * linear_num_value_heads`
- Gated DeltaNet core roofline estimate follows the FLA chunked derivation:
  `tokens * linear_num_value_heads * (6 * linear_key_head_dim * linear_value_head_dim + 2 * fla_chunk_size * (linear_key_head_dim + linear_value_head_dim))`.
  The default FLA chunk size is 64, which reduces to an 8x coefficient when
  `linear_key_head_dim == linear_value_head_dim == 128`. Backward flops remain
  a provisional `2x` forward until we split the FLA backward kernels explicitly.

## MoE Cost Model Defaults

Qwen hybrid MoE should intentionally mirror the current `qwen3_moe` modeling
style unless we explicitly broaden all MoE accounting later.

- Treat routing as metadata, not emitted sub-ops, for v1.
- Use `expert_dim` as the internal expert hidden width.
- Use `num_routed_experts`, `top_k`, and `num_shared_experts` to describe the
  sparse FFN.
- Approximate routed tokens per expert as
  `routed_tokens = tokens * top_k // num_routed_experts`.
- Forward MoE work:
  - optional `ffn_norm`;
  - optional shared expert `gate_up` matmul `d_model -> 2 * expert_dim` with
    `count=num_shared_experts`;
  - routed-token scatter memory with `fanout=top_k`;
  - routed expert `gate_up` matmul over `routed_tokens` with
    `count=num_routed_experts`;
  - SwiGLU memory for `num_shared_experts + top_k` active branches;
  - optional shared expert down matmul `expert_dim -> d_model` with
    `count=num_shared_experts`;
  - routed expert down matmul over `routed_tokens` with
    `count=num_routed_experts`;
  - routed-token gather memory with `fanin=top_k`.
- Backward MoE work follows the current module pattern: scatter/gather grads,
  down dgrad/wgrad, SwiGLU grad, and up dgrad/wgrad for routed and shared
  experts.
- Recompute should follow the current coarse policy: replay the inexpensive
  forward prefix and omit final down/gather outputs from the recompute block.
- Parameter accounting should count routed and shared expert matrices. Router
  matrices remain omitted in v1 for consistency with existing MoE accounting.
- If an upstream model has a separate shared expert gate or grouped router, keep
  it in metadata and document it as not emitted in v1.

## Test Plan

- Unit-test all preset config values and aliases against the public configs.
- Test reduced dense and MoE configs through `build_training_program`,
  `build_training_workload`, simulator summary, and recompute under tight
  memory.
- Assert the 3-linear / 1-full layer pattern, distinct compute-block keys, and
  correct dense vs MoE block counts.
- Assert key sub-op names and dimensions for Gated DeltaNet, full gated
  attention, dense FFN, coarse MoE shared/routed expert work, LM head
  forward/backward, and optimizer steps.
- Add server preset tests, UI build validation, CLI export coverage, and
  docs/examples updates.
- Run full Python test suite and `npm --prefix ui run build`.

## Assumptions

- Qwen3.5 and Qwen3.6 share the same text-backbone architecture family;
  generation and scale are preset/config data.
- Dense vs MoE is the only architecture split exposed as separate model files.
- Vision tower and multimodal paths are future work.
- Default dtype policy remains bf16.
- Existing built-in presets remain backward compatible.

## Future Fidelity Upgrades

- Add explicit router logits/top-k/scoring ops once we decide to account for
  those costs across all MoE families.
- Add shared-expert gate modeling if it proves material for Qwen hybrid
  training workloads.
- Add router auxiliary loss and MTP tasks when the workload API can expose them
  without special-casing one model family.
