# Add DeepSeek-V3 and Kimi-K2 Built-In Model Families

## Summary

Add a DeepSeekV3-style MLA + coarse MoE workload path, then expose two built-in
families: `deepseek_v3` and `kimi_k2`. Use presets `deepseek_v3_671B-37B` and
`kimi_k2_1T-32B`. Do not rename current `dense_attention` yet. Skip DeepSeek
MTP, router internals, and Kimi aux/router loss tasks for v1, but preserve
those facts in metadata.

Config sources:

- [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/config.json)
- [Kimi-K2 config](https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/main/config.json)

## Key Changes

- Add typed DeepSeekV3-like config fields: `intermediate_size`, `expert_dim`,
  `num_routed_experts`, `top_k`, `num_shared_experts`,
  `first_k_dense_replace`, `q_lora_rank`, `kv_lora_rank`,
  `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`,
  `routed_scaling_factor`, and `scoring_func`, alongside existing model fields.
  - Keep `expert_dim` as the internal expert hidden width, matching the current
    `TransformerDimensions` and `MoE` module.
  - Map upstream names such as `moe_intermediate_size`, `n_routed_experts`,
    `num_experts_per_tok`, and `n_shared_experts` into the internal vocabulary
    at preset/config load time.
- Add preset dims:
  - DeepSeek-V3: vocab `129280`, layers `61`, width `7168`, heads/KV heads
    `128`, dense prefix `3`, dense FFN `18432`, MoE expert dim `2048`,
    routed experts `256`, top-k `8`, shared experts `1`, MLA ranks
    `1536/512`, qk dims `128+64`, v dim `128`.
  - Kimi-K2: vocab `163840`, layers `61`, width `7168`, heads/KV heads `64`,
    dense prefix `1`, dense FFN `18432`, MoE expert dim `2048`, routed experts
    `384`, top-k `8`, shared experts `1`, same MLA ranks/dims.
- Add `mla_attention` module/op helpers with separate projection sub-ops:
  q low-rank projection/norm/up-projection, KV low-rank
  projection/norm/up-projection, RoPE, MLA attention with separate QK/V
  dimensions, and output projection with residual accumulation.
- Add a DeepSeek-style block module that composes MLA plus either dense SwiGLU
  FFN for prefix layers or DeepSeek MoE for later layers. Use separate compute
  block keys/names for dense-prefix and MoE layers so summaries do not collapse
  incompatible sub-op lists.
- Compose DeepSeek/Kimi blocks with existing `SwiGLUMLP` for dense prefix
  layers and existing `MoE` for suffix layers. Do not add a DeepSeek-specific
  MoE module for v1. Keep grouped/sigmoid router scoring as metadata.
- Wire `deepseek_v3` and `kimi_k2` into server presets, UI family/preset
  dropdowns, CLI export examples, docs, and package exports.

## File Inventory

Ops:

- `src/dataflow_sim/workloads/ops/forward/mla_attention.py`
- `src/dataflow_sim/workloads/ops/backward/mla_attention.py`
- Reuse existing matmul, norm, activation, movement, optimizer, and coarse MoE
  helper patterns where formulas match.
- Do not add router op files in v1.

Modules:

- `src/dataflow_sim/workloads/modules/deepseek_dimensions.py`
- `src/dataflow_sim/workloads/modules/mla_attention.py`
- `src/dataflow_sim/workloads/modules/deepseek_block.py`

Models:

- `src/dataflow_sim/workloads/models/deepseek_v3.py`
- `src/dataflow_sim/workloads/models/kimi_k2.py`

## Public Interfaces

- Extend server/UI model params to carry the new MLA/MoE-specific fields.
- Show these fields in an advanced architecture section only for `deepseek_v3`
  and `kimi_k2`.
- Keep existing Llama/Qwen/OLMoE fields and behavior unchanged.
- Keep `dense_attention` untouched for now; no GQA rename in this change.

## MLA Cost Model Defaults

Definitions: `D=hidden_size`, `H=num_attention_heads`,
`Qn=qk_nope_head_dim`, `Qr=qk_rope_head_dim`, `Q=Qn+Qr`,
`V=v_head_dim`, `QL=q_lora_rank`, and `KL=kv_lora_rank`.

- If `q_lora_rank` is set:
  - `q_a_proj`: matmul `D -> QL`;
  - `q_a_norm`: memory `2 * tokens * QL * bytes_per_element`;
  - `q_b_proj`: matmul `QL -> H * Q`.
- If `q_lora_rank` is absent, use a direct `q_proj` matmul `D -> H * Q`.
- `kv_a_proj_with_mqa`: matmul `D -> KL + Qr`.
- `kv_a_norm`: memory `2 * tokens * KL * bytes_per_element`.
- `kv_b_proj`: matmul `KL -> H * (Qn + V)`.
- MLA RoPE: provisional memory
  `2 * tokens * Qr * (H + 1) * bytes_per_element`, because query has `H`
  rotary heads and compressed key has one MQA rotary head before expansion.
- MLA attention forward: provisional FLOPs `H * (Q + V) * sum(seq_len^2)`.
- MLA attention backward: provisional FLOPs
  `H * (2 * Q + 3 * V) * sum(seq_len^2)`, with effective FLOPs
  `H * (2 * Q + 2 * V) * sum(seq_len^2)` to match the current backward
  attention convention.
- `o_proj`: matmul `H * V -> D` with `accumulate=True`.

## MoE Cost Model Defaults

DeepSeek/Kimi MoE should intentionally mirror the current `qwen3_moe` modeling
style unless we explicitly broaden all MoE accounting later.

- Treat routing as metadata, not emitted sub-ops, for v1.
- Use `expert_dim` as the internal expert hidden width.
- Use `num_routed_experts`, `top_k`, and `num_shared_experts` to describe the
  sparse FFN.
- Approximate routed tokens per expert as
  `routed_tokens = tokens * top_k // num_routed_experts`.
- Dense prefix layers use dense SwiGLU with `intermediate_size`.
- MoE suffix layers use:
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
- If an upstream model has grouped routing, sigmoid routing, scaling factors,
  or auxiliary losses, keep those values in metadata and document them as not
  emitted in v1.

## Test Plan

- Unit-test preset loading, aliases, override behavior, and exact config values
  for both new families.
- Test small reduced DeepSeek/Kimi configs through `build_training_program`,
  `build_training_workload`, simulator summary, and recompute under constrained
  memory.
- Assert dense prefix vs MoE suffix task/block counts and distinct compute block
  keys.
- Assert MLA sub-op names and key projection dimensions/FLOPs/bytes for forward,
  backward, and recompute.
- Assert coarse MoE shared/routed expert sub-op names and dimensions, matching
  the abstraction style used by existing `qwen3_moe`.
- Add server/UI preset tests and run `npm --prefix ui run build`.
- Run full Python test suite after implementation.

## Assumptions

- MTP, explicit router logits/top-k/scoring ops, router matrices, and
  aux/router losses are metadata-only for v1; no extra tasks are emitted.
- Norm/router non-matrix scalar parameters may remain omitted only if
  consistent with existing parameter accounting.
- Default dtype policy remains bf16.
- Existing models and old presets must remain backward compatible.

## Future Fidelity Upgrades

- Add explicit router logits/top-k/grouped scoring ops once we decide to account
  for those costs across all MoE families.
- Add router matrix parameter accounting together with emitted router ops, so
  params/FLOPs/bytes remain internally consistent.
- Add MTP and aux/router loss tasks when the workload API can expose them
  without special-casing one model family.
