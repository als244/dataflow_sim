# Transformer recipe — how the example app maps training onto the simulator

## Overview
A transformer training step is lowered to a single linear chain of `Task`s
(forward layers, head, backward layers) operating on `Object`s (input, weights,
saved activations, layer outputs, gradients). Per-task `runtime` is a roofline
estimate produced by decomposing each layer into compute and memory-bound
**sub-ops**, timing each against a `HardwareEnv`, and summing. `Object` sizes
come from the same dimensional math. The chain leaves all weights and gradient
buffers on host and the first input on device; a policy (`sliding_window`,
`max_reduce`, `min_grow`, etc.) decorates the bare chain with
`TransferTrigger`s and an initial device pool before `simulator.run(...)`
executes it.

## Object inventory
All byte counts use bf16 (`BYTES_PER_ELEMENT = 2`). Shared shorthand:
`T = num_seqs * seqlen` (total tokens); `d = d_model`; `hd = head_dim`;
`nh = n_heads`; `nkv = n_kv_heads`; `edim = expert_dim`;
`ns = num_shared_experts`; `nr = num_routed_experts`; `tk = top_k`.

### `input` — first-layer residual-stream input
- Dimensions: `[num_seqs, seqlen, d]`. Bytes: `T * d * 2`.
- Lifetime: declared in `initial_memory` on device; consumed only by `f_0`,
  then releasable.

### `W_i` (`i = 0..L-1`) — per-layer weight bank
- Dimensions: opaque bundle of `params_per_layer(spec) = d * (hd*(2*nh + 2*nkv)
  + 3*edim*(ns + nr))` parameters. Counts **all** experts including unused
  routed ones — the bank does not shrink with `tk`.
- Bytes: `params_per_layer(spec) * 2`.
- Lifetime: starts on host; referenced by `f_i`, `r_i`, and `b_i`. Survives
  the entire chain — the policy decides residency.

### `A_i` — saved activation for layer `i`
- Dimensions: `[num_seqs, seqlen, hd*(2*nh + 2*nkv) + 2*d + 2*(ns + tk)*edim]`.
  The routed branch contributes the **active** `tk` slice, not the full `nr`
  bank.
- Bytes: `T * (hd*(2*nh + 2*nkv) + 2*d + 2*(ns + tk)*edim) * 2`.
- Lifetime: produced by `f_i`; consumed by `r_i` and `b_i`; releasable after
  `b_i`.

### `y_i` — forward layer output (residual handoff)
- Dimensions: `[num_seqs, seqlen, d]`. Bytes: `T * d * 2`.
- Lifetime: produced by `f_i`; consumed by `f_{i+1}` (or by `head` when
  `i = L-1`); releasable after consumer.

### `W_head` — head projection weights
- Dimensions: `[d, vocab_size]`. Bytes: `d * vocab_size * 2`.
- Lifetime: starts on host; referenced once by `head`.

### `dy_head` and `dy_i` — backward residual-stream gradients
- Dimensions: `[num_seqs, seqlen, d]` (same shape as `y_i`). Bytes: `T * d * 2`.
- Lifetime: `dy_head` produced by `head`, consumed by `b_{L-1}`. `dy_i`
  produced by `b_i`, consumed by `b_{i-1}` (`dy_0` is terminal — no
  consumer).

### `dW_i` and `dW_head` — weight-gradient buffers
- Dimensions: same shape as the matching `W_i` / `W_head`. Bytes:
  `layer_weight_bytes(spec)` / `head_weight_bytes(spec)`.
- Lifetime: arrives on host carrying the previous step's accumulator. `b_i`
  declares `mutates_inputs=[dW_i]` (and `head` does the same for `dW_head`)
  so the planner writes them back to host instead of releasing.

### MoE-only `x_scatter` / `x_gather` / `dy_scatter` / `dy_gather`
When `nr > 0` and `tk > 0`, the per-layer sub-op stream gains scatter/gather
memory-bound stages sized at `T * (1 + tk) * d * 2` bytes each. They appear
in `runtime` accounting but do **not** introduce new top-level `Object`s —
the expanded `[T*(1+tk), d]` tensor is intra-task scratch and never crosses
task boundaries.

## Task chain shape
The chain is a strict sequence (one task per stage):

```
f_0, f_1, …, f_{L-1}, head, r_{L-1}, b_{L-1}, …, r_0, b_0
```

| Task    | inputs                                        | outputs                          | runtime              |
|---------|-----------------------------------------------|----------------------------------|----------------------|
| `f_0`   | `input`, `W_0`                                | `A_0`, `y_0`                     | `layer_fwd_microseconds(spec, hw, cfg)` |
| `f_i>0` | `y_{i-1}`, `W_i`                              | `A_i`, `y_i`                     | same                 |
| `head`  | `y_{L-1}`, `W_head`, `dW_head` (mutated)      | `dy_head`                        | `head_microseconds(...)` |
| `r_i`   | `A_i`, `W_i`                                  | — (touch-only, `runtime=0`)      | 0; a recompute hook  |
| `b_i`   | upstream `dy`, `A_i`, `W_i`, `dW_i` (mutated) | `dy_i`                           | `layer_bwd_microseconds(...)` |

`upstream` is `dy_head` for `i = L-1`, else `dy_{i+1}`. Runtime is a per-task
scalar in µs (1 tick = 1 µs).

### Recompute tasks (`r_i`)
Each backward step is preceded by a zero-runtime `r_i` task whose only job is
to declare `inputs=[A_i, W_i]` — that pins both objects co-resident with the
upcoming `b_i` without overlapping a compute slot. The task is created in
`app/src/dataflow_app/workloads/training.py` (the backward loop in
`build_bare_training_chain`) and currently has `runtime=0`.

To extend the model with real partial-recompute cost, define a
`recompute_subops(spec, cfg)` analogous to `forward_subops` returning the
subset of fwd sub-ops that would be re-executed, aggregate it with
`time_subop` into an `r_microseconds(...)`, and thread an `r_runtime`
parameter through `build_transformer_bare_chain` alongside the existing
`fwd_runtime` / `bwd_runtime` / `head_runtime` hooks.

## Sub-op decomposition
`runtime` for each task is the sum of `time_subop(s, hw).total_us` over the
sub-ops from `forward_subops`, `backward_subops`, and `head_subops`. Each
`SubOp` is compute-bound (rooflined against `peak_tflops * <eff_name>_eff`)
or memory-bound (rooflined against `gpu_membw_gbs * mem_eff`); per-call time
is `max(math_us, mem_us)` and totals scale by `count` (used for "one matmul
per expert"). `attn_bwd` declares `effective_flops = 4×` while doing
`flops = 5×` so the effective-TFLOPS metric isolates the flash-attention
recompute overhead.

Below, `T_r = T * tk / nr` is routed-tokens-per-expert; flops/bytes shown
are **per call**, multiplied by `count` for the total. All byte counts
already include the `* 2` bf16 factor.

### Forward sub-ops (`forward_subops`, in execution order)
- `attn_norm` — memory. Bytes `2 * T * d * 2`. Pre-attention RMSNorm.
- `qkv_proj` — compute/matmul. Flops `2 * T * d * (nh+2*nkv)*hd`. Bytes
  `(T*d + d*(nh+2*nkv)*hd + T*(nh+2*nkv)*hd) * 2`. Fused QKV projection.
- `qk_norm` — memory, optional (skipped if `qk_norm=False`). Bytes
  `2 * T * (hd*(nh+nkv)) * 2`. Per-head Q/K norm.
- `rope` — memory. Bytes `2 * T * (hd*(nh+nkv)) * 2`. Rotary on Q/K.
- `attn` — compute/attn_fwd. Flops `2 * num_seqs * nh * hd * S * S`. Bytes
  `(T*(nh+2*nkv)*hd + T*nh*hd) * 2`. Flash-attention forward.
- `attn_proj` — compute/matmul. Flops `2 * T * (nh*hd) * d`. Bytes
  `(T*nh*hd + nh*hd*d + T*d) * 2`. Output projection.
- `ffn_norm` — memory. Bytes `2 * T * d * 2`. Pre-MLP RMSNorm.
- `shared_mlp_up` (`count=ns`) — compute/matmul. Flops `2 * T * d * 2*edim`.
  Bytes `(T*d + d*2*edim + T*2*edim) * 2`. Shared experts' gate+up.
- `x_scatter` — memory, MoE-only. Bytes `T * (1+tk) * d * 2`. Dispatch tokens
  to the expanded routed-branch tensor.
- `routed_mlp_up_one_expert` (`count=nr`) — compute/matmul. Flops
  `2 * T_r * d * 2*edim`. Bytes `(T_r*d + d*2*edim + T_r*2*edim) * 2`. One
  gate+up matmul per routed expert.
- `swiglu` — memory. Bytes `3 * T * edim * (ns+tk) * 2`. Elementwise
  `SiLU(gate)*up`.
- `shared_mlp_down` (`count=ns`) — compute/matmul. Flops `2 * T * edim * d`.
  Bytes `(T*edim + edim*d + T*d) * 2`.
- `routed_mlp_down_one_expert` (`count=nr`) — compute/matmul. Flops
  `2 * T_r * edim * d`. Bytes `(T_r*edim + edim*d + T_r*d) * 2`.
- `x_gather` — memory, MoE-only. Bytes `T * (1+tk) * d * 2`. Combine routed
  outputs back to `[T, d]`.

### Backward sub-ops (`backward_subops`)
Emitted as a **DGRAD block** (reverse-fwd execution order; sits on the
critical path producing the upstream `dy_i`) followed by a **WGRAD block**
(weight grads only; does not gate downstream tasks, still summed into
`bwd_runtime`). Each fwd matmul yields a dgrad **and** a wgrad with
identical flops/bytes.

DGRAD block (in order):
- `dy_scatter` — memory, MoE-only. Bytes `T * (1+tk) * d * 2`. Undoes
  `x_gather`.
- `routed_mlp_down_one_expert_dgrad` (`count=nr`) — same flops/bytes as fwd
  `routed_mlp_down_one_expert`.
- `shared_mlp_down_dgrad` (`count=ns`) — same as fwd `shared_mlp_down`.
- `swiglu_bwd` — memory. Bytes `5 * T * edim * (ns+tk) * 2`.
- `routed_mlp_up_one_expert_dgrad` (`count=nr`) — same as fwd
  `routed_mlp_up_one_expert`.
- `dy_gather` — memory, MoE-only. Bytes `T * (1+tk) * d * 2`. Undoes
  `x_scatter`.
- `shared_mlp_up_dgrad` (`count=ns`) — same as fwd `shared_mlp_up`.
- `ffn_norm_bwd` — memory. Bytes `7 * T * d * 2`.
- `attn_proj_dgrad` — same as fwd `attn_proj`.
- `attn_bwd` — compute/attn_bwd. `flops = 5 * num_seqs * nh * hd * S * S`
  (drives `math_us`); `effective_flops = 4 * num_seqs * nh * hd * S * S`
  (drives reported effective-TFLOPS). Bytes
  `(T*(nh+2*nkv)*hd + T*nh*hd + T*(nh+2*nkv)*hd) * 2`.
- `rope_bwd` — memory. Bytes `2 * T * (hd*(nh+nkv)) * 2`.
- `qk_norm_bwd` — memory, optional. Bytes `7 * T * (hd*(nh+nkv)) * 2`.
- `qkv_proj_dgrad` — same as fwd `qkv_proj`.
- `attn_norm_bwd` — memory. Bytes `7 * T * d * 2`.

WGRAD block (down before up; mirrors the dgrad ordering):
- `routed_mlp_down_one_expert_wgrad` (`count=nr`), `shared_mlp_down_wgrad`
  (`count=ns`), `routed_mlp_up_one_expert_wgrad` (`count=nr`),
  `shared_mlp_up_wgrad` (`count=ns`), `attn_proj_wgrad`, `qkv_proj_wgrad`
  — each identical in flops/bytes to its fwd counterpart.

### Head sub-ops (`head_subops`, in execution order)
- `final_norm` — memory. Bytes `2 * T * d * 2`.
- `head_proj` — compute/matmul. Flops `2 * T * d * vocab_size`. Bytes
  `head_weight_bytes(spec) + T*d*2 + T*d*2`.
- `cross_entropy` — memory. Bytes `2 * T * vocab_size * 2`. Logits → loss +
  dlogits.
- `head_proj_dgrad` / `head_proj_wgrad` — compute/matmul. Same flops/bytes
  as `head_proj`.
- `final_norm_bwd` — memory. Bytes `7 * T * d * 2`.

### Sub-op order walkthrough
One `f_i` task executes: `attn_norm` → `qkv_proj` → `qk_norm?` → `rope` →
`attn` → `attn_proj` → `ffn_norm` → `shared_mlp_up` (×ns) → `x_scatter?` →
`routed_mlp_up_one_expert` (×nr) → `swiglu` → `shared_mlp_down` (×ns) →
`routed_mlp_down_one_expert` (×nr) → `x_gather?`.

One `b_i` task executes the DGRAD block then the WGRAD block:
`dy_scatter?` → `routed_mlp_down_dgrad` (×nr) → `shared_mlp_down_dgrad`
(×ns) → `swiglu_bwd` → `routed_mlp_up_dgrad` (×nr) → `dy_gather?` →
`shared_mlp_up_dgrad` (×ns) → `ffn_norm_bwd` → `attn_proj_dgrad` →
`attn_bwd` → `rope_bwd` → `qk_norm_bwd?` → `qkv_proj_dgrad` →
`attn_norm_bwd` (DGRAD ends; `dy_i` is materialized here) →
`routed_mlp_down_wgrad` (×nr) → `shared_mlp_down_wgrad` (×ns) →
`routed_mlp_up_wgrad` (×nr) → `shared_mlp_up_wgrad` (×ns) →
`attn_proj_wgrad` → `qkv_proj_wgrad` (`dW_i` accumulated).

## Presets
- **`model_dims.json`** (loaded by `load_model_presets()`, lru-cached) — a
  name→spec registry. Each entry sets `vocab_size`, `n_layers`, `d_model`,
  `head_dim`, `n_heads`, `n_kv_heads`, `expert_dim`, `num_shared_experts`,
  `num_routed_experts`, `top_k`, `qk_norm`. Shipped presets include
  `nanogpt_124M`, `llama3_8B`, and MoE configs (`olmoe_7Bx1B`, …). Fields
  like `datatypes` and `is_causal` in the JSON are ignored — bf16 and causal
  attention are hardcoded.
- **`HARDWARE_PRESETS`** — name→`HardwareEnv` with the parameters that drive
  every sub-op time: `peak_tflops` (compute roof), `gpu_membw_gbs` (HBM
  roof), `interconnect_bw_gbs` (host↔device link, becomes the chain's
  `bandwidth_h2d` / `bandwidth_d2h`), and the four efficiency knobs
  `matmul_eff`, `attn_fwd_eff`, `attn_bwd_eff`, `mem_eff`. Shipped:
  `H100` (989 TFLOPS / 3 TB/s / 50 GB/s PCIe-class link) and `RTX_5090`
  (210 TFLOPS / 1.5 TB/s / 30 GB/s).

## End-to-end three paths
1. **Python.** `from dataflow_app.workloads.training import
   build_transformer_bare_chain`; pass a `TransformerSpec`, `HardwareEnv`,
   `TrainingConfig`, get `(bare_chain, breakdown)`. Apply a policy
   (`apply_sliding_window_policy(bare, …)`) and hand the result to
   `simulator.run(...)`. The `breakdown` payload carries the per-sub-op
   timings used to populate the dashboard's roofline panel.
2. **Scripts.** `app/scripts/run_training.py` runs a single configuration
   end-to-end. `app/scripts/sweep_transformer.py` sweeps model × hardware ×
   `(num_seqs, seqlen)` and writes a results table.
   `app/scripts/compare_policies.py` reuses one bare chain and runs each
   registered policy against it for an apples-to-apples comparison.
3. **Webapp.** The UI POSTs a model+hardware+training-config selection to
   `POST /api/simulate`. The server builds the bare chain via
   `build_transformer_bare_chain`, applies the user-selected policy, runs the
   simulator, and returns `{events, breakdown, summary}`. The React frontend
   renders the event log as a Gantt-style trace alongside the per-sub-op
   roofline table.

## See also
- `docs/workload-recipe.md` — the general workload-construction API
  (`Object`, `Task`, `OutputAlloc`, `TaskChain`, `TransferTrigger`) that
  this recipe specializes.
- `docs/policy/README.md` — how to pick (or write) the policy that turns a
  bare chain into a runnable one.
