# Transformer recipe — how the example app maps training onto the simulator

## Overview
A transformer training run is lowered to a single linear chain of `Task`s.
Each training step contains forward layers, a head block, backward layers,
and optional optimizer tasks. The chain operates on `Object`s: inputs,
weights, saved activations, residual handoff tensors, per-step gradients, and
optimizer state. Per-task `runtime` is a roofline estimate produced by
decomposing each block into compute and memory-bound **sub-ops**, timing each
against a `HardwareEnv`, and summing. `Object` sizes come from the same
dimensional math. The chain starts persistent weights and optimizer state on
host and the first input on device; per-step gradients are produced by the
backward pass rather than loaded from host. A policy
(`sliding_window`, `max_reduce`, `min_grow`, etc.) decorates the bare chain
with `TransferTrigger`s and an initial device pool before `simulator.run(...)`
executes it.

## Object inventory
All byte counts use bf16 (`BYTES_PER_ELEMENT = 2`). Shared shorthand:
`T = num_seqs * seqlen` (total tokens); `d = d_model`; `hd = head_dim`;
`nh = n_heads`; `nkv = n_kv_heads`; `edim = expert_dim`;
`ns = num_shared_experts`; `nr = num_routed_experts`; `tk = top_k`.

`TrainingConfig.num_steps` is the outer loop. Within each step,
`TrainingConfig.grad_accum_rounds` repeats the forward/head/backward chain
once per accumulation round. Persistent model state objects remain shared
(`W_i`, `W_head`, and `O_i`), while per-microbatch data objects carry
`<step>_<accum>` indices. Per-step gradient objects carry a step index but no
accumulation index because later accumulation rounds in the same step mutate
the object produced by the first round.

### `input_<k>_<j>` — first-layer residual-stream input
- Dimensions: `[num_seqs, seqlen, d]`. Bytes: `T * d * 2`.
- Lifetime: declared in `initial_memory`; `input_0_0` starts on device and
  the rest start on host. Consumed only by `f_<k>_<j>_0`, then releasable.

### `W_i` (`i = 0..L-1`) — per-layer weight bank
- Dimensions: opaque bundle of `params_per_layer(spec) = d * (hd*(2*nh + 2*nkv)
  + 3*edim*(ns + nr))` parameters. Counts **all** experts including unused
  routed ones — the bank does not shrink with `tk`.
- Bytes: `params_per_layer(spec) * 2`.
- Lifetime: starts on host; referenced by `f_<k>_<j>_<i>`,
  `r_<k>_<j>_<i>`, and `b_<k>_<j>_<i>` across all steps. If an optimizer
  tail is enabled, `step_<k>_<i>` mutates it. If
  `TrainingConfig.final_model_state_on_host=True`,
  `final_locations[W_i] = "host"` asks the policy to return the updated bytes
  to host by chain end.

### `A_<k>_<j>_<i>` — saved activation for layer `i`
- Dimensions: `[num_seqs, seqlen, hd*(2*nh + 2*nkv) + 2*d + 2*(ns + tk)*edim]`.
  The routed branch contributes the **active** `tk` slice, not the full `nr`
  bank.
- Bytes: `T * (hd*(2*nh + 2*nkv) + 2*d + 2*(ns + tk)*edim) * 2`.
- Lifetime: produced by `f_<k>_<j>_<i>`; consumed by
  `r_<k>_<j>_<i>` and `b_<k>_<j>_<i>`; releasable after that backward task.

### `y_<k>_<j>_<i>` — forward layer output (residual handoff)
- Dimensions: `[num_seqs, seqlen, d]`. Bytes: `T * d * 2`.
- Lifetime: produced by `f_<k>_<j>_<i>`; consumed by
  `f_<k>_<j>_<i+1>` (or by `head_<k>_<j>` when `i = L-1`); releasable after
  consumer.

### `W_head` — head projection weights
- Dimensions: `[d, vocab_size]`. Bytes: `d * vocab_size * 2`.
- Lifetime: starts on host; referenced by every `head_<k>_<j>`.

### `dy_head_<k>_<j>` and `dy_<k>_<j>_<i>` — backward residual-stream gradients
- Dimensions: `[num_seqs, seqlen, d]` (same shape as `y_i`). Bytes: `T * d * 2`.
- Lifetime: `dy_head_<k>_<j>` is produced by `head_<k>_<j>` and consumed by
  `b_<k>_<j>_<L-1>`. `dy_<k>_<j>_<i>` is produced by
  `b_<k>_<j>_<i>` and consumed by `b_<k>_<j>_<i-1>` (`dy_*_*_0` is
  terminal).

### `dW_<k>_<i>` and `dW_head_<k>` — per-step weight-gradient buffers
- Dimensions: same shape as the matching `W_i` / `W_head`. Bytes:
  `layer_weight_bytes(spec)` / `head_weight_bytes(spec)`.
- Lifetime: produced during the first accumulation round of step `k`
  (`b_<k>_0_<i>` produces `dW_<k>_<i>` and `head_<k>_0` produces
  `dW_head_<k>`). Later accumulation rounds in the same step consume and
  mutate the same object. With an optimizer tail, `dW_<k>_<i>` must preserve
  its updated bytes until `step_<k>_<i>` consumes it; after that it is
  disposable unless listed in `final_locations`.

### `O_i` — per-layer optimizer state
- Present only when `TrainingConfig.optimizer` is `"adamw"` or `"muon"`.
- Bytes: `2 * layer_weight_bytes(spec)` for AdamW, representing two
  state tensors; `layer_weight_bytes(spec)` for Muon, representing one
  momentum tensor.
- Lifetime: starts on host, is consumed and mutated by `step_<k>_<i>` in
  each training step. If `TrainingConfig.final_model_state_on_host=True`,
  `final_locations[O_i] = "host"` asks the policy to return the updated bytes
  to host by chain end.

### MoE-only `x_scatter` / `x_gather` / `dy_scatter` / `dy_gather`
When `nr > 0` and `tk > 0`, the per-layer sub-op stream gains scatter/gather
memory-bound stages sized at `T * (1 + tk) * d * 2` bytes each. They appear
in `runtime` accounting but do **not** introduce new top-level `Object`s —
the expanded `[T*(1+tk), d]` tensor is intra-task scratch and never crosses
task boundaries.

## Task chain shape
The chain is strict sequence. For each training step `k`, and each
accumulation round `j` inside that step, the builder emits:

```
f_k_j_0, f_k_j_1, ..., f_k_j_{L-1},
head_k_j,
r_k_j_{L-1}, b_k_j_{L-1}, ..., r_k_j_0, b_k_j_0
```

When an optimizer mode is enabled, the builder appends
`step_k_0, step_k_1, ..., step_k_{L-1}` after all accumulation rounds for
that step. Each round's activations and residual gradients are distinct;
`dW_k_i` is shared only within step `k`. Persistent `W_i` and `O_i` carry
state across steps. By default, omitted terminal placement means the final
updated model state is disposable after its last simulated use. Set
`TrainingConfig.final_model_state_on_host=True` to add
`final_locations[W_i] = "host"` and `final_locations[O_i] = "host"` for each
optimizer-backed layer.

| Task    | inputs                                        | outputs                          | runtime              |
|---------|-----------------------------------------------|----------------------------------|----------------------|
| `f_k_j_0`   | `input_k_j`, `W_0`                       | `A_k_j_0`, `y_k_j_0`             | `layer_fwd_microseconds(spec, hw, cfg)` |
| `f_k_j_i>0` | `y_k_j_{i-1}`, `W_i`                    | `A_k_j_i`, `y_k_j_i`             | same                 |
| `head_k_0`  | `y_k_0_{L-1}`, `W_head`                 | `dy_head_k_0`, `dW_head_k`       | `head_microseconds(...)` |
| `head_k_j>0` | `y_k_j_{L-1}`, `W_head`, `dW_head_k` (mutated) | `dy_head_k_j`             | same                 |
| `r_k_j_i`   | `A_k_j_i`, `W_i`                        | — (touch-only, `runtime=0`)      | 0; a recompute hook  |
| `b_k_0_i`   | upstream `dy`, `A_k_0_i`, `W_i`         | `dy_k_0_i`, `dW_k_i`             | `layer_bwd_microseconds(...)` |
| `b_k_j>0_i` | upstream `dy`, `A_k_j_i`, `W_i`, `dW_k_i` (mutated) | `dy_k_j_i`              | same                 |
| `step_k_i` | `dW_k_i`, `W_i` (mutated), `O_i` (mutated) | —                              | `optimizer_step_microseconds(...)` |

`upstream` is `dy_head_k_j` for `i = L-1`, else `dy_k_j_{i+1}`. Runtime is a
per-task scalar in µs (1 tick = 1 µs).

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

### Optimizer sub-ops
Optimizer formulas live in `app/src/dataflow_app/workloads/optimizers.py`.
The transformer module supplies the logical matrix inventory for one layer:
`qkv_proj`, `attn_proj`, shared MLP up/down matrices, and any routed expert
up/down matrices. Routed experts are counted as the full expert bank because
optimizer state is allocated for every trainable matrix, not only for the
`top_k` experts active on a token batch.

- `adamw_step` — memory-bound. Reads `dW_i`, `W_i`, and `O_i`; writes `W_i`
  and `O_i`. Since AdamW has `O_i = 2W`, bytes are
  `W + W + 2W + W + 2W = 7W`, where `W = layer_weight_bytes(spec)`. Flops are
  currently modeled as `0`.
- `muon_step` — compute/matmul-bound when large. Muon first updates one
  momentum tensor, then applies Newton-Schulz iterations to each logical
  matrix. For a matrix transposed to shape `(n, m)` with `n <= m`, one
  iteration costs `4 n^2 m + 2 n^3 + 2 n m + 3 n^2` flops. The implementation
  uses `5` iterations and adds `10 n m` elementwise flops outside the loop.
  Bytes are approximated as bf16 persistent traffic plus scratch traffic:
  `2 * (12 n m + 5 * (5 n m + 6 n^2))`.

For `llama3_8B`, one layer has `218,103,808` parameters
(`436,207,616` bf16 bytes). AdamW state is `872,415,232` bytes and one AdamW
step accesses `3,053,453,312` bytes. Muon state is `436,207,616` bytes; one
Muon step is `20,621,211,729,920` flops and accesses `20,166,213,632` bytes
under the approximation above.

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

If an optimizer is enabled, each training step's post-accumulation tail
executes `step_k_0` → `step_k_1` → … → `step_k_{L-1}`. Each optimizer task
consumes the step-local accumulated `dW_k_i` and mutates persistent `W_i`
plus `O_i`.

## App simulation
The webapp materializes all requested `TrainingConfig.num_steps`, runs the
selected policy once on that finite chain, and reports exact simulator output
for the resulting annotated plan. Returned event snapshots may be downsampled
for response size, but the summary metrics are computed from the full event
log before compaction.

For very large chains, the app runs the final simulation with
`snapshots=False`. In that mode, makespan, peak memory, utilization metrics, and
all compute/transfer intervals are still exact. The app also requests the
simulator's compact `memory_trace`, so the GPU memory plot remains available.
Only per-event object-level memory contents and reference streams are omitted so
the response does not spend minutes constructing data that the UI would heavily
downsample.

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
   (`apply_pressurefit_policy(bare, …)`) and hand the result to
   `simulator.run(...)`. The `breakdown` payload carries the per-sub-op
   timings used to populate the dashboard's roofline panel.
2. **Scripts.** `app/scripts/run_training.py` runs a single configuration
   end-to-end. `app/scripts/sweep_transformer.py` sweeps model × hardware ×
   `(num_seqs, seqlen)` and writes a results table.
   `app/scripts/compare_policies.py` reuses one bare chain and runs each
   registered policy against it for an apples-to-apples comparison.
   `app/scripts/pressurefit_mode_sweep.py` compares PressureFit's `auto`,
   `fast`, and `full` candidate portfolios.
3. **Webapp.** The UI POSTs a model+hardware+training-config selection to
   `POST /api/simulate`. The server builds the bare chain via
   `build_transformer_bare_chain`, applies the user-selected policy, runs the
   simulator, and returns `{log, breakdown, summary, chain,
   policy_diagnostics}`. The React frontend renders the event log as a
   Gantt-style trace alongside the per-sub-op roofline table. For PressureFit,
   `policy_diagnostics` reports candidate timings and the selected candidate;
   for other policies it is `null`.

## See also
- `docs/workload-recipe.md` — the general workload-construction API
  (`Object`, `Task`, `OutputAlloc`, `TaskChain`, `TransferTrigger`) that
  this recipe specializes.
- `docs/policy/README.md` — how to pick (or write) the policy that turns a
  bare chain into a runnable one.
