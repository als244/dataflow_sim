# Datatype Controls

The webapp's **Datatypes** section applies only to built-in **Model Training**
workloads. A **Custom Dataflow Program** already contains explicit object sizes
and sub-op byte counts, so the simulator does not reinterpret those bytes with
these fields.

Datatype fields control memory footprint and memory traffic. Compute precision
fields control matmul roofline performance. These are intentionally separate:
changing a dtype changes bytes; changing compute precision changes the hardware
peak/efficiency used for math time.

Supported datatype values:

| Value | Bytes Per Element |
| --- | ---: |
| `bf16` | 2 |
| `fp8` | 1 |
| `fp4` | 0.5 |

Object sizes are rounded up to whole bytes.

## UI Fields

| UI Field | Builder Field | Controls |
| --- | --- | --- |
| Weight DType | `DTypePolicy.param` | Default parameter bytes for non-expert weights, including attention, dense MLPs, and LM head weights. |
| Activation DType | `DTypePolicy.activation` | Main activation tensors and activation-gradient tensors/traffic: `input_*`, `y_*`, `A_*`, `dy_*`, forward outputs, saved backward context, norm/activation memory terms, attention activation traffic, dense MLP activation traffic, and MoE non-dispatch activation traffic. |
| Expert Dispatch DType | `DTypePolicy.expert_dispatch` | MoE dispatch representation for forward expert routing and the symmetric routed activation-gradient dispatch path in backward. It covers `x_scatter` output traffic, `dy_scatter` output traffic, `dy_gather` input traffic, expert up-projection input reads, expert up-projection dgrad output writes, routed down-projection upstream-gradient reads, and the matching up-projection wgrad input reads. |
| Parameter Gradient DType | `DTypePolicy.gradient` | Parameter-gradient tensors and parameter-gradient memory traffic: `dW_*`, wgrad outputs, wgrad accumulation traffic, and optimizer reads of parameter gradients. It does not control `dy_*`; activation gradients follow Activation DType unless an op explicitly overrides a dispatched lane. |
| Optimizer DType | `DTypePolicy.optimizer_state` | Optimizer-state objects such as `O_*` and optimizer-state traffic inside optimizer steps. |
| Compute Precision | `DTypePolicy.compute` | Default matmul math precision for non-expert matmuls. This selects the matching hardware peak TFLOP/s and matmul efficiency field. |
| Expert Weight DType | `DTypePolicy.expert_param` | MoE expert weight bytes for both shared and routed expert matrices: `shared_mlp_up/down` and `routed_mlp_up/down`. This overrides Weight DType only for matrices marked as expert weights. |
| Expert Compute Precision | `DTypePolicy.expert_compute` | Matmul math precision for MoE expert matmuls, including shared and routed expert up/down projections. |
| Indexer Activation DType | `DTypePolicy.indexer_activation` | DeepSeek-V3.2 DSA Lightning Indexer q/k/w activation lanes and selected-score storage, including `q^I`, `k^I`, `w^I` score-kernel traffic, saved indexer activation context, and selected indexer scores. This defaults to `fp8` and is only shown for indexer families. |
| Indexer Compute Precision | `DTypePolicy.indexer_compute` | DeepSeek-V3.2 Lightning Indexer quadratic score matmul precision. This defaults to `fp8` and is only shown for indexer families. |

The webapp hides family-specific fields when they do not apply. Expert dtype
controls appear only for MoE families. Indexer dtype controls appear only for
DeepSeek-V3.2-style DSA families.

DeepSeek-V3.2-style families also expose a model-architecture checkbox,
**Train Indexer**. When unchecked, forward index scoring still runs to select
sparse KV entries, but indexer score backward, indexer projection wgrads,
indexer optimizer state, and selected indexer-score context in `A_*` are
omitted. GLM-5.2 IndexShare additionally has shared-index layers that reuse
sparse positions from full-index layers; those shared layers omit indexer
projection/scoring costs and do not add a repeated selected-index object in v1.
The dense quadratic score tensor is never materialized in any mode.

## Object Sizing

The model-training builder emits these major objects:

| Object Pattern | Role | Size Source |
| --- | --- | --- |
| `input_<step>_<round>` | activation | `tokens * input_dim * Activation DType` |
| `y_<step>_<round>_<layer>` | activation | `tokens * layer_output_dim * Activation DType` |
| `A_<step>_<round>_<layer>` | activation | `tokens * saved_activation_width * Activation DType` |
| `dy_*` | activation | shape elements times Activation DType |
| `dW_*` | gradient | parameter elements times Parameter Gradient DType |
| `W_*` | parameter | Weight DType for non-expert matrices, Expert Weight DType for expert matrices when matrix metadata is available |
| `W_head` | parameter | Weight DType |
| `O_*`, `O_head` | optimizer state | Optimizer DType |

Important: `A_*` uses **Activation DType**, not Expert Dispatch DType. `A_*`
represents saved backward context, including outputs and intermediate tensors.
It does not currently model a separate quantized expert-dispatch buffer as a
persistent saved object.

For DeepSeek-V3.2 DSA blocks, `A_*` additionally includes the Lightning
Indexer selected-state context:

```text
sum(sequence_length * min(sequence_length, index_topk)) * 4 bytes
+ Train Indexer ? sum(sequence_length * min(sequence_length, index_topk)) * Indexer Activation DType : 0
```

The first term stores selected key indices for sparse DSA attention. The second
term stores selected indexer scores only when **Train Indexer** is enabled. The
dense quadratic score tensor is assumed to be streamed by the Lightning Indexer
kernel and is not part of `A_*`. Other saved context in the same `A_*` object
follows Activation DType, except saved indexer q/k/w activation lanes, which
follow Indexer Activation DType.

For GLM-5.2 IndexShare shared-index layers, this selected-state addend is
omitted. Full-index GLM-5.2 layers follow the DeepSeek-V3.2 sizing above.

## MoE Expert Dispatch

Expert Dispatch DType is intentionally narrow. It models the dtype used for the
dispatched activation stream entering expert up projections and the matching
dispatched activation-gradient stream in backward.

It currently affects:

- `x_scatter`: reads the main activation in Activation DType and writes routed
  expert inputs in Expert Dispatch DType.
- `shared_mlp_up`: reads its activation input in Expert Dispatch DType, reads
  weights in Expert Weight DType, and writes outputs in Activation DType.
- `routed_mlp_up_one_expert`: reads dispatched routed inputs in Expert Dispatch
  DType, reads weights in Expert Weight DType, and writes outputs in Activation
  DType.
- `dy_scatter`: reads the main upstream activation gradient in Activation DType
  and writes routed upstream gradients in Expert Dispatch DType.
- `routed_mlp_down_one_expert_dgrad`: reads the routed upstream gradient side in
  Expert Dispatch DType and writes expert-intermediate activation gradients in
  Activation DType.
- `routed_mlp_down_one_expert_wgrad`: reads the routed upstream gradient side in
  Expert Dispatch DType and writes/accumulates parameter gradients in Parameter
  Gradient DType.
- `routed_mlp_up_one_expert_dgrad`: reads expert-intermediate activation
  gradients in Activation DType and writes dispatched input gradients in Expert
  Dispatch DType.
- `shared_mlp_up_dgrad`: reads expert-intermediate activation gradients in
  Activation DType and writes its input-gradient side in Expert Dispatch DType,
  mirroring the shared expert up-projection input read in forward.
- `dy_gather`: reads dispatched input gradients in Expert Dispatch DType and
  writes the main activation gradient in Activation DType.
- `shared_mlp_up_wgrad` and `routed_mlp_up_one_expert_wgrad`: read the
  up-projection input side in Expert Dispatch DType. Their upstream gradient
  operands use Activation DType and their output/accumulator traffic uses
  Parameter Gradient DType.

It does not affect:

- `A_*` saved activation objects,
- `y_*` main activation objects,
- SwiGLU memory terms,
- expert down-projection activation operands,
- `x_gather`,
- non-dispatched activation-gradient movement, which uses Activation DType,
- optimizer state.

## Expert Weights

Expert Weight DType applies to MoE expert matrices, both shared and routed. For
example, in Qwen3.5 MoE, DeepSeek/Kimi MoE, and Nemotron-H ReLU2 MoE blocks,
these matrices are expert weights:

- `shared_mlp_up`
- `shared_mlp_down`
- `routed_mlp_up`
- `routed_mlp_down`

If a preset has `num_shared_experts = 0`, changing Expert Weight DType will
only visibly affect routed expert matrices because no shared expert matrices
exist. Current Qwen3-MoE and OLMoE presets have no shared experts; Qwen3.5 MoE,
DeepSeek, Kimi, and Nemotron-H presets do have shared experts.

Expert Weight DType affects:

- `W_*` mixed parameter object size for layers whose model file exposes matrix
  metadata,
- expert matmul weight-read bytes,
- optimizer-step weight-read/write bytes for expert matrices.

It does not change parameter-gradient object dtype or optimizer-state dtype.
Those remain controlled by Parameter Gradient DType and Optimizer DType.

## Compute Precision And Hardware

Matmul roofline time is:

```text
math_time = flops / (peak_tflops_for_precision * matmul_eff_for_precision)
```

The hardware form has separate fields for:

- Peak BF16 TFLOP/s, Peak FP8 TFLOP/s, Peak FP4 TFLOP/s.
- BF16 Matmul Efficiency, FP8 Matmul Efficiency, FP4 Matmul Efficiency.

If a precision is not supported by a hardware preset, the webapp shows `--`
for that peak/efficiency field and the API payload uses `null`. A workload that
selects an unsupported compute precision fails during preview/simulation with a
clear hardware-support error. For example, the H100 preset marks FP4 matmul as
unsupported.

Compute Precision selects these fields for default non-expert matmuls.
Expert Compute Precision selects these fields for MoE expert matmuls.
Indexer Compute Precision selects these fields for DeepSeek-V3.2 Lightning
Indexer score matmuls.

Compute precision does not change object sizes or memory bytes. Dtype fields do
not change matmul peak TFLOP/s. Use both fields when modeling low-precision
training where storage format and math format differ.

Attention kernels currently use attention-specific efficiency fields rather
than the matmul precision selectors.

DeepSeek-V3.2 DSA sparse attention follows that same attention-kernel rule for
the selected sparse attention core. Only the Lightning Indexer score op uses
Indexer Compute Precision.

## Mixed-Precision Matrix Bytes

The matmul op helpers support separate byte sizes for:

- activation input,
- weight matrix,
- output activation,
- activation-gradient operands,
- activation-gradient outputs,
- parameter-gradient outputs/accumulators.

For example, an MoE routed up projection can model:

```text
input X: Expert Dispatch DType
weight W: Expert Weight DType
output Y: Activation DType
compute: Expert Compute Precision
```

This is supported for a single sub-op as long as each operand has one dtype.
The current schema does not model different dtype slices inside one logical
matrix unless the workload author splits that work into multiple sub-ops.

## Practical Checks

When changing only Expert Dispatch DType on a Qwen3.5 MoE preset, expect:

- `A_*` object sizes to stay unchanged.
- `x_scatter`, `shared_mlp_up`, and `routed_mlp_up_one_expert` byte counts to
  change.
- `dy_scatter`, `routed_mlp_down_one_expert_dgrad`,
  `routed_mlp_up_one_expert_dgrad`, `shared_mlp_up_dgrad`, `dy_gather`, and
  up-projection wgrad input byte counts to change. Routed down-projection wgrad
  also changes because it reads the dispatched upstream-gradient side.
- `swiglu`, expert down projections, and `x_gather` byte counts to stay
  unchanged.

When changing only Parameter Gradient DType, expect:

- `dW_*` object sizes and wgrad output/accumulator traffic to change.
- `dy_*` object sizes and activation-gradient dgrad traffic to stay tied to
  Activation DType.

When changing only Expert Weight DType, expect:

- expert up/down matmul weight bytes to change,
- mixed layer parameter objects `W_*` to change,
- non-expert attention/LM-head weights to stay unchanged.

When changing only Expert Compute Precision, expect:

- expert matmul math time to change,
- memory bytes and object sizes to stay unchanged.

When changing only Indexer Activation DType on DeepSeek-V3.2, expect:

- Lightning Indexer q/k/w score-kernel traffic to change,
- indexer projection output traffic to change,
- saved indexer activation context inside `A_*` to change when Train Indexer is
  enabled,
- selected score state inside `A_*` to change when Train Indexer is enabled,
- selected indices, normal activations, weights, gradients, and optimizer bytes
  to stay unchanged.

When changing only Indexer Compute Precision on DeepSeek-V3.2, expect:

- Lightning Indexer score math time to change,
- DSA sparse attention core math time to stay tied to attention forward/backward
  efficiencies,
- memory bytes and object sizes to stay unchanged.
