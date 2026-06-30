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
| Activation DType | `DTypePolicy.activation` | Main activation tensors and activation memory traffic: `input_*`, `y_*`, `A_*`, forward outputs, saved backward context, norm/activation memory terms, attention activation traffic, dense MLP activation traffic, and MoE non-dispatch activation traffic. |
| Expert Dispatch DType | `DTypePolicy.expert_dispatch` | MoE dispatch representation for forward expert routing, specifically `x_scatter` output traffic and the activation input read side of `shared_mlp_up` and `routed_mlp_up_one_expert`. The corresponding up-projection wgrad input read also uses this dtype because it reads the dispatched expert input. |
| Gradient DType | `DTypePolicy.gradient` | Gradient tensors and gradient memory traffic: `dy_*`, `dW_*`, backward gradient reads/writes, wgrad outputs, and wgrad accumulation traffic. |
| Optimizer DType | `DTypePolicy.optimizer_state` | Optimizer-state objects such as `O_*` and optimizer-state traffic inside optimizer steps. |
| Compute Precision | `DTypePolicy.compute` | Default matmul math precision for non-expert matmuls. This selects the matching hardware peak TFLOP/s and matmul efficiency field. |
| Expert Weight DType | `DTypePolicy.expert_param` | MoE expert weight bytes for both shared and routed expert matrices: `shared_mlp_up/down` and `routed_mlp_up/down`. This overrides Weight DType only for matrices marked as expert weights. |
| Expert Compute Precision | `DTypePolicy.expert_compute` | Matmul math precision for MoE expert matmuls, including shared and routed expert up/down projections. |

## Object Sizing

The model-training builder emits these major objects:

| Object Pattern | Role | Size Source |
| --- | --- | --- |
| `input_<step>_<round>` | activation | `tokens * input_dim * Activation DType` |
| `y_<step>_<round>_<layer>` | activation | `tokens * layer_output_dim * Activation DType` |
| `A_<step>_<round>_<layer>` | activation | `tokens * saved_activation_width * Activation DType` |
| `dy_*` | gradient | shape elements times Gradient DType |
| `dW_*` | gradient | parameter elements times Gradient DType |
| `W_*` | parameter | Weight DType for non-expert matrices, Expert Weight DType for expert matrices when matrix metadata is available |
| `W_head` | parameter | Weight DType |
| `O_*`, `O_head` | optimizer state | Optimizer DType |

Important: `A_*` uses **Activation DType**, not Expert Dispatch DType. `A_*`
represents saved backward context, including outputs and intermediate tensors.
It does not currently model a separate quantized expert-dispatch buffer as a
persistent saved object.

## MoE Expert Dispatch

Expert Dispatch DType is intentionally narrow. It models the dtype used for the
dispatched activation stream entering expert up projections.

It currently affects:

- `x_scatter`: reads the main activation in Activation DType and writes routed
  expert inputs in Expert Dispatch DType.
- `shared_mlp_up`: reads its activation input in Expert Dispatch DType, reads
  weights in Expert Weight DType, and writes outputs in Activation DType.
- `routed_mlp_up_one_expert`: reads dispatched routed inputs in Expert Dispatch
  DType, reads weights in Expert Weight DType, and writes outputs in Activation
  DType.
- `shared_mlp_up_wgrad` and `routed_mlp_up_one_expert_wgrad`: read the
  up-projection input side in Expert Dispatch DType, while gradient operands and
  gradient outputs use Gradient DType.

It does not affect:

- `A_*` saved activation objects,
- `y_*` main activation objects,
- SwiGLU memory terms,
- expert down-projection activation operands,
- `x_gather`,
- backward dgrad movement, which uses Gradient DType,
- optimizer state.

## Expert Weights

Expert Weight DType applies to MoE expert matrices, both shared and routed. For
example, in Qwen3.5 MoE and DeepSeek/Kimi MoE blocks, these matrices are expert
weights:

- `shared_mlp_up`
- `shared_mlp_down`
- `routed_mlp_up`
- `routed_mlp_down`

If a preset has `num_shared_experts = 0`, changing Expert Weight DType will
only visibly affect routed expert matrices because no shared expert matrices
exist. Current Qwen3-MoE and OLMoE presets have no shared experts; Qwen3.5 MoE,
DeepSeek, and Kimi presets do have shared experts.

Expert Weight DType affects:

- `W_*` mixed parameter object size for layers whose model file exposes matrix
  metadata,
- expert matmul weight-read bytes,
- optimizer-step weight-read/write bytes for expert matrices.

It does not change gradient object dtype or optimizer-state dtype. Those remain
controlled by Gradient DType and Optimizer DType.

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

Compute precision does not change object sizes or memory bytes. Dtype fields do
not change matmul peak TFLOP/s. Use both fields when modeling low-precision
training where storage format and math format differ.

Attention kernels currently use attention-specific efficiency fields rather
than the matmul precision selectors.

## Mixed-Precision Matrix Bytes

The matmul op helpers support separate byte sizes for:

- activation input,
- weight matrix,
- output activation,
- gradient operands,
- gradient outputs/accumulators.

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
- `swiglu`, expert down projections, and `x_gather` byte counts to stay
  unchanged.

When changing only Expert Weight DType, expect:

- expert up/down matmul weight bytes to change,
- mixed layer parameter objects `W_*` to change,
- non-expert attention/LM-head weights to stay unchanged.

When changing only Expert Compute Precision, expect:

- expert matmul math time to change,
- memory bytes and object sizes to stay unchanged.
