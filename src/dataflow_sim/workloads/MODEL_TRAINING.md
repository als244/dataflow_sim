# Model Training Workloads

This document describes the modular path for authoring model-training
workloads. The goal is to let users define models in layers that feel natural:
ops, modules, model families, then a generic lowering step into
`DataflowProgram v1`.

Model authoring should use the stack described here.

## End-To-End Flow

```text
ops -> modules -> model family -> training_builder -> DataflowProgram
    -> realize on hardware -> apply memory policy -> simulate -> summarize
```

The first half is hardware-free authoring. The second half is the same runtime
path used by generic dataflow programs.

1. Ops return pure `DataflowCost` sub-op specs. They do not allocate tensors,
   create tasks, choose hardware, or call planners.
2. Modules compose ops into reusable forward, backward, recompute, and
   optimizer phase factories.
3. Model-family files choose architecture: config presets, dimension overrides,
   module order, number of layers, head/loss, and any heterogeneous layer mix.
4. `training_builder` schedules the model-authored module list into ordered
   training tasks and emits `DataflowProgram v1`.
5. `realize_dataflow_program(program, hardware)` resolves sub-op costs into a
   bare `TaskChain`.
6. A memory policy annotates the bare chain with release/offload/prefetch
   decisions.
7. The simulator runs the annotated chain and `compute_workload_summary(...)`
   reports makespan, token throughput, TFLOP/s, peak fast memory, recompute
   percentage, and stream utilization.

## Authoring Layers

### Ops

Ops live under phase-specific packages:

- `dataflow_sim.workloads.ops.forward`
- `dataflow_sim.workloads.ops.backward`
- `dataflow_sim.workloads.ops.optimizer`

Each op type gets its own file for readability, for example `matmul.py`,
`attention.py`, `norm.py`, `activation.py`, `movement.py`, and `loss.py`.

Ops should return lists of `DataflowCost`. Prefer keeping them pure:

```python
from dataflow_sim.workloads.ops.forward.matmul import matmul

subops = [
    matmul("qkv_proj", tokens, d_model, (n_heads + 2 * n_kv_heads) * head_dim),
]
```

### Modules

Modules live in `dataflow_sim.workloads.modules`, one file per module. Current
examples include dense attention, MLP, MoE, stacked blocks, head/loss modules,
optimizer steps, and recompute helpers.

A module is where phase composition belongs:

- `forward_ops(...)`: useful forward compute and memory terms.
- `backward_ops(...)`: explicit backward terms.
- `recompute_ops(...)`: the work needed to recreate saved backward context.
- `optimizer_ops(...)`: optimizer update terms when the module owns parameters.

There is no automatic autodiff. Backward and recompute variants are explicit
because they often differ from forward composition.

### Model Families

Model files live under `dataflow_sim.workloads.models`. The real-family
examples are:

- `llama3.py`
- `qwen3.py`
- `qwen3_moe.py`
- `olmoe.py`
- `qwen3_hybrid_dense.py`
- `qwen3_hybrid_moe.py`
- `deepseek_v3.py`
- `deepseek_v3_2.py`
- `glm5.py`
- `glm5_2.py`
- `gpt_oss.py`
- `nemotron_h.py`

Model families own architecture. They choose how many modules exist, which
modules appear, what order they run in, and how preset dimensions map into
module dimensions.

Configs use preset-plus-overrides:

```python
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining

cfg = Llama3Config.preset("8B", n_layers=80, d_model=8192)
model = Llama3ForTraining(cfg)
```

The canonical activation shape is `[tokens, dim]`, where
`tokens = seqlen * num_seqs`. Attention, routing, KV heads, and top-k are
module metadata rather than extra batch dimensions on the main activation.

## Training Builder

`dataflow_sim.workloads.training_builder.TrainingBuilder` is generic. It is not
tied to any built-in model family. It takes a model-authored ordered list of
`TrainingLayerSpec` objects plus one `TrainingHeadSpec`.

Each layer spec provides:

- input/output dimensions for canonical `[tokens, dim]` activations,
- parameter count for `W_i`,
- saved activation width for `A_<step>_<round>_<layer>`,
- forward, backward, recompute, and optimizer op factories,
- compute-block keys and names.

The head spec provides input dimension, parameter count, separate forward and
backward op factories, and its own compute-block key/name. The generic builder
lowers these into distinct `head_fwd_*` and `head_bwd_*` tasks.

`TrainingConfig` controls loop shape:

```python
from dataflow_sim.workloads.dataflow_builder import TrainingConfig

training = TrainingConfig(
    seqlen=4096,
    num_seqs=4,
    grad_accum_rounds=1,
    num_steps=1,
    optimizer="adamw",
)
```

`DTypePolicy` controls byte sizes. The default is bf16 for parameters,
activations, expert dispatch traffic, parameter gradients, and optimizer state:

```python
from dataflow_sim.workloads.dataflow_builder import DTypePolicy

dtype_policy = DTypePolicy(
    param="bf16",
    activation="bf16",
    expert_dispatch="bf16",
    gradient="bf16",
    optimizer_state="bf16",
    expert_param="bf16",
    compute="bf16",
    expert_compute="bf16",
)
```

For the exact webapp/API semantics of each datatype and compute-precision
field, including MoE expert dispatch and expert weight overrides, see
[docs/datatypes.md](../../../docs/datatypes.md).

The builder can emit a hardware-free program:

```python
program = model.build_training_program(
    training,
    input_shape=(training.tokens, cfg.d_model),
    dtype_policy=dtype_policy,
)
```

Or it can realize directly on hardware:

```python
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS

workload = model.build_training_workload(
    training,
    HARDWARE_PRESETS["H100"],
    input_shape=(training.tokens, cfg.d_model),
)
```

## Lowering To DataflowProgram

`DataflowProgram v1` is the portable workload boundary. It contains:

- initial objects for inputs, parameters, and optimizer state,
- ordered task instances for layer forward, head/loss forward, head backward,
  optional recompute, layer backward, and optimizer steps,
- reusable `compute_blocks`,
- metrics such as total token count,
- optional final placement constraints.

Task ids are deterministic:

- `f_<step>_<round>_<layer>`: layer forward.
- `head_fwd_<step>_<round>`: head/loss forward.
- `head_bwd_<step>_<round>`: head backward.
- `r_<step>_<round>_<layer>`: recompute task, present only when that saved
  activation is recomputed.
- `b_<step>_<round>_<layer>`: layer backward.
- `step_<step>_<layer>`: optimizer update.

The forward pass follows the model-authored module order. The backward pass is
explicit and runs that order in reverse.

## Compute Blocks

A compute block is a reusable runtime/lowering grouping, not a model-definition
object. Model files choose a base key such as `layer_block`; the generic
builder appends the phase:

- `layer_block.forward`
- `layer_block.backward`
- `layer_block.recompute`
- `model_head.training`
- `optimizer_step.adamw`

`TraceContext.emit_task(...)` registers a block the first time a task references
its `compute_block_key`; later tasks can reuse that same block. This is what
lets many layer instances share one block summary while still having unique
task ids and labels.

Users can control grouping without changing planner code:

- use the same `block_key` for repeated identical module phases,
- use different `block_key` values for heterogeneous layers or alternate
  implementations,
- keep model architecture in model files and grouping choices in layer/head
  specs.

## Recompute

Recompute is expressed as workload variants. For every saved backward-context
object, the workload publishes a `RecomputeRewrite` in
`Workload.metadata["recompute_rewrites"]`.

Today the levels are binary:

- level `0`: forward saves the activation object and no recompute task is
  emitted,
- level `1`: forward does not save it and `r_<step>_<round>_<layer>` recreates
  it before backward.

The decision is per saved activation instance because memory pressure and
transfer blame are instance-specific. The available tradeoff is still
compute-block based: each rewrite records the forward compute block that saves
the object and the recompute compute block that recreates it.

The planner flow is:

1. Build a base workload with all levels at `0`.
2. Read the rewrite table.
3. Build candidate variants by passing `recompute={object_id: level}` back into
   the model builder.
4. Apply the selected memory policy to each candidate chain.
5. Simulate candidates and accept recompute conversions only when makespan
   improves.

The generic selector is
`dataflow_sim.planning.recompute.plan_with_recompute(...)`. It does not know
anything model-family specific; it only sees a variant builder, rewrite table,
and policy function.

## Hardware, Policies, And Simulation

Training programs use the same runtime path as generic dataflow programs:

```python
from dataflow_sim.engine.simulator import run
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy
from dataflow_sim.workloads.summary import compute_workload_summary

chain = apply_pressurefit_policy(
    workload.chain,
    fast_memory_capacity=40 * 1024**3,
)
log = run(chain, snapshots=False)
summary = compute_workload_summary(workload, log)
```

Important separation points:

- `DataflowProgram` is hardware-free and uploadable to the webapp.
- Realization applies `HardwareSpec` and resolves roofline costs.
- A memory policy annotates `TaskChain`; it does not mutate model definitions.
- The simulator executes the annotated chain.
- Summary metrics are produced by the simulator API helper, not recomputed by
  the UI.

## User Options

Users can customize:

- model family and preset scale,
- dimension overrides such as layers, width, heads, expert width, routed
  experts, and top-k,
- loop shape through `TrainingConfig`,
- dtype policy for params, activations, parameter gradients, and optimizer state,
- optimizer mode,
- final placement of updated model state,
- module composition and per-phase op specs,
- compute-block grouping keys,
- memory policy and fast-memory capacity,
- recompute enablement and, for experiments, explicit recompute levels.

## Examples

Export a built-in model-family training program:

```bash
python examples/model_training/builtin_arch/export_training_program.py \
  --model qwen3_moe \
  --scale 30B-3B \
  --n-layers 8 \
  --seqlen 1024 \
  --optimizer adamw \
  --out /tmp/qwen3_moe_training.dataflow.json
```

Build a custom op/module/model stack and run the simulator:

```bash
python examples/model_training/custom_arch/run_training_simulation.py \
  --n-layers 8 \
  --d-model 512 \
  --hidden-dim 2048 \
  --classes 1024 \
  --seqlen 1024 \
  --optimizer adamw \
  --fast-memory-gb 0.05 \
  --recompute \
  --out-dir /tmp/tiny_mixer_run
```

The custom example writes:

- `webapp_upload.dataflow.json`: upload this through the webapp's Custom
  Dataflow Program tab,
- `unannotated_plan.json`: realized bare `TaskChain`,
- `annotated_plan.json`: policy/recompute-annotated plan,
- `summary.json`: simulator KPI payload.
