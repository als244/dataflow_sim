# Workloads

Workloads have two public layers plus the modular training authoring stack:

1. `DataflowProgram v1`: a hardware-free schema for ordered compute over named
   memory objects.
2. Modular workload builders: phase-specific ops, reusable modules, model
   families, `training_builder` scheduling, and `dataflow_builder`
   tracing/lowering.

The simulator itself still runs `TaskChain`. A program becomes a bare
`TaskChain` only after hardware is selected.

For the end-to-end model-training path, including ops -> modules -> models,
hardware realization, memory policies, and recompute, see
[MODEL_TRAINING.md](MODEL_TRAINING.md).

## Quick Start

Run the standalone examples from the repo root:

```bash
python examples/generic_dataflow/export_program.py --out /tmp/generic_pipeline.dataflow.json
python examples/model_training/builtin_arch/export_training_program.py --out /tmp/llama3_training.dataflow.json
```

Both commands write `DataflowProgram v1` JSON that can be imported through the
webapp's **Custom Dataflow Program** tab. The first example authors the generic
program directly. The second uses the modular model-training path: choose a
model family, override scale dimensions if needed, set training parameters,
then export the program emitted through `training_builder`.

## Modular Builder Stack

The modular path is dependency-free:

- `workloads.ops.forward`, `workloads.ops.backward`, and
  `workloads.ops.optimizer` contain one-file-per-op-type helpers that return
  `DataflowCost` sub-op specs.
- `workloads.modules` composes ops into reusable modules such as attention,
  SwiGLU MLP, MoE, stacked blocks, head/loss, optimizer, and recompute phases.
- `workloads.models.llama3`, `workloads.models.qwen3`,
  `workloads.models.qwen3_moe`, `workloads.models.olmoe`,
  `workloads.models.qwen3_hybrid_dense`, `workloads.models.qwen3_hybrid_moe`,
  `workloads.models.deepseek_v3`, `workloads.models.deepseek_v3_2`,
  `workloads.models.glm5`, `workloads.models.glm5_2`,
  `workloads.models.gpt_oss`, and `workloads.models.nemotron_h` define real
  model workloads with
  preset-plus-overrides config APIs. Each model file
  constructs its ordered module list explicitly.
- `workloads.training_builder` schedules a model-authored module list into
  forward, head/loss, recompute, backward, and optimizer tasks. It does not
  decide how many blocks exist or which module types appear.
- `workloads.dataflow_builder` tracks symbolic tensors, dtypes, module phases,
  and lowering to compute blocks and tasks.

Examples:

```python
from dataflow_sim.workloads.dataflow_builder import TrainingConfig
from dataflow_sim.workloads.models.llama3 import Llama3Config, Llama3ForTraining

cfg = Llama3Config.preset("8B", n_layers=80, d_model=8192)
model = Llama3ForTraining(cfg)
program = model.build_training_program(
    TrainingConfig(seqlen=4096, num_seqs=4, optimizer="adamw")
)
```

### `training_builder` Inputs

`workloads.training_builder.TrainingBuilder` is not tied to any built-in model
family. It schedules a model-authored ordered list of `TrainingLayerSpec`
objects plus one `TrainingHeadSpec`.

Each layer spec provides:

- `input_dim` and `output_dim` for canonical `[tokens, dim]` activations.
- `param_count` for the layer parameter object `W_i`.
- `saved_activation_width` for the saved backward-context object `A_*`.
- phase op factories for forward, backward, and recompute.
- an optimizer op factory for layer-local optimizer tasks.
- compute-block keys/names, so heterogeneous module lists can group distinct
  layers separately.

`build_training_program(...)` then takes:

- `TrainingConfig`: `seqlen`, `num_seqs`, grad accumulation, steps, optimizer.
- optional `input_shape`, which must equal `[seqlen * num_seqs, input_dim]`.
- optional `DTypePolicy`, defaulting params/activations/parameter gradients/optimizer
  state to bf16.
- optional recompute levels keyed by saved activation ids such as `A_0_0_3`.

## Schema Contract

`DataflowProgram` fields:

- `schema_version`: currently `"dataflow/v1"`.
- `name`, `description`, `metadata`: display and caller-owned metadata.
- `objects[]`: initial memory objects with `id`, `size_bytes`,
  `initial_location`, and free-form `role`.
- `compute_blocks[]`: reusable structural blocks with `key`, `name`,
  `category`, `subops`, and `metadata`.
- `tasks[]`: ordered compute instances with unique `id`, unique `label`,
  `group`, `compute_block_key`, `inputs`, `outputs`, `mutates`, and optional
  inline `cost`.
- `metrics`: optional primary throughput contract, for example
  `{"primary_unit": "tokens", "primary_count": 16384}`.
- `final_locations`: optional terminal placement constraints.

`role` is intentionally generic. Training helpers use roles such as
`parameter`, `activation`, `gradient`, and `optimizer_state`; other domains can
use their own display categories. Common roles are mapped onto the simulator's
coarse object types, and unknown roles become `other`.

Cost models:

- `{"kind": "fixed", "runtime_us": 240}` for measured runtime.
- `{"kind": "roofline", "flops": ..., "memory_bytes": ..., "efficiency": ...}`
  for hardware-resolved cost.
- `{"kind": "sum", "terms": [...]}` to expose sub-ops while compiling to one
  simulator task.

Tasks should normally reference `compute_block_key`. Inline `task.cost` remains
as a convenience fallback; the compiler normalizes it into a one-off block named
`inline:<task_id>`.

## Compute Blocks

A compute block is the reusable structure behind one or more task instances.
For example, 32 layer-forward tasks can all point to one `layer.forward` block.
The UI uses blocks to show:

- per-instance runtime and sub-op timing,
- total runtime over all instances,
- total and effective FLOPs,
- read/write byte estimates,
- whether the block is compute- or memory-bound.

`task.label` names a timeline instance, such as `Step 0 Round 0 Layer 3
Forward`. `compute_block_key` names reusable structure.

In the modular training stack, compute blocks are created when
`TraceContext.emit_task(...)` registers each task's `compute_block_key`.
Model files choose the reusable block identity through `TrainingLayerSpec`
and `TrainingHeadSpec` fields such as `block_key="layer_block"`; the
generic `training_builder` appends phase suffixes like `.forward`,
`.backward`, `.recompute`, and `.training`.

Recompute choices are declared in
`Workload.metadata["recompute_rewrites"]`. Each rewrite is keyed by the saved
activation object (`A_<step>_<round>_<layer>`) and includes the forward and
recompute compute-block keys that define the tradeoff. The planner ranks
activation instances because memory pressure is instance-specific, but the
available recompute variants come from compute blocks, not from hidden module
state.

## Metrics

Generic workloads may omit `metrics`. Metric-enabled workloads get an extra
throughput summary:

```json
{
  "metrics": {
    "primary_unit": "tokens",
    "primary_count": 16384,
    "metadata": {"seqlen": 4096, "num_seqs": 4}
  }
}
```

After a workload is realized and simulated, use
`dataflow_sim.workloads.compute_workload_summary(workload, log)` to compute
the public summary payload: makespan, primary throughput, token/sec,
effective/hardware TFLOP/s, peak fast memory, recompute percentage, and memory
stream utilization. The web API uses the same helper for its `summary` field.

## JSON Example

```json
{
  "schema_version": "dataflow/v1",
  "name": "two-task-dataflow",
  "description": "Small generic workload",
  "objects": [
    {"id": "x", "size_bytes": 16777216, "initial_location": "fast", "role": "input"},
    {"id": "w", "size_bytes": 67108864, "initial_location": "backing", "role": "parameter"}
  ],
  "compute_blocks": [
    {
      "key": "projection",
      "name": "Projection",
      "category": "encoder",
      "subops": [
        {
          "kind": "roofline",
          "name": "matmul",
          "flops": 8000000000,
          "memory_bytes": 120000000,
          "efficiency": "matmul"
        }
      ]
    }
  ],
  "tasks": [
    {
      "id": "block_0",
      "label": "Block 0",
      "group": "encoder",
      "compute_block_key": "projection",
      "inputs": ["x", "w"],
      "outputs": [{"id": "h0", "size_bytes": 16777216, "role": "activation"}]
    }
  ],
  "final_locations": {}
}
```

## Python Authoring

Write helpers the way you might write model code: common ops, layer function,
then a model composition.

```python
from dataflow_sim.workloads.dataflow import (
    ComputeBlock,
    DataflowCost,
    DataflowObject,
    DataflowOutput,
    DataflowProgram,
    DataflowTask,
)


def matmul(name: str, flops: int, bytes_: int) -> DataflowCost:
    return DataflowCost(
        kind="roofline",
        name=name,
        flops=flops,
        memory_bytes=bytes_,
        efficiency="matmul",
    )


projection = ComputeBlock(
    key="projection",
    name="Projection",
    category="encoder",
    subops=[matmul("matmul", 8_000_000_000, 120_000_000)],
)


def layer(i: int, inp: str, width_bytes: int) -> DataflowTask:
    return DataflowTask(
        id=f"layer_{i}",
        label=f"Layer {i}",
        group="encoder",
        compute_block_key="projection",
        inputs=[inp, f"W_{i}"],
        outputs=[DataflowOutput(id=f"h_{i}", size_bytes=width_bytes, role="activation")],
    )


program = DataflowProgram(
    name="heterogeneous-encoder",
    objects=[
        DataflowObject(id="x", size_bytes=16_777_216, initial_location="fast", role="input"),
        DataflowObject(id="W_0", size_bytes=67_108_864, initial_location="backing", role="parameter"),
        DataflowObject(id="W_1", size_bytes=134_217_728, initial_location="backing", role="parameter"),
    ],
    compute_blocks=[projection],
    tasks=[
        layer(0, "x", 16_777_216),
        layer(1, "h_0", 33_554_432),
    ],
)
```

## Training Helpers

The built-in model-family files generate the same generic schema. Users
specify model dimensions and training parameters; the model file builds the
ordered module list, and `training_builder` emits a portable
`DataflowProgram`.

```python
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.dataflow_builder import TrainingConfig
from dataflow_sim.workloads.models.qwen3_moe import (
    Qwen3MoEConfig,
    Qwen3MoEForTraining,
)

cfg = Qwen3MoEConfig.preset("30B-3B", n_layers=8, d_model=2048)
training = TrainingConfig(seqlen=4096, num_seqs=4, optimizer="adamw")
model = Qwen3MoEForTraining(cfg)

program = model.build_training_program(
    training,
    input_shape=(training.tokens, cfg.d_model),
)
workload = model.build_training_workload(
    training,
    HARDWARE_PRESETS["H100"],
    input_shape=(training.tokens, cfg.d_model),
)
```

Custom architectures follow the same pattern by constructing their own
`TrainingLayerSpec` list and `TrainingHeadSpec`. See
`examples/model_training/custom_arch/` for a complete
op/module/model/simulation stack.

To export a custom training workload for the webapp:

```python
import json
from pathlib import Path

Path("my_model.dataflow.json").write_text(
    json.dumps(program.model_dump(mode="json"), indent=2) + "\n"
)
```

Then open the webapp, choose **Custom Dataflow Program**, import the file,
choose hardware, and click **Create Workload**. The preview endpoint returns a
normalized schema plus the bare task chain. The plan panels can export
`TaskChain` JSON for users who want to inspect or save the realized plan.

Custom training builders should follow this checklist:

- emit `DataflowProgram`, not `TaskChain`, when the workload should be portable;
- keep object ids stable and unique across parameters, activations, gradients,
  optimizer state, and inputs;
- put persistent model state in `initial_location="backing"` unless it truly
  starts in fast memory;
- use `compute_blocks` for repeated layer structure and `task.label` for unique
  timeline instances;
- attach `metrics={"primary_unit": "tokens", "primary_count": ...}` when token
  throughput should appear in summaries;
- use `final_locations` only for bytes that must end in a particular tier.

For complete runnable model-training examples, see
`examples/model_training/`.

## Webapp Flow

1. Choose a built-in model workload or paste/import a custom
   `DataflowProgram` JSON.
2. Choose hardware and click **Create Workload**. Preview returns the normalized
   schema, bare chain, workload stats, and compute block breakdown.
3. Choose planner settings and click **Run Simulation**.

Custom workloads are browser/session state only. Export the normalized JSON to
keep it outside the app.
