# Workloads

Workloads have two public layers:

1. `DataflowProgram v1`: a hardware-free schema for ordered compute over named
   memory objects.
2. Domain builders: Python helpers that emit `DataflowProgram`, such as
   transformer training builders that feel closer to model/layer code.

The simulator itself still runs `TaskChain`. A program becomes a bare
`TaskChain` only after hardware is selected.

## Quick Start

Run the standalone examples from the repo root:

```bash
python examples/export_custom_dataflow.py --out /tmp/generic_pipeline.dataflow.json
python examples/export_transformer_training_schema.py --out /tmp/heterogeneous_transformer.dataflow.json
```

Both commands write `DataflowProgram v1` JSON that can be imported through the
webapp's **Custom Schema** tab. The first example authors the generic schema
directly. The second uses the training layer: define model/layer specs and
training parameters, then export the schema emitted by the transformer builder.

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
For example, 32 layer-forward tasks can all point to one
`transformer_dense_forward` block. The UI uses blocks to show:

- per-instance runtime and sub-op timing,
- total runtime over all instances,
- total and effective FLOPs,
- read/write byte estimates,
- whether the block is compute- or memory-bound.

`task.label` names a timeline instance, such as `Step 0 Round 0 Layer 3
Forward`. `compute_block_key` names reusable structure.

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

The simulator reports `${primary_unit}/sec`. If the unit is `tokens`, the webapp
also fills the legacy `tokens_per_second` field.

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

Transformer training helpers generate the same generic schema. This is the
built-in training layer today: users specify model dimensions, layer variants,
and training parameters; the builder emits a portable `DataflowProgram`.
Non-transformer training workloads can use the same pattern by writing a small
domain helper that emits `DataflowProgram`.

```python
from dataflow_sim.workloads.common.hardware import HARDWARE_PRESETS
from dataflow_sim.workloads.models.presets import load_model_presets
from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_transformer_training_program,
    build_transformer_training_workload,
)

spec = load_model_presets()["llama3_8B"]
cfg = TrainingConfig(seqlen=4096, num_seqs=4, optimizer="adamw")

program = build_transformer_training_program(spec, cfg)
workload = build_transformer_training_workload(spec, HARDWARE_PRESETS["H100"], cfg)
```

Heterogeneous transformers can be exported as JSON too:

```python
from dataclasses import replace

from dataflow_sim.workloads.training.transformer import (
    TrainingConfig,
    build_heterogeneous_transformer_training_program,
)
from dataflow_sim.workloads.models.transformer import TransformerSpec

dense = TransformerSpec(
    vocab_size=32000, n_layers=1, d_model=512, head_dim=64,
    n_heads=8, n_kv_heads=8, expert_dim=2048,
    num_shared_experts=1, num_routed_experts=0, top_k=0,
)
moe = replace(dense, expert_dim=1536, num_routed_experts=8, top_k=2)

program = build_heterogeneous_transformer_training_program(
    [dense, dense, moe, dense],
    TrainingConfig(seqlen=128, num_seqs=1),
)
json_payload = program.model_dump(mode="json")
```

To export a custom training workload for the webapp:

```python
import json
from pathlib import Path

Path("my_transformer.dataflow.json").write_text(
    json.dumps(program.model_dump(mode="json"), indent=2) + "\n"
)
```

Then open the webapp, choose **Custom Schema**, import the file, choose
hardware, and click **Create Workload**. The preview endpoint returns a
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

For a complete runnable model-definition exporter, see
`examples/export_transformer_training_schema.py`.

## Webapp Flow

1. Choose a preset transformer workload, choose a schema preset, or paste schema
   JSON.
2. Choose hardware and click **Create Workload**. Preview returns the normalized
   schema, bare chain, workload stats, and compute block breakdown.
3. Choose planner settings and click **Run Simulation**.

Custom workloads are browser/session state only. Export the normalized JSON to
keep it outside the app.
