# Workload Export Examples

These examples show the supported authoring paths.

Run commands from the repo root. If the package is not installed in your active
environment, prefix commands with `PYTHONPATH=src`.

## Generic Dataflow

Use this when the workload is not necessarily training or even ML. Define
objects, compute blocks, ordered tasks, and optional metrics directly:

```bash
PYTHONPATH=src python examples/export_custom_dataflow.py \
  --out /tmp/generic_pipeline.dataflow.json
```

The output is a `DataflowProgram v1` JSON file that can be imported in the
webapp's **Custom Schema** tab.

## Transformer Training

Use this when the workload is transformer training. Choose a model family,
start from a scale preset, optionally override dimensions, then trace the
symbolic model into the generic schema:

```bash
PYTHONPATH=src python examples/export_transformer_training_schema.py \
  --model qwen3_moe \
  --scale 30B-3B \
  --n-layers 8 \
  --seqlen 1024 \
  --optimizer adamw \
  --out /tmp/qwen3_moe_training.dataflow.json
```

The script uses the modular workload stack:

- `dataflow_sim.workloads.ops.forward/backward/optimizer`: pure sub-op helpers.
- `dataflow_sim.workloads.modules`: reusable model modules with forward,
  backward, and recompute phases.
- `dataflow_sim.workloads.models`: real model-family definitions such as
  Llama 3, Qwen3, Qwen3 MoE, and OLMoE. These files choose the ordered module
  list for each architecture.
- `dataflow_sim.workloads.training_builder`: schedules the model-authored
  module list into training tasks, gradients, recompute slots, and optimizer
  steps.
- `dataflow_sim.workloads.dataflow_builder`: symbolic tensor/dtype tracing and
  lowering to `DataflowProgram v1`.

## Custom Training Stack

Use this path when you want to author new ops, modules, and model families
without touching the built-in Llama/Qwen examples:

```bash
PYTHONPATH=src python examples/custom_workloads/export_custom_training_schema.py \
  --n-layers 4 \
  --d-model 512 \
  --hidden-dim 2048 \
  --classes 32000 \
  --seqlen 256 \
  --optimizer adamw \
  --out /tmp/tiny_mixer.dataflow.json
```

The custom example is split across files to mirror the new workload structure:

- `custom_workloads/custom_ops.py`: add op helpers returning `DataflowCost`.
- `custom_workloads/custom_modules.py`: compose ops into reusable modules.
- `custom_workloads/custom_model.py`: define a model that owns the ordered
  module list.
- `custom_workloads/run_custom_training_simulation.py`: build the model, call
  planner/simulator APIs, and write the webapp-uploadable schema, unannotated
  plan, annotated plan, and summary stats.

End-to-end simulation:

```bash
PYTHONPATH=src python examples/custom_workloads/run_custom_training_simulation.py \
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

See `examples/custom_workloads/README.md` for the step-by-step authoring guide.

## Import Into The Webapp

1. Start the backend and frontend from the repo README.
2. Open the webapp.
3. Choose **Custom Schema** in the Workload panel.
4. Import one of the generated `.dataflow.json` files.
5. Choose hardware, click **Create Workload**, then run a planner.
