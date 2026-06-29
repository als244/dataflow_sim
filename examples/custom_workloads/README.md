# Custom Workload Authoring

This directory is a runnable miniature of the modular workload stack. It shows
how to add a new op, compose ops into modules, assemble modules into a model,
export a training `DataflowProgram`, and run the simulator to get an annotated
plan plus summary statistics.

Run commands from the repo root with `PYTHONPATH=src`.

## Files

- `custom_ops.py`: pure op helpers. Each helper returns `DataflowCost` specs.
  In the main package, helpers like these usually live under
  `dataflow_sim.workloads.ops.forward`, `backward`, or `optimizer`.
- `custom_modules.py`: module-level composition. A `MixerBlock` and
  `ClassifierHead` combine ops into forward, backward, recompute, and optimizer
  phases.
- `custom_model.py`: model-level composition. `TinyMixerForTraining` owns the
  ordered module list and passes it to `TrainingBuilder`.
- `export_custom_training_schema.py`: exports the custom model to
  `DataflowProgram v1` JSON.
- `run_custom_training_simulation.py`: builds the custom model, optionally
  plans recompute, applies a memory policy, runs the simulator, and emits the
  public summary metrics.

## Export A Program

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

## Run End-To-End Simulation

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

Outputs:

- `webapp_upload.dataflow.json`: hardware-free `DataflowProgram v1`; upload
  this in the webapp's **Custom Schema** tab.
- `program.dataflow.json`: same `DataflowProgram v1`, kept under a generic
  filename for scripts.
- `unannotated_plan.json`: realized bare `TaskChain` before memory-policy
  annotations. This is the hardware-specific unannotated plan the webapp
  creates when you click **Create Workload**.
- `annotated_plan.json`: planner-annotated `TaskChain` after policy/recompute.
- `summary.json`: simulator API summary, including makespan, token/sec,
  effective/hardware TFLOP/s, peak memory, and transfer utilization.

## Authoring Pattern

1. Add op helpers as pure functions returning `DataflowCost`.
2. Build modules that compose those helpers into phase op lists.
3. Build model files that decide the ordered module list.
4. Pass `TrainingLayerSpec` and `TrainingHeadSpec` objects to
   `TrainingBuilder`.
5. Realize, plan, simulate, and call `compute_workload_summary(workload, log)`.
