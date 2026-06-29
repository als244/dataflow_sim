# Custom Architecture Training

This directory is a runnable template for adding a new trainable architecture.
It is intentionally separate from the built-in Llama/Qwen/OLMoE examples so
you can see every layer of the API contract without reading model-specific
production code.

## Authoring Flow

```text
custom ops -> custom modules -> custom model -> TrainingBuilder
    -> DataflowProgram v1 -> hardware realization -> memory policy
    -> simulator -> summary metrics
```

The first four steps are hardware-free model authoring. The last four steps are
the same runtime path used by any uploaded dataflow program.

## Files And Contracts

### `custom_ops.py`

Op helpers are pure functions returning `DataflowCost`.

Required contract:

- return `DataflowCost` or `list[DataflowCost]`;
- include FLOPs, memory bytes, efficiency class, and optional effective FLOPs;
- do not allocate tensors;
- do not create `DataflowTask`s;
- do not inspect hardware or planner state.

This file shows local example ops. To promote one into the library, move it to
`src/dataflow_sim/workloads/ops/<phase>/<op_type>.py` and re-export it from the
phase package.

### `custom_modules.py`

Modules compose op helpers into phase-level op lists.

Required contract for layer-like modules:

- `forward_ops(tokens, seqlen, bytes_per_element) -> list[DataflowCost]`;
- `backward_ops(tokens, seqlen, bytes_per_element) -> list[DataflowCost]`;
- `recompute_ops(tokens, seqlen, bytes_per_element) -> list[DataflowCost]`;
- optional `optimizer_ops(optimizer, bytes_per_element) -> list[DataflowCost]`;
- expose enough metadata for the model file to fill `TrainingLayerSpec`:
  parameter count, input/output dimensions, and saved activation width.

There is no automatic autodiff. Backward and recompute phases are explicit
because their sub-op composition can differ from forward.

### `custom_model.py`

The model file owns architecture.

Required contract:

- define a config object for model dimensions;
- construct the ordered module list;
- decide whether repeated or heterogeneous layers share compute-block keys;
- create one `TrainingLayerSpec` per trainable layer-like module;
- create one `TrainingHeadSpec` for loss/head work;
- subclass or wrap `TrainingBuilder`.

The canonical activation shape is `[tokens, dim]`, where
`tokens = seqlen * num_seqs`.

### `export_training_program.py`

This script is the hardware-free export path.

Required contract:

- instantiate model config;
- instantiate `TrainingConfig`;
- optionally instantiate `DTypePolicy`;
- call `model.build_training_program(...)`;
- write `program.model_dump(mode="json")`.

The result can be uploaded through the webapp's **Custom Dataflow Program**
tab.

### `run_training_simulation.py`

This script is the end-to-end runtime path.

Required contract:

- build the same model/training inputs;
- call `model.build_training_workload(...)` with a `HardwareSpec`;
- optionally call `plan_with_recompute(...)`;
- apply a memory policy such as `apply_pressurefit_policy(...)`;
- run `dataflow_sim.engine.simulator.run(...)`;
- call `compute_workload_summary(workload, log)`.

## Export A Program

```bash
python examples/model_training/custom_arch/export_training_program.py \
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

Outputs:

- `webapp_upload.dataflow.json`: hardware-free `DataflowProgram v1`; upload
  this in the webapp's **Custom Dataflow Program** tab.
- `program.dataflow.json`: same `DataflowProgram v1`, kept under a generic
  filename for scripts.
- `unannotated_plan.json`: realized bare `TaskChain` before memory-policy
  annotations. This is the hardware-specific unannotated plan the webapp
  creates when you click **Create Workload**.
- `annotated_plan.json`: planner-annotated `TaskChain` after policy/recompute.
- `summary.json`: simulator API summary, including makespan, token/sec,
  effective/hardware TFLOP/s, peak memory, and transfer utilization.

## Extending This Example

To add a new custom architecture, copy the pattern but keep these boundaries:

- put op math and byte formulas in op helpers;
- put phase composition in modules;
- put model order and dimensions in the model file;
- keep hardware, memory policy, and simulator choices in scripts or server
  code, outside model definitions.
