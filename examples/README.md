# Workload Export Examples

These examples show the two supported authoring paths.

## Generic Dataflow

Use this when the workload is not necessarily training or even ML. Define
objects, compute blocks, ordered tasks, and optional metrics directly:

```bash
python examples/export_custom_dataflow.py --out /tmp/generic_pipeline.dataflow.json
```

The output is a `DataflowProgram v1` JSON file that can be imported in the
webapp's **Custom Schema** tab.

## Transformer Training

Use this when the workload is transformer training. Define model/layer specs and
training hyperparameters, then let the training helper emit the generic schema:

```bash
python examples/export_transformer_training_schema.py --out /tmp/heterogeneous_transformer.dataflow.json
```

The script defines a tiny heterogeneous model with dense and MoE layers. It
exports the same `DataflowProgram v1` format as the generic example, so the
webapp and simulator do not need a transformer-specific import path.

The built-in high-level training helper is transformer-specific today. For
another training domain, use the generic example as the target contract and
write a small Python builder that emits `DataflowProgram`.

## Import Into The Webapp

1. Start the backend and frontend from the repo README.
2. Open the webapp.
3. Choose **Custom Schema** in the Workload panel.
4. Import one of the generated `.dataflow.json` files.
5. Choose hardware, click **Create Workload**, then run a planner.
