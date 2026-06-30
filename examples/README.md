# Examples

Install the package once from the repo root before running examples:

```bash
pip install -e .
```

After that, examples can be run with plain `python` commands.

## Example Map

| Path | Use when | Output |
| --- | --- | --- |
| `generic_dataflow/` | You already know the ordered compute and memory objects. | Uploadable `DataflowProgram v1` JSON. |
| `model_training/builtin_arch/` | You want a built-in Llama/Qwen/OLMoE/DeepSeek/Kimi/GLM/GPT-OSS/Nemotron training workload with preset-plus-overrides dimensions. | Uploadable `DataflowProgram v1` JSON. |
| `model_training/custom_arch/` | You want to add custom ops, modules, and a custom trainable architecture. | Uploadable program, bare plan, annotated plan, and summary metrics. |

## Generic Dataflow Program

```bash
python examples/generic_dataflow/export_program.py \
  --out /tmp/generic_pipeline.dataflow.json
```

Use this when the workload is not necessarily training or even ML. Define the
portable dataflow contract directly: objects, compute blocks, ordered tasks,
and optional metrics.

See `examples/generic_dataflow/README.md`.

## Model Training

Built-in architecture export:

```bash
python examples/model_training/builtin_arch/export_training_program.py \
  --model deepseek_v3 \
  --scale 671B-37B \
  --n-layers 8 \
  --seqlen 1024 \
  --optimizer adamw \
  --out /tmp/deepseek_v3_training.dataflow.json
```

Custom architecture export:

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

Custom architecture end-to-end simulation:

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

See `examples/model_training/README.md` for the model-training authoring flow.

## Import Into The Webapp

1. Start the backend and frontend from the repo README.
2. Open the webapp.
3. Choose **Custom Dataflow Program** in the Workload panel.
4. Import one of the generated `.dataflow.json` files.
5. Choose hardware, click **Create Workload**, then run a planner.
