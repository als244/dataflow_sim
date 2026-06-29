# Model Training Examples

This directory contains the two supported model-training authoring paths.

## Paths

| Path | Use when |
| --- | --- |
| `builtin_arch/` | The model is one of the supported built-in families, possibly with dimension overrides. |
| `custom_arch/` | You are defining new ops, modules, or architecture composition. |

## Shared Contract

Both paths emit the same portable output:

- a hardware-free `DataflowProgram v1`;
- canonical activations shaped as `[tokens, dim]`;
- explicit forward, backward, and recompute phase costs;
- model-owned architecture order;
- generic `TrainingBuilder` lowering into forward, head/loss, recompute,
  backward, and optimizer tasks.

Runtime choices happen after export:

- hardware realization turns sub-op specs into a bare `TaskChain`;
- a memory policy annotates the chain;
- optional recompute planning chooses saved activation variants;
- the simulator returns timelines and summary metrics.

## Built-In Architecture

```bash
python examples/model_training/builtin_arch/export_training_program.py \
  --model llama3 \
  --scale 8B \
  --n-layers 4 \
  --seqlen 512 \
  --out /tmp/llama3_training.dataflow.json
```

## Custom Architecture

```bash
python examples/model_training/custom_arch/export_training_program.py \
  --n-layers 4 \
  --d-model 512 \
  --hidden-dim 2048 \
  --classes 32000 \
  --seqlen 256 \
  --out /tmp/tiny_mixer.dataflow.json
```

End-to-end simulation:

```bash
python examples/model_training/custom_arch/run_training_simulation.py \
  --n-layers 8 \
  --d-model 512 \
  --hidden-dim 2048 \
  --classes 1024 \
  --seqlen 1024 \
  --fast-memory-gb 0.05 \
  --recompute \
  --out-dir /tmp/tiny_mixer_run
```
