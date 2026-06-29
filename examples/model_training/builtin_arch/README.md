# Built-In Architecture Training

Use this example when you want a supported model family with preset-plus-
override dimensions: Llama 3, Qwen3, Qwen3 MoE, or OLMoE.

## Contract

- Choose a model family and preset scale.
- Override dimensions as needed.
- Provide `TrainingConfig` values for sequence length, microbatch, gradient
  accumulation, steps, optimizer, and final placement.
- Optionally provide a `DTypePolicy`.
- Export a hardware-free `DataflowProgram v1`.

## Run

```bash
python examples/model_training/builtin_arch/export_training_program.py \
  --model qwen3_moe \
  --scale 30B-3B \
  --n-layers 8 \
  --seqlen 1024 \
  --optimizer adamw \
  --out /tmp/qwen3_moe_training.dataflow.json
```

Upload the result through the webapp's **Custom Dataflow Program** tab, or use
the webapp's built-in workload preset dropdown for the full-size public
presets.
