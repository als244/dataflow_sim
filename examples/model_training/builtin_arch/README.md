# Built-In Architecture Training

Use this example when you want a supported model family with preset-plus-
override dimensions: Llama 3, Qwen3, Qwen3 MoE, OLMoE, Qwen3.5/3.6 hybrid,
DeepSeek-V3, or Kimi-K2.

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

Qwen3.5 hybrid dense:

```bash
python examples/model_training/builtin_arch/export_training_program.py \
  --model qwen3_hybrid_dense \
  --scale qwen3_5_27B \
  --n-layers 8 \
  --seqlen 1024 \
  --optimizer adamw \
  --out /tmp/qwen35_dense_training.dataflow.json
```

DeepSeek-V3 reduced layer count:

```bash
python examples/model_training/builtin_arch/export_training_program.py \
  --model deepseek_v3 \
  --scale 671B-37B \
  --n-layers 8 \
  --seqlen 1024 \
  --optimizer adamw \
  --out /tmp/deepseek_v3_training.dataflow.json
```

Upload the result through the webapp's **Custom Dataflow Program** tab, or use
the webapp's built-in workload preset dropdown for the full-size public
presets.
