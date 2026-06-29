# Generic Dataflow Program

Use this example when you already know the workload as ordered compute over
named memory objects. It does not assume model training.

## Contract

- Build a hardware-free `DataflowProgram v1`.
- Define initial objects, reusable compute blocks, and ordered tasks.
- Keep planner annotations out of the program; policies add releases,
  offloads, and prefetches later after hardware realization.

## Run

```bash
python examples/generic_dataflow/export_program.py \
  --out /tmp/generic_pipeline.dataflow.json
```

Upload the result through the webapp's **Custom Dataflow Program** tab.
