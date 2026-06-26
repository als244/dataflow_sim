# dataflow_sim

A discrete-event simulator for memory-constrained dataflow workloads on a two-tier memory hierarchy. The model has three parallel streams (compute, backing-from-slow, compute-to-backing) sharing a hard fast-memory cap, executing ordered compute tasks that read/write/mutate variably-sized objects. The scheduling problem — decide what to pre-place, what to evict, and when to fire each transfer so the cap holds at every continuous-time instant while makespan is minimized — has no clean classical analogue, so this repo exists to prototype and compare planning policies. DNN training is provided as a higher-level workload builder on top of the generic dataflow schema.

## Repo Layout

- `src/dataflow_sim/core/` — task-chain schema, validation, and reference-stream utilities.
- `src/dataflow_sim/engine/` — workload-agnostic event simulator.
- `src/dataflow_sim/policies/` — policies that annotate bare workloads with release/offload/prefetch plans.
- `src/dataflow_sim/workloads/` — generic workload schema, workload builders, and shared workload concepts such as hardware specs.
- `src/dataflow_sim/app/` — FastAPI backend for the current webapp.
- `ui/` — React frontend.
- `examples/` — runnable workload/schema export examples.
- `scripts/` — repo-level experiment and utility scripts.
- `docs/` — design + recipe docs.

## Setup

```bash
# 0. Activate any python environment you want to work from

# 1. Install the Python package
pip install -e

# 2. Install UI deps
cd ui && npm install
```

## Run the webapp

```bash
# Terminal 1: backend
uvicorn dataflow_sim.app.server.main:app --reload --port 8000

# Terminal 2: frontend
cd ui && npm run dev
```

Then open the URL printed by `npm run dev`.

## Run tests

```bash
pytest
```

## Where to go next

- [docs/problem.md](docs/problem.md) — the scheduling problem the simulator solves
- [src/dataflow_sim/workloads/README.md](src/dataflow_sim/workloads/README.md) — generic workload schema and authoring guide
- [examples/](examples/) — custom dataflow and transformer-training schema exporters
- [docs/workload-recipe.md](docs/workload-recipe.md) — low-level `TaskChain` reference
- [docs/transformer-recipe.md](docs/transformer-recipe.md) — how the example app maps transformer training onto the simulator
- [docs/policy/README.md](docs/policy/README.md) — the six built-in scheduling policies and which to use
- [docs/policy/pressurefit.md](docs/policy/pressurefit.md) — ***PressureFit***: An automatic policy that determines initial object placement and annotates task chains with release, offload, and prefetch triggers.
- [docs/recompute.md](docs/recompute.md) — activation recomputation: chain variants, the stall/backlog report, and evidence-directed selection layered above the policies.
