# dataflow_sim

A discrete-event simulator for memory-constrained DNN training on a single GPU. The model has three parallel streams (compute, host-to-device, device-to-host) sharing a hard device-memory cap, executing a DAG of compute tasks that read/write/mutate variably-sized objects. The scheduling problem — decide what to pre-place, what to evict, and when to fire each transfer so the cap holds at every continuous-time instant while makespan is minimized — has no clean classical analogue, so this repo exists to prototype and compare planning policies (and eventually validate them against an exact CP-SAT oracle on small configs).

## Repo Layout

- `src/dataflow_sim/core/` — task-chain schema, validation, and reference-stream utilities.
- `src/dataflow_sim/engine/` — workload-agnostic event simulator.
- `src/dataflow_sim/policies/` — policies that annotate bare workloads with release/offload/prefetch plans.
- `src/dataflow_sim/workloads/` — workload builders and shared workload concepts such as hardware specs.
- `src/dataflow_sim/app/` — FastAPI backend for the current webapp.
- `ui/` — React frontend.
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
- [docs/workload-recipe.md](docs/workload-recipe.md) — how to model your own workload via the simulator API
- [docs/transformer-recipe.md](docs/transformer-recipe.md) — how the example app maps transformer training onto the simulator
- [docs/policy/README.md](docs/policy/README.md) — the six built-in scheduling policies and which to use
- [docs/policy/pressurefit.md](docs/policy/pressurefit.md) — ***PressureFit***: An automatic policy that determines initial object placement and annotates task chains with release, offload, and prefetch triggers.
