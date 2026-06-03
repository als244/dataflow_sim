# dataflow_sim

A discrete-event simulator for memory-constrained DNN training on a single GPU. The model has three parallel streams (compute, host-to-device, device-to-host) sharing a hard device-memory cap, executing a DAG of compute tasks that read/write/mutate variably-sized objects. The scheduling problem — decide what to pre-place, what to evict, and when to fire each transfer so the cap holds at every continuous-time instant while makespan is minimized — has no clean classical analogue, so this repo exists to prototype and compare planning policies (and eventually validate them against an exact CP-SAT oracle on small configs).

## Repo layout

- `simulator/` — the discrete-event simulator package (`dataflow_sim`). Pip-installable.
- `app/` — transformer-training webapp + workloads (`dataflow_sim_app`). Depends on `dataflow_sim`.
- `docs/` — design + recipe docs.

## Setup

```bash
# 1. Install (editable, both packages)
pip install -e ./simulator
pip install -e ./app

# 2. Install UI deps
cd app/ui && npm install
```

## Run the webapp

```bash
# Terminal 1: backend
uvicorn dataflow_app.server.main:app --reload --port 8000

# Terminal 2: frontend
cd app/ui && npm run dev
```

Then open the URL printed by `npm run dev`.

## Run tests

```bash
pytest simulator/tests
pytest app/tests
```

## Where to go next

- [docs/problem.md](docs/problem.md) — the scheduling problem the simulator solves
- [docs/workload-recipe.md](docs/workload-recipe.md) — how to model your own workload via the simulator API
- [docs/transformer-recipe.md](docs/transformer-recipe.md) — how the example app maps transformer training onto the simulator
- [docs/policy/README.md](docs/policy/README.md) — the six built-in scheduling policies and which to use
