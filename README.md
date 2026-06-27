# Dataflow Sim

A discrete-event simulator for memory-constrained dataflow workloads on a two-tier memory hierarchy. The simulator uses three parallel streams (compute, slow->fast memory, fast->slow memory). Our model was originally intended for CPU<-->GPU compute/communication overlap planning; however, it is also practical for HBM<-->SRAM hierarchy (just the units differ; same high-level problem). We assume workloads are constructed as a sequential list of abstract tasks where each task contains lists of input, output, and mutated object identifiers (we assume object sizes are specified, and task runtimes can be derived). The simulator enforces that all input and mutated objects are present in fast memory before starting the task and enforces that the task stalls until there is sufficient fast memory capacity to create all output objects. The simulator manages queues to track fast<-->slow transfer requests and only one transfer (per direction) can be in-flight at a time. ***The primary objective is to minimize overall runtime when there is a hard constraint on fast memory capacity. This means ensuring a combination of (a) avoiding idle time and (b) avoiding recomputation.***

We formulate this problem as annotating a *task-chain* with **release**, **offload**, and **prefetch** directives where each contains a list of 0 or more object identifiers. After a task completes, the runtime (or simulated runtime) triggers execution of such directives.

- **Release**: Free fast-memory storage associated with that object.
- **Offload**: Enqueue object in the fast->slow transfer queue. Upon completion of transfer, the object is released.
- **Prefetch**: Enqueue object in the slow->fast transfer queue. Waits until there is sufficient fast memory to contain the object before starting transfer.

Our main policy (methodology for deciding annotations) is called [PressureFit](docs/policy/pressurefit.md).

For DNN training workloads we further apply [recompute planning](docs/recompute.md) based on memory pressure and runtime results reported by the simulator; recomputation decisions add tasks to the original set.

## Visualizing Simulated Workloads with the Webapp

The default policy is quite effective and can be [visualized](https://dataflowsim.sunshein.net/) for carrying out transformer training in memory-constrained regimes.

The simulator ingests an abstract dataflow program; we take a model architecture specification and translate it to a task chain that mimics reality. In the webapp you will see an unannotated plan that contains all of the tasks with input/output/mutated objects along with associated task runtime. After you run a simulation you can see a summary of overall metrics, the annotated plan, a timeline of events on each of the streams, composition of fast memory over time, and replayable events. The `Throughput vs. Fast Memory Capacity` sweep at the top will run simulations across different memory budgets; then choose a memory budget level that is interesting to see how events actually unfold. *The ideal case is to achieve a runtime close to that of the unlimited fast-memory-capacity regime using just a fraction of fast-memory...*

You can also [create your own dataflow program](examples/README.md) and export it to a schema that the webapp can ingest and simulate.

<!-- 
> [!NOTE]
> The space of possible planning decisions is combinatorial and becomes more difficult when the number of tasks increases and/or when memory pressure increases. Currently, our default task chain for transformer training assumes a batch of sequences is processed each forward/backward pass and goes through each transformer block (i.e. the usual framing). However, this does not have to be the case; we can break each transformer block down into finer-grained tasks, or we could split the batch into smaller chunks (cut the X matrix across rows). These are optimization opportunities, but they come with the challenge of more difficult planning and recomputation. -->

## Setup

For creating custom workloads or accessing simulator API.

```bash
# -1. Activate any python environment you want to work from

# 0. Clone this repo:
git clone git@github.com:als244/dataflow_sim.git

# 1. Install the Python package
cd dataflow_sim && pip install -e .
```

<!--
## Repo Layout

- `src/dataflow_sim/core/` - task-chain schema, validation, and reference-stream utilities.
- `src/dataflow_sim/engine/` - workload-agnostic event simulator.
- `src/dataflow_sim/policies/` - policies that annotate bare workloads with release/offload/prefetch plans.
- `src/dataflow_sim/workloads/` - generic workload schema, workload builders, and shared workload concepts such as hardware specs.
- `src/dataflow_sim/app/` - FastAPI backend for the current webapp.
- `ui/` - React frontend.
- `examples/` - runnable workload/schema export examples.
- `scripts/` - repo-level experiment and utility scripts.
- `docs/` - design + recipe docs.
-->

## TODOs

- [ ] Support distributed training simulation: Add network queues to simulator and task primivites to intiate P2P and collective communication ops (or maybe these details should be 'baked' in to intra-layer efficiency...)
- [ ] Add finer-grained transformer block task decomposition.
- [ ] Add batch/sequence chunking workload builders.
- [ ] Expand custom dataflow examples beyond transformer training.
- [ ] Add screenshots or a short walkthrough for the webapp workflow.

