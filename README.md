# Dataflow Sim

A discrete-event simulator for memory-constrained dataflow workloads on a two-tier memory hierarchy. The simulator uses three parallel streams (compute, slow->fast memory, fast->slow memory). Our model was originally intended for CPU<-->GPU compute/communication overlap planning; however, it is also practical for HBM<-->SRAM heirarchy (just the units differ; same higher level problem). We assume workloads are constructed as a sequential list of abstract tasks where each task contains lists of input, output, and mutated object identifiers (we assume object sizes are specified, and task runtimes can be derived). The simulator enforces that all input and mutated objects are present in fast memory before starting the task and enforces that the task stalls until there is sufficient fast memory capacity to create all output objects. The simulator manages queues to track fast<-->slow transfer requests and only one transfer (per direction) can be in-flight at a time. ***Our primary objective is to minimize overall runtime when there is a hard constraint on fast memory capacity. This means ensuring a combination of (a) avoiding idle time and (b) avoiding recomputation.***

We formulate this problem as annotating a *task-chain* with **release**, **offload**, and **prefetch** directives where each contains a list of 0 or more object identifiers. After a task completes, the runtime (or simulated runtime) triggers execution of such directives. 

- **Release**: Free fast-memory associated with that object
- **Offload**: Enqueue object in the fast->slow transfer queue. Upon completition of transfer, the object is released
- **Prefetch**: Enqueue object in the slow->fast transfer queue. Waits until sufficient fast memory to contain object before starting transfer. 


Our main policy (methodology for deciding annotations) is called [PressureFit](docs/policy/pressurefit.md). 

## Webapp

The policy is quite effective and can be [visualized](https://dataflowsim.sunshein.net/) for carrying out transformer training in memory constrained regimes. The simulator ingests an abstract dataflow program; we take model architecture specification and translate this to a task chain that mimics reality. In the webapp you will see an 'unannotated' plan that contains all of the tasks with input/output/mutated objects along with assoicated task runtime. After you run a simulation you can see a summary of overall metrics, the annotated plan, a timeline of events on each of the streams, composition of fast memory over time, and replayable events. *The hope is to achieve a runtime close to that of unlimited fast memory capacity regime...*

You can also [create your own dataflow program](examples/README.md) and export it to a schema that the webapp can ingest and simulate. 

## Setup

For creating custom workloads or accessing simulator API.

```bash
# -1. Activate any python environment you want to work from

# 0. Clone this repo:
git clone git@github.com:als244/dataflow_sim.git


# 1. Install the Python package
cd dataflow_sim && pip install -e
```

<!-- 
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
 -->
