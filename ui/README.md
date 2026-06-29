# Webapp

This is the React/Vite frontend for the dataflow simulator. It is a thin client:
the browser owns form state and visualization state, while FastAPI owns schema
validation, workload realization, planning, and simulation.

## Run

```bash
npm install
npm run dev
```

The dev server proxies API calls to the Python server. Run the backend with:

```bash
uvicorn dataflow_sim.app.server.main:app --reload --port 8000
```

## UI Contracts

The webapp mirrors the API split:

- Workload state: `ModelTrainingWorkloadParams | SchemaWorkloadParams`
- Hardware state: `HardwareParams`
- Planner state: `PlannerParams`
- Combined request state: `SimulationParams`

These request-side TypeScript interfaces currently live in
`src/components/InputPanel.tsx` because the form owns most editing behavior.
The backend source of truth is `src/dataflow_sim/app/server/main.py`.

Event-log response types live in `src/types.ts` and mirror the JSON form of
`dataflow_sim.core.schema.EventLog`. Component-local response types are kept
near their consumers:

- `App.tsx`: `/api/workloads/preview` and `/api/simulate` response shapes
- `SubOpBreakdownPanel.tsx`: compute block and sub-op breakdown rows
- `SummaryPanel.tsx`: top-level simulation metrics
- `AnnotatedPlanPanel.tsx`: bare/annotated `TaskChain` display shape
- `PolicyDiagnosticsPanel.tsx`: PressureFit diagnostics

When backend response fields change, update the Pydantic model or response
builder first, then update the matching TypeScript interface and renderer.

## Data Flow

1. `GET /api/presets` populates workload and hardware dropdowns.
2. Workload + hardware edits are sent to `POST /api/workloads/preview`.
3. Preview returns normalized `DataflowProgram`, bare `TaskChain`, workload
   stats, and hardware-resolved compute block summaries.
4. The left pane freezes after a simulation starts. Resetting the simulation
   unlocks workload and hardware editing again.
5. `POST /api/simulate` sends `{ workload, hardware, planner }`.
6. Simulation returns an annotated `TaskChain`, `EventLog`, summary metrics,
   compute block breakdown, and optional policy diagnostics.

The UI should never invent memory-planning annotations. It renders preview
chains and simulator responses exactly as returned by the server.

## Custom Dataflow Program Flow

The `Custom Dataflow Program` tab accepts `DataflowProgram v1` JSON. Users can
paste JSON or import a file, click `Create Workload`, inspect the normalized
program and bare plan, then run a planner. The unannotated and annotated plan
panels can export the displayed `TaskChain` JSON for local inspection or reuse
in low-level tests.

Model-training presets use the same path internally: the server turns
model-training params into a `DataflowProgram`, realizes it against hardware,
and then the UI renders it like any other dataflow program workload.

## Checks

```bash
npm run build
npm run lint
```

The current lint run reports Fast Refresh warnings for files that export both
React components and shared constants/types. They are warnings, not build
failures.
