import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ComputeTimeline } from "./components/ComputeTimeline";
import { MemoryPanel } from "./components/MemoryPanel";
import { ReferenceStream } from "./components/ReferenceStream";
import { EventControls } from "./components/EventControls";
import {
  InputPanel,
  DEFAULT_PARAMS,
  POLICY_OPTIONS,
  type SimulationParams,
  type Presets,
  type Policy,
  type OptimizerMode,
  type ModelTrainingWorkloadParams,
  type DataflowProgram,
} from "./components/InputPanel";
import { PlannerPanel } from "./components/PlannerPanel";
import { ComparePoliciesPanel } from "./components/ComparePoliciesPanel";
import { SubOpBreakdownPanel, type Breakdown } from "./components/SubOpBreakdownPanel";
import { SummaryPanel, type Summary } from "./components/SummaryPanel";
import { MemoryTimelinePanel } from "./components/MemoryTimelinePanel";
import { AnnotatedPlanPanel, type AnnotatedChain } from "./components/AnnotatedPlanPanel";
import {
  PolicyDiagnosticsPanel,
  type PressureFitDiagnostics,
} from "./components/PolicyDiagnosticsPanel";
import { MemorySweepPanel } from "./components/MemorySweepPanel";
import type { EventLog } from "./types";
import "./App.css";

type Status = "idle" | "loading" | "ok" | "error";

interface SimulateResponse {
  log: EventLog;
  breakdown: Breakdown;
  summary: Summary;
  chain: AnnotatedChain;
  workload_preview: WorkloadPreviewSummary;
  compute_blocks?: Breakdown["compute_blocks"];
  policy_diagnostics: PressureFitDiagnostics | null;
}

interface WorkloadPreviewSummary {
  name: string;
  object_count: number;
  task_count: number;
  compute_block_count: number;
  aggregate_task_runtime_us: number;
}

interface WorkloadPreviewResponse {
  schema: DataflowProgram;
  preview: WorkloadPreviewSummary;
  chain: AnnotatedChain;
  breakdown: Breakdown;
  compute_blocks: Breakdown["compute_blocks"];
  task_summaries: {
    id: string;
    label: string;
    group: string;
    compute_block_key: string;
    compute_block_name: string;
    runtime_us: number;
    inputs: number;
    outputs: number;
  }[];
}

const PARAM_STORAGE_KEY = "dataflow-sim:simulation-params:v1";

// Legacy flat URL-param encoding for nested params. Keep this reader so old
// shared links still hydrate the form, but persist new edits in local storage.
const HW_KEYS = [
  "peak_tflops", "fast_memory_bw_gbs", "from_slow_bw_gbs", "to_slow_bw_gbs",
  "matmul_eff", "attn_fwd_eff", "attn_bwd_eff", "mem_eff",
] as const;
const MODEL_NUM_KEYS = [
  "vocab_size", "n_layers", "d_model", "head_dim", "n_heads", "n_kv_heads",
  "expert_dim", "num_shared_experts", "num_routed_experts", "top_k",
] as const;

const LEGACY_QUERY_KEYS = new Set<string>([
  "hw_preset",
  "workload_preset",
  "model_preset",
  "model_qk_norm",
  "seqlen",
  "num_seqs",
  "grad_accum_rounds",
  "num_steps",
  "optimizer",
  "final_model_state_on_backing",
  "policy",
  "pressurefit_mode",
  "recompute",
  "window_size",
  "fast_memory_capacity_gb",
  ...HW_KEYS.map((k) => `hw_${k}`),
  ...MODEL_NUM_KEYS.map((k) => `model_${k}`),
]);

function cloneDefaultParams(): SimulationParams {
  return JSON.parse(JSON.stringify(DEFAULT_PARAMS)) as SimulationParams;
}

function readStoredParams(): SimulationParams | null {
  try {
    const raw = window.localStorage.getItem(PARAM_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<SimulationParams>;
    if (!parsed || !parsed.hardware || !parsed.workload || !parsed.planner) return null;
    if ((parsed.workload as { source?: string }).source === "training_transformer") {
      (parsed.workload as { source: string }).source = "model_training";
    }
    return parsed as SimulationParams;
  } catch {
    return null;
  }
}

function hasLegacyQueryParams(url: URLSearchParams): boolean {
  for (const key of LEGACY_QUERY_KEYS) {
    if (url.has(key)) return true;
  }
  return false;
}

function applyLegacyQueryParams(out: SimulationParams, url: URLSearchParams): void {
  const modelTraining = out.workload as ModelTrainingWorkloadParams;

  const hwPreset = url.get("hw_preset");
  if (hwPreset) out.hardware.preset = hwPreset;
  for (const k of HW_KEYS) {
    const v = url.get(`hw_${k}`);
    if (v !== null) {
      const n = Number(v);
      if (Number.isFinite(n)) (out.hardware as unknown as Record<string, unknown>)[k] = n;
    }
  }

  const mPreset = url.get("workload_preset") ?? url.get("model_preset");
  if (mPreset) {
    modelTraining.preset = mPreset;
    modelTraining.model.preset = mPreset;
  }
  for (const k of MODEL_NUM_KEYS) {
    const v = url.get(`model_${k}`);
    if (v !== null) {
      const n = Number(v);
      if (Number.isFinite(n)) (modelTraining.model as unknown as Record<string, unknown>)[k] = n;
    }
  }
  const qk = url.get("model_qk_norm");
  if (qk !== null) modelTraining.model.qk_norm = qk === "true";

  const seq = url.get("seqlen");
  if (seq !== null) {
    const n = Number(seq);
    if (Number.isFinite(n)) modelTraining.training.seqlen = n;
  }
  const mb = url.get("num_seqs");
  if (mb !== null) {
    const n = Number(mb);
    if (Number.isFinite(n)) modelTraining.training.num_seqs = n;
  }
  const ga = url.get("grad_accum_rounds");
  if (ga !== null) {
    const n = Number(ga);
    if (Number.isFinite(n)) modelTraining.training.grad_accum_rounds = n;
  }
  const steps = url.get("num_steps");
  if (steps !== null) {
    const n = Number(steps);
    if (Number.isFinite(n)) modelTraining.training.num_steps = n;
  }
  const optimizer = url.get("optimizer");
  if (optimizer !== null) {
    const VALID_OPTIMIZERS: OptimizerMode[] = ["none", "adamw", "muon"];
    if ((VALID_OPTIMIZERS as string[]).includes(optimizer)) {
      modelTraining.training.optimizer = optimizer as OptimizerMode;
    }
  }
  const finalBacking = url.get("final_model_state_on_backing");
  if (finalBacking !== null) modelTraining.training.final_model_state_on_backing = finalBacking === "true";
  const policy = url.get("policy");
  if (policy !== null) {
    const VALID_POLICIES: Policy[] = [
      "sliding_window",
      "belady_reactive",
      "roundtrip_planner",
      "max_reduce",
      "min_grow",
      "pressurefit",
    ];
    if ((VALID_POLICIES as string[]).includes(policy)) {
      out.planner.policy = policy as Policy;
    }
  }
  const rc = url.get("recompute");
  if (rc !== null) out.planner.recompute = rc === "true";
  const ws = url.get("window_size");
  if (ws !== null) {
    const n = Number(ws);
    if (Number.isFinite(n)) out.planner.window_size = n;
  }
  const cap = url.get("fast_memory_capacity_gb");
  if (cap === "" || cap === "null") out.planner.fast_memory_capacity_gb = null;
  else if (cap !== null) {
    const n = Number(cap);
    if (Number.isFinite(n)) out.planner.fast_memory_capacity_gb = n;
  }
}

function initialParams(): SimulationParams {
  const url = new URLSearchParams(window.location.search);
  const hasLegacyParams = hasLegacyQueryParams(url);
  const out = hasLegacyParams ? cloneDefaultParams() : readStoredParams() ?? cloneDefaultParams();
  if (hasLegacyParams) applyLegacyQueryParams(out, url);
  return out;
}

function persistParams(params: SimulationParams): void {
  try {
    window.localStorage.setItem(PARAM_STORAGE_KEY, JSON.stringify(params));
  } catch {
    /* Storage may be unavailable in private browsing or restricted contexts. */
  }
}

function cleanLegacyQueryParams(): void {
  const url = new URL(window.location.href);
  let changed = false;
  for (const key of LEGACY_QUERY_KEYS) {
    if (url.searchParams.has(key)) {
      url.searchParams.delete(key);
      changed = true;
    }
  }
  if (changed) window.history.replaceState(null, "", url.toString());
}

async function simulate(params: SimulationParams): Promise<SimulateResponse> {
  let res: Response;
  try {
    res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
  } catch (e) {
    throw new Error(
      `Network connection failed: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return res.json();
}

async function previewWorkload(params: SimulationParams): Promise<WorkloadPreviewResponse> {
  let res: Response;
  try {
    res = await fetch("/api/workloads/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workload: params.workload, hardware: params.hardware }),
    });
  } catch (e) {
    throw new Error(
      `Network connection failed: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return res.json();
}

async function fetchPresets(): Promise<Presets> {
  const res = await fetch("/api/presets");
  if (!res.ok) throw new Error(`presets HTTP ${res.status}`);
  return res.json();
}

function fmtTime(us: number): string {
  if (us === 0) return "0 µs";
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(2)} s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(1)} ms`;
  return `${us.toLocaleString()} µs`;
}

function WorkloadStatsPanel({ preview, stale }: { preview: WorkloadPreviewSummary | null; stale: boolean }) {
  if (!preview) {
    return (
      <div className="panel workload-stats-panel">
        <div className="panel-header">
          <h3>Workload Preview</h3>
        </div>
        <p className="dim">Create a workload to inspect the bare plan and compute blocks.</p>
      </div>
    );
  }
  return (
    <div className={`panel workload-stats-panel${stale ? " workload-stats-stale" : ""}`}>
      <div className="panel-header">
        <h3>Workload Preview</h3>
        {stale && <span className="tag status-stale">Stale</span>}
      </div>
      <div className="workload-stat-grid">
        <div>
          <span className="workload-stat-label">Tasks</span>
          <strong>{preview.task_count.toLocaleString()}</strong>
        </div>
        <div>
          <span className="workload-stat-label">Objects</span>
          <strong>{preview.object_count.toLocaleString()}</strong>
        </div>
        <div>
          <span className="workload-stat-label">Compute Blocks</span>
          <strong>{preview.compute_block_count.toLocaleString()}</strong>
        </div>
        <div>
          <span className="workload-stat-label">Aggregate Task Runtime</span>
          <strong>{fmtTime(preview.aggregate_task_runtime_us)}</strong>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [params, setParams] = useState<SimulationParams>(initialParams);
  const [log, setLog] = useState<EventLog | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [chain, setChain] = useState<AnnotatedChain | null>(null);
  const [simulationBreakdown, setSimulationBreakdown] = useState<Breakdown | null>(null);
  const [policyDiagnostics, setPolicyDiagnostics] = useState<PressureFitDiagnostics | null>(null);
  const [presets, setPresets] = useState<Presets | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [preview, setPreview] = useState<WorkloadPreviewResponse | null>(null);
  const [previewStatus, setPreviewStatus] = useState<Status>("idle");
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewKey, setPreviewKey] = useState<string | null>(null);

  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hoverTaskId, setHoverTaskId] = useState<string | null>(null);
  const [selectedObjId, setSelectedObjId] = useState<string | null>(null);

  // Fetch presets once.
  useEffect(() => {
    let cancelled = false;
    fetchPresets()
      .then((p) => {
        if (!cancelled) setPresets(p);
      })
      .catch((e) => {
        console.warn("presets unavailable", e);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Persist form state locally on every params change. Also clear legacy query
  // params and stale errors after the user has edited the form.
  useEffect(() => {
    persistParams(params);
    cleanLegacyQueryParams();
    if (errorMsg) setErrorMsg(null);
    if (previewError) setPreviewError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);

  const workloadKey = useMemo(
    () => JSON.stringify({ workload: params.workload, hardware: params.hardware }),
    [params.workload, params.hardware],
  );
  const previewStale = previewStatus === "ok" && previewKey !== workloadKey;
  const workloadReady = previewStatus === "ok" && !previewStale && preview !== null;

  const resetSimulation = useCallback(() => {
    setLog(null);
    setSummary(null);
    setChain(null);
    setSimulationBreakdown(null);
    setPolicyDiagnostics(null);
    setIndex(0);
    setPlaying(false);
    setStatus("idle");
    setErrorMsg(null);
  }, []);

  const handlePreview = useCallback(async () => {
    setPreviewStatus("loading");
    setPreviewError(null);
    setPlaying(false);
    try {
      const resp = await previewWorkload(params);
      setPreview(resp);
      setPreviewKey(workloadKey);
      setPreviewStatus("ok");
      resetSimulation();
    } catch (e) {
      setPreviewStatus("error");
      setPreviewError(String(e instanceof Error ? e.message : e));
    }
  }, [params, resetSimulation, workloadKey]);

  const handleSubmit = useCallback(async () => {
    if (!workloadReady) return;
    setStatus("loading");
    setErrorMsg(null);
    setPolicyDiagnostics(null);
    setPlaying(false);
    try {
      const resp = await simulate(params);
      setLog(resp.log);
      setSummary(resp.summary);
      setChain(resp.chain);
      setSimulationBreakdown(resp.breakdown);
      setPolicyDiagnostics(resp.policy_diagnostics);
      setIndex(0);
      setStatus("ok");
    } catch (e) {
      setStatus("error");
      setErrorMsg(String(e instanceof Error ? e.message : e));
    }
  }, [params, workloadReady]);

  const playRef = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || !log) return;
    playRef.current = window.setInterval(() => {
      setIndex((i) => {
        const next = i + 1;
        if (next >= log.events.length) {
          setPlaying(false);
          return i;
        }
        return next;
      });
    }, 600);
    return () => {
      if (playRef.current !== null) window.clearInterval(playRef.current);
    };
  }, [playing, log]);

  const totalDuration = useMemo(
    () => (log ? log.task_intervals.reduce((m, iv) => Math.max(m, iv.end), 0) : 0),
    [log],
  );

  const safeIndex = log && log.events.length > 0 ? Math.min(index, log.events.length - 1) : 0;
  const current = log && log.events.length > 0 ? log.events[safeIndex] : null;
  const hasSnapshots = log !== null && log.events.length > 0;
  const hasMemoryTimeline = (
    log !== null
    && (hasSnapshots || (log.memory_trace?.length ?? 0) > 0)
  );
  const simulationActive = status === "loading" || log !== null;
  const visibleBreakdown = simulationBreakdown ?? preview?.breakdown ?? null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Dataflow Simulator</h1>
        {log ? (
          <span className="dim">
            {log.task_intervals.length} tasks · {log.events.length} events · duration {totalDuration.toLocaleString()} µs
          </span>
        ) : (
          <span className="dim">Create a workload, then run a memory-planning simulation.</span>
        )}
      </header>

      <div className="workspace-shell">
        <aside className="workspace-left">
          <InputPanel
            params={params}
            setParams={setParams}
            onPreview={handlePreview}
            locked={simulationActive}
            previewStatus={previewStatus}
            previewError={previewError}
            previewStale={previewStale}
            presets={presets}
          />

          <WorkloadStatsPanel preview={preview?.preview ?? null} stale={previewStale} />

          <AnnotatedPlanPanel
            chain={preview?.chain ?? null}
            title="Unannotated Plan"
            emptyText="Create a workload to inspect the bare task chain."
          />
        </aside>

        <main className="workspace-right">
          {workloadReady && <MemorySweepPanel params={params} />}

          <PlannerPanel
            params={params}
            setParams={setParams}
            onRun={handleSubmit}
            onReset={resetSimulation}
            canRun={workloadReady}
            status={status}
            errorMsg={errorMsg}
            previewStale={previewStale}
            hasResults={log !== null}
          />

          {workloadReady && <SubOpBreakdownPanel breakdown={visibleBreakdown} />}

          {log ? (
            <>
              <SummaryPanel summary={summary} />

              <AnnotatedPlanPanel chain={chain} title="Annotated Plan" />

              {hasMemoryTimeline && (
                <MemoryTimelinePanel
                  log={log}
                  fastMemoryCapacityGb={params.planner.fast_memory_capacity_gb}
                  currentT={current?.t ?? null}
                />
              )}

              <ComputeTimeline
                intervals={log.task_intervals}
                currentT={current?.t ?? 0}
                totalDuration={totalDuration}
                activeTaskId={current?.snapshot.active_task?.id ?? null}
                hoverTaskId={hoverTaskId}
                onHoverTask={setHoverTaskId}
              />

              {hasSnapshots && current ? (
                <>
                  <EventControls
                    events={log.events}
                    index={safeIndex}
                    setIndex={setIndex}
                    playing={playing}
                    setPlaying={setPlaying}
                  />

                  <details className="panel collapsible-panel">
                    <summary className="collapsible-summary">Memory Contents &amp; Reference Stream</summary>
                    <div className="collapsible-content">
                      <div className="three-col">
                        <div className="scroll-subpanel">
                          <MemoryPanel
                            title="Slow Memory"
                            memory={current.snapshot.memory.filter((m) => m.location === "backing")}
                            highlightedIds={new Set()}
                            selectedObjId={selectedObjId}
                            onSelectObj={setSelectedObjId}
                          />
                        </div>
                        <div className="scroll-subpanel">
                          <MemoryPanel
                            title="Fast Memory"
                            memory={current.snapshot.memory.filter((m) => m.location === "fast")}
                            highlightedIds={new Set()}
                            selectedObjId={selectedObjId}
                            onSelectObj={setSelectedObjId}
                          />
                        </div>
                        <div className="scroll-subpanel">
                          <ReferenceStream
                            references={current.snapshot.reference_stream}
                            memory={current.snapshot.memory}
                            selectedObjId={selectedObjId}
                          />
                        </div>
                      </div>
                    </div>
                  </details>
                </>
              ) : (
                <div className="panel trace-note">
                  Exact summary, compute intervals, and compact fast-memory trace returned.
                  Full memory contents and reference stream were omitted for this large chain.
                </div>
              )}

              <ComparePoliciesPanel params={params} policies={POLICY_OPTIONS} />

              <PolicyDiagnosticsPanel diagnostics={policyDiagnostics} />
            </>
          ) : (
            <div className="panel empty-panel">
              {status === "loading" ? (
                <p className="dim loading-line">
                  <span className="loading-spinner" aria-hidden="true" />
                  Running simulation
                </p>
              ) : workloadReady ? (
                <p className="dim">
                  Workload ready. Choose planner settings and run the simulation.
                </p>
              ) : (
                <p className="dim">
                  Create or update the workload preview before running a simulation.
                </p>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
