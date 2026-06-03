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
} from "./components/InputPanel";
import { ComparePoliciesPanel } from "./components/ComparePoliciesPanel";
import { SubOpBreakdownPanel, type Breakdown } from "./components/SubOpBreakdownPanel";
import { SummaryPanel, type Summary } from "./components/SummaryPanel";
import { MemoryTimelinePanel } from "./components/MemoryTimelinePanel";
import { AnnotatedPlanPanel, type AnnotatedChain } from "./components/AnnotatedPlanPanel";
import type { EventLog } from "./types";
import "./App.css";

type Status = "idle" | "loading" | "ok" | "error";

interface SimulateResponse {
  log: EventLog;
  breakdown: Breakdown;
  summary: Summary;
  chain: AnnotatedChain;
}

// Flat URL-param encoding for nested params.
const HW_KEYS = [
  "peak_tflops", "gpu_membw_gbs", "interconnect_bw_gbs",
  "matmul_eff", "attn_fwd_eff", "attn_bwd_eff", "mem_eff",
] as const;
const MODEL_NUM_KEYS = [
  "vocab_size", "n_layers", "d_model", "head_dim", "n_heads", "n_kv_heads",
  "expert_dim", "num_shared_experts", "num_routed_experts", "top_k",
] as const;

function initialParams(): SimulationParams {
  const url = new URLSearchParams(window.location.search);
  const out: SimulationParams = JSON.parse(JSON.stringify(DEFAULT_PARAMS));

  const hwPreset = url.get("hw_preset");
  if (hwPreset) out.hardware.preset = hwPreset;
  for (const k of HW_KEYS) {
    const v = url.get(`hw_${k}`);
    if (v !== null) {
      const n = Number(v);
      if (Number.isFinite(n)) (out.hardware as unknown as Record<string, unknown>)[k] = n;
    }
  }

  const mPreset = url.get("model_preset");
  if (mPreset) out.model.preset = mPreset;
  for (const k of MODEL_NUM_KEYS) {
    const v = url.get(`model_${k}`);
    if (v !== null) {
      const n = Number(v);
      if (Number.isFinite(n)) (out.model as unknown as Record<string, unknown>)[k] = n;
    }
  }
  const qk = url.get("model_qk_norm");
  if (qk !== null) out.model.qk_norm = qk === "true";

  const seq = url.get("seqlen");
  if (seq !== null) {
    const n = Number(seq);
    if (Number.isFinite(n)) out.seqlen = n;
  }
  const mb = url.get("num_seqs");
  if (mb !== null) {
    const n = Number(mb);
    if (Number.isFinite(n)) out.num_seqs = n;
  }
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
      out.policy = policy as Policy;
    }
  }
  const ws = url.get("window_size");
  if (ws !== null) {
    const n = Number(ws);
    if (Number.isFinite(n)) out.window_size = n;
  }
  const cap = url.get("device_capacity_gb");
  if (cap === "" || cap === "null") out.device_capacity_gb = null;
  else if (cap !== null) {
    const n = Number(cap);
    if (Number.isFinite(n)) out.device_capacity_gb = n;
  }
  return out;
}

function syncUrl(params: SimulationParams): void {
  const url = new URL(window.location.href);
  url.searchParams.set("hw_preset", params.hardware.preset);
  for (const k of HW_KEYS) url.searchParams.set(`hw_${k}`, String(params.hardware[k]));
  url.searchParams.set("model_preset", params.model.preset);
  for (const k of MODEL_NUM_KEYS) url.searchParams.set(`model_${k}`, String(params.model[k]));
  url.searchParams.set("model_qk_norm", params.model.qk_norm ? "true" : "false");
  url.searchParams.set("seqlen", String(params.seqlen));
  url.searchParams.set("num_seqs", String(params.num_seqs));
  url.searchParams.set("policy", params.policy);
  url.searchParams.set("window_size", String(params.window_size));
  url.searchParams.set(
    "device_capacity_gb",
    params.device_capacity_gb === null ? "" : String(params.device_capacity_gb),
  );
  window.history.replaceState(null, "", url.toString());
}

async function simulate(params: SimulationParams): Promise<SimulateResponse> {
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
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

export default function App() {
  const [params, setParams] = useState<SimulationParams>(initialParams);
  const [log, setLog] = useState<EventLog | null>(null);
  const [breakdown, setBreakdown] = useState<Breakdown | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [chain, setChain] = useState<AnnotatedChain | null>(null);
  const [presets, setPresets] = useState<Presets | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

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

  // Sync URL on every params change. Also clear stale error so it doesn't
  // linger after the user has edited the form.
  useEffect(() => {
    syncUrl(params);
    if (errorMsg) setErrorMsg(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);

  const handleSubmit = useCallback(async () => {
    setStatus("loading");
    setErrorMsg(null);
    try {
      const resp = await simulate(params);
      setLog(resp.log);
      setBreakdown(resp.breakdown);
      setSummary(resp.summary);
      setChain(resp.chain);
      setIndex(0);
      setStatus("ok");
    } catch (e) {
      setStatus("error");
      setErrorMsg(String(e instanceof Error ? e.message : e));
    }
  }, [params]);

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

  const safeIndex = log ? Math.min(index, log.events.length - 1) : 0;
  const current = log ? log.events[safeIndex] : null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>dataflow simulator</h1>
        {log ? (
          <span className="dim">
            {log.task_intervals.length} tasks · {log.events.length} events · duration {totalDuration.toLocaleString()} µs
          </span>
        ) : (
          <span className="dim">no simulation yet — fill inputs and click submit</span>
        )}
      </header>

      <InputPanel
        params={params}
        setParams={setParams}
        onSubmit={handleSubmit}
        onReset={() => {
          setLog(null);
          setBreakdown(null);
          setSummary(null);
          setChain(null);
          setIndex(0);
          setPlaying(false);
          setStatus("idle");
          setErrorMsg(null);
        }}
        locked={log !== null}
        status={status}
        errorMsg={errorMsg}
        presets={presets}
      />

      {log && current ? (
        <>
          <SummaryPanel summary={summary} />

          <details className="panel collapsible-panel">
            <summary className="collapsible-summary">Compute Block Breakdown</summary>
            <div className="collapsible-content">
              <SubOpBreakdownPanel breakdown={breakdown} />
            </div>
          </details>

          <MemoryTimelinePanel
            log={log}
            deviceCapacityGb={params.device_capacity_gb}
            currentT={current.t}
          />

          <ComputeTimeline
            intervals={log.task_intervals}
            currentT={current.t}
            totalDuration={totalDuration}
            activeTaskId={current.snapshot.active_task?.id ?? null}
            hoverTaskId={hoverTaskId}
            onHoverTask={setHoverTaskId}
          />

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
                    title="host memory"
                    memory={current.snapshot.memory.filter((m) => m.location === "host")}
                    highlightedIds={new Set()}
                    selectedObjId={selectedObjId}
                    onSelectObj={setSelectedObjId}
                  />
                </div>
                <div className="scroll-subpanel">
                  <MemoryPanel
                    title="device memory"
                    memory={current.snapshot.memory.filter((m) => m.location === "device")}
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

          <AnnotatedPlanPanel chain={chain} />

          <ComparePoliciesPanel params={params} policies={POLICY_OPTIONS} />
        </>
      ) : (
        <div className="panel empty-panel">
          <p className="dim">
            press <span className="kbd">submit</span> on the inputs panel above to run a simulation.
          </p>
        </div>
      )}
    </div>
  );
}
