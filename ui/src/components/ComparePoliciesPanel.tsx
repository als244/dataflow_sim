import { useEffect, useRef, useState } from "react";
import type { SimulationParams, Policy } from "./InputPanel";
import type { Summary } from "./SummaryPanel";

interface PolicyOption {
  value: Policy;
  label: string;
}

interface Props {
  params: SimulationParams;
  policies: PolicyOption[];
}

type RowState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; summary: Summary }
  | { kind: "error"; message: string };

function fmtTime(us: number): string {
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(2)} s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(1)} ms`;
  return `${us.toLocaleString()} µs`;
}
function fmtPct(p: number): string { return `${p.toFixed(1)}%`; }
function fmtTflops(t: number): string { return t >= 100 ? t.toFixed(0) : t.toFixed(1); }
function fmtGb(g: number): string {
  if (g >= 100) return g.toFixed(0);
  if (g >= 10) return g.toFixed(1);
  return g.toFixed(2);
}
function fmtToks(t: number): string {
  if (t >= 1e6) return `${(t / 1e6).toFixed(2)}M`;
  if (t >= 1e3) return `${(t / 1e3).toFixed(1)}k`;
  return t.toFixed(0);
}
function fmtRate(summary: Summary): string {
  const rate = summary.primary_rate_per_second ?? summary.tokens_per_second;
  if (rate >= 1e9) return `${(rate / 1e9).toFixed(2)}B`;
  if (rate >= 1e6) return `${(rate / 1e6).toFixed(2)}M`;
  if (rate >= 1e3) return `${(rate / 1e3).toFixed(1)}k`;
  return rate.toFixed(0);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-stat">
      <div className="summary-stat-label">{label}</div>
      <div className="summary-stat-value">{value}</div>
    </div>
  );
}

function SummaryRow({ summary }: { summary: Summary }) {
  return (
    <div className="summary-stats">
      <Stat label="Overall Time" value={fmtTime(summary.makespan_us)} />
      <Stat
        label={summary.primary_unit ? `${summary.primary_unit}/sec` : "tok/sec"}
        value={summary.primary_unit ? fmtRate(summary) : fmtToks(summary.tokens_per_second)}
      />
      <Stat label="Effective TFLOPS" value={fmtTflops(summary.effective_tflops)} />
      <Stat label="Hardware TFLOPS" value={fmtTflops(summary.hardware_tflops)} />
      <Stat label="Peak Fast Memory (GB)" value={fmtGb(summary.peak_fast_memory_gb)} />
      <Stat label="Idle %" value={fmtPct(summary.idle_pct)} />
      <Stat label="Recompute %" value={fmtPct(summary.recompute_pct)} />
      <Stat label="From-Slow Util %" value={fmtPct(summary.from_slow_util_pct)} />
      <Stat label="To-Slow Util %" value={fmtPct(summary.to_slow_util_pct)} />
    </div>
  );
}

async function fetchSummary(params: SimulationParams, policy: Policy): Promise<Summary> {
  const body: SimulationParams = {
    ...params,
    planner: { ...params.planner, policy },
  };
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j?.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  const j = await res.json();
  return j.summary as Summary;
}

export function ComparePoliciesPanel({ params, policies }: Props) {
  const [rows, setRows] = useState<Record<string, RowState>>({});
  // Detect open/close via the <details> onToggle; track params snapshot so
  // we don't refire if user toggles the panel without changing inputs.
  const ref = useRef<HTMLDetailsElement>(null);
  const lastFetchedFor = useRef<string | null>(null);
  const paramsKey = JSON.stringify(params);

  // Whenever params change AND the panel is currently open, refetch.
  useEffect(() => {
    const el = ref.current;
    if (!el || !el.open) return;
    if (lastFetchedFor.current === paramsKey) return;
    void runAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey]);

  async function runAll() {
    lastFetchedFor.current = paramsKey;
    setRows((prev) => {
      const next: Record<string, RowState> = { ...prev };
      for (const p of policies) next[p.value] = { kind: "loading" };
      return next;
    });
    await Promise.all(
      policies.map(async (p) => {
        try {
          const summary = await fetchSummary(params, p.value);
          setRows((prev) => ({ ...prev, [p.value]: { kind: "ok", summary } }));
        } catch (e) {
          setRows((prev) => ({
            ...prev,
            [p.value]: { kind: "error", message: e instanceof Error ? e.message : String(e) },
          }));
        }
      })
    );
  }

  function onToggle() {
    const el = ref.current;
    if (!el || !el.open) return;
    if (lastFetchedFor.current === paramsKey) return;
    void runAll();
  }

  return (
    <details ref={ref} className="panel collapsible-panel compare-panel" onToggle={onToggle}>
      <summary className="collapsible-summary">Compare Policies</summary>
      <div className="collapsible-content">
        {policies.map((p) => {
          const state = rows[p.value] ?? { kind: "idle" };
          return (
            <div key={p.value} className="compare-row">
              <div className="compare-row-header">
                <span className="compare-policy-name">{p.label}</span>
                <span className="compare-policy-stem dim">{p.value}</span>
              </div>
              {state.kind === "loading" && <div className="compare-loading dim">Running...</div>}
              {state.kind === "error" && (
                <div className="compare-error">error: {state.message}</div>
              )}
              {state.kind === "ok" && <SummaryRow summary={state.summary} />}
              {state.kind === "idle" && <div className="compare-loading dim">waiting…</div>}
            </div>
          );
        })}
      </div>
    </details>
  );
}
