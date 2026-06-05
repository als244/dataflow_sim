export interface Summary {
  makespan_us: number;
  total_flops: number;
  total_effective_flops: number;
  tokens_per_second: number;
  effective_tflops: number;
  hardware_tflops: number;
  peak_memory_gb: number;
  idle_pct: number;
  recompute_pct: number;
  ingress_util_pct: number;
  egress_util_pct: number;
}

interface Props {
  summary: Summary | null;
}

function fmtTime(us: number): string {
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(2)} s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(1)} ms`;
  return `${us.toLocaleString()} µs`;
}

function fmtPct(p: number): string {
  return `${p.toFixed(1)}%`;
}

function fmtTflops(t: number): string {
  return t >= 100 ? t.toFixed(0) : t.toFixed(1);
}

function fmtToks(t: number): string {
  if (t >= 1e6) return `${(t / 1e6).toFixed(2)}M`;
  if (t >= 1e3) return `${(t / 1e3).toFixed(1)}k`;
  return t.toFixed(0);
}

function fmtGb(g: number): string {
  if (g >= 100) return g.toFixed(0);
  if (g >= 10) return g.toFixed(1);
  return g.toFixed(2);
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="summary-stat">
      <div className="summary-stat-label">{label}</div>
      <div className="summary-stat-value">{value}</div>
      {sub ? <div className="summary-stat-sub">{sub}</div> : null}
    </div>
  );
}

export function SummaryPanel({ summary }: Props) {
  if (!summary) return null;
  return (
    <div className="panel summary-panel">
      <div className="panel-header">
        <h3>Summary</h3>
      </div>
      <div className="summary-stats">
        <Stat label="overall time" value={fmtTime(summary.makespan_us)} />
        <Stat label="tok/sec" value={fmtToks(summary.tokens_per_second)} />
        <Stat
          label="effective TFLOPS"
          value={fmtTflops(summary.effective_tflops)}
        />
        <Stat
          label="hardware TFLOPS"
          value={fmtTflops(summary.hardware_tflops)}
        />
        <Stat
          label="peak memory (GB)"
          value={fmtGb(summary.peak_memory_gb)}
        />
        <Stat label="idle %" value={fmtPct(summary.idle_pct)} />
        <Stat label="recompute %" value={fmtPct(summary.recompute_pct)} />
        <Stat label="ingress util %" value={fmtPct(summary.ingress_util_pct)} />
        <Stat label="egress util %" value={fmtPct(summary.egress_util_pct)} />
      </div>
    </div>
  );
}
