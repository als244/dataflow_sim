export interface Summary {
  makespan_us: number;
  total_flops: number;
  total_effective_flops: number;
  tokens_per_second: number;
  primary_unit?: string | null;
  primary_count?: number;
  primary_rate_per_second?: number;
  effective_tflops: number;
  hardware_tflops: number;
  peak_fast_memory_gb: number;
  idle_pct: number;
  recompute_pct: number;
  from_slow_util_pct: number;
  to_slow_util_pct: number;
}

interface Props {
  summary: Summary | null;
}

function fmtTime(us: number): string {
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(2)} s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(1)} ms`;
  if (us >= 100) return `${us.toFixed(1)} µs`;
  if (us >= 1) return `${us.toFixed(3)} µs`;
  return `${us.toFixed(4)} µs`;
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

function fmtRate(t: number): string {
  if (t >= 1e9) return `${(t / 1e9).toFixed(2)}B`;
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
  const primaryUnit = summary.primary_unit;
  const primaryRate = summary.primary_rate_per_second ?? 0;
  return (
    <div className="panel summary-panel">
      <div className="panel-header">
        <h3>Summary</h3>
      </div>
      <div className="summary-stats">
        <Stat label="Overall Time" value={fmtTime(summary.makespan_us)} />
        {primaryUnit ? (
          <Stat label={`${primaryUnit}/sec`} value={fmtRate(primaryRate)} />
        ) : (
          <Stat label="tok/sec" value={fmtToks(summary.tokens_per_second)} />
        )}
        <Stat
          label="Effective TFLOPS"
          value={fmtTflops(summary.effective_tflops)}
        />
        <Stat
          label="Hardware TFLOPS"
          value={fmtTflops(summary.hardware_tflops)}
        />
        <Stat
          label="Peak Fast Memory (GB)"
          value={fmtGb(summary.peak_fast_memory_gb)}
        />
        <Stat label="Idle %" value={fmtPct(summary.idle_pct)} />
        <Stat label="Recompute %" value={fmtPct(summary.recompute_pct)} />
        <Stat label="From-Slow Util %" value={fmtPct(summary.from_slow_util_pct)} />
        <Stat label="To-Slow Util %" value={fmtPct(summary.to_slow_util_pct)} />
      </div>
    </div>
  );
}
