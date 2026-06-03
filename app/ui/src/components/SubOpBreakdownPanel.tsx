export interface SubOpTimingRow {
  name: string;
  kind: "compute" | "memory";
  flops: number;
  effective_flops: number;
  bytes: number;
  count: number;
  math_us: number | null;
  mem_us: number;
  per_call_us: number;
  per_call_us_exact: number;
  total_us: number;
  bound_by: "compute" | "memory";
  effective_tflops: number | null;
}

export interface Breakdown {
  fwd: SubOpTimingRow[];
  bwd: SubOpTimingRow[];
  head: SubOpTimingRow[];
  optimizer: SubOpTimingRow[];
  totals_us: {
    layer_fwd: number;
    layer_bwd: number;
    head: number;
    optimizer_step: number;
  };
}

interface Props {
  breakdown: Breakdown | null;
}

function fmtFlops(n: number): string {
  if (n === 0) return "—";
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  return n.toString();
}

function fmtPct(p: number): string {
  if (p >= 10) return `${p.toFixed(0)}%`;
  return `${p.toFixed(1)}%`;
}

function fmtBytes(n: number): string {
  if (n === 0) return "—";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(2)} KB`;
  return `${n} B`;
}

function fmtTflops(n: number | null): string {
  if (n === null || n === undefined) return "—";
  return `${n.toFixed(1)}`;
}

function sectionEffectiveTflops(rows: SubOpTimingRow[]): number | null {
  // Use the un-rounded per_call_us_exact so a pure compute-bound section
  // reports exactly peak × eff (no ceil bias accumulated across rows).
  const effFlops = rows.reduce((acc, r) => acc + r.effective_flops * r.count, 0);
  if (effFlops <= 0) return null;
  const totalUsExact = rows.reduce((acc, r) => acc + r.per_call_us_exact * r.count, 0);
  if (totalUsExact <= 0) return null;
  return effFlops / (totalUsExact * 1e6);
}

function Section({ title, rows, total_us }: { title: string; rows: SubOpTimingRow[]; total_us: number }) {
  const sectionTflops = sectionEffectiveTflops(rows);
  return (
    <div className="breakdown-section">
      <div className="breakdown-section-header">
        <span className="breakdown-section-title">{title}</span>
        <span className="dim">
          total {total_us.toLocaleString()} µs
          {sectionTflops !== null && (
            <> · {fmtTflops(sectionTflops)} eff. TFLOPS</>
          )}
        </span>
      </div>
      <table className="breakdown-table">
        <thead>
          <tr>
            <th>op</th>
            <th>kind</th>
            <th className="num">flops</th>
            <th className="num">eff. flops</th>
            <th className="num">bytes</th>
            <th className="num">math µs</th>
            <th className="num">mem µs</th>
            <th className="num">time µs</th>
            <th>bound by</th>
            <th className="num">eff. TFLOPS</th>
            <th className="num">% of layer</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const totalCount = r.count > 1 ? <span className="dim"> × {r.count}</span> : null;
            const isMemKind = r.kind === "memory";
            const boundChip = (
              <span className={`bound-chip bound-${r.bound_by}`}>{r.bound_by}</span>
            );
            const effFlopsTotal = r.effective_flops * r.count;
            const flopsTotal = r.flops * r.count;
            const effFlopsClass = effFlopsTotal !== flopsTotal && effFlopsTotal > 0
              ? "num eff-discount" : "num";
            const layerPct = total_us > 0 ? (r.total_us / total_us) * 100 : 0;
            return (
              <tr key={i}>
                <td>
                  {r.name}
                  {totalCount}
                </td>
                <td>
                  <span className={`kind-chip kind-${r.kind}`}>{r.kind}</span>
                </td>
                <td className="num">{fmtFlops(flopsTotal)}</td>
                <td className={effFlopsClass}>{fmtFlops(effFlopsTotal)}</td>
                <td className="num">{fmtBytes(r.bytes * r.count)}</td>
                <td className="num">{r.math_us === null ? "—" : r.math_us.toLocaleString()}</td>
                <td className="num">{(r.mem_us * r.count).toLocaleString()}</td>
                <td className="num">{r.total_us.toLocaleString()}</td>
                <td>{isMemKind ? <span className="dim">—</span> : boundChip}</td>
                <td className="num">{fmtTflops(r.effective_tflops)}</td>
                <td className="num">{fmtPct(layerPct)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function SubOpBreakdownPanel({ breakdown }: Props) {
  if (!breakdown) {
    return (
      <div className="panel breakdown-panel">
        <div className="panel-header">
          <h3>Op Breakdown</h3>
        </div>
        <p className="dim">no breakdown — submit to see live timings.</p>
      </div>
    );
  }
  const optimizerRows = breakdown.optimizer ?? [];
  const optimizerStepUs = breakdown.totals_us.optimizer_step ?? 0;
  return (
    <div className="panel breakdown-panel">
      <div className="panel-header">
        <h3>Op Breakdown (one representative layer)</h3>
        <span className="dim">
          fwd {breakdown.totals_us.layer_fwd.toLocaleString()} µs
          {" · "}
          bwd {breakdown.totals_us.layer_bwd.toLocaleString()} µs
          {" · "}
          head {breakdown.totals_us.head.toLocaleString()} µs
          {optimizerStepUs > 0 && (
            <>
              {" · "}
              opt {optimizerStepUs.toLocaleString()} µs
            </>
          )}
        </span>
      </div>
      <Section title="forward (per layer)" rows={breakdown.fwd} total_us={breakdown.totals_us.layer_fwd} />
      <Section title="head" rows={breakdown.head} total_us={breakdown.totals_us.head} />
      <Section title="backward (per layer)" rows={breakdown.bwd} total_us={breakdown.totals_us.layer_bwd} />
      {optimizerRows.length > 0 && (
        <Section title="optimizer step (per layer)" rows={optimizerRows} total_us={optimizerStepUs} />
      )}
    </div>
  );
}
