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

export interface ComputeBlockSummary {
  key: string;
  name: string;
  category: string;
  instance_count: number;
  per_instance_runtime_us: number;
  total_runtime_us: number;
  per_instance_flops: number;
  total_flops: number;
  per_instance_effective_flops: number;
  total_effective_flops: number;
  per_instance_bytes: number;
  total_bytes: number;
  hardware_tflops?: number | null;
  effective_tflops?: number | null;
  bound_by: string;
  subops: SubOpTimingRow[];
  task_ids: string[];
  task_labels: string[];
  metadata?: Record<string, unknown>;
}

export interface Breakdown {
  compute_blocks?: ComputeBlockSummary[];
  fwd: SubOpTimingRow[];
  bwd: SubOpTimingRow[];
  head: SubOpTimingRow[];
  optimizer: SubOpTimingRow[];
  totals_us: {
    layer_fwd: number;
    layer_bwd: number;
    head: number;
    optimizer_step: number;
    layer_recompute?: number;
  };
}

interface Props {
  breakdown: Breakdown | null;
  compact?: boolean;
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

function fmtUs(n: number): string {
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs === 0) return "0";
  const maximumFractionDigits =
    abs < 1 ? 3 :
    abs < 10 ? 3 :
    abs < 100 ? 2 :
    abs < 1000 ? 1 :
    0;
  return n.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  });
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

function Section({
  title,
  rows,
  total_us,
  compact = false,
  aggregateTflops,
  timeLabel,
}: {
  title: string;
  rows: SubOpTimingRow[];
  total_us: number;
  compact?: boolean;
  aggregateTflops?: number | null;
  timeLabel?: string;
}) {
  const sectionTflops = aggregateTflops ?? sectionEffectiveTflops(rows);
  return (
    <div className="breakdown-section">
      <div className="breakdown-section-header">
        <span className="breakdown-section-title">{title}</span>
        <span className="dim">
          {timeLabel ?? `total ${fmtUs(total_us)} µs`}
          {sectionTflops !== null && (
            <> · {fmtTflops(sectionTflops)} effective TFLOP/s</>
          )}
        </span>
      </div>
      <table className={`breakdown-table${compact ? " breakdown-table-compact" : ""}`}>
        <thead>
          <tr>
            <th>op</th>
            <th className="num">flops</th>
            {!compact && <th className="num">eff. flops</th>}
            <th className="num">bytes accessed</th>
            {!compact && <th className="num">math µs</th>}
            {!compact && <th className="num">mem µs</th>}
            <th className="num">time µs</th>
            <th>bound by</th>
            {!compact && <th className="num">eff. TFLOPS</th>}
            {!compact && <th className="num">% of layer</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const totalCount = r.count > 1 ? <span className="dim"> × {r.count}</span> : null;
            const boundBy = r.flops <= 0 ? "memory" : r.bound_by;
            const boundChip = (
              <span className={`bound-chip bound-${boundBy}`}>{boundBy}</span>
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
                <td className="num">{fmtFlops(flopsTotal)}</td>
                {!compact && <td className={effFlopsClass}>{fmtFlops(effFlopsTotal)}</td>}
                <td className="num">{fmtBytes(r.bytes * r.count)}</td>
                {!compact && <td className="num">{r.math_us === null ? "—" : fmtUs(r.math_us * r.count)}</td>}
                {!compact && <td className="num">{fmtUs(r.mem_us * r.count)}</td>}
                <td className="num">{fmtUs(r.total_us)}</td>
                <td>{boundChip}</td>
                {!compact && <td className="num">{fmtTflops(r.effective_tflops)}</td>}
                {!compact && <td className="num">{fmtPct(layerPct)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ComputeBlockBreakdown({ blocks, compact }: { blocks: ComputeBlockSummary[]; compact: boolean }) {
  const visibleBlocks = blocks.filter((block) =>
    block.total_runtime_us > 0 || block.subops.some((subop) => subop.total_us > 0),
  );
  const totalInstances = visibleBlocks.reduce((sum, block) => sum + block.instance_count, 0);
  const totalRuntime = visibleBlocks.reduce((sum, block) => sum + block.total_runtime_us, 0);
  return (
    <>
      <div className="panel-header">
        <h3>Compute Block Breakdown</h3>
        <span className="dim">
          {visibleBlocks.length} blocks · {totalInstances.toLocaleString()} instances · {fmtUs(totalRuntime)} µs
        </span>
      </div>
      <div className="compute-block-list">
        {visibleBlocks.map((block) => {
          const blockTflops = block.effective_tflops ?? sectionEffectiveTflops(block.subops);
          return compact ? (
            <div key={block.key} className="compute-block-card compute-block-card-compact">
              <div className="compute-block-summary compute-block-summary-static">
                <span className="compute-block-name">{block.name}</span>
                <code>{block.key}</code>
                <span className={`bound-chip bound-${block.bound_by}`}>{block.bound_by}</span>
              </div>
              <div className="compute-block-metrics">
                <span>{block.instance_count.toLocaleString()} instances</span>
                <span>{fmtUs(block.per_instance_runtime_us)} µs each</span>
                <span>{fmtUs(block.total_runtime_us)} µs total</span>
                <span>{fmtBytes(block.total_bytes)} moved/read</span>
                <span>{fmtFlops(block.total_effective_flops)} total effective FLOPs</span>
                {blockTflops !== null && <span>{fmtTflops(blockTflops)} effective TFLOP/s</span>}
              </div>
            </div>
          ) : (
            <details key={block.key} className="compute-block-card">
              <summary className="compute-block-summary">
                <span className="compute-block-name">{block.name}</span>
                <code>{block.key}</code>
                <span className={`bound-chip bound-${block.bound_by}`}>{block.bound_by}</span>
              </summary>
              <div className="compute-block-metrics">
                <span>{block.instance_count.toLocaleString()} instances</span>
                <span>{fmtUs(block.per_instance_runtime_us)} µs each</span>
                <span>{fmtUs(block.total_runtime_us)} µs total</span>
                <span>{fmtBytes(block.total_bytes)} moved/read</span>
                <span>{fmtFlops(block.total_effective_flops)} total effective FLOPs</span>
                {blockTflops !== null && <span>{fmtTflops(blockTflops)} effective TFLOP/s</span>}
              </div>
              <Section
                title="Sub-ops"
                rows={block.subops}
                total_us={block.per_instance_runtime_us}
                compact={compact}
                aggregateTflops={blockTflops}
                timeLabel={`per instance ${fmtUs(block.per_instance_runtime_us)} µs`}
              />
            </details>
          );
        })}
      </div>
    </>
  );
}

export function SubOpBreakdownPanel({ breakdown, compact = false }: Props) {
  if (!breakdown) {
    return (
      <div className="panel breakdown-panel">
        <div className="panel-header">
          <h3>Compute Block Breakdown</h3>
        </div>
        <p className="dim">Create a workload to see resolved block timings.</p>
      </div>
    );
  }
  const computeBlocks = breakdown.compute_blocks ?? [];
  if (computeBlocks.length > 0) {
    return (
      <div className="panel breakdown-panel">
        <ComputeBlockBreakdown blocks={computeBlocks} compact={compact} />
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
          fwd {fmtUs(breakdown.totals_us.layer_fwd)} µs
          {" · "}
          bwd {fmtUs(breakdown.totals_us.layer_bwd)} µs
          {" · "}
          head {fmtUs(breakdown.totals_us.head)} µs
          {optimizerStepUs > 0 && (
            <>
              {" · "}
              opt {fmtUs(optimizerStepUs)} µs
            </>
          )}
        </span>
      </div>
      <Section title="Forward (per layer)" rows={breakdown.fwd} total_us={breakdown.totals_us.layer_fwd} compact={compact} />
      <Section title="Head" rows={breakdown.head} total_us={breakdown.totals_us.head} compact={compact} />
      <Section title="Backward (per layer)" rows={breakdown.bwd} total_us={breakdown.totals_us.layer_bwd} compact={compact} />
      {optimizerRows.length > 0 && (
        <Section title="Optimizer Step (per layer)" rows={optimizerRows} total_us={optimizerStepUs} compact={compact} />
      )}
    </div>
  );
}
