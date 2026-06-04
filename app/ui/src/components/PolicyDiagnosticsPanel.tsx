export interface PressureFitCandidateDiagnostic {
  name: string;
  family: string;
  status: "valid" | "error" | "skipped" | string;
  selected: boolean;
  makespan_us: number | null;
  wall_time_s: number;
  error: string | null;
  pack_h2d: boolean | null;
  extend_h2d: boolean;
  respect_interval_start: boolean;
  latest_h2d: boolean;
  reserve_pressure: number;
  protected_count: number;
  protected_bytes: number;
  seed: string;
}

export interface PressureFitDiagnostics {
  portfolio_mode: string;
  effective_portfolio_mode: string;
  fast_portfolio: boolean;
  planning_time_s: number;
  task_count: number;
  object_count: number;
  device_capacity: number | null;
  candidate_count: number;
  valid_candidate_count: number;
  selected_candidate: string;
  selected_makespan_us: number;
  candidates: PressureFitCandidateDiagnostic[];
}

function fmtTimeUs(us: number | null): string {
  if (us === null) return "-";
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(3)} s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(3)} ms`;
  return `${us.toLocaleString()} us`;
}

function fmtWall(s: number): string {
  if (s >= 1) return `${s.toFixed(3)} s`;
  if (s >= 0.001) return `${(s * 1000).toFixed(2)} ms`;
  return `${(s * 1_000_000).toFixed(0)} us`;
}

function fmtBytes(bytes: number | null): string {
  if (bytes === null || bytes <= 0) return "-";
  const gib = bytes / (1024 ** 3);
  if (gib >= 1) return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
  const mib = bytes / (1024 ** 2);
  if (mib >= 1) return `${mib.toFixed(mib >= 10 ? 1 : 2)} MiB`;
  return `${bytes.toLocaleString()} B`;
}

function candidateKnobs(c: PressureFitCandidateDiagnostic): string {
  const knobs: string[] = [];
  if (c.pack_h2d === true) knobs.push("packed");
  if (c.pack_h2d === false) knobs.push("local");
  if (c.latest_h2d) knobs.push("latest-trigger");
  if (c.respect_interval_start) knobs.push("interval-entry");
  if (c.extend_h2d) knobs.push("extend-H2D");
  if (c.reserve_pressure > 0) knobs.push(`reserve ${fmtBytes(c.reserve_pressure)}`);
  if (c.seed !== "base") knobs.push(c.seed);
  if (c.protected_count > 0) knobs.push(`protect ${c.protected_count}`);
  return knobs.join(", ") || "-";
}

function orderCandidates(
  candidates: PressureFitCandidateDiagnostic[],
): PressureFitCandidateDiagnostic[] {
  return [...candidates].sort((a, b) => {
    if (a.selected !== b.selected) return a.selected ? -1 : 1;
    if (a.status !== b.status) {
      const rank = { valid: 0, error: 1, skipped: 2 } as Record<string, number>;
      return (rank[a.status] ?? 3) - (rank[b.status] ?? 3);
    }
    const am = a.makespan_us ?? Number.POSITIVE_INFINITY;
    const bm = b.makespan_us ?? Number.POSITIVE_INFINITY;
    if (am !== bm) return am - bm;
    return a.name.localeCompare(b.name);
  });
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="diagnostics-stat">
      <div className="diagnostics-stat-label">{label}</div>
      <div className="diagnostics-stat-value">{value}</div>
    </div>
  );
}

export function PolicyDiagnosticsPanel({
  diagnostics,
}: {
  diagnostics: PressureFitDiagnostics | null;
}) {
  if (!diagnostics) return null;
  const candidates = orderCandidates(diagnostics.candidates);

  return (
    <details className="panel collapsible-panel diagnostics-panel">
      <summary className="collapsible-summary">PressureFit Candidate Diagnostics</summary>
      <div className="collapsible-content">
        <div className="diagnostics-stats">
          <Stat
            label="mode"
            value={`${diagnostics.portfolio_mode} -> ${diagnostics.effective_portfolio_mode}`}
          />
          <Stat label="selected" value={diagnostics.selected_candidate} />
          <Stat label="selected makespan" value={fmtTimeUs(diagnostics.selected_makespan_us)} />
          <Stat label="planning wall" value={fmtWall(diagnostics.planning_time_s)} />
          <Stat
            label="valid candidates"
            value={`${diagnostics.valid_candidate_count}/${diagnostics.candidate_count}`}
          />
          <Stat
            label="chain size"
            value={`${diagnostics.task_count} tasks / ${diagnostics.object_count} objects`}
          />
        </div>

        <div className="diagnostics-table-wrap">
          <table className="data-table diagnostics-table">
            <thead>
              <tr>
                <th>candidate</th>
                <th>family</th>
                <th>status</th>
                <th>makespan</th>
                <th>wall</th>
                <th>knobs</th>
                <th>protected bytes</th>
                <th>note</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr
                  key={c.name}
                  className={[
                    c.selected ? "diagnostics-selected" : "",
                    `diagnostics-status-${c.status}`,
                  ].filter(Boolean).join(" ")}
                >
                  <td>
                    <span className="diagnostics-candidate-name">{c.name}</span>
                    {c.selected && <span className="tag diagnostics-winner">winner</span>}
                  </td>
                  <td>{c.family}</td>
                  <td>{c.status}</td>
                  <td className="num">{fmtTimeUs(c.makespan_us)}</td>
                  <td className="num">{fmtWall(c.wall_time_s)}</td>
                  <td>{candidateKnobs(c)}</td>
                  <td className="num">{fmtBytes(c.protected_bytes)}</td>
                  <td className="diagnostics-note" title={c.error ?? ""}>
                    {c.error ?? "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}
