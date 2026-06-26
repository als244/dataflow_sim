export interface PressureFitCandidateDiagnostic {
  name: string;
  status: "valid" | "error" | string;
  selected: boolean;
  makespan_us: number | null;
  wall_time_s: number;
  error: string | null;
  pack_inbound: boolean;
  extend_inbound: boolean;
  respect_interval_start: boolean;
  clamp_inbound: boolean;
}

export interface PressureFitDiagnostics {
  planning_time_s: number;
  task_count: number;
  object_count: number;
  fast_memory_capacity: number | null;
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

function scheduleKnobs(c: PressureFitCandidateDiagnostic): string {
  const knobs: string[] = [];
  if (c.pack_inbound) knobs.push("packed");
  if (c.clamp_inbound) knobs.push("pressure-clamped");
  if (c.extend_inbound) knobs.push("extend-inbound");
  if (c.respect_interval_start) knobs.push("interval-entry");
  return knobs.join(", ") || "latest-safe";
}

function orderCandidates(
  candidates: PressureFitCandidateDiagnostic[],
): PressureFitCandidateDiagnostic[] {
  return [...candidates].sort((a, b) => {
    if (a.selected !== b.selected) return a.selected ? -1 : 1;
    if (a.status !== b.status) {
      const rank = { valid: 0, error: 1 } as Record<string, number>;
      return (rank[a.status] ?? 2) - (rank[b.status] ?? 2);
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
      <summary className="collapsible-summary">PressureFit Schedule Diagnostics</summary>
      <div className="collapsible-content">
        <div className="diagnostics-stats">
          <Stat label="selected" value={diagnostics.selected_candidate} />
          <Stat label="selected makespan" value={fmtTimeUs(diagnostics.selected_makespan_us)} />
          <Stat label="planning wall" value={fmtWall(diagnostics.planning_time_s)} />
          <Stat
            label="valid schedules"
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
                <th>inbound schedule</th>
                <th>status</th>
                <th>makespan</th>
                <th>wall</th>
                <th>knobs</th>
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
                  <td>{c.status}</td>
                  <td className="num">{fmtTimeUs(c.makespan_us)}</td>
                  <td className="num">{fmtWall(c.wall_time_s)}</td>
                  <td>{scheduleKnobs(c)}</td>
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
