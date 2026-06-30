interface Object_ {
  id: string;
  size: number;
  location: "backing" | "fast";
  type: string;
}

interface OutputAlloc {
  id: string;
  size: number;
  location: "backing" | "fast";
  type: string;
}

interface TransferTrigger {
  obj_id: string;
  runtime: number | null;
}

interface Task {
  id: string;
  inputs: string[];
  outputs: OutputAlloc[];
  runtime: number;
  releases_after: string[];
  offload_after: TransferTrigger[];
  prefetch_after: TransferTrigger[];
  mutates_inputs: string[];
}

export interface AnnotatedChain {
  initial_memory: Object_[];
  tasks: Task[];
  fast_memory_capacity: number | null;
  backing_memory_capacity: number | null;
  bandwidth_from_slow: number | null;
  bandwidth_to_slow: number | null;
}

function fmtBytes(n: number): string {
  if (n === 0) return "0 B";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

function fmtTime(us: number): string {
  if (us === 0) return "0µs";
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(2)}s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(1)}ms`;
  return `${us}µs`;
}

type PlanTone = "input" | "mutate" | "output" | "release" | "offload" | "prefetch";

function ObjList({ items, tone }: { items: string[]; tone: PlanTone }) {
  if (items.length === 0) return <span className="dim">—</span>;
  return (
    <span className="plan-objs">
      {items.map((id, i) => (
        <span key={i} className={`plan-obj plan-obj-${tone}`}>{id}</span>
      ))}
    </span>
  );
}

function TriggerList({ items, tone }: { items: TransferTrigger[]; tone: PlanTone }) {
  if (items.length === 0) return <span className="dim">—</span>;
  return (
    <span className="plan-objs">
      {items.map((t, i) => (
        <span key={i} className={`plan-obj plan-obj-${tone}`} title={t.runtime != null ? `runtime override: ${t.runtime}` : undefined}>
          {t.obj_id}
        </span>
      ))}
    </span>
  );
}

interface Props {
  chain: AnnotatedChain | null;
  title?: string;
  emptyText?: string;
  exportFilename?: string;
}

function filenameFromTitle(title: string): string {
  const stem = title
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "plan";
  return `${stem}.taskchain.json`;
}

function downloadJson(filename: string, value: unknown) {
  const blob = new Blob([JSON.stringify(value, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function AnnotatedPlanPanel({
  chain,
  title = "Annotated Plan",
  emptyText = "No plan yet.",
  exportFilename,
}: Props) {
  if (!chain) {
    return (
      <div className="panel plan-panel">
        <p className="dim">{emptyText}</p>
      </div>
    );
  }
  const preplaced = chain.initial_memory.filter((o) => o.location === "fast");
  const showModelTrainingNotation = chain.tasks.some((t) =>
    /^(?:[frb]_\d+_\d+_\d+|head_(?:fwd|bwd)_\d+_\d+|head_\d+_\d+|step_\d+_\d+)$/.test(t.id),
  );
  const filename = exportFilename ?? filenameFromTitle(title);
  return (
    <details className="panel plan-panel">
      <summary className="plan-summary">
        {title}
        <span className="dim plan-summary-meta">{chain.tasks.length} tasks</span>
        <button
          type="button"
          className="reset-btn plan-export-btn"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            downloadJson(filename, chain);
          }}
        >
          Export JSON
        </button>
      </summary>
      {showModelTrainingNotation && (
        <div className="plan-notation-note">
          <strong>Model-training notation:</strong>{" "}
          <code>f_k_j_i</code> is layer forward, <code>r_k_j_i</code> is a recompute task, and{" "}
          <code>b_k_j_i</code> is layer backward. Here <code>k</code> is the training step, <code>j</code> is the gradient accumulation round, and{" "}
          <code>i</code> is the layer index. <code>head_fwd_k_j</code> runs head/loss forward,{" "}
          <code>head_bwd_k_j</code> runs head backward, and <code>step_k_i</code> applies the optimizer update.
        </div>
      )}
      <div className="plan-content">
        {preplaced.length > 0 && (
          <div className="plan-section">
            <div className="plan-section-title">Initial Fast Placement ({preplaced.length})</div>
            <div className="plan-objs">
              {preplaced.map((o) => (
                <span key={o.id} className="plan-obj plan-obj-preplaced" title={`${o.type} · ${fmtBytes(o.size)}`}>
                  {o.id}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="plan-section">
          <div className="plan-section-title">Tasks ({chain.tasks.length})</div>
          <div className="plan-tasks">
            {chain.tasks.map((t) => {
              const outIds = t.outputs.map((o) => o.id);
              const hasTriggers = (
                t.releases_after.length > 0
                || t.offload_after.length > 0
                || t.prefetch_after.length > 0
              );
              return (
                <div key={t.id} className={`plan-task${hasTriggers ? "" : " plan-task-bare"}`}>
                  <div className="plan-task-head">
                    <span className="plan-task-id">{t.id}</span>
                    {t.runtime > 0 && <span className="plan-task-runtime">{fmtTime(t.runtime)}</span>}
                  </div>
                  <div className="plan-task-row">
                    <span className="plan-task-key">Input</span>
                    <ObjList items={t.inputs} tone="input" />
                  </div>
                  {t.mutates_inputs.length > 0 && (
                    <div className="plan-task-row">
                      <span className="plan-task-key">Mutate</span>
                      <ObjList items={t.mutates_inputs} tone="mutate" />
                    </div>
                  )}
                  <div className="plan-task-row">
                    <span className="plan-task-key">Output</span>
                    <ObjList items={outIds} tone="output" />
                  </div>
                  {hasTriggers && (
                    <>
                      <div className="plan-task-divider" />
                      <div className="plan-task-triggers-label">Completion Triggers</div>
                      {t.releases_after.length > 0 && (
                        <div className="plan-task-row">
                          <span className="plan-task-key">Release</span>
                          <ObjList items={t.releases_after} tone="release" />
                        </div>
                      )}
                      {t.offload_after.length > 0 && (
                        <div className="plan-task-row">
                          <span className="plan-task-key">Offload</span>
                          <TriggerList items={t.offload_after} tone="offload" />
                        </div>
                      )}
                      {t.prefetch_after.length > 0 && (
                        <div className="plan-task-row">
                          <span className="plan-task-key">Prefetch</span>
                          <TriggerList items={t.prefetch_after} tone="prefetch" />
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </details>
  );
}
