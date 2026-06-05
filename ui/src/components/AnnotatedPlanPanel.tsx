interface Object_ {
  id: string;
  size: number;
  location: "host" | "device";
  type: string;
}

interface OutputAlloc {
  id: string;
  size: number;
  location: "host" | "device";
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
  device_capacity: number | null;
  host_capacity: number | null;
  bandwidth_h2d: number | null;
  bandwidth_d2h: number | null;
}

function fmtBytes(n: number): string {
  if (n === 0) return "0 B";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

function fmtTime(us: number): string {
  if (us === 0) return "0";
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
}

export function AnnotatedPlanPanel({ chain }: Props) {
  if (!chain) return null;
  const preplaced = chain.initial_memory.filter((o) => o.location === "device");
  return (
    <details className="panel plan-panel">
      <summary className="plan-summary">
        annotated plan
        <span className="dim plan-summary-meta">{chain.tasks.length} tasks</span>
      </summary>
      <div className="plan-content">
        {preplaced.length > 0 && (
          <div className="plan-section">
            <div className="plan-section-title">Initial Device Placement ({preplaced.length})</div>
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
          <div className="plan-section-title">tasks ({chain.tasks.length})</div>
          <div className="plan-tasks">
            {chain.tasks.map((t) => {
              const outIds = t.outputs.map((o) => o.id);
              const hasTriggers = t.releases_after.length || t.offload_after.length || t.prefetch_after.length;
              return (
                <div key={t.id} className={`plan-task${hasTriggers ? "" : " plan-task-bare"}`}>
                  <div className="plan-task-head">
                    <span className="plan-task-id">{t.id}</span>
                    <span className="plan-task-runtime">{fmtTime(t.runtime)}</span>
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
