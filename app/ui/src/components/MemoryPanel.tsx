import type { MemoryEntry, MemoryState } from "../types";
import { MemoryBreakdown, TYPE_COLORS } from "./MemoryBreakdown";

const STATE_DISPLAY: Record<MemoryState, { label: string; className: string }> = {
  live: { label: "live", className: "tag-live" },
  reserved: { label: "reserved", className: "tag-reserved" },
  pending_inbound: { label: "→ pending", className: "tag-pending-inbound" },
  inbound: { label: "→ inbound", className: "tag-inbound" },
  pending_outbound: { label: "← pending", className: "tag-pending-outbound" },
  outbound: { label: "← outbound", className: "tag-outbound" },
};

function fmtBytes(n: number): string {
  if (n === 0) return "0 B";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

// Order matches the user-facing dropdown order; empty categories are hidden.
type CategoryKey =
  | "params"
  | "activations"
  | "grads"
  | "optimizer"
  | "other"
  | "outbound"
  | "pending_outbound"
  | "inbound"
  | "reserved";

const CATEGORY_ORDER: CategoryKey[] = [
  "params",
  "activations",
  "grads",
  "optimizer",
  "other",
  "outbound",
  "pending_outbound",
  "inbound",
  "reserved",
];

const CATEGORY_LABEL: Record<CategoryKey, string> = {
  params: "Params",
  activations: "Activations",
  grads: "Grads",
  optimizer: "Optimizer",
  other: "Other",
  outbound: "Outbound",
  pending_outbound: "Pending Outbound",
  inbound: "Inbound",
  reserved: "Reserved",
};

function categorize(m: MemoryEntry): CategoryKey {
  if (m.state === "reserved") return "reserved";
  if (m.state === "outbound") return "outbound";
  if (m.state === "pending_outbound") return "pending_outbound";
  if (m.state === "inbound" || m.state === "pending_inbound") return "inbound";
  // live → bucket by type
  if (m.type === "weight") return "params";
  if (m.type === "activation") return "activations";
  if (m.type === "gradient") return "grads";
  if (m.type === "optimizer") return "optimizer";
  return "other";
}

interface Props {
  title: string;
  memory: MemoryEntry[];
  highlightedIds: Set<string>;
  selectedObjId: string | null;
  onSelectObj: (id: string | null) => void;
}

export function MemoryPanel({
  title,
  memory,
  highlightedIds,
  selectedObjId,
  onSelectObj,
}: Props) {
  const totalSize = memory.reduce((s, m) => s + m.size, 0);

  // Bucket entries by category, then sort each bucket by next_ref_t ascending
  // (nulls last, ties broken by id) — same ordering policy as before.
  const buckets: Record<CategoryKey, MemoryEntry[]> = {
    params: [], activations: [], grads: [], optimizer: [], other: [],
    outbound: [], pending_outbound: [], inbound: [], reserved: [],
  };
  for (const m of memory) buckets[categorize(m)].push(m);
  for (const key of CATEGORY_ORDER) {
    buckets[key].sort((a, b) => {
      if (a.next_ref_t === null && b.next_ref_t === null) return a.id.localeCompare(b.id);
      if (a.next_ref_t === null) return 1;
      if (b.next_ref_t === null) return -1;
      if (a.next_ref_t !== b.next_ref_t) return a.next_ref_t - b.next_ref_t;
      return a.id.localeCompare(b.id);
    });
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        <span className="badge">total {fmtBytes(totalSize)}</span>
        <span className="badge">{memory.length} objs</span>
      </div>
      <MemoryBreakdown memory={memory} />
      {memory.length === 0 ? (
        <p className="dim">empty.</p>
      ) : (
        <div className="mem-cats">
          {CATEGORY_ORDER.map((key) => {
            const rows = buckets[key];
            if (rows.length === 0) return null;
            const catBytes = rows.reduce((s, m) => s + m.size, 0);
            return (
              <details key={key} className="mem-cat">
                <summary className="mem-cat-summary">
                  <span className="mem-cat-name">{CATEGORY_LABEL[key]}</span>
                  <span className="mem-cat-badge">{rows.length}</span>
                  <span className="mem-cat-badge">{fmtBytes(catBytes)}</span>
                </summary>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>id</th>
                      <th>size</th>
                      <th>next ref</th>
                      <th>state</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((m) => {
                      const isHl = highlightedIds.has(m.id);
                      const isSel = selectedObjId === m.id;
                      const stateDisplay = STATE_DISPLAY[m.state];
                      return (
                        <tr
                          key={m.id}
                          className={
                            (isHl ? "row-hl " : "") +
                            (isSel ? "row-sel " : "") +
                            (m.state !== "live" ? "row-not-ready" : "")
                          }
                          onClick={() => onSelectObj(isSel ? null : m.id)}
                        >
                          <td>
                            <span
                              className="type-dot"
                              style={{ background: TYPE_COLORS[m.type] }}
                              title={m.type}
                            />
                            <code>{m.id}</code>
                          </td>
                          <td className="num">{fmtBytes(m.size)}</td>
                          <td className="num">
                            {m.next_ref_t === null ? (
                              <span className="dim">—</span>
                            ) : (
                              m.next_ref_t
                            )}
                          </td>
                          <td>
                            <span className={"tag " + stateDisplay.className}>
                              {stateDisplay.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </details>
            );
          })}
        </div>
      )}
    </div>
  );
}
