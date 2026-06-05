import type { MemoryEntry, ObjectType } from "../types";

const TYPE_ORDER: ObjectType[] = ["weight", "activation", "gradient", "optimizer", "other"];

export const TYPE_COLORS: Record<ObjectType, string> = {
  weight: "#7c3aed",      // violet
  activation: "#0ea5e9",  // sky
  gradient: "#dc2626",    // red
  optimizer: "#8a5a34",   // brown, matching optimizer-step timeline bars
  other: "#6b7280",       // gray
};

export const TYPE_LABEL: Record<ObjectType, string> = {
  weight: "Params",
  activation: "Activations",
  gradient: "Grads",
  optimizer: "Optimizer",
  other: "Other",
};

interface Props {
  memory: MemoryEntry[];
}

function fmtBytes(n: number): string {
  if (n === 0) return "0 B";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

export function MemoryBreakdown({ memory }: Props) {
  const sums: Record<ObjectType, number> = {
    weight: 0,
    activation: 0,
    gradient: 0,
    optimizer: 0,
    other: 0,
  };
  for (const m of memory) sums[m.type] += m.size;
  const total = TYPE_ORDER.reduce((s, t) => s + sums[t], 0);
  const segments = TYPE_ORDER.filter((t) => sums[t] > 0);

  if (total === 0) return null;

  return (
    <div className="breakdown">
      <div className="breakdown-bar">
        {segments.map((t) => (
          <div
            key={t}
            className="breakdown-seg"
            style={{
              width: `${(sums[t] / total) * 100}%`,
              background: TYPE_COLORS[t],
            }}
            title={`${TYPE_LABEL[t]}: ${fmtBytes(sums[t])} (${((sums[t] / total) * 100).toFixed(0)}%)`}
          />
        ))}
      </div>
      <div className="breakdown-legend">
        {segments.map((t) => (
          <span key={t} className="breakdown-item">
            <span
              className="breakdown-dot"
              style={{ background: TYPE_COLORS[t] }}
            />
            {TYPE_LABEL[t]} <span className="dim">{fmtBytes(sums[t])}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
