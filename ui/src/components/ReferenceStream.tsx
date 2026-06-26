import type { MemoryEntry, Reference } from "../types";

interface Props {
  references: Reference[];
  memory: MemoryEntry[];
  selectedObjId: string | null;
}

export function ReferenceStream({ references, memory, selectedObjId }: Props) {
  // "Ready" = the object has a compute-resident entry in the pool — visible OR
  // reserved (output of the active task counts: its memory is already accounted for).
  const readyOnCompute = new Set(
    memory.filter((m) => m.location === "fast").map((m) => m.id),
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>reference stream</h3>
        <span className="badge">{references.length} upcoming</span>
      </div>
      {references.length === 0 ? (
        <p className="dim">No further references.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>object</th>
              <th>next t</th>
              <th>by task</th>
              <th>ready?</th>
            </tr>
          </thead>
          <tbody>
            {references.map((r, i) => {
              const ready = readyOnCompute.has(r.obj_id);
              return (
                <tr
                  key={r.obj_id}
                  className={selectedObjId === r.obj_id ? "row-sel" : ""}
                >
                  <td className="num dim">{i + 1}</td>
                  <td>
                    <code>{r.obj_id}</code>
                  </td>
                  <td className="num">{r.ref_t}</td>
                  <td>
                    <code>{r.ref_task}</code>
                  </td>
                  <td>
                    {ready ? (
                      <span className="tag tag-ready">yes</span>
                    ) : (
                      <span className="tag tag-not-ready">no</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
