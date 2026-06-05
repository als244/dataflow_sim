import type { SimEvent } from "../types";

interface Props {
  events: SimEvent[];
  index: number;
  setIndex: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
}

export function EventControls({ events, index, setIndex, playing, setPlaying }: Props) {
  const current = events[index];
  const last = events.length - 1;

  return (
    <div className="controls">
      <div className="controls-row">
        <button onClick={() => setIndex(0)} disabled={index === 0}>
          ⏮
        </button>
        <button
          onClick={() => setIndex(Math.max(0, index - 1))}
          disabled={index === 0}
        >
          ◀ step
        </button>
        <button onClick={() => setPlaying(!playing)}>
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <button
          onClick={() => setIndex(Math.min(last, index + 1))}
          disabled={index === last}
        >
          step ▶
        </button>
        <button onClick={() => setIndex(last)} disabled={index === last}>
          ⏭
        </button>
        <div className="event-meta">
          event <strong>{index + 1}</strong> / {events.length} &nbsp;|&nbsp; t = <strong>{current.t}</strong> &nbsp;|&nbsp;{" "}
          <span className={"kind kind-" + current.kind}>{current.kind}</span>
          {current.task_id && (
            <>
              {" "}
              &nbsp;<code>{current.task_id}</code>
            </>
          )}
          {current.kind === "release" && (
            <>
              {" "}
              &nbsp;released: <code>{current.object_ids.join(", ")}</code>
            </>
          )}
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={last}
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="slider"
      />
    </div>
  );
}
