import { useEffect, useRef, useState } from "react";
import type { TaskInterval, Track } from "../types";

interface Props {
  intervals: TaskInterval[];
  currentT: number;
  totalDuration: number;
  activeTaskId: string | null;
  hoverTaskId: string | null;
  onHoverTask: (id: string | null) => void;
}

function categoryClass(iv: TaskInterval): string {
  if (iv.track === "from_slow") return "cat-from_slow";
  if (iv.track === "to_slow") return "cat-to_slow";
  const t = iv.task_id;
  if (t.startsWith("r_")) return "cat-recomp";
  if (t.startsWith("b_")) return "cat-bwd";
  if (t.startsWith("f_")) return "cat-fwd";
  if (t === "head" || t.startsWith("head_")) return "cat-head";
  if (t.startsWith("step_")) return "cat-step";
  return "";
}

/** Strip from_slow:/to_slow: prefix AND the "#N" instance suffix on transfer task_ids —
 * the lane already conveys direction, and the suffix is just for React-key
 * uniqueness across re-prefetches/re-offloads of the same object. */
function displayLabel(iv: TaskInterval): string {
  if (iv.track === "from_slow" || iv.track === "to_slow") {
    const colon = iv.task_id.indexOf(":");
    const base = colon >= 0 ? iv.task_id.slice(colon + 1) : iv.task_id;
    const hash = base.indexOf("#");
    return hash >= 0 ? base.slice(0, hash) : base;
  }
  return iv.task_id;
}

const BAR_HEIGHT = 60;
const LANE_GAP = 6;
const AXIS_GAP = 8;
const LABEL_GUTTER = 64;
const BASE_WIDTH = 1100;
const MIN_ZOOM = 1;
const MAX_ZOOM = 120;
const MAX_RENDERED_TICKS = 240;
const DRAG_MIN_PX = 6;     // ignore drags smaller than this

const LANE_ORDER: Track[] = ["from_slow", "compute", "to_slow"];
const LANE_LABELS: Record<Track, string> = {
  from_slow: "From Slow",
  compute: "Compute",
  to_slow: "To Slow",
};

/** Choose unit + decimals based on the tick STEP (µs). At step ≥ 1 s use
 * seconds, step ≥ 1 ms use ms, otherwise µs. Decimals chosen so adjacent
 * ticks always differ (3 sig figs of the step). */
function fmtTime(us: number, stepUs: number): string {
  if (us === 0) return "0";
  // Threshold to enter "seconds" unit: step ≥ 100 ms (so values like 0.5s
  // read more naturally than 500ms). Same logic for ms ≥ 100 µs.
  let unitUs: number;
  let unit: string;
  if (stepUs >= 100_000) {
    unitUs = 1_000_000;
    unit = "s";
  } else if (stepUs >= 100) {
    unitUs = 1_000;
    unit = "ms";
  } else {
    unitUs = 1;
    unit = "µs";
  }
  const v = us / unitUs;
  // Decimals needed so neighboring ticks differ: ceil(log10(unitUs / stepUs)).
  const ratio = unitUs / stepUs;
  const decimals = ratio <= 1 ? 0 : Math.min(6, Math.ceil(Math.log10(ratio)));
  return `${v.toFixed(decimals)}${unit}`;
}

interface TooltipTimeUnit {
  scaleUs: number;
  label: string;
  decimals: number;
}

function chooseTooltipTimeUnit(startUs: number, endUs: number): TooltipTimeUnit {
  const maxUs = Math.max(Math.abs(startUs), Math.abs(endUs));
  const spanUs = Math.abs(endUs - startUs);
  if (maxUs >= 1_000_000 && spanUs >= 1_000) {
    return { scaleUs: 1_000_000, label: "s", decimals: 3 };
  }
  if (maxUs >= 1_000) {
    return { scaleUs: 1_000, label: "ms", decimals: 3 };
  }
  return { scaleUs: 1, label: "µs", decimals: 0 };
}

function fmtScaledTime(us: number, unit: TooltipTimeUnit): string {
  const value = us / unit.scaleUs;
  if (unit.decimals === 0) return Math.round(value).toLocaleString();
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: unit.decimals,
  });
}

function fmtDurationAndRange(startUs: number, endUs: number): string {
  const unit = chooseTooltipTimeUnit(startUs, endUs);
  const duration = fmtScaledTime(Math.max(0, endUs - startUs), unit);
  const start = fmtScaledTime(startUs, unit);
  const end = fmtScaledTime(endUs, unit);
  const range = start === end ? `${start}` : `${start}-${end}`;
  return `${duration} ${unit.label}, ${range} ${unit.label}`;
}

function niceTickStep(durationUs: number, plotWidthPx: number): number {
  const minPx = 70;
  const maxTicks = Math.min(
    MAX_RENDERED_TICKS,
    Math.max(2, Math.floor(plotWidthPx / minPx)),
  );
  const rawStep = durationUs / maxTicks;
  if (rawStep <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let nice: number;
  if (norm <= 1) nice = 1;
  else if (norm <= 2) nice = 2;
  else if (norm <= 5) nice = 5;
  else nice = 10;
  return nice * mag;
}

function fmtZoom(z: number): string {
  if (z >= 100) return `${z.toFixed(0)}×`;
  if (z >= 10) return `${z.toFixed(1)}×`;
  return `${z.toFixed(2)}×`;
}

export function ComputeTimeline({
  intervals,
  currentT,
  totalDuration,
  activeTaskId,
  hoverTaskId,
  onHoverTask,
}: Props) {
  const [zoom, setZoom] = useState(1.0);
  const [drag, setDrag] = useState<{ startX: number; currentX: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [viewportWidth, setViewportWidth] = useState(BASE_WIDTH);

  const contentWidth = viewportWidth * zoom;
  const pxPerUnit = contentWidth / Math.max(totalDuration, 1);
  const visible = intervals.filter((iv) => iv.end > iv.start);

  const laneTop: Record<Track, number> = {
    from_slow: 0,
    compute: BAR_HEIGHT + LANE_GAP,
    to_slow: 2 * (BAR_HEIGHT + LANE_GAP),
  };
  const lanesHeight = 3 * BAR_HEIGHT + 2 * LANE_GAP;
  const trackHeight = lanesHeight + AXIS_GAP + 22;

  const tickStep = niceTickStep(totalDuration, contentWidth);
  const tickCount = Math.floor(totalDuration / tickStep) + 1;
  const cursorX = Math.min(
    Math.max(0, currentT * pxPerUnit),
    Math.max(0, contentWidth - 2),
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const updateWidth = () => {
      setViewportWidth(Math.max(1, Math.floor(el.clientWidth)));
    };
    updateWidth();
    const ResizeObserverCtor =
      typeof ResizeObserver !== "undefined" ? ResizeObserver : null;
    if (ResizeObserverCtor) {
      const ro = new ResizeObserverCtor(updateWidth);
      ro.observe(el);
      return () => ro.disconnect();
    }
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // ---- drag-to-zoom: select a sub-range with the mouse, zoom to fill viewport. ----
  useEffect(() => {
    if (!drag) return;
    function onMove(e: MouseEvent) {
      if (!innerRef.current) return;
      const rect = innerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(contentWidth, e.clientX - rect.left));
      setDrag((d) => (d ? { ...d, currentX: x } : null));
    }
    function onUp(e: MouseEvent) {
      if (!innerRef.current) return;
      const rect = innerRef.current.getBoundingClientRect();
      const endX = Math.max(0, Math.min(contentWidth, e.clientX - rect.left));
      setDrag((d) => {
        if (!d) return null;
        const x1 = Math.min(d.startX, endX);
        const x2 = Math.max(d.startX, endX);
        if (x2 - x1 < DRAG_MIN_PX) return null;
        const t1 = x1 / pxPerUnit;
        const t2 = x2 / pxPerUnit;
        const viewportW = scrollRef.current?.clientWidth ?? viewportWidth;
        const newPxPerUnit = viewportW / Math.max(t2 - t1, 1);
        const newContent = newPxPerUnit * totalDuration;
        const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, newContent / viewportW));
        setZoom(newZoom);
        requestAnimationFrame(() => {
          if (scrollRef.current) {
            scrollRef.current.scrollLeft = t1 * (viewportW * newZoom) / totalDuration;
          }
        });
        return null;
      });
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [drag, pxPerUnit, contentWidth, totalDuration, viewportWidth]);

  function onMouseDownInner(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0) return;
    if (!innerRef.current) return;
    const rect = innerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setDrag({ startX: x, currentX: x });
    e.preventDefault();
  }

  function bumpZoom(factor: number) {
    setZoom((z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * factor)));
  }

  return (
    <div className="panel timeline-panel">
      <div className="timeline-header">
        <h3>Compute &amp; Communication Timelines</h3>
        <span className="timeline-label dim">From Slow · Compute · To Slow</span>
        <div className="timeline-zoom">
          <button
            className="zoom-btn"
            onClick={() => bumpZoom(1 / 1.5)}
            disabled={zoom <= MIN_ZOOM + 1e-3}
            title="zoom out"
          >
            −
          </button>
          <span className="zoom-label">{fmtZoom(zoom)}</span>
          <button
            className="zoom-btn"
            onClick={() => bumpZoom(1.5)}
            disabled={zoom >= MAX_ZOOM - 1e-3}
            title="zoom in"
          >
            +
          </button>
          <button
            className="zoom-btn zoom-fit"
            onClick={() => {
              setZoom(1);
              requestAnimationFrame(() => {
                if (scrollRef.current) scrollRef.current.scrollLeft = 0;
              });
            }}
            disabled={zoom === 1}
            title="fit to width"
          >
            fit
          </button>
        </div>
      </div>
      <div className="timeline-body" style={{ height: trackHeight }}>
        <div className="timeline-lane-labels" style={{ width: LABEL_GUTTER, height: lanesHeight }}>
          {LANE_ORDER.map((lane) => (
            <div
              key={lane}
              className="lane-label-sticky"
              style={{ top: laneTop[lane] + BAR_HEIGHT / 2 - 8 }}
            >
              {LANE_LABELS[lane]}
            </div>
          ))}
        </div>
        <div className="timeline-scroll" ref={scrollRef}>
          <div
            ref={innerRef}
            className="timeline-bars-inner"
            style={{ width: contentWidth, height: trackHeight, cursor: "crosshair" }}
            onMouseDown={onMouseDownInner}
          >
            {visible.map((iv) => {
              const isActive = iv.task_id === activeTaskId;
              const isHover = iv.task_id === hoverTaskId;
              const rawLeft = iv.start * pxPerUnit;
              const visualWidth = Math.max(2, (iv.end - iv.start) * pxPerUnit);
              const left = Math.min(rawLeft, Math.max(0, contentWidth - visualWidth));
              const width = Math.min(visualWidth, Math.max(0, contentWidth - left));
              return (
                <div
                  key={iv.task_id}
                  className={
                    "timeline-bar " +
                    categoryClass(iv) +
                    (isActive ? " active" : "") +
                    (isHover ? " hover" : "")
                  }
                  style={{
                    left,
                    width,
                    top: laneTop[iv.track],
                    height: BAR_HEIGHT,
                  }}
                  onMouseEnter={() => onHoverTask(iv.task_id)}
                  onMouseLeave={() => onHoverTask(null)}
                  title={`${displayLabel(iv)} [${fmtDurationAndRange(iv.start, iv.end)}]`}
                >
                  <span className="bar-label">{displayLabel(iv)}</span>
                </div>
              );
            })}
            <div
              className="timeline-cursor"
              style={{ left: cursorX, height: lanesHeight }}
            />
            {drag && Math.abs(drag.currentX - drag.startX) >= 1 && (
              <div
                className="timeline-drag-rect"
                style={{
                  left: Math.min(drag.startX, drag.currentX),
                  width: Math.abs(drag.currentX - drag.startX),
                  top: 0,
                  height: lanesHeight,
                }}
              />
            )}
            <div className="timeline-axis" style={{ top: lanesHeight + AXIS_GAP }}>
              {Array.from({ length: tickCount }, (_, i) => {
                const tv = i * tickStep;
                return (
                  <div
                    key={i}
                    className="axis-tick"
                    style={{ left: tv * pxPerUnit }}
                  >
                    {fmtTime(tv, tickStep)}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
