import { useEffect, useMemo, useRef, useState } from "react";
import type { EventLog, MemoryEntry, MemoryState, ObjectType } from "../types";
import { TYPE_COLORS } from "./MemoryBreakdown";

interface Props {
  log: EventLog | null;
  fastMemoryCapacityGb: number | null;
  currentT: number | null;
}

/** Memory bands. Live/reserved bytes are bucketed by `type`; transient
 * transfer states get their own bands. Pending_inbound bytes are merged
 * into `inbound` (in the simulator both occupy fast memory the moment
 * the trigger fires, so the distinction would only be cosmetic). */
type BandKey =
  | ObjectType
  | "inbound"
  | "outbound"
  | "pending_outbound";

// User-requested bottom-up stack: inbound first, then params (weight),
// activations, gradients, then outbound and pending_outbound on top.
// Optimizer/other are kept for completeness but hidden when empty.
const STACK_ORDER: BandKey[] = [
  "inbound",
  "weight",
  "activation",
  "gradient",
  "optimizer",
  "other",
  "outbound",
  "pending_outbound",
];

// Match the timeline's transfer colors for visual consistency.
const FROM_SLOW_COLOR = "#2ee6a6";
const TO_SLOW_COLOR = "#e36bff";

const BAND_FILL: Record<BandKey, { color: string; opacity: number }> = {
  weight:           { color: TYPE_COLORS.weight,     opacity: 0.55 },
  gradient:         { color: "#f43f5e",              opacity: 0.72 },
  activation:       { color: TYPE_COLORS.activation, opacity: 0.55 },
  optimizer:        { color: TYPE_COLORS.optimizer,  opacity: 0.55 },
  other:            { color: TYPE_COLORS.other,      opacity: 0.55 },
  inbound:          { color: FROM_SLOW_COLOR,         opacity: 0.65 },
  outbound:         { color: TO_SLOW_COLOR,           opacity: 0.65 },
  pending_outbound: { color: "#facc15",              opacity: 0.45 },
};

const BAND_LABEL: Record<BandKey, string> = {
  weight: "Params",
  gradient: "Grads",
  activation: "Activations",
  optimizer: "Optimizer",
  other: "Other",
  inbound: "Inbound",
  outbound: "Outbound",
  pending_outbound: "Pending Outbound",
};

function bandKeyForEntry(m: MemoryEntry): BandKey | null {
  if (m.location !== "fast") return null;
  const s: MemoryState = m.state;
  if (s === "pending_inbound" || s === "inbound") return "inbound";
  if (s === "outbound") return "outbound";
  if (s === "pending_outbound") return "pending_outbound";
  // live or reserved → bucket by type
  return m.type;
}

interface Point {
  t: number;
  sumByBand: Record<BandKey, number>;
  cumByBand: Record<BandKey, number>; // cumulative top edge for each band
  totalBytes: number;
}

interface HoverState {
  index: number;
  x: number;
  y: number;
  clientX: number;
  clientY: number;
}

const BASE_WIDTH = 1100;
const Y_AXIS_W = 70;
const PLOT_H = 220;
const PAD_TOP = 14;
const PAD_BOT = 36;
const PLOT_AREA_H = PLOT_H - PAD_TOP - PAD_BOT;
const MIN_ZOOM = 1;
const MAX_ZOOM = 120;
const MAX_RENDERED_TICKS = 240;
const DRAG_MIN_PX = 6;

function fmtBytesGb(n: number): string {
  if (n < 1024 ** 3) return `${(n / (1024 ** 2)).toFixed(0)} MB`;
  return `${(n / (1024 ** 3)).toFixed(1)} GB`;
}

function fmtTime(us: number, stepUs: number): string {
  if (us === 0) return "0";
  let unitUs: number, unit: string;
  if (stepUs >= 100_000) { unitUs = 1_000_000; unit = "s"; }
  else if (stepUs >= 100) { unitUs = 1_000; unit = "ms"; }
  else { unitUs = 1; unit = "µs"; }
  const ratio = unitUs / stepUs;
  const decimals = ratio <= 1 ? 0 : Math.min(6, Math.ceil(Math.log10(ratio)));
  return `${(us / unitUs).toFixed(decimals)}${unit}`;
}

function fmtHoverTime(us: number): string {
  if (us >= 1_000_000) return `${(us / 1_000_000).toFixed(3)} s`;
  if (us >= 1_000) return `${(us / 1_000).toFixed(2)} ms`;
  return `${Math.round(us).toLocaleString()} µs`;
}

function fmtPct(part: number, total: number): string {
  if (total <= 0) return "0%";
  const pct = (part / total) * 100;
  if (pct < 1 && pct > 0) return "<1%";
  return `${pct.toFixed(0)}%`;
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

function nearestPointIndex(points: Point[], t: number): number {
  if (points.length <= 1) return 0;
  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (points[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  const prev = Math.max(0, lo - 1);
  return Math.abs(points[lo].t - t) < Math.abs(points[prev].t - t) ? lo : prev;
}

const ALL_KEYS_AS_RECORD = (): Record<BandKey, number> => ({
  weight: 0, gradient: 0, activation: 0, optimizer: 0, other: 0,
  inbound: 0, outbound: 0, pending_outbound: 0,
});

function pointFromBandBytes(t: number, bandBytes: Record<string, number>): Point {
  const sumByBand = ALL_KEYS_AS_RECORD();
  for (const key of STACK_ORDER) {
    sumByBand[key] = bandBytes[key] ?? 0;
  }
  const cumByBand = ALL_KEYS_AS_RECORD();
  let running = 0;
  for (const key of STACK_ORDER) {
    running += sumByBand[key];
    cumByBand[key] = running;
  }
  return { t, sumByBand, cumByBand, totalBytes: running };
}

export function MemoryTimelinePanel({ log, fastMemoryCapacityGb, currentT }: Props) {
  const points: Point[] = useMemo(() => {
    if (!log) return [];
    if (log.events.length === 0) {
      return (log.memory_trace ?? []).map((p) => (
        pointFromBandBytes(p.t, p.fast_bytes_by_band)
      ));
    }
    return log.events.map((ev) => {
      const sumByBand = ALL_KEYS_AS_RECORD();
      for (const m of ev.snapshot.memory) {
        const k = bandKeyForEntry(m);
        if (k !== null) sumByBand[k] += m.size;
      }
      const cumByBand = ALL_KEYS_AS_RECORD();
      let running = 0;
      for (const key of STACK_ORDER) {
        running += sumByBand[key];
        cumByBand[key] = running;
      }
      return { t: ev.t, sumByBand, cumByBand, totalBytes: running };
    });
  }, [log]);

  const tMax = useMemo(
    () => (points.length ? Math.max(...points.map((p) => p.t), 1) : 1),
    [points],
  );
  const peakBytes = useMemo(
    () => (points.length ? Math.max(...points.map((p) => p.totalBytes)) : 0),
    [points],
  );

  // --- zoom + drag-to-zoom ---
  const [zoom, setZoom] = useState(1.0);
  const [drag, setDrag] = useState<{ startX: number; currentX: number } | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [viewportWidth, setViewportWidth] = useState(BASE_WIDTH);

  const contentWidth = viewportWidth * zoom;
  const xScale = (t: number) => (t / tMax) * contentWidth;

  const capBytes = fastMemoryCapacityGb ? fastMemoryCapacityGb * 1024 ** 3 : null;
  const yMax = capBytes
    ? Math.max(peakBytes, capBytes) * 1.02
    : (peakBytes || 1) * 1.1;
  const yScale = (b: number) => PAD_TOP + PLOT_AREA_H - (b / yMax) * PLOT_AREA_H;

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
        const pxPerUnit = contentWidth / Math.max(tMax, 1);
        const t1 = x1 / pxPerUnit;
        const t2 = x2 / pxPerUnit;
        const viewportW = scrollRef.current?.clientWidth ?? viewportWidth;
        const newPxPerUnit = viewportW / Math.max(t2 - t1, 1);
        const newContent = newPxPerUnit * tMax;
        const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, newContent / viewportW));
        setZoom(newZoom);
        requestAnimationFrame(() => {
          if (scrollRef.current) {
            scrollRef.current.scrollLeft = t1 * (viewportW * newZoom) / tMax;
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
  }, [drag, contentWidth, tMax, viewportWidth]);

  function onMouseDownInner(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0 || !innerRef.current) return;
    const rect = innerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHover(null);
    setDrag({ startX: x, currentX: x });
    e.preventDefault();
  }
  function onMouseMoveInner(e: React.MouseEvent<HTMLDivElement>) {
    if (drag || !innerRef.current || points.length === 0) return;
    const rect = innerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(contentWidth, e.clientX - rect.left));
    const y = Math.max(PAD_TOP, Math.min(PLOT_H - PAD_BOT, e.clientY - rect.top));
    const t = (x / Math.max(contentWidth, 1)) * tMax;
    const index = nearestPointIndex(points, t);
    setHover((h) => (
      h && h.index === index && Math.abs(h.x - x) < 2 && Math.abs(h.y - y) < 2
        ? h
        : { index, x, y, clientX: e.clientX, clientY: e.clientY }
    ));
  }
  function bumpZoom(factor: number) {
    setZoom((z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * factor)));
  }

  if (!log || points.length === 0) return null;

  const tickStep = niceTickStep(tMax, contentWidth);
  const tickCount = Math.floor(tMax / tickStep) + 1;
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);

  // Build a closed step-path for one band: top edge (running cumulative through
  // this band) then bottom edge (running cumulative through the band BELOW).
  function stepPath(bandIdx: number): string {
    const key = STACK_ORDER[bandIdx];
    const belowKey = bandIdx > 0 ? STACK_ORDER[bandIdx - 1] : null;
    const topAt = (p: Point) => p.cumByBand[key];
    const botAt = (p: Point) => (belowKey ? p.cumByBand[belowKey] : 0);

    const top: string[] = [];
    const bot: string[] = [];
    top.push(`M ${xScale(0)} ${yScale(topAt(points[0]))}`);
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (i > 0) top.push(`L ${xScale(p.t)} ${yScale(topAt(points[i - 1]))}`);
      top.push(`L ${xScale(p.t)} ${yScale(topAt(p))}`);
    }
    top.push(`L ${xScale(tMax)} ${yScale(topAt(points[points.length - 1]))}`);
    bot.push(`L ${xScale(tMax)} ${yScale(botAt(points[points.length - 1]))}`);
    for (let i = points.length - 1; i >= 0; i--) {
      const p = points[i];
      bot.push(`L ${xScale(p.t)} ${yScale(botAt(p))}`);
      if (i > 0) bot.push(`L ${xScale(p.t)} ${yScale(botAt(points[i - 1]))}`);
    }
    bot.push(`L ${xScale(0)} ${yScale(botAt(points[0]))}`);
    bot.push("Z");
    return top.join(" ") + " " + bot.join(" ");
  }

  // Only render bands that ever have nonzero bytes (keeps legend tidy).
  const activeBands = STACK_ORDER.map((key, i) => ({ key, i }))
    .filter(({ key }) => points.some((p) => p.sumByBand[key] > 0));
  const hoverPoint = hover ? points[hover.index] : null;
  const hoverRows = hoverPoint
    ? STACK_ORDER
        .map((key) => ({ key, bytes: hoverPoint.sumByBand[key] }))
        .filter((row) => row.bytes > 0)
    : [];
  const tooltipWidth = 280;
  const tooltipHeight = Math.min(320, 58 + hoverRows.length * 22);
  const viewportW = typeof window !== "undefined" ? window.innerWidth : 1280;
  const viewportH = typeof window !== "undefined" ? window.innerHeight : 720;
  const tooltipLeft = hover
    ? Math.min(
        Math.max(12, hover.clientX + 14),
        Math.max(12, viewportW - tooltipWidth - 12),
      )
    : 0;
  const tooltipTop = hover
    ? Math.min(
        Math.max(12, hover.clientY + 14),
        Math.max(12, viewportH - tooltipHeight - 12),
      )
    : 0;

  return (
    <div className="panel memtl-panel">
      <div className="panel-header">
        <h3>Fast Memory Over Time</h3>
        <span className="memtl-peak dim">
          peak {fmtBytesGb(peakBytes)}
          {capBytes ? ` · cap ${fmtBytesGb(capBytes)}` : ""}
        </span>
        <div className="timeline-zoom">
          <button className="zoom-btn" onClick={() => bumpZoom(1 / 1.5)} disabled={zoom <= MIN_ZOOM + 1e-3} title="zoom out">−</button>
          <span className="zoom-label">{fmtZoom(zoom)}</span>
          <button className="zoom-btn" onClick={() => bumpZoom(1.5)} disabled={zoom >= MAX_ZOOM - 1e-3} title="zoom in">+</button>
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

      {/* Sticky y-axis on the left, scrolling band area on the right. */}
      <div className="memtl-body">
        <svg className="memtl-yaxis" width={Y_AXIS_W} height={PLOT_H}>
          {yTicks.map((tv, i) => (
            <line
              key={`g${i}`}
              x1={Y_AXIS_W - 4}
              x2={Y_AXIS_W}
              y1={yScale(tv)}
              y2={yScale(tv)}
              className="memtl-grid"
            />
          ))}
          {yTicks.map((tv, i) => (
            <text
              key={`yl${i}`}
              x={Y_AXIS_W - 8}
              y={yScale(tv) + 4}
              textAnchor="end"
              className="memtl-axis-label"
            >
              {fmtBytesGb(tv)}
            </text>
          ))}
        </svg>
        <div
          ref={scrollRef}
          className="timeline-scroll"
          style={{ flex: 1, overflowX: "auto" }}
        >
          <div
            ref={innerRef}
            className="memtl-bars-inner"
            style={{ width: contentWidth, height: PLOT_H, cursor: "crosshair" }}
            onMouseDown={onMouseDownInner}
            onMouseMove={onMouseMoveInner}
            onMouseLeave={() => setHover(null)}
          >
            <svg
              className="memtl-svg"
              width={contentWidth}
              height={PLOT_H}
            >
              {/* horizontal gridlines extending across the scroll area */}
              {yTicks.map((tv, i) => (
                <line
                  key={`g${i}`}
                  x1={0}
                  x2={contentWidth}
                  y1={yScale(tv)}
                  y2={yScale(tv)}
                  className="memtl-grid"
                />
              ))}
              {/* capacity line */}
              {capBytes !== null && capBytes <= yMax && (
                <line
                  x1={0}
                  x2={contentWidth}
                  y1={yScale(capBytes)}
                  y2={yScale(capBytes)}
                  className="memtl-cap"
                  aria-label="Fast memory budget"
                />
              )}
              {/* stacked bands */}
              {activeBands.map(({ key, i }) => {
                const fill = BAND_FILL[key];
                return (
                  <path
                    key={key}
                    d={stepPath(i)}
                    fill={fill.color}
                    fillOpacity={fill.opacity}
                    stroke="none"
                  />
                );
              })}
              {/* x ticks */}
              {Array.from({ length: tickCount }, (_, i) => {
                const tv = i * tickStep;
                return (
                  <text
                    key={`xl${i}`}
                    x={xScale(tv)}
                    y={PLOT_H - PAD_BOT + 18}
                    textAnchor={i === 0 ? "start" : i === tickCount - 1 ? "end" : "middle"}
                    className="memtl-axis-label"
                  >
                    {fmtTime(tv, tickStep)}
                  </text>
                );
              })}
              {/* current-event marker */}
              {currentT !== null && (
                <line
                  x1={xScale(currentT)}
                  x2={xScale(currentT)}
                  y1={PAD_TOP}
                  y2={PLOT_H - PAD_BOT}
                  className="memtl-cursor"
                  aria-label="Current event"
                />
              )}
              {/* hover marker */}
              {hoverPoint !== null && !drag && (
                <line
                  x1={xScale(hoverPoint.t)}
                  x2={xScale(hoverPoint.t)}
                  y1={PAD_TOP}
                  y2={PLOT_H - PAD_BOT}
                  className="memtl-hover-line"
                  aria-label="Hovered sample"
                />
              )}
              {/* drag overlay (selection rectangle for zoom) */}
              {drag && Math.abs(drag.currentX - drag.startX) >= 1 && (
                <rect
                  x={Math.min(drag.startX, drag.currentX)}
                  y={PAD_TOP}
                  width={Math.abs(drag.currentX - drag.startX)}
                  height={PLOT_AREA_H}
                  className="memtl-drag-rect"
                />
              )}
            </svg>
            {hoverPoint !== null && !drag && (
              <div
                className="memtl-tooltip"
                style={{ left: tooltipLeft, top: tooltipTop }}
              >
                <div className="memtl-tooltip-head">
                  <span>{fmtHoverTime(hoverPoint.t)}</span>
                  <strong>{fmtBytesGb(hoverPoint.totalBytes)}</strong>
                </div>
                <div className="memtl-tooltip-rows">
                  {hoverRows.map(({ key, bytes }) => {
                    const fill = BAND_FILL[key];
                    return (
                      <div key={key} className="memtl-tooltip-row">
                        <span className="memtl-tooltip-label">
                          <span
                            className="memtl-legend-swatch"
                            style={{ background: fill.color, opacity: fill.opacity }}
                          />
                          {BAND_LABEL[key]}
                        </span>
                        <span>{fmtBytesGb(bytes)}</span>
                        <span className="dim">{fmtPct(bytes, hoverPoint.totalBytes)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* legend */}
      <div className="memtl-legend">
        {capBytes !== null && (
          <span className="memtl-legend-item">
            <span className="memtl-legend-line memtl-legend-cap" />
            Fast Memory Budget
          </span>
        )}
        {currentT !== null && (
          <span className="memtl-legend-item">
            <span className="memtl-legend-line memtl-legend-cursor" />
            Current Event
          </span>
        )}
        {activeBands.map(({ key }) => {
          const fill = BAND_FILL[key];
          return (
            <span key={key} className="memtl-legend-item">
              <span
                className="memtl-legend-swatch"
                style={{ background: fill.color, opacity: fill.opacity }}
              />
              {BAND_LABEL[key]}
            </span>
          );
        })}
      </div>
    </div>
  );
}
