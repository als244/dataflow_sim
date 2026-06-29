import { useEffect, useMemo, useState } from "react";
import type { SimulationParams } from "./InputPanel";
import type { Summary } from "./SummaryPanel";

interface Props {
  params: SimulationParams;
}

interface SweepPoint {
  key: string;
  label: string;
  budgetGb: number | null;
  summary: Summary;
}

interface SweepError {
  label: string;
  message: string;
}

interface ChartSeries {
  label: string;
  color: string;
  values: {
    point: SweepPoint;
    y: number;
  }[];
}

interface XTick {
  key: string;
  value: number | null;
  label: string;
  inf: boolean;
}

interface HoverDatum {
  x: number;
  y: number;
  color: string;
  seriesLabel: string;
  budgetLabel: string;
  valueText: string;
  secondaryText?: string;
}

const DEFAULT_STEP_GB = 5;
const DEFAULT_MIN_GB = 10;
const DEFAULT_MAX_GB = 80;
const CHART_W = 520;
const CHART_H = 300;
const PLOT = { left: 68, right: 70, top: 34, bottom: 58 };
const PLOT_W = CHART_W - PLOT.left - PLOT.right;
const PLOT_H = CHART_H - PLOT.top - PLOT.bottom;

function fmtToks(t: number): string {
  if (t >= 1e6) return `${(t / 1e6).toFixed(2)}M`;
  if (t >= 1e3) return `${(t / 1e3).toFixed(1)}k`;
  return t.toFixed(0);
}

function fmtPct(p: number): string {
  return `${p.toFixed(1)}%`;
}

function fmtTflops(t: number): string {
  if (t >= 100) return t.toFixed(0);
  if (t >= 10) return t.toFixed(1);
  return t.toFixed(2);
}

function trimFixed(value: number, decimals: number): string {
  return value.toFixed(decimals).replace(/\.?0+$/, "");
}

function fmtGb(gb: number): string {
  if (Number.isInteger(gb)) return `${gb}`;
  const abs = Math.abs(gb);
  if (abs >= 10) return trimFixed(gb, 1);
  if (abs >= 1) return trimFixed(gb, 2);
  if (abs >= 0.01) return trimFixed(gb, 3);
  return trimFixed(gb, 6);
}

function parseGbInput(text: string, { allowZero }: { allowZero: boolean }): number | null {
  if (text.trim() === "") return null;
  const value = Number(text);
  if (!Number.isFinite(value)) return null;
  if (allowZero ? value < 0 : value <= 0) return null;
  return value;
}

function niceMax(value: number, floor: number): number {
  if (value <= 0) return floor;
  const raw = Math.max(value * 1.12, floor);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return nice * mag;
}

function niceTickStep(range: number, maxTicks: number): number {
  const raw = Math.max(range / Math.max(maxTicks - 1, 1), 1e-9);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return nice * mag;
}

function buildFiniteXTicks(min: number, max: number, maxTicks: number): XTick[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (Math.abs(max - min) < 1e-9) {
    return [{ key: `gb-${min}`, value: min, label: fmtGb(min), inf: false }];
  }
  const step = niceTickStep(max - min, maxTicks);
  const ticks: number[] = [];
  const push = (value: number) => {
    const rounded = Number(value.toFixed(6));
    if (!ticks.some((tick) => Math.abs(tick - rounded) < 1e-6)) {
      ticks.push(rounded);
    }
  };
  push(min);
  for (let value = Math.ceil(min / step) * step; value <= max + 1e-9; value += step) {
    if (value > min + 1e-9 && value < max - 1e-9) push(value);
  }
  push(max);
  return ticks
    .sort((a, b) => a - b)
    .map((value) => ({
      key: `gb-${value}`,
      value,
      label: fmtGb(value),
      inf: false,
    }));
}

function buildBudgets(stepGb: number, minGb: number, maxGb: number, includeInf: boolean): (number | null)[] {
  if (![stepGb, minGb, maxGb].every(Number.isFinite) || stepGb <= 0) {
    return includeInf ? [null] : [];
  }
  const step = stepGb;
  const min = Math.max(0, minGb);
  const max = Math.max(min, maxGb);
  const out: number[] = [];
  for (let gb = min; gb <= max + 1e-9; gb += step) {
    out.push(Number(gb.toFixed(6)));
  }
  if (out.length === 0 || Math.abs(out[out.length - 1] - max) > 1e-9) {
    out.push(Number(max.toFixed(6)));
  }
  return includeInf ? [...out, null] : out;
}

async function fetchSummary(params: SimulationParams): Promise<Summary> {
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  const body = await res.json();
  return body.summary as Summary;
}

function LineChart({
  title,
  yLabel,
  series,
  valueLabel,
  percent = false,
  stepGb,
  minGb,
  secondaryLabel,
  secondaryValueLabel,
  secondaryFromPrimary,
}: {
  title: string;
  yLabel: string;
  series: ChartSeries[];
  valueLabel: (v: number) => string;
  percent?: boolean;
  stepGb: number;
  minGb: number;
  secondaryLabel?: string;
  secondaryValueLabel?: (v: number) => string;
  secondaryFromPrimary?: (v: number) => number;
}) {
  const [hover, setHover] = useState<HoverDatum | null>(null);
  const allValues = series.flatMap((s) => s.values);
  if (allValues.length === 0) {
    return (
      <div className="sweep-chart">
        <div className="sweep-chart-title">{title}</div>
        <div className="sweep-chart-empty dim">No successful points</div>
      </div>
    );
  }

  const finiteMin = Math.min(
    ...allValues
      .map((v) => v.point.budgetGb)
      .filter((v): v is number => v !== null),
    minGb,
  );
  const finiteMax = Math.max(
    ...allValues.map((v) => v.point.budgetGb ?? 0),
    minGb + stepGb,
  );
  const hasInf = allValues.some((v) => v.point.budgetGb === null);
  const xMin = Math.min(finiteMin, finiteMax);
  const finiteRange = Math.max(finiteMax - xMin, stepGb, 1e-9);
  const infGap = hasInf ? Math.max(stepGb * 2, finiteRange * 0.08) : 0;
  const xMax = finiteMax + infGap;
  const yMax = percent
    ? Math.min(100, niceMax(Math.max(...allValues.map((v) => v.y)), 10))
    : niceMax(Math.max(...allValues.map((v) => v.y)), 1);

  const xFor = (gb: number | null) => {
    const xVal = gb === null ? xMax : gb;
    return PLOT.left + ((xVal - xMin) / Math.max(xMax - xMin, 1e-9)) * PLOT_W;
  };
  const yFor = (y: number) => (
    PLOT.top + PLOT_H - (y / Math.max(yMax, 1)) * PLOT_H
  );

  const makeHover = (s: ChartSeries, v: ChartSeries["values"][number]): HoverDatum => {
    const secondaryValue = (
      v.point.summary.primary_rate_per_second
      ?? (secondaryFromPrimary ? secondaryFromPrimary(v.y) : undefined)
    );
    return {
      x: xFor(v.point.budgetGb),
      y: yFor(v.y),
      color: s.color,
      seriesLabel: s.label,
      budgetLabel: v.point.budgetGb === null ? "Unlimited" : `${fmtGb(v.point.budgetGb)} GB`,
      valueText: valueLabel(v.y),
      secondaryText: secondaryLabel && secondaryValueLabel && secondaryValue !== undefined
        ? `${secondaryLabel}: ${secondaryValueLabel(secondaryValue)}`
        : undefined,
    };
  };

  const xTicks: XTick[] = [
    ...buildFiniteXTicks(xMin, finiteMax, 6),
    ...(hasInf ? [{ key: "inf", value: null, label: "Unlimited", inf: true }] : []),
  ];
  const yTicks = Array.from({ length: 5 }, (_, i) => (yMax * i) / 4);
  const axisBreakX = hasInf ? (xFor(finiteMax) + xFor(null)) / 2 : null;
  const tooltipWidth = 178;
  const tooltipHeight = hover?.secondaryText ? 72 : 56;
  const tooltip = hover ? {
    x: Math.min(Math.max(PLOT.left + 4, hover.x + 12), CHART_W - tooltipWidth - 8),
    y: Math.min(Math.max(PLOT.top + 4, hover.y - tooltipHeight - 12), CHART_H - PLOT.bottom - tooltipHeight - 4),
    width: tooltipWidth,
    height: tooltipHeight,
  } : null;

  return (
    <div className="sweep-chart">
      <div className="sweep-chart-title">{title}</div>
      <svg className="sweep-svg" viewBox={`0 0 ${CHART_W} ${CHART_H}`} role="img">
        <text x={PLOT.left} y={15} className="sweep-axis-title">{yLabel}</text>
        {secondaryLabel && (
          <text x={CHART_W - PLOT.right} y={15} textAnchor="end" className="sweep-axis-title">
            {secondaryLabel}
          </text>
        )}
        {yTicks.map((tick) => {
          const y = yFor(tick);
          return (
            <g key={`y-${tick}`}>
              {tick > 0 && tick < yMax && (
                <line x1={PLOT.left} x2={CHART_W - PLOT.right} y1={y} y2={y} className="sweep-gridline" />
              )}
              <line x1={PLOT.left - 4} x2={PLOT.left} y1={y} y2={y} className="sweep-tick" />
              <line x1={CHART_W - PLOT.right} x2={CHART_W - PLOT.right + 4} y1={y} y2={y} className="sweep-tick" />
              <text x={PLOT.left - 8} y={y + 4} textAnchor="end" className="sweep-axis-label">
                {valueLabel(tick)}
              </text>
              {secondaryFromPrimary && secondaryValueLabel && (
                <text x={CHART_W - PLOT.right + 8} y={y + 4} textAnchor="start" className="sweep-axis-label">
                  {secondaryValueLabel(secondaryFromPrimary(tick))}
                </text>
              )}
            </g>
          );
        })}
        {xTicks.map((tick) => {
          const x = xFor(tick.value);
          const labelY = CHART_H - 24;
          return (
            <g key={tick.key}>
              <line
                x1={x}
                x2={x}
                y1={PLOT.top + PLOT_H}
                y2={PLOT.top + PLOT_H + (tick.inf ? 10 : 4)}
                className={`sweep-tick${tick.inf ? " sweep-inf-tick" : ""}`}
              />
              {tick.inf && (
                <rect
                  x={x - 38}
                  y={labelY - 12}
                  width={76}
                  height={17}
                  rx={3}
                  className="sweep-inf-label-bg"
                />
              )}
              <text
                x={x}
                y={labelY}
                textAnchor="middle"
                className={`sweep-axis-label sweep-x-axis-label${tick.inf ? " sweep-axis-label-inf" : ""}`}
              >
                {tick.label}
              </text>
            </g>
          );
        })}
        <line x1={PLOT.left} x2={CHART_W - PLOT.right} y1={PLOT.top + PLOT_H} y2={PLOT.top + PLOT_H} className="sweep-axis" />
        <line x1={PLOT.left} x2={PLOT.left} y1={PLOT.top} y2={PLOT.top + PLOT_H} className="sweep-axis" />
        <text x={PLOT.left + PLOT_W / 2} y={CHART_H - 6} textAnchor="middle" className="sweep-axis-title">
          Fast memory budget (GB)
        </text>
        {secondaryLabel && (
          <line x1={CHART_W - PLOT.right} x2={CHART_W - PLOT.right} y1={PLOT.top} y2={PLOT.top + PLOT_H} className="sweep-axis sweep-secondary-axis" />
        )}
        {axisBreakX !== null && (
          <line
            x1={axisBreakX}
            x2={axisBreakX}
            y1={PLOT.top + 6}
            y2={PLOT.top + PLOT_H - 4}
            className="sweep-axis-break"
          />
        )}

        {series.map((s) => {
          const sorted = [...s.values].sort((a, b) => (
            (a.point.budgetGb ?? xMax) - (b.point.budgetGb ?? xMax)
          ));
          const path = sorted.map((v) => `${xFor(v.point.budgetGb)},${yFor(v.y)}`).join(" ");
          return (
            <g key={s.label}>
              <polyline points={path} fill="none" stroke={s.color} strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round" />
              {sorted.map((v) => (
                <g key={`${s.label}-${v.point.key}`}>
                  <circle
                    cx={xFor(v.point.budgetGb)}
                    cy={yFor(v.y)}
                    r={4.8}
                    fill={s.color}
                    className="sweep-point"
                  />
                  <circle
                    cx={xFor(v.point.budgetGb)}
                    cy={yFor(v.y)}
                    r={12}
                    className="sweep-hit-dot"
                    onMouseEnter={() => setHover(makeHover(s, v))}
                    onMouseMove={() => setHover(makeHover(s, v))}
                    onMouseLeave={() => setHover(null)}
                  />
                </g>
              ))}
            </g>
          );
        })}
        {hover && tooltip && (
          <g className="sweep-tooltip" pointerEvents="none">
            <line
              x1={hover.x}
              x2={hover.x}
              y1={PLOT.top}
              y2={PLOT.top + PLOT_H}
              className="sweep-hover-line"
            />
            <circle cx={hover.x} cy={hover.y} r={7} fill={hover.color} className="sweep-hover-point" />
            <rect x={tooltip.x} y={tooltip.y} width={tooltip.width} height={tooltip.height} rx={7} />
            <text x={tooltip.x + 10} y={tooltip.y + 18} className="sweep-tooltip-title">
              {hover.budgetLabel}
            </text>
            <text x={tooltip.x + 10} y={tooltip.y + 36} className="sweep-tooltip-row">
              {hover.seriesLabel}: {hover.valueText}
            </text>
            {hover.secondaryText && (
              <text x={tooltip.x + 10} y={tooltip.y + 54} className="sweep-tooltip-row">
                {hover.secondaryText}
              </text>
            )}
          </g>
        )}
      </svg>
      <div className="sweep-legend">
        {series.map((s) => (
          <span key={s.label} className="sweep-legend-item">
            <span className="sweep-legend-swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function MemorySweepPanel({ params }: Props) {
  const [stepGbText, setStepGbText] = useState(String(DEFAULT_STEP_GB));
  const [minGbText, setMinGbText] = useState(String(DEFAULT_MIN_GB));
  const [maxGbText, setMaxGbText] = useState(String(DEFAULT_MAX_GB));
  const [includeInf, setIncludeInf] = useState(true);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [total, setTotal] = useState(0);
  const [points, setPoints] = useState<SweepPoint[]>([]);
  const [errors, setErrors] = useState<SweepError[]>([]);

  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    setPoints([]);
    setErrors([]);
    setDone(0);
    setTotal(0);
  }, [paramsKey]);

  const parsedMinGb = parseGbInput(minGbText, { allowZero: false });
  const parsedMaxGb = parseGbInput(maxGbText, { allowZero: false });
  const parsedStepGb = parseGbInput(stepGbText, { allowZero: false });
  const sweepInputsValid = (
    parsedMinGb !== null
    && parsedMaxGb !== null
    && parsedStepGb !== null
    && parsedMaxGb >= parsedMinGb
  );
  const minGb = parsedMinGb ?? DEFAULT_MIN_GB;
  const maxGb = parsedMaxGb ?? DEFAULT_MAX_GB;
  const stepGb = parsedStepGb ?? DEFAULT_STEP_GB;
  const budgets = useMemo(
    () => (
      sweepInputsValid
        ? buildBudgets(stepGb, minGb, maxGb, includeInf)
        : includeInf ? [null] : []
    ),
    [stepGb, minGb, maxGb, includeInf, sweepInputsValid],
  );

  async function runSweep() {
    setRunning(true);
    setDone(0);
    setTotal(budgets.length);
    setPoints([]);
    setErrors([]);
    for (let i = 0; i < budgets.length; i += 1) {
      const budgetGb = budgets[i];
      const label = budgetGb === null ? "inf" : `${fmtGb(budgetGb)} GB`;
      const key = budgetGb === null ? "inf" : `${budgetGb}`;
      try {
        const summary = await fetchSummary({
          ...params,
          planner: {
            ...params.planner,
            fast_memory_capacity_gb: budgetGb,
          },
        });
        setPoints((prev) => [...prev, { key, label, budgetGb, summary }]);
      } catch (e) {
        setErrors((prev) => [
          ...prev,
          { label, message: e instanceof Error ? e.message : String(e) },
        ]);
      } finally {
        setDone(i + 1);
      }
    }
    setRunning(false);
  }

  const throughputSeries = useMemo<ChartSeries[]>(() => [
    {
      label: "effective TFLOPS",
      color: "#8fb9f0",
      values: points.map((point) => ({ point, y: point.summary.effective_tflops })),
    },
  ], [points]);

  const primaryUnit = points.find((point) => point.summary.primary_unit)?.summary.primary_unit;
  const primaryPerTflop = useMemo(() => {
    const ratios = points
      .filter((point) => point.summary.effective_tflops > 0)
      .map((point) => (point.summary.primary_rate_per_second ?? 0) / point.summary.effective_tflops)
      .filter((ratio) => ratio > 0);
    if (ratios.length === 0) return 0;
    return ratios.reduce((sum, ratio) => sum + ratio, 0) / ratios.length;
  }, [points]);

  const pctSeries = useMemo<ChartSeries[]>(() => [
    {
      label: "recompute %",
      color: "#ef4444",
      values: points.map((point) => ({ point, y: point.summary.recompute_pct })),
    },
    {
      label: "idle %",
      color: "#f0c285",
      values: points.map((point) => ({ point, y: point.summary.idle_pct })),
    },
  ], [points]);

  return (
    <details className="panel collapsible-panel sweep-panel">
      <summary className="collapsible-summary">Throughput vs. Fast Memory Budget</summary>
      <div className="collapsible-content">
        <div className="sweep-controls">
          <label className="form-field">
            <span className="form-field-label">Minimum Budget (GB)</span>
            <input
              type="number"
              min={0.000001}
              step="any"
              value={minGbText}
              disabled={running}
              onChange={(e) => setMinGbText(e.target.value)}
            />
          </label>
          <label className="form-field">
            <span className="form-field-label">Maximum Budget (GB)</span>
            <input
              type="number"
              min={0.000001}
              step="any"
              value={maxGbText}
              disabled={running}
              onChange={(e) => setMaxGbText(e.target.value)}
            />
          </label>
          <label className="form-field">
            <span className="form-field-label">Point Spacing (GB)</span>
            <input
              type="number"
              min={0.000001}
              step="any"
              value={stepGbText}
              disabled={running}
              onChange={(e) => setStepGbText(e.target.value)}
            />
          </label>
          <label className={`sweep-inf-toggle${running ? " sweep-inf-toggle-disabled" : ""}`}>
            <input
              type="checkbox"
              checked={includeInf}
              disabled={running}
              onChange={(e) => setIncludeInf(e.target.checked)}
            />
            <span>Include Unlimited</span>
          </label>
          <button className="submit-btn sweep-run-btn" onClick={runSweep} disabled={running || !sweepInputsValid}>
            {running ? `Running ${done}/${total}` : "Run Sweep"}
          </button>
        </div>

        {!sweepInputsValid && (
          <div className="input-note">Enter positive min/max budgets with max &gt;= min and point spacing &gt; 0.</div>
        )}

        {points.length > 0 && (
          <div className="sweep-grid-panel">
            <LineChart
              title="Effective Throughput"
              yLabel="effective TFLOPS"
              series={throughputSeries}
              valueLabel={fmtTflops}
              stepGb={stepGb}
              minGb={minGb}
              secondaryLabel={primaryUnit ? `${primaryUnit}/sec` : undefined}
              secondaryValueLabel={primaryUnit ? fmtToks : undefined}
              secondaryFromPrimary={primaryUnit ? (v) => v * primaryPerTflop : undefined}
            />
            <LineChart
              title="Throughput Degradation"
              yLabel="percent"
              series={pctSeries}
              valueLabel={fmtPct}
              percent
              stepGb={stepGb}
              minGb={minGb}
            />
          </div>
        )}

        {errors.length > 0 && (
          <div className="sweep-errors">
            {errors.map((err) => (
              <span key={`${err.label}-${err.message}`} className="sweep-error" title={err.message}>
                {err.label}: error
              </span>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}
