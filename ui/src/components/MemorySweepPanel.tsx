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

const DEFAULT_STEP_GB = 5;
const DEFAULT_MIN_GB = 10;
const DEFAULT_MAX_GB = 80;
const CHART_W = 520;
const CHART_H = 260;
const PLOT = { left: 62, right: 64, top: 30, bottom: 42 };
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

function fmtGb(gb: number): string {
  if (Number.isInteger(gb)) return `${gb}`;
  return gb.toFixed(1);
}

function niceMax(value: number, floor: number): number {
  if (value <= 0) return floor;
  const raw = Math.max(value * 1.12, floor);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return nice * mag;
}

function buildBudgets(stepGb: number, minGb: number, maxGb: number, includeInf: boolean): (number | null)[] {
  const step = Math.max(1, stepGb);
  const min = Math.max(0, minGb);
  const max = Math.max(min, maxGb);
  const out: number[] = [];
  for (let gb = min; gb <= max + 1e-9; gb += step) {
    out.push(Number(gb.toFixed(3)));
  }
  if (out.length === 0 || Math.abs(out[out.length - 1] - max) > 1e-9) {
    out.push(Number(max.toFixed(3)));
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
  const allValues = series.flatMap((s) => s.values);
  if (allValues.length === 0) {
    return (
      <div className="sweep-chart">
        <div className="sweep-chart-title">{title}</div>
        <div className="sweep-chart-empty dim">no successful points</div>
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
  const finiteRange = Math.max(finiteMax - xMin, stepGb, 1);
  const infGap = hasInf ? Math.max(stepGb * 2, finiteRange * 0.08) : 0;
  const xMax = finiteMax + infGap;
  const yMax = percent
    ? Math.min(100, niceMax(Math.max(...allValues.map((v) => v.y)), 10))
    : niceMax(Math.max(...allValues.map((v) => v.y)), 1);

  const xFor = (gb: number | null) => {
    const xVal = gb === null ? xMax : gb;
    return PLOT.left + ((xVal - xMin) / Math.max(xMax - xMin, 1)) * PLOT_W;
  };
  const yFor = (y: number) => (
    PLOT.top + PLOT_H - (y / Math.max(yMax, 1)) * PLOT_H
  );

  const xTicks: XTick[] = [
    ...Array.from(
      new Set(
        allValues
          .map((v) => v.point.budgetGb)
          .filter((v): v is number => v !== null),
      ),
    )
      .sort((a, b) => a - b)
      .map((value) => ({
        key: `gb-${value}`,
        value,
        label: `${fmtGb(value)}G`,
        inf: false,
      })),
    ...(hasInf ? [{ key: "inf", value: null, label: "inf", inf: true }] : []),
  ];
  const yTicks = Array.from({ length: 5 }, (_, i) => (yMax * i) / 4);
  const axisBreakX = hasInf ? (xFor(finiteMax) + xFor(null)) / 2 : null;

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
          const labelY = CHART_H - 14;
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
                  x={x - 17}
                  y={labelY - 12}
                  width={34}
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
                <circle
                  key={`${s.label}-${v.point.key}`}
                  cx={xFor(v.point.budgetGb)}
                  cy={yFor(v.y)}
                  r={4.5}
                  fill={s.color}
                >
                  <title>{`${v.point.label}: ${valueLabel(v.y)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
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
  const [stepGb, setStepGb] = useState(DEFAULT_STEP_GB);
  const [minGb, setMinGb] = useState(DEFAULT_MIN_GB);
  const [maxGb, setMaxGb] = useState(DEFAULT_MAX_GB);
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

  const budgets = useMemo(
    () => buildBudgets(stepGb, minGb, maxGb, includeInf),
    [stepGb, minGb, maxGb, includeInf],
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
          device_capacity_gb: budgetGb,
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

  const tokPerTflop = useMemo(() => {
    const ratios = points
      .filter((point) => point.summary.effective_tflops > 0)
      .map((point) => point.summary.tokens_per_second / point.summary.effective_tflops);
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
      <summary className="collapsible-summary">Throughput vs. GPU Memory Budget</summary>
      <div className="collapsible-content">
        <div className="sweep-controls">
          <label className="form-field">
            <span className="form-field-label">point spacing (GB)</span>
            <input
              type="number"
              min={1}
              step="any"
              value={String(stepGb)}
              disabled={running}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v) && v > 0) setStepGb(v);
              }}
            />
          </label>
          <label className="form-field">
            <span className="form-field-label">min budget (GB)</span>
            <input
              type="number"
              min={0}
              step={1}
              value={String(minGb)}
              disabled={running}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) setMinGb(v);
              }}
            />
          </label>
          <label className="form-field">
            <span className="form-field-label">max budget (GB)</span>
            <input
              type="number"
              min={1}
              step={1}
              value={String(maxGb)}
              disabled={running}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) setMaxGb(v);
              }}
            />
          </label>
          <label className={`sweep-inf-toggle${running ? " sweep-inf-toggle-disabled" : ""}`}>
            <input
              type="checkbox"
              checked={includeInf}
              disabled={running}
              onChange={(e) => setIncludeInf(e.target.checked)}
            />
            <span>include inf</span>
          </label>
          <button className="submit-btn sweep-run-btn" onClick={runSweep} disabled={running}>
            {running ? `running ${done}/${total}` : "run sweep"}
          </button>
        </div>

        {points.length > 0 && (
          <div className="sweep-grid-panel">
            <LineChart
              title="Effective Throughput"
              yLabel="effective TFLOPS"
              series={throughputSeries}
              valueLabel={fmtTflops}
              stepGb={stepGb}
              minGb={minGb}
              secondaryLabel="tok/sec"
              secondaryValueLabel={fmtToks}
              secondaryFromPrimary={(v) => v * tokPerTflop}
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
