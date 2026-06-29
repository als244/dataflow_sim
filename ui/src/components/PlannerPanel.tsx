import { useEffect, useState } from "react";

import {
  POLICY_OPTIONS,
  type PlannerParams,
  type Policy,
  type SimulationParams,
} from "./InputPanel";

interface Props {
  params: SimulationParams;
  setParams: (p: SimulationParams) => void;
  onRun: () => void;
  onReset: () => void;
  canRun: boolean;
  status: "idle" | "loading" | "ok" | "error";
  errorMsg: string | null;
  previewStale: boolean;
  hasResults: boolean;
}

export function PlannerPanel({
  params,
  setParams,
  onRun,
  onReset,
  canRun,
  status,
  errorMsg,
  previewStale,
  hasResults,
}: Props) {
  function setPlanner<K extends keyof PlannerParams>(key: K, value: PlannerParams[K]) {
    setParams({ ...params, planner: { ...params.planner, [key]: value } });
  }

  const controlsLocked = status === "loading" || hasResults;
  const runLabel = hasResults ? "Reset Simulation" : status === "loading" ? "Running..." : "Run Simulation";
  const runAction = hasResults ? onReset : onRun;
  const [fastMemoryBudgetText, setFastMemoryBudgetText] = useState(
    params.planner.fast_memory_capacity_gb === null
      ? ""
      : String(params.planner.fast_memory_capacity_gb),
  );
  const fastMemoryBudgetValid = (
    fastMemoryBudgetText.trim() === ""
    || (Number.isFinite(Number(fastMemoryBudgetText)) && Number(fastMemoryBudgetText) > 0)
  );
  const runDisabled = hasResults
    ? false
    : !canRun || status === "loading" || !fastMemoryBudgetValid;

  useEffect(() => {
    setFastMemoryBudgetText(
      params.planner.fast_memory_capacity_gb === null
        ? ""
        : String(params.planner.fast_memory_capacity_gb),
    );
  }, [params.planner.fast_memory_capacity_gb]);

  return (
    <div className="panel planner-panel">
      <div className="panel-header">
        <h3>Simulation</h3>
        {status === "loading" && <span className="loading-spinner" aria-hidden="true" />}
        <span className={`tag status-${status}`}>{status === "ok" ? "Complete" : status === "error" ? "Error" : status === "loading" ? "Running" : "Idle"}</span>
      </div>

      <div className="form-grid planner-grid">
        <label className="form-field form-field-wide">
          <span className="form-field-label">Planner Policy</span>
          <select
            value={params.planner.policy}
            disabled={controlsLocked}
            onChange={(e) => setPlanner("policy", e.target.value as Policy)}
          >
            {POLICY_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
        {params.planner.policy === "sliding_window" && (
          <label className="form-field">
            <span className="form-field-label">Weight Window</span>
            <input
              type="number"
              min={1}
              step={1}
              disabled={controlsLocked}
              value={String(params.planner.window_size)}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) setPlanner("window_size", v);
              }}
            />
          </label>
        )}
        <label className="form-field">
          <span className="form-field-label">Fast Memory Budget (GB)</span>
          <input
            type="number"
            min={0.000001}
            step="any"
            placeholder="Unlimited"
            disabled={controlsLocked}
            value={fastMemoryBudgetText}
            onChange={(e) => {
              const text = e.target.value;
              setFastMemoryBudgetText(text);
              if (text === "") {
                setPlanner("fast_memory_capacity_gb", null);
                return;
              }
              const v = Number(text);
              if (Number.isFinite(v) && v > 0) setPlanner("fast_memory_capacity_gb", v);
            }}
          />
        </label>
      </div>

      {previewStale && (
        <div className="input-note">The workload preview is stale. Update the workload before running.</div>
      )}
      {!fastMemoryBudgetValid && (
        <div className="input-note">Fast memory budget must be positive, or blank for unlimited.</div>
      )}
      {errorMsg && <div className="input-error">{errorMsg}</div>}

      <div className="planner-actions">
        <button className="submit-btn" onClick={runAction} disabled={runDisabled}>
          {runLabel}
        </button>
      </div>
    </div>
  );
}
