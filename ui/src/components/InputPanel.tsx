export type Policy = "sliding_window" | "belady_reactive" | "roundtrip_planner" | "max_reduce" | "min_grow" | "pressurefit";
export type OptimizerMode = "none" | "adamw" | "muon";

export interface HardwareParams {
  preset: string;
  peak_tflops: number;
  gpu_membw_gbs: number;
  interconnect_bw_gbs: number;
  matmul_eff: number;
  attn_fwd_eff: number;
  attn_bwd_eff: number;
  mem_eff: number;
}

export interface ModelParams {
  preset: string;
  vocab_size: number;
  n_layers: number;
  d_model: number;
  head_dim: number;
  n_heads: number;
  n_kv_heads: number;
  expert_dim: number;
  num_shared_experts: number;
  num_routed_experts: number;
  top_k: number;
  qk_norm: boolean;
}

export interface SimulationParams {
  hardware: HardwareParams;
  model: ModelParams;
  seqlen: number;
  num_seqs: number;
  grad_accum_rounds: number;
  num_steps: number;
  optimizer: OptimizerMode;
  final_model_state_on_host: boolean;
  policy: Policy;
  window_size: number;
  device_capacity_gb: number | null;
  recompute: boolean;
}

export interface Presets {
  models: Record<string, Omit<ModelParams, "preset">>;
  hardware: Record<string, Omit<HardwareParams, "preset">>;
}

export const DEFAULT_HARDWARE: HardwareParams = {
  preset: "H100",
  peak_tflops: 989,
  gpu_membw_gbs: 3000,
  interconnect_bw_gbs: 50,
  matmul_eff: 0.65,
  attn_fwd_eff: 0.6,
  attn_bwd_eff: 0.5,
  mem_eff: 0.9,
};

export const DEFAULT_MODEL: ModelParams = {
  preset: "llama3_8B",
  vocab_size: 128256,
  n_layers: 32,
  d_model: 4096,
  head_dim: 128,
  n_heads: 32,
  n_kv_heads: 8,
  expert_dim: 14336,
  num_shared_experts: 1,
  num_routed_experts: 0,
  top_k: 0,
  qk_norm: false,
};

export const DEFAULT_PARAMS: SimulationParams = {
  hardware: DEFAULT_HARDWARE,
  model: DEFAULT_MODEL,
  seqlen: 4096,
  num_seqs: 4,
  grad_accum_rounds: 1,
  num_steps: 1,
  optimizer: "none",
  final_model_state_on_host: false,
  policy: "pressurefit",
  window_size: 2,
  device_capacity_gb: null,
  recompute: true,
};

export const POLICY_OPTIONS: { value: Policy; label: string; hint: string }[] = [
  { value: "pressurefit", label: "PressureFit", hint: "pressure-fit interval planning; picks the fastest of four verified inbound schedules" },
  { value: "max_reduce", label: "Max-reduce", hint: "analytic top-down: start at MAX residency, split most-overloaded boundary until cap fits" },
  { value: "min_grow", label: "Min-grow", hint: "MIN-seeded over-shrink + beam search using the simulator as cost oracle" },
  { value: "belady_reactive", label: "Reactive Belady", hint: "shadow-simulator walk; evicts farthest-next-use when capacity binds" },
  { value: "roundtrip_planner", label: "Round-trip planner", hint: "constructively enumerates (offload, prefetch) round-trips and packs them onto streams" },
  { value: "sliding_window", label: "Sliding window", hint: "hand-crafted fixed-width window over W/dW/A; tune `weight window`" },
];

const OPTIMIZER_OPTIONS: { value: OptimizerMode; label: string }[] = [
  { value: "none", label: "None" },
  { value: "adamw", label: "AdamW" },
  { value: "muon", label: "Muon" },
];

const HW_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "peak_tflops", label: "peak TFLOPS", min: 0.1, step: 1 },
  { key: "gpu_membw_gbs", label: "GPU mem-bw (GB/s)", min: 1, step: 10 },
  { key: "interconnect_bw_gbs", label: "interconnect BW (GB/s)", min: 0.1, step: 1 },
  { key: "matmul_eff", label: "matmul eff", min: 0.01, step: 0.01 },
  { key: "attn_fwd_eff", label: "attn fwd eff", min: 0.01, step: 0.01 },
  { key: "attn_bwd_eff", label: "attn bwd eff", min: 0.01, step: 0.01 },
  { key: "mem_eff", label: "mem eff", min: 0.01, step: 0.01 },
];

const MODEL_FIELDS: { key: keyof Omit<ModelParams, "preset" | "qk_norm">; label: string }[] = [
  { key: "vocab_size", label: "vocab size" },
  { key: "n_layers", label: "L (layers)" },
  { key: "d_model", label: "d_model" },
  { key: "head_dim", label: "head_dim" },
  { key: "n_heads", label: "n_heads" },
  { key: "n_kv_heads", label: "n_kv_heads" },
  { key: "expert_dim", label: "expert_dim" },
  { key: "num_shared_experts", label: "shared experts" },
  { key: "num_routed_experts", label: "routed experts" },
  { key: "top_k", label: "top_k" },
];

interface Props {
  params: SimulationParams;
  setParams: (p: SimulationParams) => void;
  onSubmit: () => void;
  onReset: () => void;
  locked: boolean;
  status: "idle" | "loading" | "ok" | "error";
  errorMsg: string | null;
  presets: Presets | null;
}

export function InputPanel({ params, setParams, onSubmit, onReset, locked, status, errorMsg, presets }: Props) {
  // Hardware preset change -> overwrite numeric fields
  function onHardwarePreset(preset: string) {
    if (preset === "custom") {
      setParams({ ...params, hardware: { ...params.hardware, preset: "custom" } });
      return;
    }
    const hwPreset = presets?.hardware[preset];
    if (!hwPreset) return;
    setParams({ ...params, hardware: { preset, ...hwPreset } });
  }

  function onModelPreset(preset: string) {
    if (preset === "custom") {
      setParams({ ...params, model: { ...params.model, preset: "custom" } });
      return;
    }
    const mPreset = presets?.models[preset];
    if (!mPreset) return;
    setParams({ ...params, model: { preset, ...mPreset } });
  }

  function setHardware<K extends keyof HardwareParams>(key: K, value: HardwareParams[K]) {
    // Editing any field -> preset becomes "custom"
    setParams({
      ...params,
      hardware: { ...params.hardware, [key]: value, preset: key === "preset" ? (value as string) : "custom" },
    });
  }

  function setModel<K extends keyof ModelParams>(key: K, value: ModelParams[K]) {
    setParams({
      ...params,
      model: { ...params.model, [key]: value, preset: key === "preset" ? (value as string) : "custom" },
    });
  }

  const activePolicyHint =
    POLICY_OPTIONS.find((o) => o.value === params.policy)?.hint ?? "";

  const statusLabel =
    status === "loading" ? "running…" :
    status === "error" ? "error" :
    status === "ok" ? "ready" : "idle";

  return (
    <div className="panel input-panel">
      <div className="panel-header">
        <h3>inputs</h3>
        {status === "loading" && <span className="loading-spinner" aria-hidden="true" />}
        <span className={`tag status-${status}`}>{statusLabel}</span>
        <div className="header-buttons">
          {locked ? (
            <button className="reset-btn" onClick={onReset} title="clear results and unlock the form">
              reset
            </button>
          ) : (
            <button className="submit-btn" onClick={onSubmit} disabled={status === "loading"}>
              {status === "loading" ? "running..." : "submit"}
            </button>
          )}
        </div>
      </div>

      <fieldset className="form-sections" disabled={locked}>
        {/* Hardware */}
        <section className="form-section">
          <header className="form-section-header">
            <span className="form-section-title">hardware</span>
            <select
              className="form-preset"
              value={params.hardware.preset}
              onChange={(e) => onHardwarePreset(e.target.value)}
            >
              {presets &&
                Object.keys(presets.hardware).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              <option value="custom">Custom</option>
            </select>
          </header>
          <div className="form-grid">
            {HW_FIELDS.map((f) => (
              <label key={f.key} className="form-field">
                <span className="form-field-label">{f.label}</span>
                <input
                  type="number"
                  min={f.min}
                  step={f.step ?? 1}
                  value={String(params.hardware[f.key])}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v)) setHardware(f.key, v);
                  }}
                />
              </label>
            ))}
          </div>
        </section>

        {/* Model */}
        <section className="form-section">
          <header className="form-section-header">
            <span className="form-section-title">model</span>
            <select
              className="form-preset"
              value={params.model.preset}
              onChange={(e) => onModelPreset(e.target.value)}
            >
              {presets &&
                Object.keys(presets.models).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              <option value="custom">Custom</option>
            </select>
          </header>
          <div className="form-grid">
            {MODEL_FIELDS.map((f) => (
              <label key={f.key} className="form-field">
                <span className="form-field-label">{f.label}</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={String(params.model[f.key])}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v)) setModel(f.key, v);
                  }}
                />
              </label>
            ))}
            <label className="form-field form-field-checkbox">
              <span className="form-field-label">qk_norm</span>
              <input
                type="checkbox"
                checked={params.model.qk_norm}
                onChange={(e) => setModel("qk_norm", e.target.checked)}
              />
            </label>
          </div>
        </section>

        {/* Training */}
        <section className="form-section">
          <header className="form-section-header">
            <span className="form-section-title">training</span>
          </header>
          <div className="form-grid">
            <label className="form-field">
              <span className="form-field-label">seqlen</span>
              <input
                type="number" min={1} step={1}
                value={String(params.seqlen)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setParams({ ...params, seqlen: v });
                }}
              />
            </label>
            <label className="form-field">
              <span className="form-field-label">microbatch size</span>
              <input
                type="number" min={1} step={1}
                value={String(params.num_seqs)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setParams({ ...params, num_seqs: v });
                }}
              />
            </label>
            <label className="form-field">
              <span className="form-field-label">grad. accum rounds</span>
              <input
                type="number" min={1} step={1}
                value={String(params.grad_accum_rounds)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setParams({ ...params, grad_accum_rounds: v });
                }}
              />
            </label>
            <label className="form-field">
              <span className="form-field-label">num steps</span>
              <input
                type="number" min={1} step={1}
                value={String(params.num_steps)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setParams({ ...params, num_steps: v });
                }}
              />
            </label>
            <label className="form-field">
              <span className="form-field-label">optimizer</span>
              <select
                value={params.optimizer}
                onChange={(e) => setParams({ ...params, optimizer: e.target.value as OptimizerMode })}
              >
                {OPTIMIZER_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="form-field form-field-checkbox">
              <span className="form-field-label">final model state on host</span>
              <input
                type="checkbox"
                checked={params.final_model_state_on_host}
                onChange={(e) => setParams({
                  ...params,
                  final_model_state_on_host: e.target.checked,
                })}
              />
            </label>
          </div>
        </section>

        {/* Memory */}
        <section className="form-section">
          <header className="form-section-header">
            <span className="form-section-title">memory</span>
          </header>
          <div className="form-grid">
            <label className="form-field form-field-wide">
              <span className="form-field-label">policy</span>
              <select
                value={params.policy}
                onChange={(e) => setParams({ ...params, policy: e.target.value as Policy })}
              >
                {POLICY_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            {params.policy === "sliding_window" && (
              <label className="form-field">
                <span className="form-field-label">weight window</span>
                <input
                  type="number" min={1} step={1}
                  value={String(params.window_size)}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v)) setParams({ ...params, window_size: v });
                  }}
                />
              </label>
            )}
            <label className="form-field">
              <span className="form-field-label">GPU mem budget (GB)</span>
              <input
                type="number" min={0.1} step={1}
                placeholder="unlimited"
                value={params.device_capacity_gb === null ? "" : String(params.device_capacity_gb)}
                onChange={(e) => {
                  const text = e.target.value;
                  if (text === "") {
                    setParams({ ...params, device_capacity_gb: null });
                    return;
                  }
                  const v = Number(text);
                  if (Number.isFinite(v)) setParams({ ...params, device_capacity_gb: v });
                }}
              />
            </label>
            <label className="form-field form-field-checkbox">
              <span className="form-field-label">Allow Recompute</span>
              <input
                type="checkbox"
                checked={params.recompute}
                onChange={(e) => setParams({
                  ...params,
                  recompute: e.target.checked,
                })}
              />
            </label>
          </div>
          <div className="form-section-hint dim">{activePolicyHint}</div>
        </section>

      </fieldset>
      {errorMsg && <div className="input-error">{errorMsg}</div>}
    </div>
  );
}
