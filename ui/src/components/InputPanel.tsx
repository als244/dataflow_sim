import { useEffect, useMemo, useState, type ChangeEvent, type ReactNode } from "react";

export type Policy = "sliding_window" | "belady_reactive" | "roundtrip_planner" | "max_reduce" | "min_grow" | "pressurefit";
export type OptimizerMode = "none" | "adamw" | "muon";
export type ModelFamily = "llama3" | "qwen3" | "qwen3_moe" | "olmoe";

export interface HardwareParams {
  preset: string;
  peak_tflops: number;
  fast_memory_bw_gbs: number;
  from_slow_bw_gbs: number;
  to_slow_bw_gbs: number;
  matmul_eff: number;
  attn_fwd_eff: number;
  attn_bwd_eff: number;
  mem_eff: number;
}

export interface ModelParams {
  preset: string;
  family: ModelFamily;
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

export interface TrainingParams {
  seqlen: number;
  num_seqs: number;
  grad_accum_rounds: number;
  num_steps: number;
  optimizer: OptimizerMode;
  final_model_state_on_backing: boolean;
}

export interface DataflowCost {
  kind: "fixed" | "roofline" | "sum";
  name?: string | null;
  runtime_us?: number | null;
  flops?: number;
  memory_bytes?: number;
  efficiency?: "matmul" | "attention" | "attention_fwd" | "attention_bwd" | "memory" | "custom";
  count?: number;
  effective_flops?: number | null;
  compute_eff?: number | null;
  mem_eff?: number | null;
  terms?: DataflowCost[];
}

export interface DataflowObject {
  id: string;
  size_bytes: number;
  initial_location: "backing" | "fast";
  role: string;
}

export interface DataflowOutput {
  id: string;
  size_bytes: number;
  location?: "backing" | "fast";
  role?: string;
  metadata?: Record<string, unknown>;
}

export interface DataflowTask {
  id: string;
  label?: string | null;
  group?: string;
  compute_block_key?: string | null;
  inputs?: string[];
  outputs?: DataflowOutput[];
  mutates?: string[];
  cost?: DataflowCost | null;
  metadata?: Record<string, unknown>;
}

export interface DataflowMetrics {
  primary_unit: string;
  primary_count: number;
  metadata?: Record<string, unknown>;
}

export interface ComputeBlock {
  key: string;
  name: string;
  category: string;
  subops: DataflowCost[];
  metadata?: Record<string, unknown>;
}

export interface DataflowProgram {
  schema_version: "dataflow/v1";
  name: string;
  description?: string;
  metadata?: Record<string, unknown>;
  metrics?: DataflowMetrics | null;
  objects: DataflowObject[];
  compute_blocks?: ComputeBlock[];
  tasks: DataflowTask[];
  final_locations?: Record<string, "backing" | "fast">;
}

export interface TransformerWorkloadParams {
  source: "training_transformer";
  preset: string;
  model: ModelParams;
  training: TrainingParams;
}

export interface SchemaWorkloadParams {
  source: "schema";
  schema: DataflowProgram;
}

export type WorkloadParams = TransformerWorkloadParams | SchemaWorkloadParams;

export interface PlannerParams {
  policy: Policy;
  window_size: number;
  fast_memory_capacity_gb: number | null;
  recompute: boolean;
}

export interface SimulationParams {
  workload: WorkloadParams;
  hardware: HardwareParams;
  planner: PlannerParams;
}

export interface TransformerWorkloadPreset {
  source: "training_transformer";
  preset: string;
  model: Omit<ModelParams, "preset">;
  training: TrainingParams;
  description: string;
}

export interface SchemaWorkloadPreset {
  source: "schema";
  preset: string;
  schema: DataflowProgram;
  description: string;
}

export type WorkloadPreset = TransformerWorkloadPreset | SchemaWorkloadPreset;

export interface Presets {
  workloads: Record<string, WorkloadPreset>;
  models?: Record<string, Omit<ModelParams, "preset">>;
  hardware: Record<string, Omit<HardwareParams, "preset">>;
}

export const DEFAULT_HARDWARE: HardwareParams = {
  preset: "H100",
  peak_tflops: 989,
  fast_memory_bw_gbs: 3000,
  from_slow_bw_gbs: 50,
  to_slow_bw_gbs: 50,
  matmul_eff: 0.65,
  attn_fwd_eff: 0.6,
  attn_bwd_eff: 0.5,
  mem_eff: 0.9,
};

export const DEFAULT_MODEL: ModelParams = {
  preset: "llama3_8B",
  family: "llama3",
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

export const DEFAULT_TRAINING: TrainingParams = {
  seqlen: 4096,
  num_seqs: 4,
  grad_accum_rounds: 1,
  num_steps: 1,
  optimizer: "none",
  final_model_state_on_backing: false,
};

export const EXAMPLE_SCHEMA: DataflowProgram = {
  schema_version: "dataflow/v1",
  name: "two-task-dataflow",
  description: "Small generic workload",
  metadata: {},
  objects: [
    { id: "x", size_bytes: 16_777_216, initial_location: "fast", role: "activation" },
    { id: "w0", size_bytes: 67_108_864, initial_location: "backing", role: "parameter" },
    { id: "w1", size_bytes: 67_108_864, initial_location: "backing", role: "parameter" },
  ],
  tasks: [
    {
      id: "block_0",
      label: "block 0",
      group: "encoder",
      inputs: ["x", "w0"],
      outputs: [{ id: "h0", size_bytes: 16_777_216, role: "activation" }],
      cost: { kind: "roofline", name: "matmul", flops: 8_000_000_000, memory_bytes: 120_000_000, efficiency: "matmul" },
    },
    {
      id: "block_1",
      label: "block 1",
      group: "encoder",
      inputs: ["h0", "w1"],
      outputs: [{ id: "h1", size_bytes: 16_777_216, role: "activation" }],
      cost: { kind: "fixed", name: "measured_block", runtime_us: 240 },
    },
  ],
  final_locations: {},
};

export const DEFAULT_PARAMS: SimulationParams = {
  workload: {
    source: "training_transformer",
    preset: "llama3_8B",
    model: DEFAULT_MODEL,
    training: DEFAULT_TRAINING,
  },
  hardware: DEFAULT_HARDWARE,
  planner: {
    policy: "pressurefit",
    window_size: 2,
    fast_memory_capacity_gb: null,
    recompute: true,
  },
};

export const POLICY_OPTIONS: { value: Policy; label: string; hint: string }[] = [
  { value: "pressurefit", label: "PressureFit", hint: "Pressure-fit interval planning; picks the fastest of four verified inbound schedules" },
  { value: "max_reduce", label: "Max-Reduce", hint: "Analytic top-down planning: start at max residency, split the most overloaded boundary until the cap fits" },
  { value: "min_grow", label: "Min-Grow", hint: "Min-seeded shrink plus beam search using the simulator as cost oracle" },
  { value: "belady_reactive", label: "Reactive Belady", hint: "Shadow-simulator walk; evicts the farthest next use when capacity binds" },
  { value: "roundtrip_planner", label: "Round-Trip Planner", hint: "Constructively enumerates offload and prefetch round trips and packs them onto streams" },
  { value: "sliding_window", label: "Sliding Window", hint: "Fixed-width window over model state, gradients, and activations" },
];

const OPTIMIZER_OPTIONS: { value: OptimizerMode; label: string }[] = [
  { value: "none", label: "None" },
  { value: "adamw", label: "AdamW" },
  { value: "muon", label: "Muon" },
];

const MODEL_FAMILY_OPTIONS: { value: ModelFamily; label: string }[] = [
  { value: "llama3", label: "Llama 3" },
  { value: "qwen3", label: "Qwen3 Dense" },
  { value: "qwen3_moe", label: "Qwen3 MoE" },
  { value: "olmoe", label: "OLMoE" },
];

const HW_ACCELERATOR_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "peak_tflops", label: "Peak TFLOPS", min: 0.1, step: 1 },
  { key: "fast_memory_bw_gbs", label: "Fast Memory BW (GB/s)", min: 1, step: 10 },
];

const HW_SLOW_MEMORY_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "from_slow_bw_gbs", label: "From-Slow BW (GB/s)", min: 0.1, step: 1 },
  { key: "to_slow_bw_gbs", label: "To-Slow BW (GB/s)", min: 0.1, step: 1 },
];

const HW_KERNEL_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "mem_eff", label: "Memory Efficiency", min: 0.01, step: 0.01 },
  { key: "matmul_eff", label: "Matmul Efficiency", min: 0.01, step: 0.01 },
  { key: "attn_fwd_eff", label: "Attention Forward Efficiency", min: 0.01, step: 0.01 },
  { key: "attn_bwd_eff", label: "Attention Backward Efficiency", min: 0.01, step: 0.01 },
];

type HardwareField = {
  key: keyof Omit<HardwareParams, "preset">;
  label: string;
  step?: number;
  min?: number;
};

const HW_FIELD_SECTIONS: { title: string; fields: HardwareField[] }[] = [
  { title: "Accelerator Specs", fields: HW_ACCELERATOR_FIELDS },
  { title: "Slow Memory Specs", fields: HW_SLOW_MEMORY_FIELDS },
  { title: "Kernel Efficiency", fields: HW_KERNEL_FIELDS },
];

const MODEL_FIELDS: { key: keyof Omit<ModelParams, "preset" | "family" | "qk_norm">; label: string }[] = [
  { key: "vocab_size", label: "Vocabulary Size" },
  { key: "n_layers", label: "Layers" },
  { key: "d_model", label: "Model Width" },
  { key: "head_dim", label: "Head Dim" },
  { key: "n_heads", label: "Attention Heads" },
  { key: "n_kv_heads", label: "KV Heads" },
  { key: "expert_dim", label: "Expert Dim" },
  { key: "num_shared_experts", label: "Shared Experts" },
  { key: "num_routed_experts", label: "Routed Experts" },
  { key: "top_k", label: "Top K" },
];

interface Props {
  params: SimulationParams;
  setParams: (p: SimulationParams) => void;
  onPreview: () => void;
  locked: boolean;
  previewStatus: "idle" | "loading" | "ok" | "error";
  previewError: string | null;
  previewStale: boolean;
  presets: Presets | null;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function FormSubsection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="form-subsection">
      <div className="form-subsection-title">{title}</div>
      {children}
    </div>
  );
}

export function InputPanel({
  params,
  setParams,
  onPreview,
  locked,
  previewStatus,
  previewError,
  previewStale,
  presets,
}: Props) {
  const [schemaText, setSchemaText] = useState(() => JSON.stringify(EXAMPLE_SCHEMA, null, 2));
  const [schemaError, setSchemaError] = useState<string | null>(null);

  useEffect(() => {
    if (params.workload.source === "schema") {
      setSchemaText(JSON.stringify(params.workload.schema, null, 2));
      setSchemaError(null);
    }
  }, [params.workload]);

  function onHardwarePreset(preset: string) {
    if (preset === "custom") {
      setParams({ ...params, hardware: { ...params.hardware, preset: "custom" } });
      return;
    }
    const hwPreset = presets?.hardware[preset];
    if (!hwPreset) return;
    setParams({ ...params, hardware: { preset, ...hwPreset } });
  }

  function onWorkloadPreset(preset: string) {
    const p = presets?.workloads[preset];
    if (!p) return;
    if (p.source === "schema") {
      const schema = clone(p.schema);
      setSchemaText(JSON.stringify(schema, null, 2));
      setSchemaError(null);
      setParams({ ...params, workload: { source: "schema", schema } });
      return;
    }
    setParams({
      ...params,
      workload: {
        source: "training_transformer",
        preset,
        model: { preset, ...p.model },
        training: p.training,
      },
    });
  }

  function setHardware<K extends keyof HardwareParams>(key: K, value: HardwareParams[K]) {
    setParams({
      ...params,
      hardware: { ...params.hardware, [key]: value, preset: key === "preset" ? (value as string) : "custom" },
    });
  }

  function setTransformerModel<K extends keyof ModelParams>(key: K, value: ModelParams[K]) {
    if (params.workload.source !== "training_transformer") return;
    setParams({
      ...params,
      workload: {
        ...params.workload,
        preset: "custom",
        model: {
          ...params.workload.model,
          [key]: value,
          preset: key === "preset" ? (value as string) : "custom",
        },
      },
    });
  }

  function setTraining<K extends keyof TrainingParams>(key: K, value: TrainingParams[K]) {
    if (params.workload.source !== "training_transformer") return;
    setParams({
      ...params,
      workload: {
        ...params.workload,
        training: { ...params.workload.training, [key]: value },
      },
    });
  }

  function setRecompute(value: boolean) {
    setParams({
      ...params,
      planner: { ...params.planner, recompute: value },
    });
  }

  function useTransformerWorkload() {
    if (params.workload.source === "training_transformer") return;
    setParams({
      ...params,
      workload: {
        source: "training_transformer",
        preset: DEFAULT_MODEL.preset,
        model: DEFAULT_MODEL,
        training: DEFAULT_TRAINING,
      },
    });
  }

  function useSchemaWorkload() {
    if (params.workload.source === "schema") return;
    const schema = clone(EXAMPLE_SCHEMA);
    setSchemaText(JSON.stringify(schema, null, 2));
    setSchemaError(null);
    setParams({ ...params, workload: { source: "schema", schema } });
  }

  function onSchemaText(next: string) {
    setSchemaText(next);
    try {
      const parsed = JSON.parse(next) as DataflowProgram;
      if (parsed.schema_version !== "dataflow/v1") {
        setSchemaError("schema_version must be dataflow/v1");
        return;
      }
      setSchemaError(null);
      setParams({ ...params, workload: { source: "schema", schema: parsed } });
    } catch (e) {
      setSchemaError(e instanceof Error ? e.message : String(e));
    }
  }

  async function onSchemaImport(e: ChangeEvent<HTMLInputElement>) {
    const input = e.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      onSchemaText(text);
    } catch (err) {
      setSchemaError(err instanceof Error ? err.message : String(err));
    } finally {
      input.value = "";
    }
  }

  const statusLabel =
    previewStatus === "loading" ? "Creating" :
    previewStatus === "error" ? "Needs Fix" :
    previewStatus === "ok" && !previewStale ? "Ready" :
    previewStatus === "ok" && previewStale ? "Stale" : "Draft";

  const transformerPresets = useMemo(
    () => Object.keys(presets?.workloads ?? {}),
    [presets],
  );
  const transformerWorkload =
    params.workload.source === "training_transformer" ? params.workload : null;

  return (
    <div className="panel input-panel">
      <div className="panel-header">
        <h3>Workload Workspace</h3>
        {previewStatus === "loading" && <span className="loading-spinner" aria-hidden="true" />}
        <span className={`tag status-${previewStatus}${previewStale ? " status-stale" : ""}`}>{statusLabel}</span>
        {locked && <span className="tag status-locked">Locked</span>}
      </div>

      <fieldset className="form-sections" disabled={locked || previewStatus === "loading"}>
        <section className="form-section">
          <header className="form-section-header">
            <span className="form-section-title">Workload</span>
            <div className="segmented">
              <button
                type="button"
                className={params.workload.source === "training_transformer" ? "active" : ""}
                onClick={useTransformerWorkload}
              >
                Transformer Training
              </button>
              <button
                type="button"
                className={params.workload.source === "schema" ? "active" : ""}
                onClick={useSchemaWorkload}
              >
                Custom Schema
              </button>
            </div>
          </header>

          {transformerWorkload ? (
            <>
              <FormSubsection title="Preset">
                <div className="form-row">
                  <label className="form-field form-field-wide">
                    <span className="form-field-label">Workload Preset</span>
                    <select
                      value={transformerWorkload.preset}
                      onChange={(e) => onWorkloadPreset(e.target.value)}
                    >
                      {transformerPresets.map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                      <option value="custom">Custom</option>
                    </select>
                  </label>
                </div>
              </FormSubsection>

              <FormSubsection title="Model Architecture Dimensions">
                <div className="form-row">
                  <label className="form-field form-field-wide">
                    <span className="form-field-label">Model Family</span>
                    <select
                      value={transformerWorkload.model.family}
                      onChange={(e) => setTransformerModel("family", e.target.value as ModelFamily)}
                    >
                      {MODEL_FAMILY_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="form-grid">
                  {MODEL_FIELDS.map((f) => (
                    <label key={f.key} className="form-field">
                      <span className="form-field-label">{f.label}</span>
                      <input
                        type="number"
                        min={0}
                        step={1}
                        value={String(transformerWorkload.model[f.key])}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          if (Number.isFinite(v)) setTransformerModel(f.key, v);
                        }}
                      />
                    </label>
                  ))}
                  <label className="form-field form-field-checkbox">
                    <span className="form-field-label">QK Norm</span>
                    <input
                      type="checkbox"
                      checked={transformerWorkload.model.qk_norm}
                      onChange={(e) => setTransformerModel("qk_norm", e.target.checked)}
                    />
                  </label>
                </div>
              </FormSubsection>

              <FormSubsection title="Data Sizing">
                <div className="form-grid">
                  <label className="form-field">
                    <span className="form-field-label">Sequence Length</span>
                    <input
                      type="number" min={1} step={1}
                      value={String(transformerWorkload.training.seqlen)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (Number.isFinite(v)) setTraining("seqlen", v);
                      }}
                    />
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Microbatch Size</span>
                    <input
                      type="number" min={1} step={1}
                      value={String(transformerWorkload.training.num_seqs)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (Number.isFinite(v)) setTraining("num_seqs", v);
                      }}
                    />
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Gradient Accumulation</span>
                    <input
                      type="number" min={1} step={1}
                      value={String(transformerWorkload.training.grad_accum_rounds)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (Number.isFinite(v)) setTraining("grad_accum_rounds", v);
                      }}
                    />
                  </label>
                </div>
              </FormSubsection>

              <FormSubsection title="Training Procedure">
                <div className="form-grid">
                  <label className="form-field">
                    <span className="form-field-label">Optimizer</span>
                    <select
                      value={transformerWorkload.training.optimizer}
                      onChange={(e) => setTraining("optimizer", e.target.value as OptimizerMode)}
                    >
                      {OPTIMIZER_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Training Steps</span>
                    <input
                      type="number" min={1} step={1}
                      value={String(transformerWorkload.training.num_steps)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (Number.isFinite(v)) setTraining("num_steps", v);
                      }}
                    />
                  </label>
                  <label className="form-field form-field-checkbox">
                    <span className="form-field-label">Allow Recompute</span>
                    <input
                      type="checkbox"
                      checked={params.planner.recompute}
                      onChange={(e) => setRecompute(e.target.checked)}
                    />
                  </label>
                </div>
              </FormSubsection>

              <FormSubsection title="Terminal State">
                <label className="form-field form-field-checkbox">
                  <span className="form-field-label">Final State On Slow Memory</span>
                  <input
                    type="checkbox"
                    checked={transformerWorkload.training.final_model_state_on_backing}
                    onChange={(e) => setTraining("final_model_state_on_backing", e.target.checked)}
                  />
                </label>
              </FormSubsection>
            </>
          ) : (
            <div className="schema-editor-wrap">
              <p className="schema-note">
                Please see our{" "}
                <a
                  href="https://github.com/als244/dataflow_sim/tree/master"
                  target="_blank"
                  rel="noreferrer"
                >
                  repo
                </a>{" "}
                for more details on generating a dataflow program. We included some basic{" "}
                <a
                  href="https://github.com/als244/dataflow_sim/blob/master/examples/README.md"
                  target="_blank"
                  rel="noreferrer"
                >
                  examples
                </a>
                .
              </p>
              <div className="schema-editor-toolbar">
                <span className="dim">Common schema JSON</span>
                <label className="reset-btn schema-import-btn">
                  Import JSON
                  <input
                    type="file"
                    accept="application/json,.json"
                    onChange={onSchemaImport}
                  />
                </label>
              </div>
              <textarea
                className="schema-editor"
                spellCheck={false}
                value={schemaText}
                onChange={(e) => onSchemaText(e.target.value)}
              />
              {schemaError && <div className="input-error">schema: {schemaError}</div>}
            </div>
          )}
        </section>

        <section className="form-section">
          <header className="form-section-header">
            <span className="form-section-title">Hardware Environment</span>
          </header>
          <FormSubsection title="Preset">
            <div className="form-row">
              <label className="form-field form-field-wide">
                <span className="form-field-label">Hardware Preset</span>
                <select
                  value={params.hardware.preset}
                  onChange={(e) => onHardwarePreset(e.target.value)}
                >
                  {presets &&
                    Object.keys(presets.hardware).map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  <option value="custom">Custom</option>
                </select>
              </label>
            </div>
          </FormSubsection>
          {HW_FIELD_SECTIONS.map(({ title, fields }) => (
            <FormSubsection key={title} title={title}>
              <div className="form-grid">
                {fields.map((f) => (
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
            </FormSubsection>
          ))}
        </section>

        <div className="workspace-actions">
          <button
            className="submit-btn"
            onClick={onPreview}
            disabled={previewStatus === "loading" || schemaError !== null}
            type="button"
          >
            {previewStatus === "loading" ? "Creating..." : previewStatus === "ok" ? "Update Workload" : "Create Workload"}
          </button>
        </div>
      </fieldset>
      {schemaError && <div className="input-error">Schema: {schemaError}</div>}
      {previewError && <div className="input-error">{previewError}</div>}
    </div>
  );
}
