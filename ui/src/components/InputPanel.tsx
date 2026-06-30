import { useEffect, useMemo, useState, type ChangeEvent, type ReactNode } from "react";

export type Policy = "sliding_window" | "belady_reactive" | "roundtrip_planner" | "max_reduce" | "min_grow" | "pressurefit";
export type OptimizerMode = "none" | "adamw" | "muon";
export type ModelFamily =
  | "llama3"
  | "qwen3"
  | "qwen3_moe"
  | "olmoe"
  | "qwen3_hybrid_dense"
  | "qwen3_hybrid_moe"
  | "deepseek_v3";

export interface HardwareParams {
  preset: string;
  peak_tflops_bf16: number;
  peak_tflops_fp8: number;
  peak_tflops_fp4: number | null;
  fast_memory_bw_gbs: number;
  from_slow_bw_gbs: number;
  to_slow_bw_gbs: number;
  matmul_eff_bf16: number;
  matmul_eff_fp8: number;
  matmul_eff_fp4: number | null;
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
  intermediate_size?: number;
  full_attention_interval?: number;
  linear_num_key_heads?: number;
  linear_key_head_dim?: number;
  linear_num_value_heads?: number;
  linear_value_head_dim?: number;
  linear_conv_kernel_dim?: number;
  gdn_chunk_size?: number;
  router_aux_loss_coef?: number;
  mtp_num_hidden_layers?: number;
  first_k_dense_replace?: number;
  q_lora_rank?: number;
  kv_lora_rank?: number;
  qk_nope_head_dim?: number;
  qk_rope_head_dim?: number;
  v_head_dim?: number;
  routed_scaling_factor?: number;
  scoring_func?: string;
}

export interface TrainingParams {
  seqlen: number;
  num_seqs: number;
  grad_accum_rounds: number;
  num_steps: number;
  optimizer: OptimizerMode;
  final_model_state_on_backing: boolean;
}

export type DTypeName = "bf16" | "fp8" | "fp4";

export interface DatatypeParams {
  weight_dtype: DTypeName;
  activation_dtype: DTypeName;
  expert_dispatch_dtype: DTypeName;
  gradient_dtype: DTypeName;
  optimizer_dtype: DTypeName;
  compute_precision: DTypeName;
  expert_weight_dtype: DTypeName;
  expert_compute_precision: DTypeName;
}

export interface DataflowCost {
  kind: "fixed" | "roofline" | "sum";
  name?: string | null;
  runtime_us?: number | null;
  flops?: number;
  memory_bytes?: number;
  efficiency?: "matmul" | "matmul_bf16" | "matmul_fp8" | "matmul_fp4" | "attention" | "attention_fwd" | "attention_bwd" | "memory" | "custom";
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

export interface ModelTrainingWorkloadParams {
  source: "model_training";
  preset: string;
  model: ModelParams;
  training: TrainingParams;
  datatypes: DatatypeParams;
}

export interface SchemaWorkloadParams {
  source: "schema";
  schema: DataflowProgram;
}

export type WorkloadParams = ModelTrainingWorkloadParams | SchemaWorkloadParams;

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

export interface ModelTrainingWorkloadPreset {
  source: "model_training";
  preset: string;
  model: Omit<ModelParams, "preset">;
  training: TrainingParams;
  datatypes: DatatypeParams;
  description: string;
}

export interface SchemaWorkloadPreset {
  source: "schema";
  preset: string;
  schema: DataflowProgram;
  description: string;
}

export type WorkloadPreset = ModelTrainingWorkloadPreset | SchemaWorkloadPreset;

export interface ModelFieldDescriptor {
  key: keyof ModelParams;
  label: string;
  kind?: "number" | "boolean" | "text";
  min?: number;
  step?: number;
  advanced?: boolean;
}

export interface ModelFamilyDescriptor {
  key: ModelFamily;
  label: string;
  presets: string[];
  fields: ModelFieldDescriptor[];
}

export interface Presets {
  workloads: Record<string, WorkloadPreset>;
  model_families?: Record<string, ModelFamilyDescriptor>;
  hardware: Record<string, Omit<HardwareParams, "preset">>;
}

export const DEFAULT_HARDWARE: HardwareParams = {
  preset: "H100",
  peak_tflops_bf16: 989,
  peak_tflops_fp8: 1978,
  peak_tflops_fp4: null,
  fast_memory_bw_gbs: 3000,
  from_slow_bw_gbs: 50,
  to_slow_bw_gbs: 50,
  matmul_eff_bf16: 0.65,
  matmul_eff_fp8: 0.65,
  matmul_eff_fp4: null,
  attn_fwd_eff: 0.6,
  attn_bwd_eff: 0.5,
  mem_eff: 0.9,
};

export const DEFAULT_MODEL: ModelParams = {
  preset: "qwen3_5_35B-A3B",
  family: "qwen3_hybrid_moe",
  vocab_size: 248320,
  n_layers: 40,
  d_model: 2048,
  head_dim: 256,
  n_heads: 16,
  n_kv_heads: 2,
  expert_dim: 512,
  num_shared_experts: 1,
  num_routed_experts: 256,
  top_k: 8,
  qk_norm: true,
  intermediate_size: 0,
  full_attention_interval: 4,
  linear_num_key_heads: 16,
  linear_key_head_dim: 128,
  linear_num_value_heads: 32,
  linear_value_head_dim: 128,
  linear_conv_kernel_dim: 4,
  gdn_chunk_size: 64,
  router_aux_loss_coef: 0.001,
  mtp_num_hidden_layers: 1,
  first_k_dense_replace: 0,
  q_lora_rank: 0,
  kv_lora_rank: 0,
  qk_nope_head_dim: 0,
  qk_rope_head_dim: 0,
  v_head_dim: 0,
  routed_scaling_factor: 1,
  scoring_func: "sigmoid",
};

export const DEFAULT_TRAINING: TrainingParams = {
  seqlen: 4096,
  num_seqs: 4,
  grad_accum_rounds: 1,
  num_steps: 1,
  optimizer: "none",
  final_model_state_on_backing: false,
};

export const DEFAULT_DATATYPES: DatatypeParams = {
  weight_dtype: "bf16",
  activation_dtype: "bf16",
  expert_dispatch_dtype: "bf16",
  gradient_dtype: "bf16",
  optimizer_dtype: "bf16",
  compute_precision: "bf16",
  expert_weight_dtype: "bf16",
  expert_compute_precision: "bf16",
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
    source: "model_training",
    preset: "qwen3_5_35B-A3B",
    model: DEFAULT_MODEL,
    training: DEFAULT_TRAINING,
    datatypes: DEFAULT_DATATYPES,
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

const DTYPE_OPTIONS: { value: DTypeName; label: string }[] = [
  { value: "bf16", label: "BF16" },
  { value: "fp8", label: "FP8" },
  { value: "fp4", label: "FP4" },
];

const MODEL_FAMILY_OPTIONS: { value: ModelFamily; label: string }[] = [
  { value: "llama3", label: "Llama 3" },
  { value: "qwen3", label: "Qwen3 Dense" },
  { value: "qwen3_moe", label: "Qwen3 MoE" },
  { value: "olmoe", label: "OLMoE" },
  { value: "qwen3_hybrid_dense", label: "Qwen3.5/3.6 Dense" },
  { value: "qwen3_hybrid_moe", label: "Qwen3.5/3.6 MoE" },
  { value: "deepseek_v3", label: "DeepSeek-V3" },
];

const HW_ACCELERATOR_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "peak_tflops_bf16", label: "Peak BF16 TFLOP/s", min: 0.1, step: 1 },
  { key: "peak_tflops_fp8", label: "Peak FP8 TFLOP/s", min: 0.1, step: 1 },
  { key: "peak_tflops_fp4", label: "Peak FP4 TFLOP/s", min: 0.1, step: 1 },
  { key: "fast_memory_bw_gbs", label: "Fast Memory BW (GB/s)", min: 1, step: 10 },
];

const HW_SLOW_MEMORY_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "from_slow_bw_gbs", label: "From-Slow BW (GB/s)", min: 0.1, step: 1 },
  { key: "to_slow_bw_gbs", label: "To-Slow BW (GB/s)", min: 0.1, step: 1 },
];

const HW_KERNEL_FIELDS: { key: keyof Omit<HardwareParams, "preset">; label: string; step?: number; min?: number }[] = [
  { key: "mem_eff", label: "Memory Efficiency", min: 0.01, step: 0.01 },
  { key: "matmul_eff_bf16", label: "BF16 Matmul Efficiency", min: 0.01, step: 0.01 },
  { key: "matmul_eff_fp8", label: "FP8 Matmul Efficiency", min: 0.01, step: 0.01 },
  { key: "matmul_eff_fp4", label: "FP4 Matmul Efficiency", min: 0.01, step: 0.01 },
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

const NULLABLE_HW_FIELDS = new Set<keyof HardwareParams>([
  "peak_tflops_fp4",
  "matmul_eff_fp4",
]);

const FALLBACK_MODEL_FIELDS: ModelFieldDescriptor[] = [
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
  { key: "qk_norm", label: "QK Norm", kind: "boolean" },
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
        source: "model_training",
        preset,
        model: { preset, ...p.model },
        training: p.training,
        datatypes: p.datatypes,
      },
    });
  }

  function setHardware<K extends keyof HardwareParams>(key: K, value: HardwareParams[K]) {
    setParams({
      ...params,
      hardware: { ...params.hardware, [key]: value, preset: key === "preset" ? (value as string) : "custom" },
    });
  }

  function setModelTrainingModel<K extends keyof ModelParams>(key: K, value: ModelParams[K]) {
    if (params.workload.source !== "model_training") return;
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
    if (params.workload.source !== "model_training") return;
    setParams({
      ...params,
      workload: {
        ...params.workload,
        training: { ...params.workload.training, [key]: value },
      },
    });
  }

  function setDatatype<K extends keyof DatatypeParams>(key: K, value: DatatypeParams[K]) {
    if (params.workload.source !== "model_training") return;
    setParams({
      ...params,
      workload: {
        ...params.workload,
        datatypes: { ...params.workload.datatypes, [key]: value },
      },
    });
  }

  function setRecompute(value: boolean) {
    setParams({
      ...params,
      planner: { ...params.planner, recompute: value },
    });
  }

  function useModelTrainingWorkload() {
    if (params.workload.source === "model_training") return;
    setParams({
      ...params,
      workload: {
        source: "model_training",
        preset: DEFAULT_MODEL.preset,
        model: DEFAULT_MODEL,
        training: DEFAULT_TRAINING,
        datatypes: DEFAULT_DATATYPES,
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

  const modelTrainingPresets = useMemo(
    () => Object.keys(presets?.workloads ?? {}),
    [presets],
  );
  const modelTrainingWorkload =
    params.workload.source === "model_training" ? params.workload : null;
  const modelFamilyOptions = useMemo(
    () => {
      const families = presets?.model_families;
      if (!families) return MODEL_FAMILY_OPTIONS;
      return Object.values(families).map((family) => ({
        value: family.key,
        label: family.label,
      }));
    },
    [presets],
  );
  const activeModelFields = useMemo(
    () => {
      if (!modelTrainingWorkload) return FALLBACK_MODEL_FIELDS;
      return presets?.model_families?.[modelTrainingWorkload.model.family]?.fields ?? FALLBACK_MODEL_FIELDS;
    },
    [modelTrainingWorkload, presets],
  );

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
                className={params.workload.source === "model_training" ? "active" : ""}
                onClick={useModelTrainingWorkload}
              >
                Model Training
              </button>
              <button
                type="button"
                className={params.workload.source === "schema" ? "active" : ""}
                onClick={useSchemaWorkload}
              >
                Custom Dataflow Program
              </button>
            </div>
          </header>

          {modelTrainingWorkload ? (
            <>
              <FormSubsection title="Preset">
                <div className="form-row">
                  <label className="form-field form-field-wide">
                    <span className="form-field-label">Workload Preset</span>
                    <select
                      value={modelTrainingWorkload.preset}
                      onChange={(e) => onWorkloadPreset(e.target.value)}
                    >
                      {modelTrainingPresets.map((name) => (
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
                      value={modelTrainingWorkload.model.family}
                      onChange={(e) => setModelTrainingModel("family", e.target.value as ModelFamily)}
                    >
                      {modelFamilyOptions.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="form-grid">
                  {activeModelFields.map((f) => {
                    const key = f.key;
                    const kind = f.kind ?? "number";
                    const value = modelTrainingWorkload.model[key];
                    if (kind === "boolean") {
                      return (
                        <label key={key} className="form-field form-field-checkbox">
                          <span className="form-field-label">{f.label}</span>
                          <input
                            type="checkbox"
                            checked={Boolean(value)}
                            onChange={(e) => setModelTrainingModel(key, e.target.checked as ModelParams[typeof key])}
                          />
                        </label>
                      );
                    }
                    if (kind === "text") {
                      return (
                        <label key={key} className="form-field">
                          <span className="form-field-label">{f.label}</span>
                          <input
                            type="text"
                            value={String(value ?? "")}
                            onChange={(e) => setModelTrainingModel(key, e.target.value as ModelParams[typeof key])}
                          />
                        </label>
                      );
                    }
                    return (
                      <label key={key} className="form-field">
                        <span className="form-field-label">{f.label}</span>
                        <input
                          type="number"
                          min={f.min ?? 0}
                          step={f.step ?? 1}
                          value={String(value ?? 0)}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            if (Number.isFinite(v)) setModelTrainingModel(key, v as ModelParams[typeof key]);
                          }}
                        />
                      </label>
                    );
                  })}
                </div>
              </FormSubsection>

              <FormSubsection title="Datatypes">
                <p className="form-note dim">
                  See{" "}
                  <a
                    href="https://github.com/als244/dataflow_sim/blob/master/docs/datatypes.md"
                    target="_blank"
                    rel="noreferrer"
                  >
                    datatype option docs
                  </a>
                  {" "}for exact byte and compute-precision semantics.
                </p>
                <div className="form-grid">
                  <label className="form-field">
                    <span className="form-field-label">Weight DType</span>
                    <select
                      value={modelTrainingWorkload.datatypes.weight_dtype}
                      onChange={(e) => setDatatype("weight_dtype", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Activation DType</span>
                    <select
                      value={modelTrainingWorkload.datatypes.activation_dtype}
                      onChange={(e) => setDatatype("activation_dtype", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Parameter Gradient DType</span>
                    <select
                      value={modelTrainingWorkload.datatypes.gradient_dtype}
                      onChange={(e) => setDatatype("gradient_dtype", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Optimizer DType</span>
                    <select
                      value={modelTrainingWorkload.datatypes.optimizer_dtype}
                      onChange={(e) => setDatatype("optimizer_dtype", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Compute Precision</span>
                    <select
                      value={modelTrainingWorkload.datatypes.compute_precision}
                      onChange={(e) => setDatatype("compute_precision", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Expert Weight DType</span>
                    <select
                      value={modelTrainingWorkload.datatypes.expert_weight_dtype}
                      onChange={(e) => setDatatype("expert_weight_dtype", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Expert Compute Precision</span>
                    <select
                      value={modelTrainingWorkload.datatypes.expert_compute_precision}
                      onChange={(e) => setDatatype("expert_compute_precision", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="form-field">
                    <span className="form-field-label">Expert Dispatch DType</span>
                    <select
                      value={modelTrainingWorkload.datatypes.expert_dispatch_dtype}
                      onChange={(e) => setDatatype("expert_dispatch_dtype", e.target.value as DTypeName)}
                    >
                      {DTYPE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
              </FormSubsection>

              <FormSubsection title="Data Sizing">
                <div className="form-grid">
                  <label className="form-field">
                    <span className="form-field-label">Sequence Length</span>
                    <input
                      type="number" min={1} step={1}
                      value={String(modelTrainingWorkload.training.seqlen)}
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
                      value={String(modelTrainingWorkload.training.num_seqs)}
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
                      value={String(modelTrainingWorkload.training.grad_accum_rounds)}
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
                      value={modelTrainingWorkload.training.optimizer}
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
                      value={String(modelTrainingWorkload.training.num_steps)}
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
                    checked={modelTrainingWorkload.training.final_model_state_on_backing}
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
                <span className="dim">DataflowProgram JSON</span>
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
                {fields.map((f) => {
                  const value = params.hardware[f.key];
                  const nullable = NULLABLE_HW_FIELDS.has(f.key as keyof HardwareParams);
                  const unsupportedPresetValue = value === null && params.hardware.preset !== "custom";
                  return (
                    <label key={f.key} className="form-field">
                      <span className="form-field-label">{f.label}</span>
                      <input
                        type={unsupportedPresetValue ? "text" : "number"}
                        min={f.min}
                        step={f.step ?? 1}
                        placeholder={nullable ? "--" : undefined}
                        disabled={unsupportedPresetValue}
                        value={value === null ? (unsupportedPresetValue ? "--" : "") : String(value)}
                        onChange={(e) => {
                          if (e.target.value === "" && nullable) {
                            setHardware(f.key, null as HardwareParams[typeof f.key]);
                            return;
                          }
                          const v = Number(e.target.value);
                          if (Number.isFinite(v)) setHardware(f.key, v as HardwareParams[typeof f.key]);
                        }}
                      />
                    </label>
                  );
                })}
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
