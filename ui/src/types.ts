// Mirrors the JSON shape of dataflow_sim.core.schema.EventLog.

export type Location = "backing" | "fast";
export type ObjectType = "weight" | "activation" | "gradient" | "optimizer" | "other";
export type MemoryState =
  | "live"
  | "reserved"
  | "pending_inbound"
  | "inbound"
  | "pending_outbound"
  | "outbound";

export interface MemoryEntry {
  id: string;
  size: number;
  location: Location;
  type: ObjectType;
  state: MemoryState;
  next_ref_t: number | null;
}

export interface ActiveTask {
  id: string;
  ends_at: number;
}

export interface Reference {
  obj_id: string;
  ref_t: number;
  ref_task: string;
}

export interface Snapshot {
  memory: MemoryEntry[];
  total_size: number;
  active_task: ActiveTask | null;
  reference_stream: Reference[];
}

export type TransferDirection = "from_slow" | "to_slow";
export type EventKind =
  | "task_start"
  | "task_end"
  | "release"
  | "transfer_enqueue"
  | "transfer_start"
  | "transfer_end"
  | "transfer_deferred";

export interface SimEvent {
  t: number;
  kind: EventKind;
  snapshot: Snapshot;
  task_id: string | null;
  object_ids: string[];
  transfer_obj: string | null;
  transfer_direction: TransferDirection | null;
}

export type Track = "compute" | "from_slow" | "to_slow";

export interface TaskInterval {
  task_id: string;
  start: number;
  end: number;
  track: Track;
}

export interface MemoryTracePoint {
  t: number;
  fast_bytes_by_band: Record<string, number>;
}

export interface EventLog {
  task_intervals: TaskInterval[];
  events: SimEvent[];
  peak_fast_memory_bytes: number;
  memory_trace: MemoryTracePoint[];
}
