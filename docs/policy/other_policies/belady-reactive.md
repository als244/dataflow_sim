# belady_reactive

A reactive, oracle-aware eviction policy that takes a *bare* task chain (compute tasks with `inputs / outputs / runtime`, all weights on backing, zero triggers) and produces a fully annotated `TaskChain` consumable by the simulator. It places `release / offload / prefetch` triggers by walking the chain forward through a `ShadowSimulator` that mirrors the real simulator's state machine, evicting the furthest-next-use object (Belady) when the compute is full and chaining offload→prefetch round-trips against actual completion times rather than ideal ones.

Implementation: `src/dataflow_sim/policies/belady_reactive.py`.

## Mechanism

Six phases, applied to a bare chain:

- **Phase 0a — reference stream.** `_compute_uses` builds a per-object sorted list of input-use timestamps from cumulative ideal start times. `_next_use_after` does bisect lookup. This is the Belady oracle.
- **Phase 0b — recompute level.** Stub at `k_o = 0` (no recompute decisions). The signature is in place for a future K-level extension.
- **Phase 1 — initial placement.** Must-place `in(T_1) ∪ out(T_1)` on compute. Greedy-fill the remainder by `first_use` ascending, stopping at `fast_memory_capacity − widest_task_footprint + t1_pinned_size` so cascade and late-arriving prefetches have headroom (slack-aware).
- **Phase 2 — forward simulation with Belady eviction.** Walk the chain inside a `ShadowSimulator`. Before each task, ensure inputs are compute-live and the compute has room for outputs; otherwise pick a Belady victim and emit an offload+prefetch pair. Cascade-aware prefetch placement walks past boundaries when the current one is full. Opportunistic GC uses task-index-based liveness (`last_use_task_idx[obj] <= i`), which is robust to actual-vs-ideal time drift.
- **Phase 3 — trigger-task assignment.** Releases → offloads → prefetches, ordered against `boundary_pool_size[k]` snapshots so retroactive placements respect peak compute usage over the affected boundary range. `offload_done_t` chains prefetches after their paired offload completes; per-stream busy-until is respected.
- **Phase 4 — cascade resolution.** `_ensure_prefetch_v2` walks past boundaries when the immediate boundary won't fit, with a `safe_after` victim filter so an evicted object is never needed by an intervening task.
- **Phase 5 — verify and refine.** Run the real simulator; parse `"cannot prefetch X: insufficient compute capacity"` and `"X is being offloaded"` errors. The first triggers `_shift_prefetch_earlier`; the second drops the over-eager offload+prefetch pair. Loops up to 20 iterations.

### ShadowSimulator

A state mirror of the simulator (`src/dataflow_sim/engine/simulator.py`), forked rather than refactored from a shared base — the simulator's hot path is per-event chronological processing, while the planner's hot path is predicate queries about future state. Advances per task boundary, not per event. Exposes a "decide-and-record" API: trigger decisions are recorded as `(boundary_task_index, kind, obj_id)` and shadow state updates immediately so subsequent decisions see the consequences. Key state: `boundary_pool_size[k]` snapshots, `actual_boundary_end[k]`, per-object `_Entry.producer_task_idx` and `appeared_at`.

## When it wins

Sweep results (default bandwidths bw=8, validated 2026-05-29):

| L  | cap        | sliding         | belady_reactive   | winner            |
|----|------------|-----------------|-------------------|-------------------|
| 3  | ∞ to 500   | ms=100 off=4    | **ms=92 off=0**   | **belady**        |
| 5  | ∞ to 1000  | ms=160 off=8    | **ms=152 off=0**  | **belady**        |
| 5  | 800        | ms=160 off=8    | **ms=152 off=1**  | **belady**        |
| 5  | 600        | ms=160 off=8    | ms=162 off=3      | sliding by 2      |
| 5  | 500        | FAIL            | ms=164 off=3      | **belady only**   |
| 10 | ∞          | ms=310 off=18   | **ms=302 off=0**  | **belady**        |
| 10 | 1500–1000  | ms=310 off=18   | **ms=302 off=4-5**| **belady**        |
| 10 | 800        | ms=310 off=18   | ms=316 off=9      | sliding by 6      |
| 10 | 600        | ms=310 off=18   | ms=322 off=13     | sliding by 12     |
| 10 | 500        | FAIL            | ms=324            | **belady only**   |

The structural win is **tail-end optimization**: belady skips the unnecessary trailing `dW_*` offload tail that the hand-tuned sliding-window pattern always emits regardless of capacity pressure (~5–8% makespan improvement at loose caps). At very tight caps, belady's envelope is strictly wider than sliding's (L=5/10 at cap=500).

## Key design decisions

1. **`boundary_pool_size[k]` snapshots** with peak-over-range checks let the planner correctly cap-check retroactive trigger placements without re-simulating.
2. **Task-index-based GC** (not time-based) — actual-vs-ideal drift would otherwise falsely flag chronologically-live objects dead.
3. **`safe_after` victim filter** — cascade victims at past boundary k must have no use in `(boundary_end, deadline)`, preventing evictions of objects the next task immediately reads.
4. **Drift-aware to_slow subtraction** — offload completion uses `actual_boundary_end[k] >= completion_t + drift_at_offload`, not the ideal boundary end.
5. **Cascade not restricted to most-recent boundary** — past boundaries are eligible as long as the victim's `appeared_at <= boundary_end_k`.
6. **Task-index-based producer filter** — `_Entry.producer_task_idx` rejects trigger placement at boundaries before the object exists; replaces an earlier time-based check that broke under drift.
7. **Slack-aware initial placement** — leaves headroom for `dW_head` / late-arriving weights at tight caps.
8. **Phase 5 handles both error classes** — capacity errors shift prefetch earlier; "object being offloaded" errors remove the over-eager offload+prefetch pair.
9. **`_evict_v2` safety filters** — rejects offload boundaries before the producer task, before the victim's last prior use, or where the pending-outbound window would conflict; cascade retries via `tried_victims`.

## Limitations

- **Makespan trails sliding by 4–16 ticks at L=10 cap ∈ {600, 800}.** Phase 5's shift-earlier strategy fixes feasibility but doesn't optimize trigger placement for makespan. A Phase 5 that also moves OFFLOAD triggers earlier (freeing bps budget for later prefetches) would close this gap.
- **~1.5–2× gap to theoretical minimum capacity** (widest `b_i` working set ≈ 224 for L=3, 320 for L=5/10) due to (a) slack-aware initial-placement greedy-fill and (b) inability to dynamically schedule offloads early enough to free space for late-arriving `dW_head` / `W_*` prefetches at very tight caps.
- **`k_o = 0`** — no recompute decisions. The hook is in place for a future K-level extension.
- **Belady oracle uses cumsum-based `next_use`**, which diverges from actual `next_use` under stalls. Mitigated in practice by the shadow's actual-time tracking, but not eliminated.
