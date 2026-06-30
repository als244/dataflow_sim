# Recompute Selection

Activation recomputation trades compute-stream time for memory and transfer
relief: instead of saving a layer's activations (keeping them resident or
round-tripping them over tier-link), the chain re-produces them in the layer's
recompute task right before backward. Recompute and offload are competing
ways to regenerate the same bytes, and which is cheaper depends
on the whole schedule — model shape, sequence length, hardware ratios, and
memory capacity — so the choice is made per configuration, by measurement.

## Architecture

Three layers, deliberately separated:

1. **Workload variants (mechanics).**
   A `TrainingWorkloadBuilder`, for example `Llama3ForTraining`, builds the
   chain for any per-activation choice through
   `model.build_training_workload(training, hw, recompute={...})`. For a
   recomputed layer instance, the forward task stops producing the saved
   activation and a recompute task re-produces it from the layer input:

   ```text
   f_k_j_i = (deps=[in_act, W_i], out=[y_k_j_i])
   r_k_j_i = (deps=[in_act, W_i], out=[A_k_j_i], runtime=R)
   ```

   For a saved layer instance, no `r_k_j_i` task is emitted.

   `R` comes from the layer's `.recompute` compute block; backward tasks are
   untouched because their memory ops already assume recomputed activations are
   HBM-hot. The
   workload also publishes a rewrite table
   (`metadata["recompute_rewrites"]`): per saved-activation object, the
   discrete options available, including the forward and recompute
   `compute_block_key` values that define the tradeoff. Today the options are
   binary (level 0 = save-full, level 1 = recompute-full); partial
   recomputation adds intermediate levels with `saved_bytes` between the
   extremes without changing any interface.

2. **Stall/backlog evidence (information).**
   `dataflow_sim.engine.stall_report.build_stall_report(chain, log)` turns a
   simulator run into ground-truth blame: compute stalls split into
   input-wait (attributed to the arriving object) vs capacity-wait, stream
   busy time, backlog windows (periods with enqueued-but-unstarted
   transfers), and each object's transfer time inside backlog windows. It
   works on snapshot-free logs and is policy-agnostic.

3. **Selection (decision).** `dataflow_sim.planning.recompute.plan_with_recompute`
   takes a chain-variant builder, the rewrite table, and a policy function.
   It evaluates a fixed seed family (none / all / every-other) so it never
   loses to a trivial choice, then runs an evidence loop from the all-saved
   plan: rank activations by blame (stall + backlog overlap) minus added
   recompute time, convert the best half of the positive-net candidates,
   replan, and accept only if the simulated makespan improves. The best plan
   seen anywhere wins. Residency policies (PressureFit etc.) are untouched —
   a recompute variant is just another `TaskChain`.

## Measured behavior (scripts/recompute_sweep.py)

The optimal choice swings across model family, sequence length, hardware, and
capacity cap. Current sweeps should cover the public preset families
(`llama3_*`, `qwen3_*`, `qwen3_moe_30B-3B`, and `olmoe_7B-1B`) across multiple
scales rather than relying on one tiny stress case:

- **MoE models at shorter sequence lengths:** recompute can be attractive
  because expert state dominates memory and activations are cheap to recreate.
- **Longer sequences on slower compute hardware:** recompute can lose because
  attention-heavy replay is more expensive than the avoided transfers.
- **Dense models:** often prefer little or no recompute, with isolated
  activation instances selected when memory pressure is highly localized.
- **Middle-capacity regimes:** partial recompute frequently beats fixed
  none/all/half choices because the evidence loop can target the most
  transfer-blamed saved activations.

Selection costs well under a second per config on top of normal planning.

## Known limitations / next steps

- **Blame is transfer-based.** An activation that stays resident produces
  no stall or backlog evidence even when recomputing it would free pool
  headroom for other traffic; the seed family covers that regime today. A
  residency-pressure term in the report would let the evidence loop find
  those conversions directly.
- The loop only converts (never un-converts); starting points other than
  all-saved are reached via seeds. Bidirectional moves become more useful
  once partial levels exist.
- Per-instance decisions are independent; for grad accumulation the loop
  usually converges to symmetric choices across rounds on its own.
