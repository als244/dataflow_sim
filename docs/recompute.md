# Recompute Selection

Activation recomputation trades compute-stream time for memory and transfer
relief: instead of saving a layer's activations (keeping them resident or
round-tripping them over PCIe), the chain re-produces them in the layer's
recompute slot right before backward. Recompute and offload are competing
rematerialization channels for the same bytes, and which is cheaper depends
on the whole schedule — model shape, sequence length, hardware ratios, and
memory capacity — so the choice is made per configuration, by measurement.

## Architecture

Three layers, deliberately separated:

1. **Workload variants (mechanics).**
   `build_transformer_training_workload(spec, hw, cfg, recompute={...})`
   builds the chain for any per-activation choice. For a recomputed layer
   instance, the forward task stops producing the saved activation and the
   recompute slot re-produces it from the layer input:

   ```text
   f_k_j_i = (deps=[in_act, W_i], out=[y_k_j_i])
   r_k_j_i = (deps=[in_act, W_i], out=[A_k_j_i], runtime=R)
   ```

   `R` is the layer's forward compute minus the final down projection(s)
   (`layer_recompute_microseconds`); backward tasks are untouched — their
   memory ops already assume recomputed activations are HBM-hot. The
   workload also publishes a rewrite table
   (`metadata["recompute_rewrites"]`): per saved-activation object, the
   discrete options available. Today the options are binary (level 0 =
   save-full, level 1 = recompute-full); partial recomputation (save part
   of the activation, recompute the rest) adds intermediate levels with
   `saved_bytes` between the extremes without changing any interface.

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

The optimal choice swings across a 24-config grid (llama3-8B,
sparse_16Bx3B, qwen3_30Bx3B × H100/RTX_5090 × two caps × S∈{4K, 16K}):

- **qwen3_30Bx3B (MoE) at S=4K:** recompute *everything* — expert weights
  dominate memory and activations are cheap to re-produce.
- **The same model at S=16K on RTX_5090:** recompute *nothing* — slow
  compute makes the attention-heavy recompute too expensive.
- **Dense llama3-8B:** mostly none, with small evidence-found refinements
  (k=1–10 layers).
- **In between** (sparse_16Bx3B, qwen3 at 16K on H100): partial counts
  (k=11–19) win, and on several configs the evidence loop beats every fixed
  choice (up to ~4% over the best of none/all/half).

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
