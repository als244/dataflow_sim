# research framing

Academic framing for the auto-policy problem. Pairs with [docs/problem.md](problem.md) (concrete simulator contract) and [docs/policy/README.md](policy/README.md) (policy catalog).

## Problem space

Modern ML training compilers have **complete forward visibility** into the task graph: tensor sizes, kernel runtimes, and transfer placements are known at planning time. Given a finite fast-memory budget, the runtime decides what to keep resident, what to offload to backing, what to prefetch back, and (optionally) what to discard and recompute.

These decisions look like classical paging but classical theory does not capture them. Belady-MIN and online-paging models assume a single uniform cache, fixed miss penalty, and free eviction. The training setting has *two parallel FIFO transfer streams with bandwidth-proportional latency, mandatory write-back of mutated state, output reservation at task start, and a per-intermediate recompute trade-off curve*.

This is an **offline oracle-aware scheduling** problem, not online prediction. See [docs/problem.md](problem.md) for the exact simulator contract the policy must produce annotated chains against.

## Key decision axes

The auto-policy has eight degrees of freedom, not three:

1. **Initial placement** — backing/compute residency at t=0.
2. **Release decisions** — drop a compute entry whose value is dead or backing-backed.
3. **Offload decisions** — to-slow write-back (semantically required for mutated state like gradients).
4. **Prefetch decisions** — from-slow reload before next use.
5. **Trigger-task assignment** — *which prior boundary* fires a release/offload/prefetch. Late triggers minimize compute occupancy; early triggers give streams slack.
6. **Stream FIFO ordering** — when multiple triggers co-fire on the same stream, queue order determines which downstream task is unblocked first.
7. **Cascade dependencies** — some prefetches can only fit after a paired offload completes; "issue after" semantics are required, not just "issue at."
8. **Recomputation level** `k_o ∈ {0..K}` — per-intermediate point on a (stored-size, recompute-time) curve. Generalizes Chen-2016 binary checkpointing to a K-level continuum. Compute allocation stays `s(o)`; only backing occupancy and transfer cost vary.

Axes 5–8 are the ones a naive "what to evict / what to load" framing collapses or omits.

## Prior-work pointers

Closest neighbors and where each diverges (see archived original for the full table):

- **Belady-MIN (1966)** — optimal offline paging; single resource, uniform miss cost, free eviction. Ours: asymmetric per-object transfer time, real eviction cost.
- **Sleator–Tarjan (1985)** — competitive online paging; we have an oracle, so competitive ratios are wrong framing.
- **Activation checkpointing (Chen 2016)** — binary store/recompute; no offload, no stream contention.
- **ZeRO-Offload / ZeRO-Infinity (Ren 2021; Rajbhandari 2021)** — heuristic, transformer-tuned, not oracle-aware in our sense.
- **Capuchin (Peng 2020)** — online profiling, binary swap-vs-recompute.
- **FlexGen (Sheng 2023)** — inference-time, coarse block-level.
- **Register allocation (Chaitin 1981; Poletto–Sarkar 1999)** — cleanest classical analog; uniform-cost single resource, no direction-asymmetric queueing.
- **POET (Patil 2022)** — closest in spirit; MILP-based, doesn't model two parallel FIFO streams with queue contention.

This work is the only entry that combines: oracle + multi-stream + bidirectional write-back + output reservation + multi-resource + K-level recompute.

## Complexity and feasibility highlights

**Hardness.** The decision problem subsumes multi-machine scheduling with precedence constraints (NP-hard, Garey–Johnson 1979). Per-object integer `k_o` adds a combinatorial layer. Practical approach: greedy approximations with verification re-planning.

**Feasibility theorem (lazy strategy).** Let `W = max_i ∑_{o ∈ in(T_i) ∪ out(T_i)} s(o)` be the *widest single-task compute footprint*. If `C_fast ≥ W` and backing has room for the chosen stored fragments, a feasible schedule exists: between tasks, evict everything not needed by `T_{i+1}`; offload anything needed later (at max `k_o`); prefetch + recompute at each task start. At most one task's footprint is resident at any instant.

The bound is tight — `C_fast < W` makes `T_i` impossible because compute requires its full input set fast-memory-resident plus output reservation simultaneously.

**Optimality conjecture.** Greedy-Belady (eviction by furthest-next-use, modulated by stream load) is conjectured within `O(τ_max / r_min)` of optimal makespan — at worst one transfer's bandwidth wasted vs. an oracle. Unproven; natural open problem alongside ILP-baseline benchmarks for small chains.

## Open extensions

Online variants (dynamic graphs), joint `k_o` + cache optimization, multi-compute, multi-tier memory (NVMe / network), bandwidth contention (concurrent transfers per stream), and tighter approximation bounds.

---

*Full derivation, mathematical formulation (§6), and six-phase greedy template (§7) are archived verbatim at [docs/internal/research-original.md](internal/research-original.md). This live doc is the framing a reader needs alongside [problem.md](problem.md) and the [policy catalog](policy/README.md).*
