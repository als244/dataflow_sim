# The simulator scheduling problem — formal note

Policy-agnostic problem statement for the dataflow_sim. Any auto-policy (`belady_reactive`, `roundtrip_planner`, `max_reduce`, `min_grow`, `pressurefit`, ...) is a candidate algorithm for the optimization problem defined here. This note exists so per-policy design docs can refer to "the problem" instead of redefining it, and so we have a shared vocabulary for what "optimal" would mean and why it is hard.

---

## 1. Setting

### Compute graph
A linear-ish DAG of compute tasks, all executed on a single **compute stream** (serial, no compute-parallelism):

- `f_1, ..., f_L` (forward per layer)
- `head` (combined head fwd+bwd)
- `b_L, ..., b_1` (backward per layer, reverse order)
- optional `r_i` tasks that recreate saved activations when recompute is selected

For an L=32 model with no recompute: **2L+1 = 65 active compute tasks**.
Selected recompute instances add real `r_i` tasks immediately before their
matching backward task.

Each task has fixed (known at planning time) attributes:
- `runtime` (deterministic, µs)
- `reads`: object IDs whose bytes must be compute-resident at task start
- `writes`: object IDs produced on-compute by this task
- `mutates_inputs`: subset of `reads` whose bytes are overwritten in place (canonical case: backward writes into the pre-allocated `dW_i`)

### Object set
Roughly `~5L` objects of widely varying size (MBs to GBs):
- `input` (backing-init, used by `f_1`)
- `W_1..W_L` (backing-init weights; read by both `f_i` and `b_i`; mutated nowhere in step — re-loadable from backing)
- `dW_1..dW_L` (pre-allocated grads; mutated by `b_i`; must be offloaded to backing at end of step — not re-loadable as-was)
- `A_1..A_L` (saved activations; produced by `f_i`, consumed by `b_i`; on-compute origin)
- `y_1..y_{L-1}`, `dy_1..dy_{L-1}` (layer outputs / output grads; on-compute origin)
- A few head-specific objects (`head_W`, `head_dW`, `head_A`, etc.)

Each object has a known `size_bytes`, a known set of producing/consuming tasks, and (for backing-init objects) a known backing source.

### Hardware abstraction
- **Compute stream**: serial; task `T` cannot start until (a) prior compute task ended AND (b) all `T.reads ∪ T.writes ∪ T.mutates_inputs` are compute-resident.
- **from-slow stream**: FIFO; serves prefetches backing→compute; one transfer at a time; transfer time = `bytes / from_slow_bandwidth`.
- **to-slow stream**: FIFO; serves offloads compute→backing; same model.
- All three streams run in **parallel** with each other.
- **Compute memory pool**: hard cap (bytes); the sum of resident-or-in-transit object bytes must not exceed the cap at any continuous-time instant.

### Lifecycle semantics (one nuance that bites)
An object is "in the pool" from the moment its **from_slow starts** (not finishes) until the moment its **release or to_slow finishes** (not starts). The pool is debited by an in-flight outbound transit until the to_slow completes — this "to_slow tail" is one of the main contention sources policies must reason about.

---

## 2. Formal statement

### Decision variables
Given the inputs above, a **schedule** is a tuple `(R, H, D)` where:

- `R ⊆ BackingInitObjects` — the subset pre-placed on compute at `t=0`.
- `H = [(obj_id, start_time)]` — sequence of from_slow events on the inbound FIFO.
- `D = [(obj_id, start_time)]` — sequence of to_slow events on the outbound FIFO.

A schedule implicitly determines:
- The compute task start times (each task starts at the latest of its predecessor's end and its inputs' arrival).
- The resident set at every instant (pre-placement + transfer effects).
- The total makespan `M(R, H, D)` = end time of `b_1` (or last compute task).

### Constraints
1. **Residency at use**: for every compute task `T` and every `o ∈ T.reads ∪ T.writes ∪ T.mutates_inputs`, `o` is resident at `T.start`.
2. **Cap**: for every instant `t`, `Σ size(o) for o resident-or-in-transit at t ≤ cap`.
3. **Stream FIFO**: events on H and D do not overlap (one transfer per stream at a time).
4. **Conservation**: every prefetched object must have a backing source available; every offloaded object's bytes are durably valid (writeback for mutated objects, no-op for un-mutated backing-source objects).
5. **Mutation correctness**: a mutated object's writeback must happen **after** the mutating task and **before** the backing's next read of it (in our setup: end of step).

### Objective
`minimize M(R, H, D)`

---

## 3. What is NOT a decision

A non-trivial fraction of the schedule is **forced** by the workload and not subject to optimization. Recognizing this shrinks the apparent decision space:

- **Forced residency**: at task `T`, `T.reads ∪ T.writes ∪ T.mutates_inputs` are all resident — no algorithm has a choice.
- **Forced presence interval**: the *minimum* residency interval of an object is `[first_use, last_use]`. Any contiguous "always resident" plan is feasible; any plan that releases mid-interval must arrange a prefetch back in time for the next use.
- **Forced production**: outputs of compute tasks materialize on compute at task end. They cannot be "scheduled elsewhere."
- **Forced consumption order**: the compute DAG is fixed (no task reordering in scope).
- **Forced minimum writeback**: every mutated object with a backing home (i.e., every `dW_i`) must complete a to_slow at some point before the step's end. The *time* is a decision; the *occurrence* is not.

The **actual decision space** is therefore:
1. Which backing-init objects to pre-place (yes/no per object).
2. For each non-backing-init or "release-and-reload-eligible" object, which of its inter-use gaps to drop residency in (and via what mechanism: release vs offload).
3. The temporal ordering of the resulting H and D transfer queues.

For a stacked training chain this is on the order of a few hundred binary decisions plus their scheduling — small in absolute terms, but combinatorially explosive when coupled with the timing constraints below.

---

## 4. Why "we have perfect information" does not imply tractable

The natural intuition — "we know all sizes, runtimes, dependencies ahead of time, so we should be able to plan optimally" — is misleading. Perfect information removes online uncertainty, but doesn't remove the structural combinatorics. Three reasons:

### 4.1 Self-referential timing
The cost of a decision depends on the schedule, and the schedule depends on the decisions. Concretely: "should I evict object `o` at boundary `k`?" depends on "how busy will the from_slow FIFO be when I need to reload `o`?", which depends on "what else did I evict?". `max_reduce` breaks this loop by using a static heuristic (residency intervals modeled at task-boundary granularity, transit time ignored) — which is also why it can mispredict in regimes with heavy stream contention.

### 4.2 Continuous-time pool constraint
The cap is enforced at every continuous-time instant, not just at task boundaries. A schedule that looks fine at every `task.start` can still violate the cap *during* a task because of in-flight transit (the to_slow tail). The simulator's drain loop catches this and stalls, but a planner that ignores it predicts a wrong makespan.

### 4.3 Two parallel transfer streams with shared resource budget
H and D are independent in terms of bandwidth (no contention between them) but coupled through the pool: a to_slow that's "running for free in parallel" still occupies bytes until it lands. Optimizing H and D separately and then composing them gives the wrong answer.

---

## 5. Relation to known computer-science problems

There is **no exact match** in the classical literature. The closest analogues, with the gaps:

### 5.1 Offline paging (Belady, 1966)
**The classic**: given a sequence of page references and a cache of `k` pages, minimize the number of cache misses. Belady's rule (evict the page used furthest in future) is optimal in polynomial time.

**Why it doesn't fit**:
- **Uniform page size** vs. our MB–GB variable-size objects → ours is a *weighted* problem (NP-hard even offline when sizes are arbitrary; this is **Weighted Caching**, solvable optimally via LP but not in linear time and the value model is different).
- **Instantaneous misses** vs. our bandwidth-bound transfers → minimizing miss *count* ≠ minimizing makespan. A single huge miss can be worse than many small ones.
- **One operation at a time** vs. our compute+from-slow+to-slow parallelism → Belady doesn't address overlap.

### 5.2 Weighted caching / k-server
Generalizes Belady to weighted pages. Offline optimum solvable via LP (assignment-style). Still assumes instantaneous fetches and a single serial machine — same gap as 5.1 on the parallelism and continuous-time fronts.

### 5.3 Pebble games (black-white pebbling)
Models recomputation: pebbles = memory slots, moves = compute steps; black pebbles = recomputable values, white pebbles = stored values. Black-white pebbling models offload+reload.

**Closer than paging** because it captures the "recompute vs store" tradeoff and the cap. But:
- Even **single-machine** pebbling is **PSPACE-complete** for general DAGs (Hopcroft-Paul-Valiant variants). For trees / series-parallel graphs there are poly algorithms — our DAG is *nearly* a chain, so this is encouraging, but the variable-size and bandwidth aspects aren't modeled at all.
- **Pebble moves are unit-cost**; we have weighted moves (bytes/bandwidth).

### 5.4 Resource-Constrained Project Scheduling (RCPSP)
DAG of tasks with durations, renewable resource pool (e.g., `R` units of resource type `r`), minimize makespan subject to "resource consumption at every instant ≤ `R`".

**Strongly relevant** — this is the framework that captures the cap constraint and the DAG simultaneously. **NP-hard**. Standard MILP formulations exist; commercial solvers handle ~hundreds of tasks for small resource dimensions.

**Why our problem extends RCPSP**:
- In RCPSP, each task's resource demand is a *parameter* of the task. In our problem, the resource demand of the compute stream during task `T` depends on *what else is resident*, which depends on prior transfer scheduling — i.e., the resource consumption is itself a decision variable, not an input. This is sometimes called RCPSP with **flexible resource profiles** and is materially harder.
- RCPSP has no notion of "data" — we additionally have to schedule the H and D streams (which are themselves RCPSP-like) and link them to the compute stream via residency.

### 5.5 Job-shop scheduling with precedence (3-machine variant)
Treat compute / from_slow / to_slow as three "machines"; tasks need to visit machines in some order with precedence. NP-hard for `≥3` machines. Doesn't model the shared cap constraint.

### 5.6 Register allocation with spilling
Classic compiler problem: graph coloring + spill cost optimization. NP-hard. The "spill / reload" mechanic is structurally identical to our offload/prefetch. The crucial differences:
- Spill costs are per-instruction; we have continuous time and bandwidth.
- Register pressure is in *register count*; ours is in *bytes*.

### 5.7 The active research area: memory-constrained DNN training
This **is** the field our problem lives in. The relevant works:

- **Checkmate** (Jain et al., MLSys 2020) — formulates fwd-recompute scheduling as an **MILP**; provably optimal but computationally feasible only for graphs up to ~100 nodes with hours of solver time. Doesn't model bandwidth-aware overlap.
- **SwapAdvisor** (Huang et al., ASPLOS 2020) — genetic algorithm over swap decisions; bandwidth-aware; heuristic, no optimality claim.
- **Capuchin** (Peng et al., ASPLOS 2020) — online + measurement-driven; greedy.
- **ZeRO-Offload / ZeRO-Infinity** — engineering-driven, not formally optimal.
- **POET** (Patil et al., ICML 2022) — MILP for joint paging+recompute on edge computes.

The literature's **consistent verdict**: optimal MILP is intractable past ~100 graph nodes; production systems use greedy or learned heuristics. **Our problem is exactly this problem**, with the added wrinkle that we want offline planning (not online), which moves us closer to Checkmate's regime — but Checkmate ignores bandwidth, which is the bottleneck in our regressions.

---

## 6. What specifically makes our scenario hard

Pulling the threads above together, our problem has **six** structural traits that any one of them is enough to break a clean optimum:

1. **Variable-size resources** (KB–GB objects). Defeats Belady; turns the cap constraint into a bin-pack at every instant.
2. **Bandwidth-bound transfers**. Optimizing "miss count" is the wrong objective — we optimize wall-clock, which depends on *which* misses are on the critical path of overlap.
3. **Three parallel streams sharing the cap**. Compute progresses while transfers happen; transfers compete for pool budget. Cannot decompose into per-stream subproblems.
4. **Continuous-time cap constraint** including in-flight transit. Discrete-time-at-boundaries planners systematically under-predict pool pressure.
5. **DAG with precedence**. The compute order is fixed but the *deadline* each transfer must meet is determined by both the DAG and the schedule of all other transfers — they couple.
6. **Mutation semantics** (in-place gradient writeback). Object identity changes mid-life: the same object ID is a "pure" weight before its `b_i` and a "dirty must-writeback" gradient after. Forces asymmetric reload costs (`W_i` can be re-fetched; `dW_i` after mutation cannot be discarded).

Any single one of these is well-studied in isolation; their combination has no canonical solution in the literature.

---

## 7. Can we solve it exactly? Formulations and the limits of estimating their runtime

The problem **is** expressible as a mathematical program — there's no formal barrier to writing down an exact solver. The questions are (a) which formulation is least painful, (b) how big the resulting program is at our scale, and (c) how long it would take to solve. (a) and (b) we can answer now; (c) is genuinely uncertain in a way worth explaining.

Notation used below for L=32, num_seqs=4 (our target config): `T = 2L+1 = 65` compute tasks, `O ≈ 5L = 160` objects (input, weights, grads, activations, layer outputs/grads, head objects), `E` = the number of potential transfer events (from_slow + to_slow) the schedule might include. `E` is itself a function of the policy — worst case it's `O · K` where `K` is the maximum times an object is fetched, but for our workload `K ≤ 2` per object (e.g., W_i fetched at most for `f_i` and `b_i`), so `E ≲ 4·L ≈ 130` is a reasonable upper bound.

### 7.1 Three candidate formulations

#### 7.1.1 Time-indexed MILP (most direct, worst scaling)

Discretize time to a grid of `P` ticks (e.g., µs granularity over a ~1-second makespan ⇒ `P ≈ 10⁶`). Binary variables:

| Variable | Meaning | Count |
|---|---|---|
| `x[o, p] ∈ {0,1}` | object `o` resident at tick `p` | `O · P ≈ 1.6·10⁸` |
| `start_h[o, p] ∈ {0,1}` | from_slow for `o` starts at tick `p` | `O · P` |
| `start_d[o, p] ∈ {0,1}` | to_slow for `o` starts at tick `p` | `O · P` |
| `start_t[task, p] ∈ {0,1}` | task starts at tick `p` | `T · P ≈ 6.5·10⁷` |

Constraints linearize cleanly (cap is a sum at each `p`, FIFO is a sum over active transfers at each `p`). Hopeless in this form — hundreds of millions of binaries. **Coarsening the grid** (e.g., ticks at multiples of 100 µs, or at task-boundary granularity) cuts `P` to `T` or `2T`, giving `~30k` binaries — tractable in principle but loses the continuous-time cap constraint that the to_slow-tail problem hinges on.

#### 7.1.2 Event-time MILP (continuous time, discrete events — most compact)

Don't discretize time at all. Introduce continuous start-time variables and use binary variables only for combinatorial choices.

| Variable | Meaning | Count |
|---|---|---|
| `s_task[t] ∈ ℝ⁺` | start time of compute task `t` | `T = 65` |
| `s_h[e] ∈ ℝ⁺`, `s_d[e] ∈ ℝ⁺` | start times of transfer events | `E ≈ 130` |
| `present[o, k] ∈ {0,1}` | object `o` resident at task-boundary `k` | `O · T ≈ 10⁴` |
| `use_h[o, k] ∈ {0,1}` | from_slow for `o` occurs before boundary `k` | `O · T ≈ 10⁴` |
| `order_h[e1, e2] ∈ {0,1}` | from_slow event `e1` precedes `e2` (FIFO order) | `O(E²) ≈ 1.7·10⁴` |
| `order_d[e1, e2] ∈ {0,1}` | same for to_slow | `O(E²) ≈ 1.7·10⁴` |

**Total integer variables: ~5·10⁴.** Continuous variables: ~200. This is a reasonable size — comparable to mid-difficulty RCPSP benchmarks that commercial solvers handle.

The constraint pain points are (a) the continuous-time cap (needs disjunctive constraints over event-pairs to express "object `o` is in pool during interval `[s_h[e], s_d[e']]`" then sum object weights inside that interval) and (b) FIFO non-overlap (needs disjunctive `order_h` constraints with big-M). Both are standard MILP modeling techniques but inflate the formulation 2–5× in constraint count.

#### 7.1.3 CP-SAT with interval variables (cleanest model, unknown scaling)

Constraint Programming solvers (Google OR-Tools CP-SAT, IBM CP Optimizer) have **first-class support** for scheduling with cumulative resources via `IntervalVar` and `Cumulative` constraints. Our model would be:

- An `IntervalVar` per compute task (fixed duration, decision = start time).
- An optional `IntervalVar` per potential transfer event (existence is a decision, duration is fixed given the object).
- `Cumulative` constraint on the compute stream with capacity 1 (serial).
- `Cumulative` constraint on from_slow with capacity 1 (serial FIFO).
- `Cumulative` constraint on to_slow with capacity 1.
- `Cumulative` constraint on the pool, where each object's "demand interval" is the union of `[from_slow.start, to_slow.end]` segments (where present) and demand height = `size_bytes`.
- Precedence: every task's input transfer events must complete before task start; every output's to_slow must start after task end.

CP-SAT is **purpose-built** for exactly this constraint shape. The model is dramatically more compact than MILP (no big-M, no order-pair binaries) and CP-SAT's learning + propagation often beats MILP on scheduling. But — and this is the honest part — **CP-SAT performance on novel scheduling instances is hard to predict without running it.**

### 7.2 How to actually estimate runtime — there is no shortcut

NP-hardness tells you worst-case exponential. It does **not** tell you what real solvers do on real instances of *your* problem. The only reliable methods:

1. **Implement and scale.** Write the formulation (7.1.2 or 7.1.3). Solve on `L=2, M=1, cap=large` (~10 tasks, ~10 objects) → seconds, near-trivial. Scale up `L` and tighten `cap`; fit the wall-clock curve. If it's polynomial in practice (e.g., `~L³` for CP-SAT, which empirically happens for well-structured scheduling problems), L=32 is reachable. If it's exponential (`2^L`), it stops around `L=10`. **Until we run it, we don't know which.**
2. **Compare against analogous published benchmarks.** Checkmate (recompute MILP, ~100-node graphs, ~minutes–hours depending on memory budget tightness) is the closest reference, but solves a **strictly easier** problem: no bandwidth model, no parallel streams, no continuous-time cap. Our problem is harder per-node, so Checkmate's runtimes are a **lower bound** on what to expect, not a prediction.
3. **Solve a relaxation as a lower bound.** The LP relaxation of 7.1.2 (drop integrality) gives a polynomial-time lower bound on the makespan and a *fractional* schedule. If the LP optimum is close to the auto policies' makespan, we have evidence we're near optimal without needing to solve the IP. This is the **cheapest signal** about how much room remains and worth doing regardless of whether we attempt a full exact solver.

### 7.3 Practical recommendation

**Implement 7.1.3 (CP-SAT) on toy configs** as a verification oracle, not a production policy. Concrete plan:

| Config | Compute tasks (`2L+1`) | Expected solve time | Purpose |
|---|---|---|---|
| L=2, cap=loose | 5 | <1s | Sanity check the model |
| L=4, cap=tight | 9 | 1–10s | Catch formulation bugs vs. simulator replay |
| L=8, cap=tight | 17 | 10s–min | First real scaling signal |
| L=16, cap=tight | 33 | min–hour? | Largest config we have any hope of |
| L=32 (production) | 65 | unknown | Probably out of reach; try anyway with 1h limit |

Note: `M` (num_seqs) and `seqlen` **don't change task count** — they scale object sizes and per-task runtimes, which affects cap-tightness and per-transfer durations. The oracle should sweep `M` and `seqlen` at each `L` row above, since a config that solves quickly with loose cap may explode with tight cap at the same `L`.

Use it to (a) **validate a policy's plan against ground-truth optimum** on small configs, (b) **quantify the gap** between any policy and optimum (within X% on cases we can solve → likely near-optimal on cases we can't), (c) **debug regressions** — if one policy loses to another on some config and the oracle confirms the better policy is near-optimal, we know the loser has a real bug rather than the problem being hard.

This is also how Checkmate validates its solver: solve small instances optimally, compare heuristic to oracle, extrapolate confidence.

---

## 8. Where the structural good news comes from

Despite the above, several properties of *our specific* problem make near-optimal much cheaper than full MILP:

1. **The DAG is nearly a chain.** This dramatically restricts the "freedom" that makes RCPSP and pebbling hard. Many results in scheduling theory become poly-time for series-parallel or interval graphs; ours is closer to those than to general DAGs.
2. **Forced residency dominates**. A large fraction of the schedule is determined by `T.reads`; the optional "should I keep this lingering?" decisions are a minority.
3. **Object lifetimes are short and structured**. Most activations are used in exactly two places: produced at `f_i`, consumed at `b_i`. Most weights are used in exactly two places: `f_i` and `b_i`. The interesting decision is per-pair, not per-instant.
4. **The cap is binary-ish**. Either an object fits in cap or it doesn't; either a prefetch can hide behind compute or it can't. There are only a few "tight" boundaries per schedule (the worst-pressure points), and decisions far from those boundaries are free.
5. **Simulator is cheap.** Replaying a candidate schedule through the simulator costs ~ms. This makes simulation-in-the-loop search affordable: we can evaluate hundreds–thousands of candidate plans before MILP would have finished its preprocessing.

These observations don't yield "the optimum" but they do justify a **two-level approach**: (a) a search/enumeration over the small set of genuinely optional residency decisions, (b) a deterministic stream-scheduling step given a fixed residency plan, (c) the simulator as the cost oracle.

---

## 9. What "optimal" means here

Three precise notions of optimality, from strongest to weakest, that a search-based policy could target:

### 9.1 Optimal-by-MILP
Solve the full integer program. **Not pursued** for production; useful only as an offline verification oracle for small benchmark configs.

### 9.2 Optimal-given-residency-plan
**Tractable.** Given a fixed residency plan (which objects are resident over which intervals), the question "what is the best ordering of transfers on H and D to make this plan happen with minimum makespan" has known polynomial solutions in pieces:
- For a single stream with hard deadlines: **EDF (earliest-deadline-first) is provably optimal** (Liu & Layland, 1973; for our case: one stream as a single-machine real-time problem).
- For two streams with shared cap: combined heuristics, but the search space is small (`|H|! × |D|!`) at our scale and pruneable.

This is the regime `max_reduce`'s trigger-placement phase operates in (from_slow only). Extending to to_slow and adding contention-aware guards would close most of the gap.

### 9.3 Optimal-by-search-over-plans
**Tractable with the right structure.** Enumerate or beam-search over residency plans; for each, derive the optimal-given-residency schedule (9.2); pick the lowest-makespan plan via simulator replay.

If the search space is structured (e.g., "for each W_i, decide: pre-place or not; for each A_i, decide: keep across forward or offload"), beam search over ~O(L) decisions with simulator scoring is feasible in seconds. `min_grow` is the policy in this family.

A search-based policy's design space is essentially: **how to organize 9.3** so that the search converges to (or proves it has converged to) something within a small known gap of 9.1.

---

## 10. Design implications for any search-based policy

A reader of any per-policy design doc should expect these three principles to drive its structure:

1. **Separate "what's resident" (plan) from "when transfers fire" (schedule).** The plan is a discrete combinatorial object; the schedule is a deterministic derivative. Conflating them (as some early attempts did in their reduce phase) prevents either from being reasoned about cleanly.
2. **Use the simulator as the cost oracle, not a heuristic key.** Every plan-scoring decision should ultimately be backed by simulator replay rather than a static rank function. The static rank can be a tiebreaker / candidate generator, not the final word.
3. **Model the cap in continuous time, including transit.** Any plan that only checks the cap at task boundaries will systematically under-predict pool pressure and produce stall-prone schedules.

Each per-policy design doc lays out the specific algorithm (search structure, plan representation, cost function) that operationalizes these principles.

---

## Appendix: glossary

| Symbol / term | Meaning |
|---|---|
| `L` | number of layer blocks |
| `M` | num_seqs (microbatch); affects tensor sizes, not task count |
| `S` | seqlen; affects tensor sizes |
| `f_i`, `b_i`, `r_i`, `head` | forward / backward / recompute / head compute tasks |
| `W_i`, `dW_i`, `A_i`, `y_i`, `dy_i` | weights, gradients, saved activations, layer outputs, output grads |
| from-slow / to-slow | inbound / outbound transfer streams |
| Residency interval `[a, b]` | object is on compute between task boundaries `a` and `b` |
| Cap | hard fast memory limit (bytes) |
| Belady | optimal offline page-replacement (evict furthest-in-future) |
| RCPSP | Resource-Constrained Project Scheduling Problem |
| EDF | Earliest Deadline First scheduling |
| MILP | Mixed Integer Linear Programming |
