# sliding_window

A hand-crafted, workload-specific policy that annotates a bare model-training
chain with a fixed-pattern weight/activation/gradient trigger schedule. This
is the only non-auto policy in the suite: it does not reason about residency or
generalize beyond the recognized task/object id conventions emitted by the
modular training workload builder. It keys directly off ids such as
`f_0_0_i`, `b_0_0_i`, `W_i`, `dW_0_i`, and `A_0_0_i`, plus a tunable
`window_size` parameter that controls how many weights are kept
compute-resident simultaneously during forward/backward.

## Mechanism

Given a bare training chain with all model state on backing and zero triggers, the policy infers `L` (layer count) from the number of `f_i` tasks and emits the following pattern:

- **Initial compute residency:** `W_0..W_{w-1}` (so forward can start), plus `W_head`. Legacy chains also keep `dW_head` compute-resident and may pre-place layer `dW_j` entries not covered by the forward dW preamble or backward dW cascade.
- **After `f_i`** (`i ≤ L-3`): offload `A_i`.
- **After `f_i`** (`i + w ≤ L-1`): release `W_i`; prefetch `W_{i+w}` — the forward window slide.
- **After `f_{L-2}`:** prefetch `dW_{L-1}` (forward dW preamble).
- **After `f_{L-1}`:** prefetch `dW_{L-2}`.
- **After `b_i`:** release upstream activation, release `A_i`, release `W_i`; offload `dW_i` (writeback).
- **After `b_i`** (`i - w ≥ 0`): prefetch `W_{i-w}` — the backward window slide.
- **After `b_i`** (`i - 2 ≥ 0`): prefetch `dW_{i-2}` (dW cascade).
- **After `b_i`** (`i - 2 ∈ offloaded_A`): prefetch `A_{i-2}` (activation recompute path).

For current modular chains, `head_fwd` is left untouched and `head_bwd` gets a
release of `y_{L-1}` plus any produced head-gradient object. Legacy chains use
one combined `head` task for the same release point.

## Legacy worked example: L=4, window_size=2

Chain: `f_0, f_1, f_2, f_3, head, b_3, b_2, b_1, b_0`. Activations `A_0, A_1` are in the offload set (`i ≤ L-3 = 1`); `A_2, A_3` stay resident. Initial compute residency: `{W_0, W_1, W_head, dW_head}` (no extra `dW_j` pre-placed since the f-preamble + b-cascade cover all of `dW_0..dW_3`).

Per-task triggers emitted by the policy:

| Task | releases | offload | prefetch |
|---|---|---|---|
| `f_0` | `input, W_0` | `A_0` | `W_2` |
| `f_1` | `y_0, W_1` | `A_1` | `W_3` |
| `f_2` | `y_1` | — | `dW_3` (f-preamble) |
| `f_3` | `y_2` | — | `dW_2` (f-preamble) |
| `head` | `y_3` | — | — |
| `b_3` | `dy_head, A_3, W_3` | `dW_3` | `W_1, dW_1, A_1` |
| `b_2` | `dy_3, A_2, W_2` | `dW_2` | `W_0, dW_0, A_0` |
| `b_1` | `dy_2, A_1, W_1` | `dW_1` | — |
| `b_0` | `dy_1, A_0, W_0` | `dW_0` | — |

Compute-resident weight/grad/activation footprint at each task boundary (excluding intermediate `y_*` / `dy_*` tensors; `(↓)` = outbound writeback in flight, `(↑)` = inbound prefetch in flight):

```
boundary    | W resident                | dW resident                  | A resident
------------|---------------------------|------------------------------|------------------
initial     | W_0, W_1, W_head          | dW_head                      | —
after f_0   | W_1, W_2(↑), W_head       | dW_head                      | A_0(↓)
after f_1   | W_2, W_3(↑), W_head       | dW_head                      | A_1(↓)
after f_2   | W_2, W_3, W_head          | dW_head, dW_3(↑)             | A_2
after f_3   | W_2, W_3, W_head          | dW_head, dW_3, dW_2(↑)       | A_2, A_3
after head  | W_2, W_3, W_head          | dW_head, dW_3, dW_2          | A_2, A_3
after b_3   | W_2, W_1(↑), W_head       | dW_head, dW_2, dW_1(↑), dW_3(↓) | A_2, A_1(↑)
after b_2   | W_1, W_0(↑), W_head       | dW_head, dW_1, dW_0(↑), dW_2(↓) | A_1, A_0(↑)
after b_1   | W_0, W_head               | dW_head, dW_0, dW_1(↓)       | A_0
after b_0   | W_head                    | dW_head, dW_0(↓)             | —
```

The steady-state invariant is visible: at every forward/backward boundary the resident non-head weight count hovers at `w = 2` (one live + one inbound during a slide), and `dW_head` / `W_head` are pinned throughout. Reader can verify by tracing `_annotate_forward` / `_annotate_backward` for each `i`.

## Effect of `window_size`

`window_size` (`w`) is the only knob. It directly sets the number of non-head weights resident in steady state and trades stall risk against the capacity floor:

- **`w = 1`:** maximally aggressive. Only one non-head `W` resident at a time; every `f_i` releases `W_i` and prefetches `W_{i+1}` with no overlap. Lowest cap floor but highest stall risk — a slow from-slow link will block the very next compute.
- **`w = 2`** (default): one weight live, one inbound. Hides a single prefetch behind one compute step. Sweet spot for the synthetic chains used in `experiments/`.
- **`w ≥ 3`:** keeps more weights warm, more overlap slack, fewer stalls when from-slow is slow — at the cost of a higher cap floor (each extra `w` adds one weight's worth of resident bytes).
- **`w ≥ L`:** degenerates — `i + w ≤ L-1` is never satisfied, so no forward slide fires and all weights stay compute-resident for the whole forward pass (effectively "keep everything"). Useful as a sanity baseline on loose caps.

## Why this policy is chain-specific

Unlike the `*_auto` planners (which take any `TaskChain` and reason about
residency from the reference stream + capacity), `sliding_window` is hard-coded
to the modular model-training chain shape:

- It **infers `L` by counting `f_i` tasks** and assumes a symmetric `f_0..f_{L-1}, head_fwd/head_bwd, b_{L-1}..b_0` structure — any deviation (missing head, asymmetric backward, fused tasks) silently produces wrong triggers or raises.
- It **keys triggers off chain position** (forward window slide uses `i + w`, dW cascade uses `i - 2`), not off any property visible in the `schema.Task` interface (`inputs`, `outputs`, `mutates_inputs`).
- It **assumes the `W_i` / `dW_i` / `A_i` id convention** literally — `_annotate_backward` builds the string `f"dW_{i}"`. A chain that uses different ids (or non-integer layer indices) won't match.
- It **assumes the activation recompute pattern** `A_{i-2}` needed by `b_i` — a workload with different reuse distances would prefetch the wrong tensor.

Generalizing would require recovering all of this from the schema-level signals (object types, `mutates_inputs`, reference stream distances), which is exactly what the `*_auto` policies do. `sliding_window` exists as a known-good reference point: it shows what the right schedule looks like for this workload so the auto planners can be evaluated against it.

## When it wins / when it loses

| Regime | Outcome |
|---|---|
| Tight `fast_memory_capacity` (≈ 1.5–2× widest `b_i` working set) | Wins — predictable envelope, no planning overhead, succeeds where reactive planners stall on cap errors. |
| Mid caps (e.g. `L=10, cap ∈ {600, 800}`) | Wins by 4–16 ticks vs `belady_reactive` due to better forward/backward overlap. |
| Loose caps (`cap = ∞` or generous) | Loses — emits an unconditional trailing `dW_*` writeback tail that auto-policies skip; ~5–8% makespan overhead. |
| Non-training chains | Inapplicable — raises on the `f_i` count check. |
| Tuning `window_size` | Larger `w` = more weights resident = lower stall risk but higher cap floor; `w=2` is the default. |

## Implementation

`src/dataflow_sim/policies/sliding_window.py` — entry: `apply_sliding_window_policy(bare, *, window_size=2, fast_memory_capacity=None)`.
