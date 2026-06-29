# Policies

Six built-in scheduling policies. All implement `apply_<name>_policy(bare: TaskChain, ...) -> TaskChain` and live in `src/dataflow_sim/policies/`.

See [principles.md](principles.md) for the correctness invariants and resource-conservation principles a good policy follows, and the tensions a new policy must take a stance on.

## Which policy should I use?

| Policy | Approach | When it wins | When to avoid |
|---|---|---|---|
| [sliding-window.md](other_policies/sliding-window.md) | Hand-crafted fixed-width window | Recognized chain-shaped training workload you tune by hand | Bad for non-chain workloads |
| [belady-reactive.md](other_policies/belady-reactive.md) | Shadow-sim Belady eviction | General bare chains, fast | Misses proactive opportunities |
| [roundtrip-planner.md](other_policies/roundtrip-planner.md) | Constructive round-trip packing | Workloads with reusable objects | High-pressure regimes |
| [max-reduce.md](other_policies/max-reduce.md) | Analytic top-down from MAX residency | Mid-to-loose capacity, deterministic | Tight caps where MAX-not-feasible |
| [min-grow.md](other_policies/min-grow.md) | MIN + beam search with simulator oracle | Tight cap, large models | Slow (10s+ budget); mid-cap unstable |
| [pressurefit.md](pressurefit.md) | Pressure-fit interval planning; fastest of four verified inbound schedules | General chains needing fast planning | Cases needing proactive global to-slow packing |

## File / function / enum convention

Every policy follows the same convention: `<stem>.py` in policy/, function `apply_<stem>_policy`, server enum `"<stem>"`, UI value `"<stem>"`, doc `docs/policy/<stem-with-dashes>.md`.
