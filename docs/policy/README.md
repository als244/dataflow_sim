# Policies

Seven built-in scheduling policies. All implement `apply_<name>_policy(bare: TaskChain, ...) -> TaskChain` and live in `simulator/src/dataflow_sim/policy/`.

See [principles.md](principles.md) for the correctness invariants and resource-conservation principles a good policy follows, and the tensions a new policy must take a stance on.

## Which policy should I use?

| Policy | Approach | When it wins | When to avoid |
|---|---|---|---|
| [sliding-window.md](sliding-window.md) | Hand-crafted fixed-width window | Transformer chain you tune by hand | Bad for non-chain workloads |
| [belady-reactive.md](belady-reactive.md) | Shadow-sim Belady eviction | General bare chains, fast | Misses proactive opportunities |
| [roundtrip-planner.md](roundtrip-planner.md) | Constructive round-trip packing | Workloads with reusable objects | High-pressure regimes |
| [race-best.md](race-best.md) | Race belady + roundtrip, keep better | General safety choice — robust | Pays 2× planning cost |
| [max-reduce.md](max-reduce.md) | Analytic top-down from MAX residency | Mid-to-loose capacity, deterministic | Tight caps where MAX-not-feasible |
| [min-grow.md](min-grow.md) | MIN + beam search with simulator oracle | Tight cap, large models | Slow (10s+ budget); mid-cap unstable |
| [pressurefit.md](pressurefit.md) | Pressure-fit interval planning + bounded candidate specs + deadline-aware H2D scheduling | General chains needing fast planning | Cases needing proactive global D2H packing |

## File / function / enum convention

Every policy follows the same convention: `<stem>.py` in policy/, function `apply_<stem>_policy`, server enum `"<stem>"`, UI value `"<stem>"`, doc `docs/policy/<stem-with-dashes>.md`.
