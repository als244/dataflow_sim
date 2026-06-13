"""Transformer math + sub-op catalog.

Pure functions/dataclasses that translate transformer model dimensions +
hardware specs + training params into per-task runtimes and per-object
byte sizes for the simulator's `TaskChain`.

Design goal: this module is the ONLY place arch-aware math lives. Adding
new architecture features (MLA, partial recompute, datatype changes, etc.)
should be a local edit here, not a sprawl across the codebase.

All quantities are integers; time is in microseconds (`1 tick = 1 µs`).
bf16 = 2 bytes/element (hardcoded for now).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal

from dataflow_sim.workloads.training.optimizers import (
    OptimizerMatrix,
    OptimizerMode,
    adamw_step_bytes as _adamw_step_bytes,
    muon_step_flops_bytes as _muon_step_flops_bytes,
    optimizer_state_bytes,
)
from dataflow_sim.workloads.common.hardware import HardwareSpec

if TYPE_CHECKING:
    from dataflow_sim.workloads.training.transformer import TrainingConfig

BYTES_PER_ELEMENT = 2  # bf16


# ---------- dataclasses ----------

@dataclass(frozen=True)
class TransformerSpec:
    vocab_size: int
    n_layers: int
    d_model: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    expert_dim: int
    num_shared_experts: int
    num_routed_experts: int
    top_k: int
    qk_norm: bool = True


SubOpKind = Literal["compute", "memory"]
EffName = Literal["matmul", "attn_fwd", "attn_bwd", "none"]


@dataclass(frozen=True)
class SubOp:
    """A unit of work modeled as either a roofline-bounded compute op or a
    pure memory-bound op. `count` lets us model serial dispatches (e.g. one
    matmul per routed expert).

    `effective_flops` is the flop count that counts as *useful* work — for
    most sub-ops this equals `flops`, but for `attn_bwd` (flash-attention
    backward) the actual op does 5× fwd flops while only 4× is non-recompute
    work; the extra 1× is forward-recompute that doesn't contribute to the
    effective-TFLOPS rate. `flops` still drives the math_us calculation
    (the GPU genuinely executes them), but `effective_flops` is what's
    aggregated into the model-wide effective-TFLOPS metric.

    `compute_eff` / `mem_eff` are optional per-op efficiency overrides. When
    None, time_subop falls back to the HW-level eff for that op's `eff_name`
    (compute) and `hw.mem_eff` (memory). Useful for modeling a specific
    op's bad kernel without globally degrading the HW envelope.
    """
    name: str
    kind: SubOpKind
    flops: int            # 0 for memory-bound
    bytes: int            # read + write
    eff_name: EffName     # "none" for memory-bound
    count: int = 1
    effective_flops: int | None = None  # defaults to `flops` when None
    compute_eff: float | None = None    # override hw.<eff_name>_eff
    mem_eff: float | None = None        # override hw.mem_eff


@dataclass
class SubOpTiming:
    """Resolved per-sub-op timing breakdown for the UI panel + aggregators."""
    name: str
    kind: SubOpKind
    flops: int                     # 0 for memory-bound
    effective_flops: int           # 0 for memory-bound; == flops except attn_bwd
    bytes: int
    count: int
    math_us: int | None            # None for memory-bound
    mem_us: int
    per_call_us: int               # max(math_us, mem_us), ceil-rounded to int µs
    per_call_us_exact: float       # un-rounded float µs of the binding term —
                                   # use for section-level effective-TFLOPS so
                                   # the result doesn't accumulate ceil errors
    total_us: int
    bound_by: SubOpKind            # always "memory" for memory-bound sub-ops
    effective_tflops: float | None # None for memory-bound; uses effective_flops
                                   # and the un-rounded binding time so pure
                                   # matmul-bound ops show exactly peak × eff

    def asdict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "flops": self.flops,
            "effective_flops": self.effective_flops,
            "bytes": self.bytes,
            "count": self.count,
            "math_us": self.math_us,
            "mem_us": self.mem_us,
            "per_call_us": self.per_call_us,
            "per_call_us_exact": self.per_call_us_exact,
            "total_us": self.total_us,
            "bound_by": self.bound_by,
            "effective_tflops": self.effective_tflops,
        }


# ---------- param / byte helpers ----------

def layer_weight_matrices(spec: TransformerSpec) -> list[OptimizerMatrix]:
    """Logical matrix inventory for one transformer layer.

    Dense layers have QKV, attention output, shared MLP up, and shared MLP
    down matrices. MoE layers additionally model the full routed expert bank;
    optimizer state is allocated for every expert, not only the `top_k` active
    experts used by a particular token batch.
    """
    mats: list[OptimizerMatrix] = []

    def add(name: str, rows: int, cols: int, count: int = 1) -> None:
        if rows > 0 and cols > 0 and count > 0:
            mats.append(OptimizerMatrix(name=name, rows=rows, cols=cols, count=count))

    d = spec.d_model
    hd = spec.head_dim
    nh, nkv = spec.n_heads, spec.n_kv_heads
    edim = spec.expert_dim

    add("qkv_proj", d, (nh + 2 * nkv) * hd)
    add("attn_proj", nh * hd, d)
    add("shared_mlp_up", d, 2 * edim, spec.num_shared_experts)
    add("shared_mlp_down", edim, d, spec.num_shared_experts)
    add("routed_mlp_up", d, 2 * edim, spec.num_routed_experts)
    add("routed_mlp_down", edim, d, spec.num_routed_experts)
    return mats


def params_per_layer(spec: TransformerSpec) -> int:
    """Total parameter count per transformer layer (full expert bank)."""
    return sum(m.rows * m.cols * m.count for m in layer_weight_matrices(spec))


def active_params_per_layer(spec: TransformerSpec) -> int:
    """Active parameter count per layer (only `top_k` routed experts fire)."""
    attn = spec.head_dim * (2 * spec.n_heads + 2 * spec.n_kv_heads)
    mlp = 3 * spec.expert_dim * (spec.num_shared_experts + spec.top_k)
    return spec.d_model * (attn + mlp)


def head_params(spec: TransformerSpec) -> int:
    return spec.d_model * spec.vocab_size


def input_bytes(spec: TransformerSpec, cfg: TrainingConfig) -> int:
    """First-layer residual stream input. Embedding is out of scope."""
    return cfg.num_seqs * cfg.seqlen * spec.d_model * BYTES_PER_ELEMENT


def layer_output_bytes(spec: TransformerSpec, cfg: TrainingConfig) -> int:
    """`y_i` (forward residual stream output) and `dy_i` (backward gradient)."""
    return cfg.num_seqs * cfg.seqlen * spec.d_model * BYTES_PER_ELEMENT


def activation_bytes(spec: TransformerSpec, cfg: TrainingConfig) -> int:
    """`A_i` — the saved activation consumed by `b_i`. Uses `2 * d_model` factor."""
    elements = cfg.num_seqs * cfg.seqlen * (
        spec.head_dim * (2 * spec.n_heads + 2 * spec.n_kv_heads)
        + 2 * spec.d_model
        + 2 * (spec.num_shared_experts + spec.top_k) * spec.expert_dim
    )
    return elements * BYTES_PER_ELEMENT


def layer_weight_bytes(spec: TransformerSpec) -> int:
    """Full per-layer weight bank (all experts, including unused routed ones)."""
    return params_per_layer(spec) * BYTES_PER_ELEMENT


def head_weight_bytes(spec: TransformerSpec) -> int:
    return head_params(spec) * BYTES_PER_ELEMENT


def optimizer_state_bytes_per_layer(spec: TransformerSpec, optimizer: OptimizerMode) -> int:
    """Bytes in the per-layer optimizer-state object `O_i`.

    AdamW carries two state tensors (e.g. first and second moments). Muon
    carries one momentum tensor. Both use the same bf16 element size as the
    layer weights in this model.
    """
    return optimizer_state_bytes(layer_weight_bytes(spec), optimizer)


# ---------- sub-op enumerators ----------

def _matmul_subop(name: str, flops: int, bytes_total: int, count: int = 1) -> SubOp:
    return SubOp(name=name, kind="compute", flops=flops, bytes=bytes_total,
                 eff_name="matmul", count=count)


def _mem_subop(name: str, bytes_total: int) -> SubOp:
    return SubOp(name=name, kind="memory", flops=0, bytes=bytes_total,
                 eff_name="none", count=1)


def _mlp_up_bytes(expert_tokens: int, d: int, edim: int) -> int:
    """Bytes for one fused-gate+up matmul: input + weights + output."""
    return (
        expert_tokens * d                # input  [T, d_model]
        + d * 2 * edim                   # weight [d_model, 2*expert_dim]
        + expert_tokens * 2 * edim       # output [T, 2*expert_dim]
    ) * BYTES_PER_ELEMENT


def _mlp_down_bytes(expert_tokens: int, d: int, edim: int) -> int:
    """Bytes for one down matmul: input + weights + output."""
    return (
        expert_tokens * edim             # input  [T, expert_dim]
        + edim * d                       # weight [expert_dim, d_model]
        + expert_tokens * d              # output [T, d_model]
    ) * BYTES_PER_ELEMENT


def _mlp_up_flops(expert_tokens: int, d: int, edim: int) -> int:
    """Flops for one fused-gate+up matmul: 2 * M_dim * K * N where
    M=expert_tokens, K=d_model, N=2*expert_dim."""
    return 2 * expert_tokens * d * (2 * edim)


def _mlp_down_flops(expert_tokens: int, d: int, edim: int) -> int:
    """Flops for one down matmul: 2 * M * K * N where M=expert_tokens,
    K=expert_dim, N=d_model."""
    return 2 * expert_tokens * edim * d


def forward_subops(spec: TransformerSpec, cfg: TrainingConfig) -> list[SubOp]:
    """Per-layer fwd sub-ops in execution order. MLP is split into the fused
    gate+up projection (output dim = 2 * expert_dim) and the down projection
    (input dim = expert_dim, output dim = d_model). SwiGLU is the elementwise
    activation between them."""
    M, S = cfg.num_seqs, cfg.seqlen
    tt = M * S  # total tokens
    d = spec.d_model
    hd = spec.head_dim
    nh, nkv = spec.n_heads, spec.n_kv_heads
    edim = spec.expert_dim
    ns, nr, tk = spec.num_shared_experts, spec.num_routed_experts, spec.top_k

    out: list[SubOp] = []
    # 1. attn_norm (memory)
    out.append(_mem_subop("attn_norm", 2 * tt * d * BYTES_PER_ELEMENT))
    # 2. qkv_proj (compute)
    qkv_out_dim = (nh + 2 * nkv) * hd
    qkv_flops = 2 * M * S * d * qkv_out_dim
    qkv_bytes = (M * S * d + d * qkv_out_dim + M * S * qkv_out_dim) * BYTES_PER_ELEMENT
    out.append(_matmul_subop("qkv_proj", qkv_flops, qkv_bytes))
    # 3. qk_norm (memory, optional)
    if spec.qk_norm:
        qk_norm_bytes = 2 * tt * (hd * (nh + nkv)) * BYTES_PER_ELEMENT
        out.append(_mem_subop("qk_norm", qk_norm_bytes))
    # 4. rope (memory)
    rope_bytes = 2 * tt * (hd * (nh + nkv)) * BYTES_PER_ELEMENT
    out.append(_mem_subop("rope", rope_bytes))
    # 5. attn (compute)
    attn_flops = 2 * M * nh * hd * S * S
    attn_bytes = (M * S * (nh + 2 * nkv) * hd + M * S * nh * hd) * BYTES_PER_ELEMENT
    out.append(SubOp(name="attn", kind="compute", flops=attn_flops, bytes=attn_bytes,
                     eff_name="attn_fwd", count=1))
    # 6. attn_proj (compute)
    attn_in_dim = nh * hd
    attn_proj_flops = 2 * M * S * attn_in_dim * d
    attn_proj_bytes = (M * S * attn_in_dim + attn_in_dim * d + M * S * d) * BYTES_PER_ELEMENT
    out.append(_matmul_subop("attn_proj", attn_proj_flops, attn_proj_bytes))
    # 7. ffn_norm (memory)
    out.append(_mem_subop("ffn_norm", 2 * tt * d * BYTES_PER_ELEMENT))

    # ---- MLP up: shared (all tokens) + routed (top_k/num_routed tokens each)
    # 8. shared_mlp_up (one matmul per shared expert)
    if ns > 0:
        out.append(_matmul_subop(
            "shared_mlp_up",
            _mlp_up_flops(tt, d, edim),
            _mlp_up_bytes(tt, d, edim),
            count=ns,
        ))
    # 9. x_scatter (memory) — MoE only: dispatch tokens to routed experts.
    # Sizes the expanded [tt*(1+tk), d_model] tensor that the routed branch
    # consumes (read tt*d input + write tt*tk*d expanded copy = tt*(1+tk)*d).
    if nr > 0 and tk > 0:
        out.append(_mem_subop("x_scatter", tt * (1 + tk) * d * BYTES_PER_ELEMENT))
    # 10. routed_mlp_up (one matmul per routed expert; expert_tokens = tt*tk/nr)
    if nr > 0 and tk > 0:
        T = M * S * tk // nr
        if T > 0:
            out.append(_matmul_subop(
                "routed_mlp_up_one_expert",
                _mlp_up_flops(T, d, edim),
                _mlp_up_bytes(T, d, edim),
                count=nr,
            ))
    # 11. swiglu (memory) — elementwise SiLU(gate) * up, shared + routed
    swiglu_factor = ns + tk
    if swiglu_factor > 0:
        out.append(_mem_subop("swiglu", 3 * tt * edim * swiglu_factor * BYTES_PER_ELEMENT))
    # 12. shared_mlp_down
    if ns > 0:
        out.append(_matmul_subop(
            "shared_mlp_down",
            _mlp_down_flops(tt, d, edim),
            _mlp_down_bytes(tt, d, edim),
            count=ns,
        ))
    # 13. routed_mlp_down
    if nr > 0 and tk > 0:
        T = M * S * tk // nr
        if T > 0:
            out.append(_matmul_subop(
                "routed_mlp_down_one_expert",
                _mlp_down_flops(T, d, edim),
                _mlp_down_bytes(T, d, edim),
                count=nr,
            ))
    # 14. x_gather (memory) — MoE only: combine routed-expert outputs back
    # to per-token [tt, d_model]. Mirrors x_scatter sizing.
    if nr > 0 and tk > 0:
        out.append(_mem_subop("x_gather", tt * (1 + tk) * d * BYTES_PER_ELEMENT))
    return out


def backward_subops(spec: TransformerSpec, cfg: TrainingConfig) -> list[SubOp]:
    """Per-layer bwd sub-ops. Order:
      (a) DGRAD group + memory-bound bwd ops + attn_bwd, in reverse execution
          order (critical path back through the layer).
      (b) WGRAD group at the bottom in reverse execution order — wgrads
          compute weight gradients only and don't gate downstream tasks, so
          they're separated visually.

    Each fwd matmul becomes (dgrad + wgrad) with the SAME flops/bytes as the
    fwd version. Memory-bound bwd ops appear ONCE.
    """
    M, S = cfg.num_seqs, cfg.seqlen
    tt = M * S
    d = spec.d_model
    hd = spec.head_dim
    nh, nkv = spec.n_heads, spec.n_kv_heads
    edim = spec.expert_dim
    ns, nr, tk = spec.num_shared_experts, spec.num_routed_experts, spec.top_k

    # Precompute per-matmul flop/byte tuples so dgrad + wgrad share them.
    qkv_out_dim = (nh + 2 * nkv) * hd
    qkv_flops = 2 * M * S * d * qkv_out_dim
    qkv_bytes = (M * S * d + d * qkv_out_dim + M * S * qkv_out_dim) * BYTES_PER_ELEMENT
    attn_in_dim = nh * hd
    attn_proj_flops = 2 * M * S * attn_in_dim * d
    attn_proj_bytes = (M * S * attn_in_dim + attn_in_dim * d + M * S * d) * BYTES_PER_ELEMENT
    routed_T = M * S * tk // nr if (nr > 0 and tk > 0) else 0

    dgrads: list[SubOp] = []
    wgrads: list[SubOp] = []

    # ---- DGRAD + memory + attn_bwd ----
    # MLP dgrad order is the autograd reverse of fwd: x_gather → routed_down →
    # shared_down → swiglu_bwd → routed_up → x_scatter → shared_up. Scatter
    # and gather mirror their fwd counterparts: dy_scatter undoes x_gather
    # (precedes the down dgrads); dy_gather undoes x_scatter (follows the
    # routed up_dgrad).
    swiglu_factor = ns + tk
    if nr > 0 and tk > 0:
        dgrads.append(_mem_subop("dy_scatter", tt * (1 + tk) * d * BYTES_PER_ELEMENT))
    if nr > 0 and routed_T > 0:
        dgrads.append(_matmul_subop(
            "routed_mlp_down_one_expert_dgrad",
            _mlp_down_flops(routed_T, d, edim),
            _mlp_down_bytes(routed_T, d, edim),
            count=nr,
        ))
    if ns > 0:
        dgrads.append(_matmul_subop(
            "shared_mlp_down_dgrad",
            _mlp_down_flops(tt, d, edim),
            _mlp_down_bytes(tt, d, edim),
            count=ns,
        ))
    if swiglu_factor > 0:
        dgrads.append(_mem_subop(
            "swiglu_bwd", 5 * tt * edim * swiglu_factor * BYTES_PER_ELEMENT,
        ))
    if nr > 0 and routed_T > 0:
        dgrads.append(_matmul_subop(
            "routed_mlp_up_one_expert_dgrad",
            _mlp_up_flops(routed_T, d, edim),
            _mlp_up_bytes(routed_T, d, edim),
            count=nr,
        ))
    if nr > 0 and tk > 0:
        dgrads.append(_mem_subop("dy_gather", tt * (1 + tk) * d * BYTES_PER_ELEMENT))
    if ns > 0:
        dgrads.append(_matmul_subop(
            "shared_mlp_up_dgrad",
            _mlp_up_flops(tt, d, edim),
            _mlp_up_bytes(tt, d, edim),
            count=ns,
        ))
    dgrads.append(_mem_subop("ffn_norm_bwd", 7 * tt * d * BYTES_PER_ELEMENT))
    dgrads.append(_matmul_subop("attn_proj_dgrad", attn_proj_flops, attn_proj_bytes))
    # attn_bwd: 5× flops (1× is fwd recompute, only 4× is "useful").
    attn_bwd_flops = 5 * M * nh * hd * S * S
    attn_bwd_effective_flops = 4 * M * nh * hd * S * S
    attn_bwd_bytes_read = (M * S * (nh + 2 * nkv) * hd + M * S * nh * hd) * BYTES_PER_ELEMENT
    attn_bwd_bytes_write = M * S * (nh + 2 * nkv) * hd * BYTES_PER_ELEMENT
    dgrads.append(SubOp(
        name="attn_bwd", kind="compute", flops=attn_bwd_flops,
        effective_flops=attn_bwd_effective_flops,
        bytes=attn_bwd_bytes_read + attn_bwd_bytes_write,
        eff_name="attn_bwd", count=1,
    ))
    dgrads.append(_mem_subop("rope_bwd", 2 * tt * (hd * (nh + nkv)) * BYTES_PER_ELEMENT))
    if spec.qk_norm:
        dgrads.append(_mem_subop("qk_norm_bwd", 7 * tt * (hd * (nh + nkv)) * BYTES_PER_ELEMENT))
    dgrads.append(_matmul_subop("qkv_proj_dgrad", qkv_flops, qkv_bytes))
    dgrads.append(_mem_subop("attn_norm_bwd", 7 * tt * d * BYTES_PER_ELEMENT))

    # ---- WGRAD group ----
    # MLP wgrad order: down before up (reverse of fwd; matches the dgrad
    # block above).
    if nr > 0 and routed_T > 0:
        wgrads.append(_matmul_subop(
            "routed_mlp_down_one_expert_wgrad",
            _mlp_down_flops(routed_T, d, edim),
            _mlp_down_bytes(routed_T, d, edim),
            count=nr,
        ))
    if ns > 0:
        wgrads.append(_matmul_subop(
            "shared_mlp_down_wgrad",
            _mlp_down_flops(tt, d, edim),
            _mlp_down_bytes(tt, d, edim),
            count=ns,
        ))
    if nr > 0 and routed_T > 0:
        wgrads.append(_matmul_subop(
            "routed_mlp_up_one_expert_wgrad",
            _mlp_up_flops(routed_T, d, edim),
            _mlp_up_bytes(routed_T, d, edim),
            count=nr,
        ))
    if ns > 0:
        wgrads.append(_matmul_subop(
            "shared_mlp_up_wgrad",
            _mlp_up_flops(tt, d, edim),
            _mlp_up_bytes(tt, d, edim),
            count=ns,
        ))
    wgrads.append(_matmul_subop("attn_proj_wgrad", attn_proj_flops, attn_proj_bytes))
    wgrads.append(_matmul_subop("qkv_proj_wgrad", qkv_flops, qkv_bytes))

    return dgrads + wgrads


def head_subops(spec: TransformerSpec, cfg: TrainingConfig) -> list[SubOp]:
    """Head block sub-ops in execution order:
      1. final_norm        (memory-bound; pre-head normalization)
      2. head_proj         (compute; fwd matmul, 2*tt*d*vocab flops)
      3. cross_entropy     (memory-bound; logits → loss + dlogits)
      4. head_proj_dgrad   (compute; bwd input-grad, 2*tt*d*vocab flops)
      5. head_proj_wgrad   (compute; bwd weight-grad, 2*tt*d*vocab flops)
      6. final_norm_bwd    (memory-bound; gradient through the norm)
    """
    M, S = cfg.num_seqs, cfg.seqlen
    tt = M * S
    d = spec.d_model

    final_norm = _mem_subop(
        "final_norm", 2 * tt * d * BYTES_PER_ELEMENT,
    )
    head_one_matmul_flops = 2 * tt * d * spec.vocab_size
    head_one_matmul_bytes = (
        head_weight_bytes(spec)
        + tt * d * BYTES_PER_ELEMENT
        + tt * d * BYTES_PER_ELEMENT
    )
    head_proj = _matmul_subop("head_proj", head_one_matmul_flops, head_one_matmul_bytes)
    cross_entropy = _mem_subop(
        "cross_entropy", 2 * tt * spec.vocab_size * BYTES_PER_ELEMENT,
    )
    head_proj_dgrad = _matmul_subop(
        "head_proj_dgrad", head_one_matmul_flops, head_one_matmul_bytes,
    )
    head_proj_wgrad = _matmul_subop(
        "head_proj_wgrad", head_one_matmul_flops, head_one_matmul_bytes,
    )
    final_norm_bwd = _mem_subop(
        "final_norm_bwd", 7 * tt * d * BYTES_PER_ELEMENT,
    )
    return [
        final_norm, head_proj, cross_entropy,
        head_proj_dgrad, head_proj_wgrad, final_norm_bwd,
    ]


def optimizer_step_subops(spec: TransformerSpec, cfg: TrainingConfig) -> list[SubOp]:
    """Per-layer optimizer step sub-ops.

    The task builder emits one `step_i` task per layer after all gradient
    accumulation rounds. This returns the timing model for one such task.
    """
    if cfg.optimizer == "none":
        return []
    if cfg.optimizer == "adamw":
        return [_mem_subop("adamw_step", _adamw_step_bytes(layer_weight_bytes(spec)))]
    if cfg.optimizer == "muon":
        flops, bytes_total = _muon_step_flops_bytes(
            layer_weight_matrices(spec),
            bytes_per_element=BYTES_PER_ELEMENT,
        )
        return [
            SubOp(
                name="muon_step",
                kind="compute",
                flops=flops,
                effective_flops=flops,
                bytes=bytes_total,
                eff_name="matmul",
            )
        ]
    raise ValueError(f"unknown optimizer mode: {cfg.optimizer!r}")


# ---------- timing ----------

_EFF_LOOKUP = {
    "matmul": "matmul_eff",
    "attn_fwd": "attn_fwd_eff",
    "attn_bwd": "attn_bwd_eff",
}


def _eff_value(hw: HardwareSpec, eff_name: EffName) -> float:
    if eff_name == "none":
        return 1.0
    return getattr(hw, _EFF_LOOKUP[eff_name])


def time_subop(subop: SubOp, hw: HardwareSpec) -> SubOpTiming:
    """Resolve a `SubOp` to a `SubOpTiming` given hardware. Computes math
    (compute-only), memory time always, then takes max per call and scales
    by count for the total. `effective_flops` (defaulting to `flops`) drives
    the effective-TFLOPS rate, isolating recompute overhead (e.g. attn_bwd)
    from the "useful work" rate.

    Per-op efficiency overrides on `subop` take precedence over the HW values
    (so a single bad-kernel op can be modeled without globally degrading the
    HW envelope). Effective TFLOPS is computed from the un-rounded binding
    time so a fully-compute-bound matmul reports exactly `peak × matmul_eff`
    regardless of its size (no ceil-induced quantization).
    """
    eff_flops = subop.effective_flops if subop.effective_flops is not None else subop.flops
    mem_eff = subop.mem_eff if subop.mem_eff is not None else hw.mem_eff
    # Memory time (always computed; uses gpu_membw × mem_eff).
    if subop.bytes > 0 and hw.gpu_membw_gbs > 0 and mem_eff > 0:
        mem_seconds = subop.bytes / (hw.gpu_membw_gbs * 1e9 * mem_eff)
        mem_us_exact = mem_seconds * 1e6
        mem_us = max(1, math.ceil(mem_us_exact))
    else:
        mem_seconds = 0.0
        mem_us_exact = 0.0
        mem_us = 0

    if subop.kind == "compute":
        eff = subop.compute_eff if subop.compute_eff is not None else _eff_value(hw, subop.eff_name)
        math_seconds = subop.flops / (hw.peak_tflops * 1e12 * eff)
        math_us_exact = math_seconds * 1e6
        math_us = max(1, math.ceil(math_us_exact))
        per_call_us = max(math_us, mem_us)
        per_call_us_exact = max(math_us_exact, mem_us_exact)
        total_us = per_call_us * subop.count
        bound_by: SubOpKind = "memory" if mem_us > math_us else "compute"
        # Effective TFLOPS uses the un-rounded binding seconds so a pure
        # compute-bound matmul shows exactly peak × eff (no ceil bias).
        binding_seconds = max(math_seconds, mem_seconds)
        if binding_seconds > 0:
            effective_tflops = eff_flops / (binding_seconds * 1e12)
        else:
            effective_tflops = 0.0
        return SubOpTiming(
            name=subop.name, kind="compute", flops=subop.flops,
            effective_flops=eff_flops, bytes=subop.bytes,
            count=subop.count, math_us=math_us, mem_us=mem_us,
            per_call_us=per_call_us, per_call_us_exact=per_call_us_exact,
            total_us=total_us, bound_by=bound_by,
            effective_tflops=effective_tflops,
        )
    # memory-bound
    per_call_us = mem_us
    per_call_us_exact = mem_us_exact
    total_us = per_call_us * subop.count
    return SubOpTiming(
        name=subop.name, kind="memory", flops=0, effective_flops=0,
        bytes=subop.bytes,
        count=subop.count, math_us=None, mem_us=mem_us,
        per_call_us=per_call_us, per_call_us_exact=per_call_us_exact,
        total_us=total_us, bound_by="memory",
        effective_tflops=None,
    )


# ---------- aggregators ----------

def layer_fwd_breakdown(spec: TransformerSpec, hw: HardwareSpec,
                        cfg: TrainingConfig) -> list[SubOpTiming]:
    return [time_subop(s, hw) for s in forward_subops(spec, cfg)]


def layer_bwd_breakdown(spec: TransformerSpec, hw: HardwareSpec,
                        cfg: TrainingConfig) -> list[SubOpTiming]:
    return [time_subop(s, hw) for s in backward_subops(spec, cfg)]


def head_breakdown(spec: TransformerSpec, hw: HardwareSpec,
                   cfg: TrainingConfig) -> list[SubOpTiming]:
    return [time_subop(s, hw) for s in head_subops(spec, cfg)]


def optimizer_step_breakdown(spec: TransformerSpec, hw: HardwareSpec,
                             cfg: TrainingConfig) -> list[SubOpTiming]:
    return [time_subop(s, hw) for s in optimizer_step_subops(spec, cfg)]


def layer_fwd_microseconds(spec: TransformerSpec, hw: HardwareSpec,
                           cfg: TrainingConfig) -> int:
    return max(1, sum(t.total_us for t in layer_fwd_breakdown(spec, hw, cfg)))


# Sub-ops skipped when recomputing a layer's activations: the down
# projections feed the saved residual handoff (`y_i`), which is never
# discarded, so they are not re-executed; `x_gather` only reassembles that
# same output.
_RECOMPUTE_SKIP_SUBOPS = frozenset({
    "shared_mlp_down",
    "routed_mlp_down_one_expert",
    "x_gather",
})


def layer_recompute_microseconds(spec: TransformerSpec, hw: HardwareSpec,
                                 cfg: TrainingConfig) -> int:
    """Runtime of re-running a layer's forward compute minus the final down
    projection(s), used when the layer's saved activations are recomputed."""
    return max(1, sum(
        t.total_us for t in layer_fwd_breakdown(spec, hw, cfg)
        if t.name not in _RECOMPUTE_SKIP_SUBOPS
    ))


def layer_bwd_microseconds(spec: TransformerSpec, hw: HardwareSpec,
                           cfg: TrainingConfig) -> int:
    return max(1, sum(t.total_us for t in layer_bwd_breakdown(spec, hw, cfg)))


def head_microseconds(spec: TransformerSpec, hw: HardwareSpec,
                      cfg: TrainingConfig) -> int:
    return max(1, sum(t.total_us for t in head_breakdown(spec, hw, cfg)))


def optimizer_step_microseconds(spec: TransformerSpec, hw: HardwareSpec,
                                cfg: TrainingConfig) -> int:
    timings = optimizer_step_breakdown(spec, hw, cfg)
    if not timings:
        return 0
    return max(1, sum(t.total_us for t in timings))
