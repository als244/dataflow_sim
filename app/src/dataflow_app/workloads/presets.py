"""Model + hardware preset registries.

Loaded once at import time. Both registries are also exported via the
HTTP layer (`GET /api/presets`) so the UI can populate its dropdowns.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from dataflow_app.workloads.transformer import HardwareEnv, TransformerSpec


_TRANSFORMER_FIELDS = {
    "vocab_size", "n_layers", "d_model", "head_dim", "n_heads", "n_kv_heads",
    "expert_dim", "num_shared_experts", "num_routed_experts", "top_k", "qk_norm",
}


@lru_cache(maxsize=1)
def load_model_presets() -> dict[str, TransformerSpec]:
    """Read `model_dims.json` and return a name→TransformerSpec registry.
    Ignores `datatypes` and `is_causal` (causal is implicit; bf16 is hardcoded).
    """
    raw = json.loads((files("dataflow_app") / "model_dims.json").read_text())
    out: dict[str, TransformerSpec] = {}
    for name, body in raw.items():
        kwargs = {k: body[k] for k in _TRANSFORMER_FIELDS if k in body}
        out[name] = TransformerSpec(**kwargs)
    return out


HARDWARE_PRESETS: dict[str, HardwareEnv] = {
    "H100": HardwareEnv(
        peak_tflops=989.0,
        gpu_membw_gbs=3000.0,
        interconnect_bw_gbs=50.0,
        matmul_eff=0.65,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.5,
        mem_eff=0.9,
    ),
    "RTX_5090": HardwareEnv(
        peak_tflops=210.0,
        gpu_membw_gbs=1500.0,
        interconnect_bw_gbs=30.0,
        matmul_eff=0.95,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.3,
        mem_eff=0.9,
    ),
}
