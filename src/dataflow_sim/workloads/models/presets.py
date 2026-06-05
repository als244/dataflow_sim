"""Transformer model preset registry.

Loaded once at import time and exported through the HTTP layer so the UI can
populate its model dropdown.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from dataflow_sim.workloads.models.transformer import TransformerSpec


_TRANSFORMER_FIELDS = {
    "vocab_size", "n_layers", "d_model", "head_dim", "n_heads", "n_kv_heads",
    "expert_dim", "num_shared_experts", "num_routed_experts", "top_k", "qk_norm",
}


@lru_cache(maxsize=1)
def load_model_presets() -> dict[str, TransformerSpec]:
    """Read `model_dims.json` and return a name→TransformerSpec registry.
    Ignores `datatypes` and `is_causal` (causal is implicit; bf16 is hardcoded).
    """
    raw = json.loads(
        (files("dataflow_sim.workloads.models") / "model_dims.json").read_text()
    )
    out: dict[str, TransformerSpec] = {}
    for name, body in raw.items():
        kwargs = {k: body[k] for k in _TRANSFORMER_FIELDS if k in body}
        out[name] = TransformerSpec(**kwargs)
    return out
