from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    peak_tflops: float
    fast_memory_bw_gbs: float
    from_slow_bw_gbs: float
    to_slow_bw_gbs: float
    matmul_eff: float
    attn_fwd_eff: float
    attn_bwd_eff: float
    mem_eff: float = 0.9


def gbs_to_bytes_per_microsecond(gbs: float) -> int:
    """Convert GB/s to bytes/us. 1 GB/s = 1000 B/us."""
    return max(1, round(gbs * 1000))


HARDWARE_PRESETS: dict[str, HardwareSpec] = {
    "H100": HardwareSpec(
        peak_tflops=989.0,
        fast_memory_bw_gbs=3000.0,
        from_slow_bw_gbs=50.0,
        to_slow_bw_gbs=50.0,
        matmul_eff=0.65,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.5,
        mem_eff=0.9,
    ),
    "RTX_5090": HardwareSpec(
        peak_tflops=210.0,
        fast_memory_bw_gbs=1500.0,
        from_slow_bw_gbs=30.0,
        to_slow_bw_gbs=30.0,
        matmul_eff=0.95,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.3,
        mem_eff=0.9,
    ),
}
