from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    peak_tflops_bf16: float
    peak_tflops_fp8: float
    peak_tflops_fp4: float | None
    fast_memory_bw_gbs: float
    from_slow_bw_gbs: float
    to_slow_bw_gbs: float
    matmul_eff_bf16: float
    matmul_eff_fp8: float
    matmul_eff_fp4: float | None
    attn_fwd_eff: float
    attn_bwd_eff: float
    mem_eff: float = 0.9
    scale_up_bw_gbs: float = 400.0

    @property
    def peak_tflops(self) -> float:
        return self.peak_tflops_bf16

    @property
    def matmul_eff(self) -> float:
        return self.matmul_eff_bf16


def gbs_to_bytes_per_microsecond(gbs: float) -> int:
    """Convert GB/s to bytes/us. 1 GB/s = 1000 B/us."""
    return max(1, round(gbs * 1000))


HARDWARE_PRESETS: dict[str, HardwareSpec] = {
    "H100": HardwareSpec(
        peak_tflops_bf16=989.0,
        peak_tflops_fp8=1978.0,
        peak_tflops_fp4=None,
        fast_memory_bw_gbs=3000.0,
        scale_up_bw_gbs=400.0,
        from_slow_bw_gbs=50.0,
        to_slow_bw_gbs=50.0,
        matmul_eff_bf16=0.65,
        matmul_eff_fp8=0.65,
        matmul_eff_fp4=None,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.5,
        mem_eff=0.9,
    ),
    "GB300": HardwareSpec(
        peak_tflops_bf16=2500.0,
        peak_tflops_fp8=5000.0,
        peak_tflops_fp4=15000.0,
        fast_memory_bw_gbs=8000.0,
        scale_up_bw_gbs=800.0,
        from_slow_bw_gbs=400.0,
        to_slow_bw_gbs=400.0,
        matmul_eff_bf16=0.65,
        matmul_eff_fp8=0.65,
        matmul_eff_fp4=0.65,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.5,
        mem_eff=0.9,
    ),
    "RTX_5090": HardwareSpec(
        peak_tflops_bf16=210.0,
        peak_tflops_fp8=420.0,
        peak_tflops_fp4=840.0,
        fast_memory_bw_gbs=1500.0,
        scale_up_bw_gbs=30.0,
        from_slow_bw_gbs=30.0,
        to_slow_bw_gbs=30.0,
        matmul_eff_bf16=0.95,
        matmul_eff_fp8=0.95,
        matmul_eff_fp4=0.95,
        attn_fwd_eff=0.6,
        attn_bwd_eff=0.3,
        mem_eff=0.9,
    ),
    "SRAM Accelerator": HardwareSpec(
        peak_tflops_bf16=3200.0,
        peak_tflops_fp8=6400.0,
        peak_tflops_fp4=12800.0,
        fast_memory_bw_gbs=40000.0,
        scale_up_bw_gbs=800.0,
        from_slow_bw_gbs=3000.0,
        to_slow_bw_gbs=3000.0,
        matmul_eff_bf16=1.0,
        matmul_eff_fp8=1.0,
        matmul_eff_fp4=1.0,
        attn_fwd_eff=1.0,
        attn_bwd_eff=1.0,
        mem_eff=1.0,
    ),
}
