"""
Universal 64 KB Zero-Padding Stripe Geometry & Configuration for SplitSlab.
"""

from __future__ import annotations

from dataclasses import dataclass

NVME_STRIPE_BYTES: int = 65536  # Exact 64.0 KB NVMe page / GDS DMA unit


@dataclass(frozen=True)
class SplitSlabGeometry:
    """
    Exact zero-fragmentation Split-Plane geometry derived from a model's KV-head count
    `num_kv_heads` and `head_dim` (assuming FP16/BF16 2-byte elements):
      - `bytes_per_token_per_layer = 2 (K+V) * num_kv_heads * head_dim * 2`
      - `plane1_residual_tokens = 65536 // bytes_per_token_per_layer`
      - For target Plane-0 compaction ratio `rho = 0.20` (`80%` offloaded to Plane 1):
        `plane0_anchor_tokens = round(plane1_residual_tokens * rho / (1 - rho))`
        `block_tokens = plane0_anchor_tokens + plane1_residual_tokens`

    Concrete exact mappings (`d_h = 128`, `rho = 0.20`):
      - `H_kv = 2` (`Qwen2.5-3B`):   1,024 B/tok -> M = 80 (Plane 0 = 16, Plane 1 = 64 -> 65,536 B exact)
      - `H_kv = 4` (`Qwen2.5-7B`):   2,048 B/tok -> M = 40 (Plane 0 =  8, Plane 1 = 32 -> 65,536 B exact)
      - `H_kv = 8` (`Granite-8B`, `Llama-8B`, `Qwen2.5-14B`):
                                     4,096 B/tok -> M = 20 (Plane 0 =  4, Plane 1 = 16 -> 65,536 B exact)
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_tokens: int
    plane0_anchor_tokens: int
    plane1_residual_tokens: int
    bytes_per_token_per_layer: int
    stripe_bytes: int = NVME_STRIPE_BYTES
    local_window_tokens: int = 48

    @classmethod
    def from_model_config(
        cls,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        anchor_ratio: float = 0.20,
        local_window_tokens: int = 48,
    ) -> "SplitSlabGeometry":
        b_tok = 2 * num_kv_heads * head_dim * 2  # FP16 K + V
        if NVME_STRIPE_BYTES % b_tok != 0:
            raise ValueError(
                f"64KB stripe ({NVME_STRIPE_BYTES}B) not evenly divisible by {b_tok}B/tok"
            )
        p1_toks = NVME_STRIPE_BYTES // b_tok
        p0_toks = max(2, int(round(p1_toks * anchor_ratio / (1.0 - anchor_ratio))))
        blk_toks = p0_toks + p1_toks
        return cls(
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_tokens=blk_toks,
            plane0_anchor_tokens=p0_toks,
            plane1_residual_tokens=p1_toks,
            bytes_per_token_per_layer=b_tok,
            stripe_bytes=NVME_STRIPE_BYTES,
            local_window_tokens=local_window_tokens,
        )
