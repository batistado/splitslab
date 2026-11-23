"""
Generalized Subspace Routing Sketch Codec (`16B / 32B / 64B / 128B`) for SplitSlab.
Supports arbitrary `(H_kv, H_q, d_h)` GQA architectures and configurable byte budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import numpy as np
import torch
from transformers import PreTrainedModel


@dataclass(frozen=True)
class SketchSpec:
    """
    Bit-exact layout for a `total_bytes` Subspace Routing Sketch:
      - `num_clusters = 2` (2 float16 sub-cluster L2 norms = 4-byte header, or 1 cluster = 2B header for 16B)
      - Remaining `payload_bytes = total_bytes - header_bytes` holds `payload_bits = 8 * payload_bytes`
        packed 1-bit sign projections across `num_clusters * num_kv_heads * dims_per_kv_head`.
    """

    total_bytes: int
    num_clusters: int
    num_kv_heads: int
    dims_per_kv_head: int
    header_bytes: int
    payload_bytes: int

    @classmethod
    def create(cls, total_bytes: int, num_kv_heads: int, head_dim: int = 128) -> "SketchSpec":
        if total_bytes == 16:
            # 1 sub-cluster: 2B fp16 norm + 14B (112 bits) -> e.g. H_kv=4 -> 28 dims/head; H_kv=8 -> 14 dims/head; H_kv=2 -> 56 dims/head
            n_clust = 1
            hdr_b = 2
        else:
            # 2 sub-clusters: 4B (2 x fp16 norms) + (total_bytes - 4)B sign bits
            # Note: for total_bytes = 64 -> 60B = 480 bits = 2 clusters * 240 bits/cluster:
            #   - H_kv = 2 -> 120 dims/head
            #   - H_kv = 4 ->  60 dims/head
            #   - H_kv = 8 ->  30 dims/head
            # For total_bytes = 32 -> 28B = 224 bits = 2 clusters * 112 bits/cluster:
            #   - H_kv = 2 -> 56 dims/head; H_kv = 4 -> 28 dims/head; H_kv = 8 -> 14 dims/head
            # For total_bytes = 128 -> 124B = 992 bits: cap dims_per_kv_head <= head_dim
            n_clust = 2
            hdr_b = 4

        pay_b = total_bytes - hdr_b
        bits_per_clust = (pay_b * 8) // n_clust
        dims_kv = min(head_dim, bits_per_clust // num_kv_heads)
        return cls(
            total_bytes=total_bytes,
            num_clusters=n_clust,
            num_kv_heads=num_kv_heads,
            dims_per_kv_head=dims_kv,
            header_bytes=hdr_b,
            payload_bytes=pay_b,
        )


def build_model_subspaces(
    model: PreTrainedModel,
    num_res_tokens: int,
    sketch_bytes: int = 64,
    device: str = "mps",
) -> tuple[torch.Tensor, SketchSpec]:
    """
    Constructs static, prompt-independent head-space `W_Q/W_K` principal subspaces
    `P[L, H_kv, d_h, dims_per_kv_head]` weighted by sub-cluster RoPE phase coherence.
    100% deterministic from static pretrained weights -> identical across all fleet nodes.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_q
    gqa_groups = num_q // num_kv

    spec = SketchSpec.create(total_bytes=sketch_bytes, num_kv_heads=num_kv, head_dim=head_dim)
    sub_span = max(4, num_res_tokens // spec.num_clusters)

    backbone = model.model
    inv_freq = backbone.rotary_emb.inv_freq.float().cpu()  # [head_dim // 2]
    t_pos = torch.arange(sub_span, dtype=torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)
    coh = torch.sqrt(torch.cos(t_pos).mean(0) ** 2 + torch.sin(t_pos).mean(0) ** 2)
    coh_d = torch.cat([coh, coh], dim=0)  # [head_dim]

    P_all = torch.zeros((num_layers, num_kv, head_dim, spec.dims_per_kv_head), dtype=torch.float32)
    for l in range(num_layers):
        attn = backbone.layers[l].self_attn
        Wq = attn.q_proj.weight.detach().float().cpu().view(num_kv, gqa_groups, head_dim, -1)
        Wk = attn.k_proj.weight.detach().float().cpu().view(num_kv, head_dim, -1)
        for g in range(num_kv):
            cov_k = torch.matmul(Wk[g], Wk[g].T)
            cov_q = torch.einsum("rqh,rkh->qk", Wq[g], Wq[g]) / float(gqa_groups)
            energy = (coh_d.unsqueeze(1) * (cov_k @ cov_q + cov_q @ cov_k)) * coh_d.unsqueeze(0)
            _, vecs = torch.linalg.eigh(0.5 * (energy + energy.T))
            P_all[l, g] = vecs[:, -spec.dims_per_kv_head :].flip(-1)

    return P_all.to(device=device, dtype=torch.float32), spec


def pack_subspace_sketch(
    k_res_pure: torch.Tensor,  # [H_kv, N_res, d_h] fp32 (post-bias cancellation)
    P_layer: torch.Tensor,     # [H_kv, d_h, dims_kv] fp32
    spec: SketchSpec,
) -> bytes:
    """
    Packs a Cold Residual stripe into an exact `spec.total_bytes`-byte binary sketch.
    """
    h_kv, n_res, d_h = k_res_pure.shape
    span = n_res // spec.num_clusters
    kc = k_res_pure[:, : spec.num_clusters * span, :].view(h_kv, spec.num_clusters, span, d_h)

    tok_w = torch.softmax(kc.norm(dim=-1) * 0.35, dim=-1).unsqueeze(-1)
    mu = (kc * tok_w).sum(dim=2)  # [H_kv, C, d_h]
    proj = torch.einsum("gsc,gcb->gsb", mu, P_layer)  # [H_kv, C, dims_kv]
    active_bits = h_kv * spec.dims_per_kv_head
    proj_c = proj.permute(1, 0, 2).reshape(spec.num_clusters, active_bits)

    norms_fp16 = proj_c.norm(dim=-1).half().cpu().tolist()
    hdr_fmt = f"<{spec.num_clusters}e"
    header = struct.pack(hdr_fmt, *norms_fp16)

    flat_bits = (proj_c >= 0).to(torch.uint8).cpu().numpy().reshape(-1)
    total_payload_bits = spec.payload_bytes * 8
    if flat_bits.size < total_payload_bits:
        padded = np.zeros(total_payload_bits, dtype=np.uint8)
        padded[: flat_bits.size] = flat_bits
        flat_bits = padded
    packed_bits = np.packbits(flat_bits[:total_payload_bits]).tobytes()

    raw = header + packed_bits
    if len(raw) != spec.total_bytes:
        raise ValueError(f"Sketch byte mismatch: expected {spec.total_bytes}, got {len(raw)}")
    return raw


def unpack_subspace_sketches(
    sketch_bytes_list: list[bytes],
    spec: SketchSpec,
    device: str,
) -> torch.Tensor:
    """
    Unpacks M binary sketches into `[H_kv, num_clusters, M, dims_per_kv_head]` on GPU HBM.
    """
    m_count = len(sketch_bytes_list)
    hdr_fmt = f"<{spec.num_clusters}e"
    norms = np.zeros((m_count, spec.num_clusters), dtype=np.float32)
    bits_u8 = np.zeros((m_count, spec.payload_bytes), dtype=np.uint8)

    for i, raw in enumerate(sketch_bytes_list):
        norms[i] = struct.unpack(hdr_fmt, raw[: spec.header_bytes])
        bits_u8[i] = np.frombuffer(raw[spec.header_bytes :], dtype=np.uint8)

    active_bits = spec.num_kv_heads * spec.dims_per_kv_head
    used_total_bits = spec.num_clusters * active_bits
    signs_all = np.unpackbits(bits_u8, axis=1)[:, :used_total_bits].astype(np.float32) * 2.0 - 1.0
    signs_t = torch.from_numpy(signs_all).to(device=device).view(
        m_count, spec.num_clusters, spec.num_kv_heads, spec.dims_per_kv_head
    )
    norms_t = torch.from_numpy(norms).to(device=device)
    scale = (norms_t / math.sqrt(float(active_bits)) * math.sqrt(math.pi / 2.0)).unsqueeze(-1).unsqueeze(-1)
    return (signs_t * scale).permute(2, 1, 0, 3).contiguous()
