#!/usr/bin/env python3
"""
SplitSlab Quickstart Demo
=========================
Demonstrates SplitSlab's core storage, sketching, and theoretical primitives in <2 seconds:
  1. Universal zero-padding 64.0 KB (65,536 B) stripe geometry across GQA H_kv in {2, 4, 8}.
  2. Direct uncached NVMe CAS storage (`fcntl.F_NOCACHE`) achieving WAF = 1.000 and 100% cross-agent dedup.
  3. Compact 64-byte binary subspace routing sketch pack/unpack (`1,024x` compression).
  4. Non-asymptotic Theorem 1 output perturbation bound & Theorem 2 re-compaction WAF comparison.
  5. RecallRegret-Bench incident dataset inspection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recallregret_bench import TRAJECTORY_SPECS, score_delayed_recall_response
from splitslab import (
    NVME_STRIPE_BYTES,
    DirectNVMeSlabStore,
    SketchSpec,
    SplitSlabGeometry,
    analytical_recompaction_waf,
    deserialize_stripe_fp16,
    pack_subspace_sketch,
    serialize_stripe_fp16,
    unpack_subspace_sketches,
    verify_empirical_perturbation_bound,
)


def main() -> None:
    print("=== 1. Universal Zero-Padding 64.0 KB Split-Plane Geometry ===")
    configs = [
        ("Qwen2.5-3B (16:2)", 36, 16, 2, 128),
        ("Qwen2.5-7B (28:4)", 28, 28, 4, 128),
        ("Granite-3.1-8B (32:8)", 40, 32, 8, 128),
        ("Qwen2.5-Coder-14B (40:8)", 48, 40, 8, 128),
    ]
    for label, n_layers, n_q, n_kv, d_h in configs:
        geom = SplitSlabGeometry.from_model_config(n_layers, n_q, n_kv, d_h, anchor_ratio=0.20)
        assert geom.stripe_bytes == NVME_STRIPE_BYTES
        print(
            f"  {label:<26} -> M={geom.block_tokens:2d} "
            f"(Plane-0={geom.plane0_anchor_tokens:2d}, Plane-1={geom.plane1_residual_tokens:2d}) | "
            f"Stripe={geom.stripe_bytes:,} B (0 B padding)"
        )

    print("\n=== 2. Direct Uncached NVMe CAS Store (fcntl.F_NOCACHE) ===")
    geom_7b = SplitSlabGeometry.from_model_config(28, 28, 4, 128)
    torch.manual_seed(42)
    k_res = torch.randn(geom_7b.num_kv_heads, geom_7b.plane1_residual_tokens, geom_7b.head_dim, dtype=torch.float16)
    v_res = torch.randn_like(k_res)
    payload = serialize_stripe_fp16(k_res, v_res)
    digest = hashlib.sha256(b"repo_snapshot_block_00_layer_00").hexdigest()

    with tempfile.TemporaryDirectory(prefix="splitslab_demo_") as tmp_dir:
        store = DirectNVMeSlabStore(Path(tmp_dir), direct_nocache=True, sync_mode="fsync")
        off_a, hit_a = store.put_immutable_stripe(digest, payload, count_dedup=False)
        off_b, hit_b = store.put_immutable_stripe(digest, payload, count_dedup=True)
        raw_back = store.pread_stripe(off_a)
        k_rec, v_rec = deserialize_stripe_fp16(
            raw_back, geom_7b.num_kv_heads, geom_7b.plane1_residual_tokens, geom_7b.head_dim, device="cpu"
        )
        assert torch.equal(k_res, k_rec.squeeze(0)) and torch.equal(v_res, v_rec.squeeze(0))
        waf = store.physical_bytes_written / float(store.logical_unique_bytes_stored)
        dedup_pct = 100.0 * store.dedup_hits / max(1, store.dedup_queries)
        print(
            f"  F_NOCACHE active={store.nocache_enabled} | Agent-A write offset={off_a} (hit={hit_a}) | "
            f"Agent-B CAS hit={hit_b} (offset={off_b})"
        )
        print(
            f"  Physical WAF={waf:.3f}x | Cross-Agent Dedup={dedup_pct:.1f}% | "
            f"24-Turn Re-Compaction WAF={analytical_recompaction_waf(24, 0.817, 65536):.2f}x"
        )
        store.close()

    print("\n=== 3. Strict 64-Byte Subspace Routing Sketch (1,024x Compression) ===")
    spec = SketchSpec.create(total_bytes=64, num_kv_heads=4, head_dim=128)
    P_ortho = torch.linalg.qr(torch.randn(4, 128, spec.dims_per_kv_head))[0]
    sketch_bytes = pack_subspace_sketch(k_res.float(), P_ortho, spec)
    unpacked = unpack_subspace_sketches([sketch_bytes], spec, device="cpu")
    print(
        f"  Packed sketch size={len(sketch_bytes)} B ({NVME_STRIPE_BYTES // len(sketch_bytes)}x compression) | "
        f"Unpacked tensor shape={tuple(unpacked.shape)}"
    )

    print("\n=== 4. Theorem 1 Exact Non-Asymptotic Perturbation Check ===")
    q = torch.randn(128)
    k_anc = torch.randn(8, 128)
    v_anc = torch.randn(8, 128)
    audit = verify_empirical_perturbation_bound(q, k_anc, v_anc, k_res[0].float(), v_res[0].float())
    print(
        f"  Omitted alpha_R={audit['omitted_alpha_mass']:.4f} | "
        f"Empirical L2 error={audit['empirical_l2_error']:.4f} <= "
        f"Bound={audit['theoretical_upper_bound']:.4f} (holds={audit['bound_holds']})"
    )

    print("\n=== 5. RecallRegret-Bench Incident Sample ===")
    sample = TRAJECTORY_SPECS[0]
    sc = score_delayed_recall_response(f"{sample.const_name} | {sample.param_val}", sample)
    print(
        f"  Incident {sample.traj_id} ({sample.domain}): trap={sample.hex_code} -> "
        f"expected='{sample.const_name} | {sample.param_val}' (verified exact_match={sc['exact_match']})"
    )


if __name__ == "__main__":
    main()
