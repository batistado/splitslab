"""
Unit & Integration Tests for SplitSlab Storage Geometry, Direct-NVMe (`F_NOCACHE`) Store,
Subspace Routing Sketches (`16B/32B/64B/128B`), and Theorems 1 & 2.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splitslab import (
    NVME_STRIPE_BYTES,
    DirectNVMeSlabStore,
    SketchSpec,
    SplitSlabGeometry,
    analytical_recompaction_waf,
    async_lookahead_overlap_ratio,
    attention_output_perturbation_bound,
    deserialize_stripe_fp16,
    greedy_margin_equivalence_condition,
    pack_subspace_sketch,
    serialize_stripe_fp16,
    unpack_subspace_sketches,
    verify_empirical_perturbation_bound,
)


@pytest.mark.parametrize(
    "num_kv_heads,expected_block,expected_p0,expected_p1",
    [
        (2, 80, 16, 64),  # Qwen2.5-3B
        (4, 40, 8, 32),   # Qwen2.5-7B
        (8, 20, 4, 16),   # Granite-3.1-8B / Llama-8B / Qwen2.5-Coder-14B
    ],
)
def test_universal_zero_padding_64kb_geometry(
    num_kv_heads: int, expected_block: int, expected_p0: int, expected_p1: int
) -> None:
    geom = SplitSlabGeometry.from_model_config(
        num_layers=32,
        num_q_heads=32,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        anchor_ratio=0.20,
    )
    assert geom.block_tokens == expected_block
    assert geom.plane0_anchor_tokens == expected_p0
    assert geom.plane1_residual_tokens == expected_p1
    assert geom.plane1_residual_tokens * geom.bytes_per_token_per_layer == NVME_STRIPE_BYTES


@pytest.mark.parametrize("num_kv_heads,n_res", [(2, 64), (4, 32), (8, 16)])
def test_stripe_fp16_roundtrip_exact_64kb(num_kv_heads: int, n_res: int) -> None:
    k = torch.randn((num_kv_heads, n_res, 128), dtype=torch.float16)
    v = torch.randn((num_kv_heads, n_res, 128), dtype=torch.float16)
    blob = serialize_stripe_fp16(k, v)
    assert len(blob) == 65536
    k_out, v_out = deserialize_stripe_fp16(blob, num_kv_heads, n_res, 128, device="cpu")
    assert torch.equal(k_out[0], k)
    assert torch.equal(v_out[0], v)


def test_direct_nvme_fnocache_waf_and_dedup(tmp_path: Path) -> None:
    store_dir = tmp_path / "nvme_cas"
    store = DirectNVMeSlabStore(store_dir, direct_nocache=True, sync_mode="fsync", io_workers=4)
    try:
        if sys.platform == "darwin":
            assert store.nocache_enabled is True
        payload_a = b"\x5a" * NVME_STRIPE_BYTES
        payload_b = b"\xa5" * NVME_STRIPE_BYTES

        off_a1, hit_a1 = store.put_immutable_stripe("sha_block_0", payload_a, count_dedup=False)
        off_b1, hit_b1 = store.put_immutable_stripe("sha_block_1", payload_b, count_dedup=False)
        assert not hit_a1 and not hit_b1
        assert store.waf == pytest.approx(1.0, abs=1e-9)

        # Collaborating Agent B probes exact same canonical SHA-256 prefix blocks -> 100% dedup!
        off_a2, hit_a2 = store.put_immutable_stripe("sha_block_0", payload_a, count_dedup=True, count_waf_logical=False)
        off_b2, hit_b2 = store.put_immutable_stripe("sha_block_1", payload_b, count_dedup=True, count_waf_logical=False)
        assert hit_a2 and hit_b2
        assert off_a2 == off_a1 and off_b2 == off_b1
        assert store.dedup_hits == 2 and store.dedup_queries == 2
        assert store.waf == pytest.approx(1.0, abs=1e-9)

        # Parallel multi-queue pread check
        blobs = store.pread_stripes_parallel([off_a1, off_b1], NVME_STRIPE_BYTES)
        assert blobs[0] == payload_a and blobs[1] == payload_b

        # Mutable rewrite increases physical WAF linearly!
        store.rewrite_mutable_page(off_a1, payload_b)
        assert store.waf == pytest.approx(1.5, abs=1e-9)
    finally:
        store.close()
        shutil.rmtree(store_dir, ignore_errors=True)


@pytest.mark.parametrize("sketch_bytes", [16, 32, 64, 128])
@pytest.mark.parametrize("num_kv_heads,n_res", [(2, 64), (4, 32), (8, 16)])
def test_sketch_codecs_exact_byte_budget_and_correlation(
    sketch_bytes: int, num_kv_heads: int, n_res: int
) -> None:
    torch.manual_seed(42)
    spec = SketchSpec.create(total_bytes=sketch_bytes, num_kv_heads=num_kv_heads, head_dim=128)
    # Build orthonormal P_layer [H_kv, 128, dims_kv]
    P_layer = torch.zeros((num_kv_heads, 128, spec.dims_per_kv_head))
    for g in range(num_kv_heads):
        q_o, _ = torch.linalg.qr(torch.randn(128, spec.dims_per_kv_head))
        P_layer[g] = q_o

    # Create 10 stripes where Stripe 3 has a strong embedded signal aligned with query `q`
    q_signal = torch.randn(num_kv_heads, 128)
    raw_sketches = []
    for m in range(10):
        k_m = torch.randn(num_kv_heads, n_res, 128) * 0.4
        if m == 3:
            k_m[:, :8, :] += q_signal.unsqueeze(1) * 3.5
        b = pack_subspace_sketch(k_m, P_layer, spec)
        assert len(b) == sketch_bytes
        raw_sketches.append(b)

    unpacked = unpack_subspace_sketches(raw_sketches, spec, device="cpu")
    assert unpacked.shape == (num_kv_heads, spec.num_clusters, 10, spec.dims_per_kv_head)

    if sketch_bytes >= 32:
        q_proj = torch.einsum("gd,gdb->gb", q_signal, P_layer)
        scores = torch.einsum("gb,gcmb->cm", q_proj, unpacked).amax(dim=0)
        assert int(torch.argmax(scores).item()) == 3


def test_theorem1_exact_perturbation_and_margin_bounds() -> None:
    torch.manual_seed(2026)
    for _ in range(100):
        q = torch.randn(128)
        k_anc = torch.randn(16, 128)
        v_anc = torch.randn(16, 128) * 2.5
        k_res = torch.randn(32, 128) + torch.randn(1) * 1.5  # Vary alpha_res across [0.01, 0.99]
        v_res = torch.randn(32, 128) * 2.5
        chk = verify_empirical_perturbation_bound(q, k_anc, v_anc, k_res, v_res)
        assert chk["bound_holds"] is True
        assert chk["empirical_l2_error"] <= chk["theoretical_upper_bound"] + 1e-5

    assert greedy_margin_equivalence_condition(
        logit_margin=1.5, downstream_lipschitz=2.0, omitted_alpha_delta=0.05, max_v_norm=3.0
    ) is True
    assert greedy_margin_equivalence_condition(
        logit_margin=0.5, downstream_lipschitz=2.0, omitted_alpha_delta=0.05, max_v_norm=3.0
    ) is False


def test_theorem2_waf_and_async_overlap_formulas() -> None:
    assert analytical_recompaction_waf(1, 0.0, 65536, 65536) == pytest.approx(1.0)
    assert analytical_recompaction_waf(10, 0.8, 65536, 65536) == pytest.approx(8.2)

    ov = async_lookahead_overlap_ratio(
        num_recalled_stripes=3,
        stripe_bytes=65536,
        nvme_bandwidth_gbps=6.0,
        nvme_base_latency_us=165.0,
        per_layer_compute_us=580.0,
        lookahead_layers=2,
    )
    assert ov["fully_hidden"] is True
    assert ov["exposed_stall_us"] == 0.0
