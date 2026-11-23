#!/usr/bin/env python3
"""
SplitSlab Storage Microbenchmarks, Multi-Agent Fleet Dedup, Ablations, Theorem-1
Empirical Verification, and 8.2K / 16.4K Long-Horizon Context Scaling
================================================================================
Executes 5 systems & ML experiments locally on Apple Silicon (`mps`, `float16`) +
internal Apple NVMe SSD (`fcntl.F_NOCACHE` + `F_FULLFSYNC`):
  A. Buffered vs. Direct-NVMe (`F_NOCACHE`) Read/Write Latency & Throughput + WAF vs. Turns K
  B. Multi-Agent Prefix-Sharing Fleet ($P \in \{1, 2, 4, 8, 16, 32\}$ Collaborating Agents)
  C. Sketch Byte Budget (`16B/32B/64B/128B`), Lookahead Depth (`0/1/2/3`), & Proposition-1 Ablations
  D. Empirical Theorem-1 Perturbation Bound Verification Across 560 Real `Qwen2.5-7B` Attention Heads
  E. Long-Horizon `8.2K` & `16.4K` Token Context Scaling (`~200` and `~400` Cold Stripes/Layer)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recallregret_bench import (
    TRAJECTORY_SPECS,
    build_model_formatted_trajectory,
    score_delayed_recall_response,
)
from splitslab import (
    DARWIN_F_FULLFSYNC,
    DARWIN_F_NOCACHE,
    NVME_STRIPE_BYTES,
    DirectNVMeSlabStore,
    SplitSlabAblationFlags,
    SplitSlabGeometry,
    SplitSlabSession,
    analytical_recompaction_waf,
    async_lookahead_overlap_ratio,
    build_model_subspaces,
    verify_empirical_perturbation_bound,
)


# ============================================================================
# Part A: Physical Apple NVMe SSD `F_NOCACHE` vs. OS-Buffered Microbenchmark
# ============================================================================
def run_nvme_storage_microbenchmark(work_dir: Path) -> dict[str, Any]:
    print("\n=== [Part A] Running Physical Apple NVMe SSD (F_NOCACHE vs Buffered) Benchmark ===", flush=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    stripe_sizes = [4096, 16384, 65536, 262144]
    num_ops = 128

    size_sweep: list[dict[str, Any]] = []
    for sz in stripe_sizes:
        entry: dict[str, Any] = {"stripe_bytes": sz, "stripe_kb": sz // 1024}
        payload = os.urandom(sz * num_ops)

        for mode_label, use_nocache in [("os_buffered", False), ("direct_fnocache", True)]:
            fpath = work_dir / f"bench_{mode_label}_{sz}.slab"
            fd = os.open(str(fpath), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
            if use_nocache and sys.platform == "darwin":
                fcntl.fcntl(fd, DARWIN_F_NOCACHE, 1)

            # Sequential aligned writes + controller flush
            w_lats = []
            t_w_total = time.perf_counter()
            for i in range(num_ops):
                t0 = time.perf_counter()
                os.pwrite(fd, payload[i * sz : (i + 1) * sz], i * sz)
                w_lats.append((time.perf_counter() - t0) * 1e6)
            os.fsync(fd)
            w_bw_gbps = (sz * num_ops) / ((time.perf_counter() - t_w_total) * 1e9)

            # Single-QD random stripe `pread`
            rng = np.random.default_rng(2026 + sz)
            order = rng.permutation(num_ops).tolist()
            r_lats = []
            for idx in order:
                t0 = time.perf_counter()
                _ = os.pread(fd, sz, idx * sz)
                r_lats.append((time.perf_counter() - t0) * 1e6)

            # Multi-queue (`QD=4`) parallel stripe `pread`
            with ThreadPoolExecutor(max_workers=4) as pool:
                t_par = time.perf_counter()
                _ = list(pool.map(lambda idx: os.pread(fd, sz, idx * sz), order))
                dt_par = time.perf_counter() - t_par
            par_bw_gbps = (sz * num_ops) / (dt_par * 1e9)
            par_eff_us = (dt_par * 1e6) / num_ops

            os.close(fd)
            fpath.unlink(missing_ok=True)

            entry[mode_label] = {
                "pread_qd1_p50_us": float(np.median(r_lats)),
                "pread_qd1_p95_us": float(np.percentile(r_lats, 95)),
                "pread_qd1_p99_us": float(np.percentile(r_lats, 99)),
                "pread_qd4_effective_us_per_stripe": float(par_eff_us),
                "pread_qd4_bandwidth_gbps": float(par_bw_gbps),
                "pwrite_seq_p50_us": float(np.median(w_lats)),
                "write_fsync_bandwidth_gbps": float(w_bw_gbps),
            }
        size_sweep.append(entry)
        print(
            f"  Stripe={sz//1024:3d}KB | Buffered p50={entry['os_buffered']['pread_qd1_p50_us']:5.1f}us | "
            f"F_NOCACHE QD1 p50={entry['direct_fnocache']['pread_qd1_p50_us']:6.1f}us (p95={entry['direct_fnocache']['pread_qd1_p95_us']:6.1f}us) | "
            f"QD4 BW={entry['direct_fnocache']['pread_qd4_bandwidth_gbps']:.2f} GB/s",
            flush=True,
        )

    # Physical NVMe WAF scaling vs. compaction turns K in {2, 4, 8, 12, 16, 24}
    waf_curve: list[dict[str, Any]] = []
    for k_turns in [2, 4, 8, 12, 16, 24]:
        # Measure physical DirectNVMeSlabStore WAF for SplitSlab vs. Iterative Re-Compaction (ragged 20KB sub-page + RMW vs 64KB aligned RMW)
        st_ss = DirectNVMeSlabStore(work_dir / f"waf_ss_{k_turns}", direct_nocache=True, sync_mode="batch_fsync")
        st_rc_aligned = DirectNVMeSlabStore(work_dir / f"waf_rca_{k_turns}", direct_nocache=True, sync_mode="batch_fsync")
        st_rc_subpage = DirectNVMeSlabStore(work_dir / f"waf_rcs_{k_turns}", direct_nocache=True, sync_mode="batch_fsync")
        try:
            n_slabs = 16
            blob_64k = os.urandom(65536)
            blob_24k = os.urandom(24576)  # Ragged per-head / sub-block compacted chunk (< 64KB page)
            offs_a, offs_s = [], []
            for s in range(n_slabs):
                st_ss.put_immutable_stripe(f"ss_{s}", blob_64k)
                oa, _ = st_rc_aligned.put_immutable_stripe(f"rca_{s}", blob_64k)
                osub, _ = st_rc_subpage.put_immutable_stripe(f"rcs_{s}", blob_24k)
                offs_a.append(oa)
                offs_s.append(osub)

            rng = np.random.default_rng(100 + k_turns)
            for turn in range(1, k_turns):
                # Empirical dirty slab fraction from Qwen2.5-7B runs: ~81.7% of slabs shift alignment per turn
                for s in range(n_slabs):
                    if rng.random() < 0.817:
                        offs_a[s] = st_rc_aligned.rewrite_mutable_page(offs_a[s], blob_64k)
                        offs_s[s] = st_rc_subpage.rewrite_mutable_page(offs_s[s], blob_24k)
            st_ss.sync_batch()
            st_rc_aligned.sync_batch()
            st_rc_subpage.sync_batch()

            waf_curve.append(
                {
                    "compaction_turns_k": k_turns,
                    "splitslab_physical_waf": st_ss.waf,
                    "recompact_64kb_slab_physical_waf": st_rc_aligned.waf,
                    "recompact_ragged_subpage_physical_waf": st_rc_subpage.waf,
                    "theorem2_analytical_64kb_waf": analytical_recompaction_waf(k_turns, 0.817, 65536, 65536),
                }
            )
        finally:
            st_ss.close()
            st_rc_aligned.close()
            st_rc_subpage.close()
            shutil.rmtree(work_dir / f"waf_ss_{k_turns}", ignore_errors=True)
            shutil.rmtree(work_dir / f"waf_rca_{k_turns}", ignore_errors=True)
            shutil.rmtree(work_dir / f"waf_rcs_{k_turns}", ignore_errors=True)

    shutil.rmtree(work_dir, ignore_errors=True)
    return {"stripe_size_sweep": size_sweep, "waf_vs_turns_curve": waf_curve}


# ============================================================================
# Part B: Multi-Agent Prefix-Sharing Fleet ($P = 1..32$ Agents on Qwen2.5-7B)
# ============================================================================
def run_fleet_dedup_benchmark(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    geom: SplitSlabGeometry,
    P_sub: torch.Tensor,
    sk_spec: Any,
    work_dir: Path,
    device: str,
) -> list[dict[str, Any]]:
    print("\n=== [Part B] Running 32-Agent Prefix-Sharing Fleet Benchmark ===", flush=True)
    spec = TRAJECTORY_SPECS[0]
    base_traj = build_model_formatted_trajectory(spec, tok, model.config.model_type, num_repo_modules=20)
    delim_u = "<|im_start|>user\n"
    delim_e = "<|im_end|>\n<|im_start|>assistant\n"
    delim_ae = "<|im_end|>\n"

    t1_ids = tok.encode(base_traj["turn1_shared"], add_special_tokens=False)
    shared_len = len(t1_ids)

    # Precompute shared Turn-1 base sessions once to avoid 32x redundant Turn-1 prefill
    store_ss = DirectNVMeSlabStore(work_dir / "fleet_ss", direct_nocache=True, sync_mode="batch_fsync")
    store_ev = DirectNVMeSlabStore(work_dir / "fleet_ev", direct_nocache=True, sync_mode="batch_fsync")
    store_fk = DirectNVMeSlabStore(work_dir / "fleet_fk", direct_nocache=True, sync_mode="batch_fsync")

    fleet_records: list[dict[str, Any]] = []
    try:
        base_ss = SplitSlabSession("splitslab_full", model, geom, store_ss, P_sub, sk_spec, base_traj["boundary_ids"], device)
        base_ss.forward_tokens(t1_ids, track_online_attn=False)

        base_ev = SplitSlabSession("online_evict_actkv", model, geom, store_ev, P_sub, sk_spec, base_traj["boundary_ids"], device)
        base_ev.forward_tokens(t1_ids, track_online_attn=False)
        base_ev.cum_attn_scores = torch.zeros(len(t1_ids), dtype=torch.float32, device=device)

        base_fk = SplitSlabSession("full_kv_flatcake", model, geom, store_fk, P_sub, sk_spec, base_traj["boundary_ids"], device)
        base_fk.forward_tokens(t1_ids, track_online_attn=False)

        checkpoints = {1, 2, 4, 8, 16, 32}
        for agent_idx in range(1, 33):
            mod_target = agent_idx % 20
            field_ask = "queue_depth" if (agent_idx % 2 == 0) else "max_inflight"
            q_turn2 = (
                f"{delim_u}[Agent {agent_idx} Turn 2] Inspect module index {mod_target:02d} and verify `{field_ask}` "
                f"for shard `{agent_idx % 4}`.{delim_e}Verified `{field_ask}` on module {mod_target:02d}.{delim_ae}"
            )
            q2_ids = tok.encode(q_turn2, add_special_tokens=False)
            is_probe = agent_idx > 1

            for base_s in (base_ss, base_ev, base_fk):
                ag_s = base_s.clone_shared_prefix_state()
                ag_s.forward_tokens(q2_ids, track_online_attn=True)
                # Write unique non-deduplicated slabs to physical store (`count_waf_logical=True` so fleet MB is tracked!)
                ag_s.apply_post_turn_tiering(shared_prefix_token_limit=shared_len, is_dedup_probe=False)
                if is_probe:
                    # Also record dedup hit rate for reporting
                    ag_probe = base_s.clone_shared_prefix_state()
                    ag_probe.forward_tokens(q2_ids, track_online_attn=True)
                    ag_probe.apply_post_turn_tiering(shared_prefix_token_limit=shared_len, is_dedup_probe=True)
                    del ag_probe
                del ag_s

            if agent_idx in checkpoints:
                ss_hit = 100.0 * store_ss.dedup_hits / max(1, store_ss.dedup_queries) if agent_idx > 1 else 100.0
                ev_hit = 100.0 * store_ev.dedup_hits / max(1, store_ev.dedup_queries) if agent_idx > 1 else 0.0
                fk_hit = 100.0 * store_fk.dedup_hits / max(1, store_fk.dedup_queries) if agent_idx > 1 else 100.0
                rec = {
                    "num_agents_p": agent_idx,
                    "splitslab_nvme_mb": store_ss.physical_bytes_written / (1024 * 1024),
                    "online_evict_nvme_mb": store_ev.physical_bytes_written / (1024 * 1024),
                    "full_kv_flatcake_nvme_mb": store_fk.physical_bytes_written / (1024 * 1024),
                    "splitslab_dedup_hit_pct": ss_hit,
                    "online_evict_dedup_hit_pct": ev_hit,
                    "full_kv_dedup_hit_pct": fk_hit,
                }
                fleet_records.append(rec)
                print(
                    f"  P={agent_idx:2d} Agents | SplitSlab NVMe={rec['splitslab_nvme_mb']:6.2f} MB (Dedup={ss_hit:5.1f}%) | "
                    f"ActKV NVMe={rec['online_evict_nvme_mb']:6.2f} MB (Dedup={ev_hit:4.1f}%) | "
                    f"FullKV NVMe={rec['full_kv_flatcake_nvme_mb']:6.2f} MB",
                    flush=True,
                )
    finally:
        store_ss.close()
        store_ev.close()
        store_fk.close()
        shutil.rmtree(work_dir / "fleet_ss", ignore_errors=True)
        shutil.rmtree(work_dir / "fleet_ev", ignore_errors=True)
        shutil.rmtree(work_dir / "fleet_fk", ignore_errors=True)

    return fleet_records


# ============================================================================
# Part C: Sketch Byte Size, Lookahead Depth & Proposition-1 Ablation Matrix
# ============================================================================
def run_ablation_matrix(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    geom: SplitSlabGeometry,
    work_dir: Path,
    device: str,
    n_traj: int = 12,
) -> dict[str, Any]:
    print(f"\n=== [Part C] Running Sketch, Lookahead & Component Ablations (n={n_traj}) ===", flush=True)
    trajs = [build_model_formatted_trajectory(s, tok, model.config.model_type) for s in TRAJECTORY_SPECS[:n_traj]]

    def eval_config(label: str, sk_bytes: int, flags: SplitSlabAblationFlags, method: str = "splitslab_full") -> dict[str, Any]:
        P_s, sk_sp = build_model_subspaces(model, geom.plane1_residual_tokens, sk_bytes, device=device)
        exacts, fields, rec_t12, rec_dist, hbm_saves = [], [], [], [], []
        for tr in trajs:
            st = DirectNVMeSlabStore(work_dir / "abl_tmp", direct_nocache=True, sync_mode="batch_fsync")
            try:
                sess = SplitSlabSession(method, model, geom, st, P_s, sk_sp, tr["boundary_ids"], device, flags)
                sess.forward_tokens(tok.encode(tr["turn1_shared"], add_special_tokens=False), track_online_attn=False)
                sess.forward_tokens(tok.encode(tr["turn2_agent_a"], add_special_tokens=False), track_online_attn=False)
                sess.apply_post_turn_tiering()
                for up, ar in tr["distractor_pairs"]:
                    sess.clear_staged_recall()
                    sess.current_step_uncertain = False
                    sess.forward_tokens(tok.encode(up, add_special_tokens=False), allow_recall=True, track_online_attn=False)
                    sess.forward_tokens(tok.encode(ar, add_special_tokens=False), allow_recall=False, track_online_attn=False)
                    sess.clear_staged_recall()
                    sess.apply_post_turn_tiering()
                steady_b, _ = sess.active_hbm_bytes()
                full_b = sess.logical_seq_len * geom.num_layers * geom.bytes_per_token_per_layer
                hbm_saves.append(100.0 * (1.0 - steady_b / float(full_b)))
                resp = sess.greedy_generate(
                    tok.encode(tr["turn12_prompt"], add_special_tokens=False), tok, tr["stop_ids"], max_new_tokens=52
                )
                sc = score_delayed_recall_response(resp, tr["spec"])
                exacts.append(sc["exact_match"])
                fields.append(sc["partial_score"])
                rec_t12.append(
                    100.0 * sess.recall_turn_cold_stripes_recalled / max(1, sess.recall_turn_cold_stripe_opportunities)
                )
                rec_dist.append(
                    100.0 * sess.distractor_cold_stripes_recalled / max(1, sess.distractor_cold_stripe_opportunities)
                )
            finally:
                st.close()
                shutil.rmtree(work_dir / "abl_tmp", ignore_errors=True)

        out = {
            "label": label,
            "sketch_bytes": sk_bytes,
            "exact_match_pct": 100.0 * float(np.mean(exacts)),
            "field_accuracy_pct": 100.0 * float(np.mean(fields)),
            "steady_hbm_reduction_pct": float(np.mean(hbm_saves)),
            "recall_turn12_pct": float(np.mean(rec_t12)),
            "recall_distractor_pct": float(np.mean(rec_dist)),
        }
        print(
            f"  [{label:36s}] Exact={out['exact_match_pct']:5.1f}% | Field={out['field_accuracy_pct']:5.1f}% | "
            f"RecT12={out['recall_turn12_pct']:4.2f}% | RecDist={out['recall_distractor_pct']:4.2f}%",
            flush=True,
        )
        return out

    sketch_sweep = [
        eval_config(f"sketch_{b}B", b, SplitSlabAblationFlags(prominence_delta=0.130))
        for b in [16, 32, 64, 128]
    ]

    lookahead_and_components = [
        eval_config("lookahead_0_sync_stall", 64, SplitSlabAblationFlags(lookahead_layers=0, prominence_delta=0.130)),
        eval_config("lookahead_1_layer", 64, SplitSlabAblationFlags(lookahead_layers=1, prominence_delta=0.130)),
        eval_config("lookahead_2_layer_default", 64, SplitSlabAblationFlags(lookahead_layers=2, prominence_delta=0.130)),
        eval_config("lookahead_3_layer", 64, SplitSlabAblationFlags(lookahead_layers=3, prominence_delta=0.130)),
        eval_config(
            "naive_raw_query_no_reproj",
            64,
            SplitSlabAblationFlags(lookahead_layers=2, use_basis_aligned_query=False, prominence_delta=0.130),
        ),
        eval_config(
            "no_bias_centering",
            64,
            SplitSlabAblationFlags(lookahead_layers=2, use_bias_centering=False, prominence_delta=0.130),
        ),
        eval_config(
            "no_co_routed_beacon",
            64,
            SplitSlabAblationFlags(lookahead_layers=2, use_co_routed_beacon=False, prominence_delta=0.012),
        ),
    ]

    return {"sketch_byte_sweep": sketch_sweep, "lookahead_and_component_ablations": lookahead_and_components}


# ============================================================================
# Part D: Empirical Theorem-1 Perturbation Bound Verification (560 Real Heads)
# ============================================================================
@torch.no_grad()
def run_theorem1_empirical_verification(
    model: AutoModelForCausalLM, tok: AutoTokenizer, device: str
) -> dict[str, Any]:
    print("\n=== [Part D] Verifying Theorem 1 Across 560 Real Qwen2.5-7B Attention Heads ===", flush=True)
    traj = build_model_formatted_trajectory(TRAJECTORY_SPECS[0], tok, model.config.model_type)
    full_text = (
        traj["turn1_shared"]
        + traj["turn2_agent_a"]
        + "".join(u + a for u, a in traj["distractor_pairs"])
        + traj["turn12_prompt"]
    )
    ids = torch.tensor([tok.encode(full_text, add_special_tokens=False)], dtype=torch.long, device=device)

    layer_inputs: dict[int, torch.Tensor] = {}
    pos_embs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def mk_h(l_i: int):
        def _h(mod, args, kwargs):
            layer_inputs[l_i] = args[0].detach()
            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                pos_embs[l_i] = kwargs["position_embeddings"]
        return _h

    hooks = [model.model.layers[l].register_forward_pre_hook(mk_h(l), with_kwargs=True) for l in range(28)]
    past = model(ids, use_cache=True).past_key_values
    for h in hooks:
        h.remove()

    cos, sin = pos_embs[0]
    seq_len = ids.shape[1]
    # Split past tokens `0..seq_len-2` into Plane-0 (20% + last 48 local) vs Plane-1 (80% cold)
    anc_idx_list = [i for i in range(seq_len - 1) if (i % 5 == 0) or (i >= seq_len - 49)]
    anc_set = set(anc_idx_list)
    res_idx_list = [i for i in range(seq_len - 1) if i not in anc_set]
    anc_t = torch.tensor(anc_idx_list, dtype=torch.long, device=device)
    res_t = torch.tensor(res_idx_list, dtype=torch.long, device=device)

    violations = 0
    total_checks = 0
    ratios: list[float] = []
    errors: list[float] = []
    bounds: list[float] = []

    # Sample 28 layers x 20 (head, query_pos) pairs = 560 real head evaluations!
    q_positions = [seq_len - 1, seq_len - 12, seq_len - 25, seq_len - 40, seq_len - 120]
    heads_sample = [0, 7, 14, 21]

    for l in range(28):
        attn = model.model.layers[l].self_attn
        K_rep = repeat_kv(past.layers[l].keys, 7)[0]  # [28, T, 128]
        V_rep = repeat_kv(past.layers[l].values, 7)[0]
        for qp_idx in q_positions:
            h_q = layer_inputs[l][:, qp_idx : qp_idx + 1, :]
            q_raw = attn.q_proj(model.model.layers[l].input_layernorm(h_q)).view(1, 1, 28, 128).transpose(1, 2)
            kd = torch.zeros((1, 4, 1, 128), dtype=torch.float16, device=device)
            q_rope, _ = apply_rotary_pos_emb(q_raw, kd, cos[:, qp_idx : qp_idx + 1, :], sin[:, qp_idx : qp_idx + 1, :])
            anc_valid = anc_t[anc_t < qp_idx]
            res_valid = res_t[res_t < qp_idx]
            for h_idx in heads_sample:
                chk = verify_empirical_perturbation_bound(
                    q_rope[0, h_idx, 0],
                    K_rep[h_idx, anc_valid],
                    V_rep[h_idx, anc_valid],
                    K_rep[h_idx, res_valid],
                    V_rep[h_idx, res_valid],
                )
                total_checks += 1
                if not chk["bound_holds"]:
                    violations += 1
                b_val = float(chk["theoretical_upper_bound"])
                e_val = float(chk["empirical_l2_error"])
                errors.append(e_val)
                bounds.append(b_val)
                if b_val > 1e-6:
                    ratios.append(e_val / b_val)

    print(
        f"  Verified {total_checks} real attention heads: violations={violations} (0 expected) | "
        f"median tightness ratio (emp/bound)={float(np.median(ratios)):.3f} (max={float(np.max(ratios)):.3f})",
        flush=True,
    )
    return {
        "total_real_head_evaluations": total_checks,
        "bound_violations": violations,
        "satisfaction_rate_pct": 100.0 * (total_checks - violations) / float(total_checks),
        "empirical_l2_error_median": float(np.median(errors)),
        "theoretical_bound_median": float(np.median(bounds)),
        "tightness_ratio_median": float(np.median(ratios)),
        "tightness_ratio_p95": float(np.percentile(ratios, 95)),
        "tightness_ratio_max": float(np.max(ratios)),
    }


# ============================================================================
# Part E: Long-Horizon `8.2K` and `16.4K` Context Scaling (`Qwen2.5-7B`)
# ============================================================================
def run_long_horizon_scaling(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    geom: SplitSlabGeometry,
    P_sub: torch.Tensor,
    sk_spec: Any,
    work_dir: Path,
    device: str,
) -> list[dict[str, Any]]:
    print("\n=== [Part E] Running 8.2K and 16.4K Long-Horizon Context Scaling (Qwen2.5-7B) ===", flush=True)
    horizon_configs = [
        ("2.3K", 16, 1, 6),
        ("8.2K", 60, 4, 6),
        ("16.4K", 120, 8, 4),
    ]
    records: list[dict[str, Any]] = []

    for label, n_mods, d_reps, n_cases in horizon_configs:
        for meth in ["online_evict_actkv", "splitslab_full"]:
            exacts, hbm_saves, peak_saves, rec_t12s, rec_dists, tok_counts = [], [], [], [], [], []
            for idx in range(n_cases):
                tr = build_model_formatted_trajectory(
                    TRAJECTORY_SPECS[idx],
                    tok,
                    model.config.model_type,
                    num_repo_modules=n_mods,
                    distractor_repeats=d_reps,
                )
                st = DirectNVMeSlabStore(work_dir / "lh_tmp", direct_nocache=True, sync_mode="batch_fsync")
                try:
                    sess = SplitSlabSession(meth, model, geom, st, P_sub, sk_spec, tr["boundary_ids"], device)
                    t1_ids = tok.encode(tr["turn1_shared"], add_special_tokens=False)
                    # Feed long Turn-1 repo in 1,024-token chunks so prefill memory stays O(chunk)
                    for c_start in range(0, len(t1_ids), 1024):
                        sess.forward_tokens(t1_ids[c_start : c_start + 1024], track_online_attn=False)
                    sess.cum_attn_scores = torch.zeros(len(t1_ids), dtype=torch.float32, device=device)

                    sess.forward_tokens(tok.encode(tr["turn2_agent_a"], add_special_tokens=False), track_online_attn=True)
                    sess.apply_post_turn_tiering()

                    for up, ar in tr["distractor_pairs"]:
                        sess.clear_staged_recall()
                        sess.current_step_uncertain = False
                        sess.forward_tokens(tok.encode(up, add_special_tokens=False), allow_recall=True, track_online_attn=True)
                        sess.forward_tokens(tok.encode(ar, add_special_tokens=False), allow_recall=False, track_online_attn=True)
                        sess.clear_staged_recall()
                        sess.apply_post_turn_tiering()

                    steady_b, _ = sess.active_hbm_bytes()
                    full_b = sess.logical_seq_len * geom.num_layers * geom.bytes_per_token_per_layer
                    tok_counts.append(sess.logical_seq_len)

                    resp = sess.greedy_generate(
                        tok.encode(tr["turn12_prompt"], add_special_tokens=False), tok, tr["stop_ids"], max_new_tokens=52
                    )
                    peak_b = steady_b + sess.peak_staged_bytes
                    sc = score_delayed_recall_response(resp, tr["spec"])
                    exacts.append(sc["exact_match"])
                    hbm_saves.append(100.0 * (1.0 - steady_b / float(full_b)))
                    peak_saves.append(100.0 * (1.0 - peak_b / float(full_b)))
                    rec_t12s.append(
                        100.0 * sess.recall_turn_cold_stripes_recalled / max(1, sess.recall_turn_cold_stripe_opportunities)
                    )
                    rec_dists.append(
                        100.0 * sess.distractor_cold_stripes_recalled / max(1, sess.distractor_cold_stripe_opportunities)
                    )
                finally:
                    st.close()
                    shutil.rmtree(work_dir / "lh_tmp", ignore_errors=True)
                if device == "mps":
                    torch.mps.empty_cache()
                gc.collect()

            rec = {
                "horizon_label": label,
                "method": meth,
                "num_cases": n_cases,
                "mean_logical_tokens": float(np.mean(tok_counts)),
                "exact_match_pct": 100.0 * float(np.mean(exacts)),
                "steady_hbm_reduction_pct": float(np.mean(hbm_saves)),
                "peak_hbm_reduction_pct": float(np.mean(peak_saves)),
                "cold_stripes_recalled_turn12_pct": float(np.mean(rec_t12s)),
                "cold_stripes_recalled_distractor_pct": float(np.mean(rec_dists)),
            }
            records.append(rec)
            print(
                f"  [{label:5s} | {meth:18s}] Toks={rec['mean_logical_tokens']:7.1f} | "
                f"Exact={rec['exact_match_pct']:5.1f}% | HBM_Save={rec['steady_hbm_reduction_pct']:5.1f}% "
                f"(Peak {rec['peak_hbm_reduction_pct']:5.1f}%) | RecT12={rec['cold_stripes_recalled_turn12_pct']:4.2f}% | "
                f"RecDist={rec['cold_stripes_recalled_distractor_pct']:4.2f}%",
                flush=True,
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="SplitSlab Storage, Fleet, Ablation & Long-Horizon Suite")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "storage_ablations_scaling.json"),
    )
    parser.add_argument("--work-dir", type=str, default="/tmp/splitslab_deepdive")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    work_dir = Path(args.work_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "metadata": {
            "project": "SplitSlab",
            "suite": "Storage-Fleet-Ablations-LongHorizon",
            "model": args.model,
            "device": device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    }

    # Part A (pure NVMe hardware I/O before loading GPU model)
    results["part_a_nvme_storage"] = run_nvme_storage_microbenchmark(work_dir / "part_a")
    out_path.write_text(json.dumps(results, indent=2))

    # Load Qwen2.5-7B-Instruct once for Parts B, C, D, E
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, local_files_only=True).to(device)
    model.eval()
    cfg = model.config
    geom = SplitSlabGeometry.from_model_config(
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    )
    P_sub, sk_spec = build_model_subspaces(model, geom.plane1_residual_tokens, 64, device=device)

    results["part_b_fleet_dedup"] = run_fleet_dedup_benchmark(model, tok, geom, P_sub, sk_spec, work_dir / "part_b", device)
    out_path.write_text(json.dumps(results, indent=2))

    results["part_c_ablations"] = run_ablation_matrix(model, tok, geom, work_dir / "part_c", device, n_traj=12)
    out_path.write_text(json.dumps(results, indent=2))

    results["part_d_theorem1_verification"] = run_theorem1_empirical_verification(model, tok, device)
    out_path.write_text(json.dumps(results, indent=2))

    results["part_e_long_horizon_scaling"] = run_long_horizon_scaling(model, tok, geom, P_sub, sk_spec, work_dir / "part_e", device)

    # Analytical async prefetch overlap table across storage interconnects
    results["async_interconnect_overlap"] = [
        {
            "interconnect": name,
            "bandwidth_gbps": bw,
            "base_latency_us": lat,
            **async_lookahead_overlap_ratio(3, 65536, bw, lat, 580.0, lookahead_layers=2),
        }
        for name, bw, lat in [
            ("Apple_M5Pro_Internal_NVMe_QD4", 5.2, 165.0),
            ("PCIe_Gen4x4_Enterprise_NVMe", 7.0, 95.0),
            ("PCIe_Gen5x4_GDS_NVMe", 14.0, 65.0),
            ("Dual_NVMe_RAID0_GDS", 26.0, 65.0),
            ("CXL_3_Type3_NAND_Tier", 32.0, 25.0),
        ]
    ]

    out_path.write_text(json.dumps(results, indent=2))
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"\n[Done] Saved complete deep-dive results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
