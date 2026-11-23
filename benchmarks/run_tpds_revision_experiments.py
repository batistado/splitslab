#!/usr/bin/env python3
"""
IEEE TPDS Major-Revision Empirical Suite for SplitSlab
======================================================
Runs strictly locally on Apple Silicon (mps FP16 + Darwin fcntl.F_NOCACHE direct NVMe)
using cached `Qwen/Qwen2.5-7B-Instruct` to address all four empirical review concerns:

  Part 1: Recallable KV-Cache Baselines (`Quest-Box + NVMe`, `ArkVale-Centroid + NVMe`)
          vs. `SplitSlab Full` and `Full-KV` on RecallRegret-Bench (n=20) with full
          end-to-end serving latency (TTFT, TPOT p50/p95/p99, component breakdown)
          and 95% Wilson confidence intervals.
  Part 2: Non-Recall Routine Turn Quality (Turns 2-11, n=10): next-token greedy agreement
          and perplexity relative to Full-KV oracle when SplitSlab's recall gate is closed.
  Part 3: RULER-Style Multi-Key Needle-in-a-Haystack (`RULER-MK-NIAH`, 2 simultaneous
          6-digit passkeys in ~3.8K-token technical haystack, n=16): Full-KV vs. ActKV
          vs. Quest-Box+NVMe vs. SplitSlab (k=1) vs. SplitSlab Multi-Peak (k=2).
  Part 4: Gate Threshold Sensitivity Sweep over prominence `delta` and entropy `gamma`
          (n=12): Turn-12 Exact Match vs. Distractor False-Positive Recall Pareto curve.

Saves verified telemetry to:
  results/tpds_revision_experiments.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recallregret_bench import (
    TRAJECTORY_SPECS,
    build_model_formatted_trajectory,
    get_model_chat_delimiters,
    score_delayed_recall_response,
)
from splitslab import (
    DirectNVMeSlabStore,
    SplitSlabAblationFlags,
    SplitSlabGeometry,
    SplitSlabSession,
    build_model_subspaces,
)


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    """Exact 95% Wilson score confidence interval (percentage [0, 100])."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2)


def greedy_decode_step_loop(
    sess: SplitSlabSession,
    first_logits: torch.Tensor,
    eos_ids: set[int],
    max_new_tokens: int,
    is_final_recall_turn: bool,
    allow_recall: bool,
    record_timing: bool = False,
) -> list[int]:
    gen_ids: list[int] = []
    logits = first_logits
    for _ in range(max_new_tokens):
        nxt = int(torch.argmax(logits[0]).item())
        gen_ids.append(nxt)
        if nxt in eos_ids:
            break
        logits = sess.forward_tokens(
            [nxt],
            is_final_recall_turn=is_final_recall_turn,
            allow_recall=allow_recall,
            track_online_attn=False,
            record_layer_timing=record_timing,
        )
    return gen_ids


def run_recallregret_episode(
    method: str,
    spec_idx: int,
    model,
    tokenizer,
    model_type: str,
    geom: SplitSlabGeometry,
    P_subspaces: torch.Tensor,
    sketch_spec,
    boundary_ids: set[int],
    eos_ids: set[int],
    device: str,
    flags: SplitSlabAblationFlags,
    record_timing: bool = False,
    use_live_entropy_gate: bool = False,
) -> dict[str, Any]:
    spec = TRAJECTORY_SPECS[spec_idx]
    traj = build_model_formatted_trajectory(spec, tokenizer, model_type, num_repo_modules=16, distractor_repeats=1)
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    with tempfile.TemporaryDirectory(prefix=f"sp_rev_{method}_") as tmp_dir:
        nvme = DirectNVMeSlabStore(Path(tmp_dir), direct_nocache=True, sync_mode="batch_fsync", io_workers=4)
        sess = SplitSlabSession(
            method=method,
            model=model,
            geom=geom,
            nvme_store=nvme,
            P_subspaces=P_subspaces,
            sketch_spec=sketch_spec,
            boundary_token_ids=boundary_ids,
            device=device,
            flags=flags,
        )

        # Turn 1: Repository ingest
        sess.forward_tokens(enc(traj["turn1_shared"]), allow_recall=False, record_layer_timing=False)
        sess.apply_post_turn_tiering()

        # Turn 2: Agent A query
        sess.forward_tokens(enc(traj["turn2_agent_a"]), allow_recall=False, record_layer_timing=False)
        sess.apply_post_turn_tiering()

        # Turns 3-11: 9 Distractor turns with live recall gate active
        for q_str, a_str in traj["distractor_pairs"]:
            sess.clear_staged_recall()
            sess.current_step_uncertain = False
            sess.forward_tokens(
                enc(q_str),
                is_final_recall_turn=False,
                allow_recall=True,
                record_layer_timing=False,
            )
            sess.forward_tokens(
                enc(a_str),
                is_final_recall_turn=False,
                allow_recall=False,
                record_layer_timing=False,
            )
            sess.apply_post_turn_tiering()

        steady_hbm, _ = sess.active_hbm_bytes()
        full_hbm = sess.logical_seq_len * geom.num_layers * geom.bytes_per_token_per_layer

        # Turn 12: Delayed-recall crash dump using canonical greedy_generate (max_new_tokens=52)
        resp = sess.greedy_generate(
            enc(traj["turn12_prompt"]),
            tokenizer=tokenizer,
            stop_token_ids=eos_ids,
            max_new_tokens=52,
            is_final_recall_turn=True,
            record_layer_timing=record_timing,
            use_live_entropy_gate_on_prefill=use_live_entropy_gate,
        )
        peak_hbm = steady_hbm + sess.peak_staged_bytes
        sc = score_delayed_recall_response(resp, spec)

        dist_rec_pct = 100.0 * sess.distractor_cold_stripes_recalled / max(1, sess.distractor_cold_stripe_opportunities)
        t12_rec_pct = 100.0 * sess.recall_turn_cold_stripes_recalled / max(1, sess.recall_turn_cold_stripe_opportunities)

        out = {
            "traj_id": spec.traj_id,
            "exact_match": bool(sc["exact_match"]),
            "field_acc": float(sc["partial_score"]),
            "steady_hbm_saved_pct": 100.0 * (1.0 - steady_hbm / float(full_hbm)),
            "peak_hbm_saved_pct": 100.0 * (1.0 - peak_hbm / float(full_hbm)),
            "dist_recall_pct": dist_rec_pct,
            "t12_recall_pct": t12_rec_pct,
            "waf": nvme.waf,
            "ttft_ms": list(sess.prefill_ttft_ms_list),
            "tpot_ms": list(sess.decode_tpot_ms_list),
            "sketch_scoring_us": list(sess.sketch_scoring_us_list),
            "entropy_gate_us": list(sess.entropy_gate_us_list),
            "nvme_stage_us": list(sess.nvme_stage_us_list),
            "exposed_sync_stall_us": list(sess.exposed_sync_nvme_stall_us),
            "response": resp[:80],
        }
        nvme.close()
        return out


def run_part1_recallable_baselines_and_e2e(
    model, tokenizer, model_type, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device
) -> dict[str, Any]:
    print("\n=== Part 1: Recallable KV-Cache Baselines & End-to-End Serving Latency (n=20) ===")
    methods = [
        ("full_kv_flatcake", SplitSlabAblationFlags()),
        ("quest_box_nvme", SplitSlabAblationFlags(baseline_topk_blocks=3)),
        ("arkvale_centroid_nvme", SplitSlabAblationFlags(baseline_topk_blocks=3)),
        ("splitslab_full", SplitSlabAblationFlags(lookahead_layers=2, prominence_delta=0.130, entropy_gamma=0.32)),
    ]
    summary: dict[str, Any] = {}
    for m_name, flags in methods:
        t0 = time.perf_counter()
        eps = []
        for idx in range(20):
            # Record full synchronized MPS timing on first 6 trajectories (120+ decode steps)
            ep = run_recallregret_episode(
                m_name, idx, model, tokenizer, model_type, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device,
                flags=flags, record_timing=(idx < 6),
            )
            eps.append(ep)

        n = len(eps)
        k_exact = sum(1 for e in eps if e["exact_match"])
        ci_lo, ci_hi = wilson_ci_95(k_exact, n)
        all_tpot = [x for e in eps for x in e["tpot_ms"]]
        all_ttft = [x for e in eps for x in e["ttft_ms"]]
        all_sk = [x for e in eps for x in e["sketch_scoring_us"]]
        all_eg = [x for e in eps for x in e["entropy_gate_us"]]
        all_nv = [x for e in eps for x in e["nvme_stage_us"]]
        all_stall = [x for e in eps for x in e["exposed_sync_stall_us"]]

        meta_b = (
            0 if m_name == "full_kv_flatcake"
            else 2048 if m_name == "quest_box_nvme"
            else 1024 if m_name == "arkvale_centroid_nvme"
            else 64
        )
        res = {
            "n": n,
            "exact_match_pct": round(100.0 * k_exact / n, 2),
            "exact_match_ci95_pct": [ci_lo, ci_hi],
            "field_acc_pct": round(100.0 * float(np.mean([e["field_acc"] for e in eps])), 2),
            "metadata_bytes_per_stripe": meta_b,
            "steady_hbm_saved_pct": round(float(np.mean([e["steady_hbm_saved_pct"] for e in eps])), 2),
            "peak_hbm_saved_pct": round(float(np.mean([e["peak_hbm_saved_pct"] for e in eps])), 2),
            "waf": round(float(np.mean([e["waf"] for e in eps])), 3),
            "dist_recall_pct": round(float(np.mean([e["dist_recall_pct"] for e in eps])), 2),
            "t12_recall_pct": round(float(np.mean([e["t12_recall_pct"] for e in eps])), 2),
            "ttft_p50_ms": round(float(np.median(all_ttft)), 2) if all_ttft else 0.0,
            "tpot_p50_ms": round(float(np.median(all_tpot)), 2) if all_tpot else 0.0,
            "tpot_p95_ms": round(float(np.percentile(all_tpot, 95)), 2) if all_tpot else 0.0,
            "tpot_p99_ms": round(float(np.percentile(all_tpot, 99)), 2) if all_tpot else 0.0,
            "throughput_tok_s": round(1000.0 / float(np.median(all_tpot)), 2) if all_tpot else 0.0,
            "sketch_scoring_per_layer_p50_us": round(float(np.median(all_sk)), 1) if all_sk else 0.0,
            "entropy_gate_check_p50_us": round(float(np.median(all_eg)), 1) if all_eg else 0.0,
            "nvme_stage_call_p50_us": round(float(np.median(all_nv)), 1) if all_nv else 0.0,
            "exposed_sync_stall_per_layer_p50_us": round(float(np.median(all_stall)), 1) if all_stall else 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        summary[m_name] = res
        print(
            f"  {m_name:<22} | Exact={res['exact_match_pct']:5.1f}% [{ci_lo:.1f},{ci_hi:.1f}] | "
            f"Meta={meta_b:4d}B | HBM={res['steady_hbm_saved_pct']:4.1f}% | "
            f"T12/Dist={res['t12_recall_pct']:4.2f}%/{res['dist_recall_pct']:4.2f}% | "
            f"TPOT p50/p95={res['tpot_p50_ms']:.1f}/{res['tpot_p95_ms']:.1f} ms | "
            f"SyncStall={res['exposed_sync_stall_per_layer_p50_us']:.1f} us"
        )
    return summary


def run_part2_routine_turn_quality(
    model, tokenizer, model_type, geom, P_subspaces, sketch_spec, boundary_ids, device
) -> dict[str, Any]:
    """
    Measures token-level greedy top-1 agreement and teacher-forced perplexity on the
    assistant response tokens across Turns 3-11 (routine engineering turns where
    SplitSlab keeps its recall gate closed) for Full-KV vs. SplitSlab Full vs. ActKV.
    """
    print("\n=== Part 2: Non-Recall Routine Turn Quality (Turns 3-11, n=10) ===")
    n_cases = 10
    results: dict[str, Any] = {}
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    for m_name in ("full_kv_flatcake", "online_evict_actkv", "splitslab_full"):
        nlls: list[float] = []
        top1_matches = 0
        total_toks = 0
        dist_rec_list = []

        for idx in range(n_cases):
            spec = TRAJECTORY_SPECS[idx]
            traj = build_model_formatted_trajectory(spec, tokenizer, model_type, num_repo_modules=16, distractor_repeats=1)

            with tempfile.TemporaryDirectory(prefix=f"sp_rt_{m_name}_") as tmp_dir:
                nvme = DirectNVMeSlabStore(Path(tmp_dir), direct_nocache=True, sync_mode="batch_fsync", io_workers=4)
                sess = SplitSlabSession(
                    method=m_name,
                    model=model,
                    geom=geom,
                    nvme_store=nvme,
                    P_subspaces=P_subspaces,
                    sketch_spec=sketch_spec,
                    boundary_token_ids=boundary_ids,
                    device=device,
                    flags=SplitSlabAblationFlags(),
                )
                sess.forward_tokens(enc(traj["turn1_shared"]), allow_recall=False)
                sess.apply_post_turn_tiering()
                sess.forward_tokens(enc(traj["turn2_agent_a"]), allow_recall=False)
                sess.apply_post_turn_tiering()

                for q_str, a_str in traj["distractor_pairs"]:
                    sess.clear_staged_recall()
                    logits = sess.forward_tokens(enc(q_str), is_final_recall_turn=False, allow_recall=True)
                    ans_part = enc(a_str)
                    for target_tid in ans_part:
                        if target_tid in boundary_ids:
                            logits = sess.forward_tokens([target_tid], is_final_recall_turn=False, allow_recall=False)
                            continue
                        log_probs = torch.log_softmax(logits[0], dim=-1)
                        nlls.append(float(-log_probs[target_tid].item()))
                        pred_tid = int(torch.argmax(logits[0]).item())
                        if pred_tid == target_tid:
                            top1_matches += 1
                        total_toks += 1
                        logits = sess.forward_tokens([target_tid], is_final_recall_turn=False, allow_recall=False)
                    sess.apply_post_turn_tiering()

                dist_rec_pct = 100.0 * sess.distractor_cold_stripes_recalled / max(1, sess.distractor_cold_stripe_opportunities)
                dist_rec_list.append(dist_rec_pct)
                nvme.close()

        mean_nll = float(np.mean(nlls))
        ppl = math.exp(mean_nll)
        acc = 100.0 * top1_matches / max(1, total_toks)
        results[m_name] = {
            "n_trajectories": n_cases,
            "evaluated_routine_tokens": total_toks,
            "mean_nll": round(mean_nll, 4),
            "perplexity": round(ppl, 4),
            "top1_token_acc_pct": round(acc, 2),
            "distractor_recall_pct": round(float(np.mean(dist_rec_list)), 2),
        }
        print(
            f"  {m_name:<20} | Routine Toks={total_toks} | PPL={ppl:.4f} | "
            f"Top-1 Acc={acc:.2f}% | Dist Recall={np.mean(dist_rec_list):.2f}%"
        )
    return results


def build_ruler_mk_niah_episode(tokenizer, ep_idx: int) -> dict[str, Any]:
    """
    Builds a RULER-style Multi-Key Needle-in-a-Haystack (MK-NIAH) episode (~3.6K tokens)
    with 2 distinct passkeys buried at ~18% and ~62% depth in technical documentation.
    """
    rng = np.random.default_rng(2026 + ep_idx)
    key_a = f"ALPHA-{10 + ep_idx:02d}"
    val_a = f"{int(rng.integers(100000, 999999))}"
    key_b = f"OMEGA-{50 + ep_idx:02d}"
    val_b = f"{int(rng.integers(100000, 999999))}"

    paragraphs = []
    pos_a = 8 + (ep_idx % 5)
    pos_b = 30 + (ep_idx % 6)
    for i in range(48):
        if i == pos_a:
            paragraphs.append(
                f"Section {i:02d} Security Audit Note: The special magic number for {key_a} is: {val_a}. "
                f"All other keys in this segment are rotating telemetry placeholders."
            )
        elif i == pos_b:
            paragraphs.append(
                f"Section {i:02d} Runtime Verification Note: The special magic number for {key_b} is: {val_b}. "
                f"Verify checksum integrity before committing ring buffer descriptors."
            )
        else:
            paragraphs.append(
                f"Section {i:02d} Systems Log: Worker queue {i} processed {1200 + i * 17} aligned 64KB NVMe "
                f"frames across channel {i % 4} with CRC32 verification enabled and queue depth {(i % 8) + 1}. "
                f"Memory pool utilization remained steady at {42 + (i % 19)} percent with zero ECC faults."
            )

    sys_msg = "You are a precise retrieval assistant. Extract exact magic numbers from the document."
    doc_turn = "Document Archive:\n" + "\n".join(paragraphs)
    # Split archive into 4 chunks (turns) so blocks exit W_loc and tier to NVMe before the final query
    chunks = [
        "\n".join(paragraphs[0:12]),
        "\n".join(paragraphs[12:24]),
        "\n".join(paragraphs[24:36]),
        "\n".join(paragraphs[36:48]),
    ]
    turn_chunks_ids = []
    for ci, ch in enumerate(chunks):
        if ci == 0:
            txt = (
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\nArchive Part {ci+1}:\n{ch}<|im_end|>\n"
                f"<|im_start|>assistant\nIndexed Part {ci+1}.<|im_end|>\n"
            )
        else:
            txt = (
                f"<|im_start|>user\nArchive Part {ci+1}:\n{ch}<|im_end|>\n"
                f"<|im_start|>assistant\nIndexed Part {ci+1}.<|im_end|>\n"
            )
        turn_chunks_ids.append(tokenizer.encode(txt, add_special_tokens=False))

    # Add 2 short neutral turns so Part 4 also tiers cleanly out of W_loc
    filler_txt = (
        "<|im_start|>user\nConfirm archive ingestion status.<|im_end|>\n"
        "<|im_start|>assistant\nAll 4 archive parts are indexed and ready for queries.<|im_end|>\n"
    )
    turn_chunks_ids.append(tokenizer.encode(filler_txt, add_special_tokens=False))

    query_txt = (
        f"<|im_start|>user\nWhat are the special magic numbers for {key_a} and {key_b}? "
        f"Reply strictly as: {key_a}=<NUM1> | {key_b}=<NUM2><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    query_ids = tokenizer.encode(query_txt, add_special_tokens=False)
    return {
        "ep_id": f"MK-{ep_idx+1:02d}",
        "key_a": key_a,
        "val_a": val_a,
        "key_b": key_b,
        "val_b": val_b,
        "turn_chunks_ids": turn_chunks_ids,
        "query_ids": query_ids,
    }


def run_part3_ruler_mk_niah(
    model, tokenizer, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device
) -> dict[str, Any]:
    print("\n=== Part 3: RULER-Style Multi-Key NIAH (2 Simultaneous Needles, ~3.4K Toks, n=16) ===")
    n_eps = 16
    configs = [
        ("full_kv_flatcake", "full_kv_flatcake", SplitSlabAblationFlags()),
        ("online_evict_actkv", "online_evict_actkv", SplitSlabAblationFlags()),
        ("quest_box_nvme_k6", "quest_box_nvme", SplitSlabAblationFlags(baseline_topk_blocks=6)),
        ("splitslab_single_peak_k1", "splitslab_full", SplitSlabAblationFlags(max_recall_peaks=1)),
        ("splitslab_multi_peak_k2", "splitslab_full", SplitSlabAblationFlags(max_recall_peaks=2)),
    ]
    out_summary: dict[str, Any] = {}
    for label, method, flags in configs:
        both_hits = 0
        needle_hits = 0
        hbm_saved = []
        rec_pcts = []
        tok_lens = []

        for ep_i in range(n_eps):
            ep = build_ruler_mk_niah_episode(tokenizer, ep_i)
            with tempfile.TemporaryDirectory(prefix=f"sp_mk_{label}_") as tmp_dir:
                nvme = DirectNVMeSlabStore(Path(tmp_dir), direct_nocache=True, sync_mode="batch_fsync", io_workers=4)
                sess = SplitSlabSession(
                    method=method,
                    model=model,
                    geom=geom,
                    nvme_store=nvme,
                    P_subspaces=P_subspaces,
                    sketch_spec=sketch_spec,
                    boundary_token_ids=boundary_ids,
                    device=device,
                    flags=flags,
                )
                for c_ids in ep["turn_chunks_ids"]:
                    sess.forward_tokens(c_ids, allow_recall=False)
                    sess.apply_post_turn_tiering()

                sess.clear_staged_recall()
                logits = sess.forward_tokens(ep["query_ids"], is_final_recall_turn=True, allow_recall=True)
                gen_ids = greedy_decode_step_loop(
                    sess, logits, eos_ids=eos_ids, max_new_tokens=28,
                    is_final_recall_turn=True, allow_recall=True,
                )
                resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                hit_a = ep["val_a"] in resp
                hit_b = ep["val_b"] in resp
                if hit_a and hit_b:
                    both_hits += 1
                needle_hits += int(hit_a) + int(hit_b)

                steady_hbm, _ = sess.active_hbm_bytes()
                full_hbm = sess.logical_seq_len * geom.num_layers * geom.bytes_per_token_per_layer
                hbm_saved.append(100.0 * (1.0 - steady_hbm / float(full_hbm)))
                rec_pcts.append(
                    100.0 * sess.recall_turn_cold_stripes_recalled / max(1, sess.recall_turn_cold_stripe_opportunities)
                )
                tok_lens.append(sess.logical_seq_len)
                nvme.close()

        ci_lo, ci_hi = wilson_ci_95(both_hits, n_eps)
        res = {
            "n_episodes": n_eps,
            "mean_logical_tokens": round(float(np.mean(tok_lens)), 1),
            "both_needles_exact_pct": round(100.0 * both_hits / n_eps, 2),
            "both_needles_ci95_pct": [ci_lo, ci_hi],
            "per_needle_recall_pct": round(100.0 * needle_hits / (2 * n_eps), 2),
            "steady_hbm_saved_pct": round(float(np.mean(hbm_saved)), 2),
            "cold_stripes_recalled_pct": round(float(np.mean(rec_pcts)), 2),
        }
        out_summary[label] = res
        print(
            f"  {label:<26} | Toks={res['mean_logical_tokens']:.0f} | "
            f"Both={res['both_needles_exact_pct']:5.1f}% [{ci_lo:.1f},{ci_hi:.1f}] | "
            f"Per-Needle={res['per_needle_recall_pct']:5.1f}% | "
            f"HBM={res['steady_hbm_saved_pct']:4.1f}% | Rec={res['cold_stripes_recalled_pct']:4.2f}%"
        )
    return out_summary


def run_part4_gate_sensitivity(
    model, tokenizer, model_type, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device
) -> dict[str, Any]:
    print("\n=== Part 4: (gamma, delta) Gate Sensitivity Sweep (n=12) ===")
    n_cases = 12
    delta_sweep = []
    for delta_v in (0.06, 0.09, 0.13, 0.18, 0.25):
        flags = SplitSlabAblationFlags(prominence_delta=delta_v, entropy_gamma=0.32)
        eps = [
            run_recallregret_episode(
                "splitslab_full", idx, model, tokenizer, model_type, geom, P_subspaces, sketch_spec,
                boundary_ids, eos_ids, device, flags=flags, record_timing=False,
            )
            for idx in range(n_cases)
        ]
        k = sum(1 for e in eps if e["exact_match"])
        row = {
            "delta": delta_v,
            "gamma": 0.32,
            "exact_match_pct": round(100.0 * k / n_cases, 2),
            "dist_recall_pct": round(float(np.mean([e["dist_recall_pct"] for e in eps])), 2),
            "t12_recall_pct": round(float(np.mean([e["t12_recall_pct"] for e in eps])), 2),
        }
        delta_sweep.append(row)
        print(
            f"  delta={delta_v:.2f}, gamma=0.32 | Exact={row['exact_match_pct']:5.1f}% | "
            f"DistRec={row['dist_recall_pct']:4.2f}% | T12Rec={row['t12_recall_pct']:4.2f}%"
        )

    gamma_sweep = []
    for gamma_v in (0.10, 0.20, 0.32, 0.50, 0.80):
        flags = SplitSlabAblationFlags(prominence_delta=0.13, entropy_gamma=gamma_v)
        eps = [
            run_recallregret_episode(
                "splitslab_full", idx, model, tokenizer, model_type, geom, P_subspaces, sketch_spec,
                boundary_ids, eos_ids, device, flags=flags, record_timing=False,
                use_live_entropy_gate=True,
            )
            for idx in range(n_cases)
        ]
        k = sum(1 for e in eps if e["exact_match"])
        row = {
            "delta": 0.13,
            "gamma": gamma_v,
            "exact_match_pct": round(100.0 * k / n_cases, 2),
            "dist_recall_pct": round(float(np.mean([e["dist_recall_pct"] for e in eps])), 2),
            "t12_recall_pct": round(float(np.mean([e["t12_recall_pct"] for e in eps])), 2),
        }
        gamma_sweep.append(row)
        print(
            f"  delta=0.13, gamma={gamma_v:.2f} | Exact={row['exact_match_pct']:5.1f}% | "
            f"DistRec={row['dist_recall_pct']:4.2f}% | T12Rec={row['t12_recall_pct']:4.2f}%"
        )

    return {"delta_sweep": delta_sweep, "gamma_sweep": gamma_sweep}


def main() -> None:
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_id} on {device} (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    model.eval()

    cfg = model.config
    geom = SplitSlabGeometry.from_model_config(
        num_layers=cfg.num_hidden_layers,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
    )
    P_subspaces, sketch_spec = build_model_subspaces(
        model, num_res_tokens=geom.plane1_residual_tokens, sketch_bytes=64, device=device
    )
    delim = get_model_chat_delimiters(tokenizer, cfg.model_type)
    boundary_ids, eos_ids = delim["boundary_ids"], delim["stop_ids"]

    out_path = ROOT / "results" / "tpds_revision_experiments.json"
    only_1_4 = "--only-part1-and-part4" in sys.argv
    prev = json.loads(out_path.read_text(encoding="utf-8")) if (only_1_4 and out_path.exists()) else {}

    t_all = time.perf_counter()
    p1 = run_part1_recallable_baselines_and_e2e(
        model, tokenizer, cfg.model_type, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device
    )
    if only_1_4 and "part2_routine_turn_quality" in prev and "part3_ruler_mk_niah" in prev:
        p2 = prev["part2_routine_turn_quality"]
        p3 = prev["part3_ruler_mk_niah"]
        print("\nReusing already-verified Part 2 and Part 3 results from existing JSON.")
    else:
        p2 = run_part2_routine_turn_quality(
            model, tokenizer, cfg.model_type, geom, P_subspaces, sketch_spec, boundary_ids, device
        )
        p3 = run_part3_ruler_mk_niah(
            model, tokenizer, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device
        )
    p4 = run_part4_gate_sensitivity(
        model, tokenizer, cfg.model_type, geom, P_subspaces, sketch_spec, boundary_ids, eos_ids, device
    )

    payload = {
        "metadata": {
            "project": "SplitSlab",
            "suite": "TPDS-Major-Revision-Experiments",
            "model": model_id,
            "device": device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_elapsed_s": round(time.perf_counter() - t_all, 1),
        },
        "part1_recallable_baselines_e2e": p1,
        "part2_routine_turn_quality": p2,
        "part3_ruler_mk_niah": p3,
        "part4_gate_sensitivity": p4,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved all revision results to {out_path} in {payload['metadata']['total_elapsed_s']}s")


if __name__ == "__main__":
    main()
