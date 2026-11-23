#!/usr/bin/env python3
"""
IEEE TPDS Minor-Revision Targeted Checks
========================================
1. Vectorized Detrending TTFT & Long-Horizon TTFT Scaling on Qwen2.5-7B (2.3K, 8.2K, 16.4K).
2. Recallable KV Baselines (Quest-Box+NVMe, ArkVale-Cent+NVMe) on bias-free
   ibm-granite/granite-3.1-8b-instruct (b_K = 0, n=20) to isolate pre-RoPE bias removal
   from SplitSlab's 64B SVD sketch and 2-layer async lookahead.

Saves verified telemetry to:
  results/tpds_minor_revision_checks.json
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


def wilson_ci_95(k: int, n: int) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = k / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return [round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2)]


def run_episode(
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
    num_repo_modules: int = 16,
    distractor_repeats: int = 1,
    record_timing: bool = False,
) -> dict[str, Any]:
    spec = TRAJECTORY_SPECS[spec_idx]
    traj = build_model_formatted_trajectory(
        spec, tokenizer, model_type, num_repo_modules=num_repo_modules, distractor_repeats=distractor_repeats
    )
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    with tempfile.TemporaryDirectory(prefix=f"sp_mrev_{method}_") as tmp_dir:
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
        sess.forward_tokens(enc(traj["turn1_shared"]), allow_recall=False)
        sess.apply_post_turn_tiering()
        sess.forward_tokens(enc(traj["turn2_agent_a"]), allow_recall=False)
        sess.apply_post_turn_tiering()

        for q_str, a_str in traj["distractor_pairs"]:
            sess.clear_staged_recall()
            sess.current_step_uncertain = False
            sess.forward_tokens(enc(q_str), is_final_recall_turn=False, allow_recall=True)
            sess.forward_tokens(enc(a_str), is_final_recall_turn=False, allow_recall=False)
            sess.apply_post_turn_tiering()

        steady_hbm, _ = sess.active_hbm_bytes()
        full_hbm = sess.logical_seq_len * geom.num_layers * geom.bytes_per_token_per_layer
        resp = sess.greedy_generate(
            enc(traj["turn12_prompt"]),
            tokenizer=tokenizer,
            stop_token_ids=eos_ids,
            max_new_tokens=52,
            is_final_recall_turn=True,
            record_layer_timing=record_timing,
        )
        peak_hbm = steady_hbm + sess.peak_staged_bytes
        sc = score_delayed_recall_response(resp, spec)

        out = {
            "traj_id": spec.traj_id,
            "logical_tokens": sess.logical_seq_len,
            "exact_match": bool(sc["exact_match"]),
            "field_acc": float(sc["partial_score"]),
            "steady_hbm_saved_pct": 100.0 * (1.0 - steady_hbm / float(full_hbm)),
            "peak_hbm_saved_pct": 100.0 * (1.0 - peak_hbm / float(full_hbm)),
            "dist_recall_pct": 100.0 * sess.distractor_cold_stripes_recalled / max(1, sess.distractor_cold_stripe_opportunities),
            "t12_recall_pct": 100.0 * sess.recall_turn_cold_stripes_recalled / max(1, sess.recall_turn_cold_stripe_opportunities),
            "ttft_ms": list(sess.prefill_ttft_ms_list),
            "tpot_ms": list(sess.decode_tpot_ms_list),
            "sketch_scoring_us": list(sess.sketch_scoring_us_list),
            "exposed_sync_stall_us": list(sess.exposed_sync_nvme_stall_us),
        }
        nvme.close()
        return out


def run_vectorized_ttft_scaling(device: str) -> dict[str, Any]:
    print("\n=== Part A: Vectorized Detrending TTFT & Context Scaling (Qwen2.5-7B) ===")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(device)
    model.eval()
    cfg = model.config
    geom = SplitSlabGeometry.from_model_config(
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    )
    P_sub, sk_spec = build_model_subspaces(model, geom.plane1_residual_tokens, sketch_bytes=64, device=device)
    delim = get_model_chat_delimiters(tokenizer, cfg.model_type)

    horizons = [
        ("2.3K", 16, 1, 6),
        ("8.2K", 60, 4, 2),
        ("16.4K", 120, 7, 2),
    ]
    results = {}
    for h_lbl, n_mod, d_rep, n_eps in horizons:
        h_res = {}
        for m_name in ("full_kv_flatcake", "splitslab_full"):
            eps = [
                run_episode(
                    m_name, i, model, tokenizer, cfg.model_type, geom, P_sub, sk_spec,
                    delim["boundary_ids"], delim["stop_ids"], device, SplitSlabAblationFlags(),
                    num_repo_modules=n_mod, distractor_repeats=d_rep, record_timing=True,
                )
                for i in range(n_eps)
            ]
            ttfts = [x for e in eps for x in e["ttft_ms"]]
            tpots = [x for e in eps for x in e["tpot_ms"]]
            sks = [x for e in eps for x in e["sketch_scoring_us"]]
            h_res[m_name] = {
                "n": n_eps,
                "mean_tokens": round(float(np.mean([e["logical_tokens"] for e in eps])), 1),
                "exact_match_pct": round(100.0 * sum(e["exact_match"] for e in eps) / n_eps, 2),
                "ttft_p50_ms": round(float(np.median(ttfts)), 2) if ttfts else 0.0,
                "tpot_p50_ms": round(float(np.median(tpots)), 2) if tpots else 0.0,
                "sketch_scoring_per_layer_p50_us": round(float(np.median(sks)), 1) if sks else 0.0,
            }
            print(
                f"  [{h_lbl}] {m_name:<18} | Toks={h_res[m_name]['mean_tokens']:.0f} | "
                f"Exact={h_res[m_name]['exact_match_pct']:5.1f}% | "
                f"TTFT p50={h_res[m_name]['ttft_p50_ms']:.1f} ms | "
                f"TPOT p50={h_res[m_name]['tpot_p50_ms']:.1f} ms | "
                f"Sketch/layer={h_res[m_name]['sketch_scoring_per_layer_p50_us']:.1f} us"
            )
        results[h_lbl] = h_res

    del model
    if device == "mps":
        torch.mps.empty_cache()
    return results


def run_granite_unbiased_baselines(device: str) -> dict[str, Any]:
    print("\n=== Part B: Recallable Baselines on Bias-Free Granite-3.1-8B (b_K = 0, n=20) ===")
    model_id = "ibm-granite/granite-3.1-8b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(device)
    model.eval()
    cfg = model.config
    geom = SplitSlabGeometry.from_model_config(
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    )
    P_sub, sk_spec = build_model_subspaces(model, geom.plane1_residual_tokens, sketch_bytes=64, device=device)
    delim = get_model_chat_delimiters(tokenizer, cfg.model_type)

    # Note: Granite has M=20 (vs Qwen-7B M=40), so 5 blocks = 100 tokens (matching SplitSlab's 5-block micro-window!)
    methods = [
        ("quest_box_nvme", SplitSlabAblationFlags(baseline_topk_blocks=5)),
        ("arkvale_centroid_nvme", SplitSlabAblationFlags(baseline_topk_blocks=5)),
        ("splitslab_full", SplitSlabAblationFlags()),
    ]
    out = {}
    for m_name, flags in methods:
        eps = [
            run_episode(
                m_name, i, model, tokenizer, cfg.model_type, geom, P_sub, sk_spec,
                delim["boundary_ids"], delim["stop_ids"], device, flags,
                num_repo_modules=16, distractor_repeats=1, record_timing=(i < 4),
            )
            for i in range(20)
        ]
        k = sum(e["exact_match"] for e in eps)
        ci = wilson_ci_95(k, 20)
        tpots = [x for e in eps for x in e["tpot_ms"]]
        stalls = [x for e in eps for x in e["exposed_sync_stall_us"]]
        meta_b = 4096 if m_name == "quest_box_nvme" else 2048 if m_name == "arkvale_centroid_nvme" else 64
        res = {
            "n": 20,
            "exact_match_pct": round(100.0 * k / 20, 2),
            "exact_match_ci95_pct": ci,
            "field_acc_pct": round(100.0 * float(np.mean([e["field_acc"] for e in eps])), 2),
            "metadata_bytes_per_stripe": meta_b,
            "steady_hbm_saved_pct": round(float(np.mean([e["steady_hbm_saved_pct"] for e in eps])), 2),
            "dist_recall_pct": round(float(np.mean([e["dist_recall_pct"] for e in eps])), 2),
            "t12_recall_pct": round(float(np.mean([e["t12_recall_pct"] for e in eps])), 2),
            "tpot_p50_ms": round(float(np.median(tpots)), 2) if tpots else 0.0,
            "exposed_sync_stall_per_layer_p50_us": round(float(np.median(stalls)), 1) if stalls else 0.0,
        }
        out[m_name] = res
        print(
            f"  [Granite-8B] {m_name:<22} | Exact={res['exact_match_pct']:5.1f}% [{ci[0]:.1f},{ci[1]:.1f}] | "
            f"Field={res['field_acc_pct']:5.1f}% | Meta={meta_b:4d}B | HBM={res['steady_hbm_saved_pct']:4.1f}% | "
            f"T12/Dist={res['t12_recall_pct']:4.2f}%/{res['dist_recall_pct']:4.2f}% | "
            f"TPOT={res['tpot_p50_ms']:.1f} ms | SyncStall={res['exposed_sync_stall_per_layer_p50_us']:.1f} us"
        )
    return out


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    part_a = run_vectorized_ttft_scaling(device)
    part_b = run_granite_unbiased_baselines(device)

    out_path = ROOT / "results" / "tpds_minor_revision_checks.json"
    payload = {
        "metadata": {
            "project": "SplitSlab",
            "suite": "TPDS-Minor-Revision-Checks",
            "device": device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        },
        "part_a_vectorized_ttft_scaling_qwen7b": part_a,
        "part_b_unbiased_granite8b_baselines": part_b,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved minor-revision checks to {out_path} in {payload['metadata']['elapsed_s']}s")


if __name__ == "__main__":
    main()
