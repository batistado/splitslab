#!/usr/bin/env python3
"""
Multi-Architecture & Multi-Scale End-to-End SplitSlab Benchmark (`F_NOCACHE` Direct NVMe)
=========================================================================================
Evaluates all 5 KV-tiering architectures with Darwin `fcntl.F_NOCACHE` OS page-cache bypass
across 4 locally cached models spanning 3B to 14B parameters and 2 distinct families:
  1. `Qwen/Qwen2.5-3B-Instruct`         (3B,  Qwen2 GQA 16:2, biased keys,   M=80 -> 64KB exact)
  2. `Qwen/Qwen2.5-7B-Instruct`         (7B,  Qwen2 GQA 28:4, biased keys,   M=40 -> 64KB exact)
  3. `ibm-granite/granite-3.1-8b-instruct` (8B, Granite uP GQA 32:8, unbiased, M=20 -> 64KB exact)
  4. `Qwen/Qwen2.5-Coder-14B-Instruct`  (14B, Qwen2 GQA 40:8, 48 layers,     M=20 -> 64KB exact)
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recallregret_bench import (
    TRAJECTORY_SPECS,
    build_model_formatted_trajectory,
    score_delayed_recall_response,
)
from splitslab import (
    DirectNVMeSlabStore,
    SplitSlabAblationFlags,
    SplitSlabGeometry,
    SplitSlabSession,
    build_model_subspaces,
)

DEFAULT_MODEL_MATRIX: list[tuple[str, int]] = [
    ("Qwen/Qwen2.5-3B-Instruct", 20),
    ("Qwen/Qwen2.5-7B-Instruct", 20),
    ("ibm-granite/granite-3.1-8b-instruct", 20),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", 12),
]

METHODS = [
    "full_kv_flatcake",
    "online_evict_actkv",
    "iterative_recompact",
    "splitslab_anchor_only",
    "splitslab_full",
]


def run_one_trajectory(
    method: str,
    traj: dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    geom: SplitSlabGeometry,
    P_subspaces: torch.Tensor,
    sk_spec: Any,
    nvme_root: Path,
    device: str,
    record_timing: bool = False,
) -> dict[str, Any]:
    spec = traj["spec"]
    m_dir = nvme_root / f"{method}_{spec.traj_id}"
    if m_dir.exists():
        shutil.rmtree(m_dir)

    store = DirectNVMeSlabStore(m_dir, direct_nocache=True, sync_mode="batch_fsync", io_workers=4)
    t_start = time.perf_counter()
    try:
        sess_a = SplitSlabSession(
            method=method,
            model=model,
            geom=geom,
            nvme_store=store,
            P_subspaces=P_subspaces,
            sketch_spec=sk_spec,
            boundary_token_ids=traj["boundary_ids"],
            device=device,
            flags=SplitSlabAblationFlags(),
        )

        # Turn 1: Shared repository snapshot
        t1_ids = tokenizer.encode(traj["turn1_shared"], add_special_tokens=False)
        shared_prefix_len = len(t1_ids)
        sess_a.forward_tokens(t1_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=False)
        sess_a.cum_attn_scores = torch.zeros(len(t1_ids), dtype=torch.float32, device=device)

        # Branch collaborating Agent B before Turn-2 query divergence
        sess_b = sess_a.clone_shared_prefix_state()

        # Agent A executes Turn 2 and offloads shared Turn-1 slabs
        t2a_ids = tokenizer.encode(traj["turn2_agent_a"], add_special_tokens=False)
        sess_a.forward_tokens(t2a_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=True)
        sess_a.apply_post_turn_tiering(shared_prefix_token_limit=shared_prefix_len, is_dedup_probe=False)

        # Agent B executes Turn 2 and probes SHA-256 deduplication on shared Turn-1 slabs
        t2b_ids = tokenizer.encode(traj["turn2_agent_b"], add_special_tokens=False)
        sess_b.forward_tokens(t2b_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=True)
        sess_b.apply_post_turn_tiering(shared_prefix_token_limit=shared_prefix_len, is_dedup_probe=True)
        del sess_b

        # Agent A executes Distractor Turns 3..11 incrementally with live SplitSlab gate
        for u_prompt, a_resp in traj["distractor_pairs"]:
            up_ids = tokenizer.encode(u_prompt, add_special_tokens=False)
            ar_ids = tokenizer.encode(a_resp, add_special_tokens=False)
            sess_a.clear_staged_recall()
            sess_a.current_step_uncertain = False
            sess_a.forward_tokens(up_ids, is_final_recall_turn=False, allow_recall=True, track_online_attn=True)
            sess_a.forward_tokens(ar_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=True)
            sess_a.clear_staged_recall()
            sess_a.apply_post_turn_tiering(is_dedup_probe=False)

        steady_hbm_b, _ = sess_a.active_hbm_bytes()
        full_equiv_b = sess_a.logical_seq_len * geom.num_layers * geom.bytes_per_token_per_layer

        # Turn 12: Production Stack-Trace Delayed-Recall Trigger
        t12_ids = tokenizer.encode(traj["turn12_prompt"], add_special_tokens=False)
        response = sess_a.greedy_generate(
            t12_ids,
            tokenizer=tokenizer,
            stop_token_ids=traj["stop_ids"],
            max_new_tokens=52,
            is_final_recall_turn=True,
            record_layer_timing=record_timing,
        )
        peak_hbm_b = steady_hbm_b + sess_a.peak_staged_bytes
        acc = score_delayed_recall_response(response, spec)
        elapsed_s = time.perf_counter() - t_start

        dedup_pct = (
            100.0 * store.dedup_hits / float(store.dedup_queries) if store.dedup_queries > 0 else 0.0
        )
        rec_t12_pct = (
            100.0 * sess_a.recall_turn_cold_stripes_recalled / float(sess_a.recall_turn_cold_stripe_opportunities)
            if sess_a.recall_turn_cold_stripe_opportunities > 0
            else 0.0
        )
        rec_dist_pct = (
            100.0 * sess_a.distractor_cold_stripes_recalled / float(sess_a.distractor_cold_stripe_opportunities)
            if sess_a.distractor_cold_stripe_opportunities > 0
            else 0.0
        )

        return {
            "traj_id": spec.traj_id,
            "domain": spec.domain,
            "expected": f"{spec.const_name} | {spec.param_val}",
            "response": response,
            **acc,
            "logical_tokens_pre_t12": sess_a.logical_seq_len,
            "steady_hbm_reduction_pct": 100.0 * (1.0 - steady_hbm_b / float(full_equiv_b)),
            "peak_hbm_reduction_pct": 100.0 * (1.0 - peak_hbm_b / float(full_equiv_b)),
            "nvme_waf": store.waf,
            "nvme_physical_bytes_written": store.physical_bytes_written,
            "cross_agent_dedup_pct": dedup_pct,
            "cold_stripes_recalled_turn12_pct": rec_t12_pct,
            "cold_stripes_recalled_distractor_pct": rec_dist_pct,
            "fnocache_pread_p50_us": float(np.median(store.read_latencies_us)) if store.read_latencies_us else 0.0,
            "fnocache_pread_p95_us": float(np.percentile(store.read_latencies_us, 95)) if store.read_latencies_us else 0.0,
            "fnocache_pwrite_p50_us": float(np.median(store.write_latencies_us)) if store.write_latencies_us else 0.0,
            "per_layer_decode_compute_p50_us": (
                float(np.median(sess_a.layer_compute_latencies_us)) if sess_a.layer_compute_latencies_us else 0.0
            ),
            "elapsed_sec": elapsed_s,
        }
    finally:
        store.close()
        shutil.rmtree(m_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SplitSlab Multi-Model F_NOCACHE Benchmark Suite")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "multimodel_fnocache_results.json"),
    )
    parser.add_argument("--nvme-dir", type=str, default="/tmp/splitslab_mm_fnocache")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    nvme_root = Path(args.nvme_dir)
    nvme_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "metadata": {
            "benchmark": "SplitSlab-MultiModel-DirectNVMe",
            "device": device,
            "dtype": "float16",
            "darwin_f_nocache": True,
            "sync_mode": "fsync",
            "nvme_stripe_bytes": 65536,
            "sketch_bytes": 64,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "models": {},
    }

    for model_id, n_traj in DEFAULT_MODEL_MATRIX:
        print(f"\n######################################################################")
        print(f"# Loading Model: {model_id} (n_trajectories={n_traj})")
        print(f"######################################################################", flush=True)

        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, local_files_only=True
        ).to(device)
        model.eval()
        cfg = model.config
        geom = SplitSlabGeometry.from_model_config(
            num_layers=cfg.num_hidden_layers,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
        )
        P_subspaces, sk_spec = build_model_subspaces(model, geom.plane1_residual_tokens, 64, device=device)
        print(
            f"Loaded in {time.perf_counter() - t0:.2f}s | arch={cfg.model_type} | "
            f"L={geom.num_layers} Qh={geom.num_q_heads} KVh={geom.num_kv_heads} | "
            f"Block={geom.block_tokens} (P0={geom.plane0_anchor_tokens}, P1={geom.plane1_residual_tokens} -> 64KB exact)",
            flush=True,
        )

        trajs = [
            build_model_formatted_trajectory(s, tok, cfg.model_type)
            for s in TRAJECTORY_SPECS[:n_traj]
        ]

        model_entry: dict[str, Any] = {
            "geometry": {
                "model_type": cfg.model_type,
                "num_layers": geom.num_layers,
                "num_q_heads": geom.num_q_heads,
                "num_kv_heads": geom.num_kv_heads,
                "head_dim": geom.head_dim,
                "block_tokens": geom.block_tokens,
                "plane0_anchor_tokens": geom.plane0_anchor_tokens,
                "plane1_residual_tokens": geom.plane1_residual_tokens,
                "bytes_per_token_per_layer": geom.bytes_per_token_per_layer,
                "sketch_dims_per_kv_head": sk_spec.dims_per_kv_head,
            },
            "summary_by_method": {},
            "per_trajectory": {},
        }

        for method in METHODS:
            print(f"\n--- [{model_id}] Method: {method} ---", flush=True)
            runs: list[dict[str, Any]] = []
            for idx, tr in enumerate(trajs):
                r = run_one_trajectory(
                    method=method,
                    traj=tr,
                    model=model,
                    tokenizer=tok,
                    geom=geom,
                    P_subspaces=P_subspaces,
                    sk_spec=sk_spec,
                    nvme_root=nvme_root,
                    device=device,
                    record_timing=(idx < 3 and method in ("full_kv_flatcake", "splitslab_full")),
                )
                runs.append(r)
                print(
                    f"  [{idx+1:02d}/{n_traj:02d}] {r['traj_id']} | Exact={int(r['exact_match'])} "
                    f"HBM={r['steady_hbm_reduction_pct']:5.1f}% WAF={r['nvme_waf']:4.2f}x "
                    f"Dedup={r['cross_agent_dedup_pct']:5.1f}% RecT12={r['cold_stripes_recalled_turn12_pct']:4.2f}% "
                    f"RecDist={r['cold_stripes_recalled_distractor_pct']:4.2f}% | {r['response'][:38]}",
                    flush=True,
                )
                if device == "mps":
                    torch.mps.empty_cache()
                gc.collect()

            read_p50s = [x["fnocache_pread_p50_us"] for x in runs if x["fnocache_pread_p50_us"] > 0]
            read_p95s = [x["fnocache_pread_p95_us"] for x in runs if x["fnocache_pread_p95_us"] > 0]
            write_p50s = [x["fnocache_pwrite_p50_us"] for x in runs if x["fnocache_pwrite_p50_us"] > 0]
            lay_p50s = [x["per_layer_decode_compute_p50_us"] for x in runs if x["per_layer_decode_compute_p50_us"] > 0]

            summ = {
                "num_trajectories": n_traj,
                "exact_match_pct": 100.0 * float(np.mean([x["exact_match"] for x in runs])),
                "field_accuracy_pct": 100.0 * float(np.mean([x["partial_score"] for x in runs])),
                "const_accuracy_pct": 100.0 * float(np.mean([x["has_const"] for x in runs])),
                "val_accuracy_pct": 100.0 * float(np.mean([x["has_val"] for x in runs])),
                "steady_hbm_reduction_pct": float(np.mean([x["steady_hbm_reduction_pct"] for x in runs])),
                "peak_hbm_reduction_pct": float(np.mean([x["peak_hbm_reduction_pct"] for x in runs])),
                "nvme_waf": float(np.mean([x["nvme_waf"] for x in runs])),
                "nvme_physical_mb_written_total": float(
                    np.sum([x["nvme_physical_bytes_written"] for x in runs]) / (1024 * 1024)
                ),
                "cross_agent_dedup_pct": float(np.mean([x["cross_agent_dedup_pct"] for x in runs])),
                "cold_stripes_recalled_turn12_pct": float(
                    np.mean([x["cold_stripes_recalled_turn12_pct"] for x in runs])
                ),
                "cold_stripes_recalled_distractor_pct": float(
                    np.mean([x["cold_stripes_recalled_distractor_pct"] for x in runs])
                ),
                "fnocache_pread_p50_us": float(np.median(read_p50s)) if read_p50s else 0.0,
                "fnocache_pread_p95_us": float(np.median(read_p95s)) if read_p95s else 0.0,
                "fnocache_pwrite_p50_us": float(np.median(write_p50s)) if write_p50s else 0.0,
                "per_layer_decode_compute_p50_us": float(np.median(lay_p50s)) if lay_p50s else 0.0,
                "two_layer_compute_window_us": 2.0 * float(np.median(lay_p50s)) if lay_p50s else 0.0,
                "mean_trajectory_sec": float(np.mean([x["elapsed_sec"] for x in runs])),
            }
            model_entry["summary_by_method"][method] = summ
            model_entry["per_trajectory"][method] = runs
            print(
                f"==> [{model_id} | {method}] Exact={summ['exact_match_pct']:.1f}% | "
                f"Field={summ['field_accuracy_pct']:.1f}% | HBM_Save={summ['steady_hbm_reduction_pct']:.1f}% | "
                f"WAF={summ['nvme_waf']:.2f}x | Dedup={summ['cross_agent_dedup_pct']:.1f}% | "
                f"RecT12={summ['cold_stripes_recalled_turn12_pct']:.2f}% | "
                f"RecDist={summ['cold_stripes_recalled_distractor_pct']:.2f}% | "
                f"F_NOCACHE_pread_p50={summ['fnocache_pread_p50_us']:.1f}us | "
                f"2LayerWindow={summ['two_layer_compute_window_us']:.1f}us",
                flush=True,
            )

        payload["models"][model_id] = model_entry
        # Checkpoint JSON after each model completes!
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[Checkpoint] Wrote {model_id} results to {out_path}", flush=True)

        del model, P_subspaces
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()

    shutil.rmtree(nvme_root, ignore_errors=True)
    print(f"\n[Done] Complete multi-model F_NOCACHE suite saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
