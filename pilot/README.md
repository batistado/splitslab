# SplitSlab — Phase 0/1 End-to-End Pilot (`RecallRegret-Pilot`)

Empirical Phase 0/1 evaluation of **SplitSlab** (*Write-Once Canonical-Hash Residual Tiering to Eliminate Write Amplification and Compaction Regret in Disaggregated KV Stores*) on Apple Silicon (`mps`, `float16`) + internal Apple NVMe SSD using locally cached `Qwen/Qwen2.5-7B-Instruct`.

## 1. Reproduction

```bash
python3 \
  pilot/splitslab_pilot.py \
  --num-trajectories 20 \
  --out results/splitslab_qwen7b_pilot.json
```

## 2. Physical Geometry & Key Pilot Findings (`Qwen2.5-7B-Instruct`)

1. **Exact Zero-Fragmentation `64 KB` NVMe Stripe Geometry:**
   - `Qwen2.5-7B-Instruct` has $L=28$ layers, $H_{kv}=4$ KV heads, $d=128$, `fp16` ($2,048\text{ B}$ per token per layer).
   - Partitioning each completed $M=40$-token logical block into **Plane 0** ($8$ anchor tokens = $20\%$ kept in GPU HBM) and **Plane 1** ($32$ residual tokens = $80\%$ offloaded once to NVMe) yields **exact `65,536`-byte (`64.0 KB`) layer-major stripes** with zero internal padding (`WAF = 1.000`).
2. **Basis-Aligned 2-Layer Speculative Lookahead ($\tilde{q}_t^{(l \to l+2)}$):**
   - Naive cross-layer scoring ($q_t^{(l)}$ vs. Layer $l+2$ keys) suffers orthogonal head-basis mismatch (`cosine similarity = 0.0102`).
   - Projecting the post-Layer-$l$ residual stream through Layer $(l+2)$'s static `RMSNorm` + $W_Q^{(l+2)}$ + RoPE achieves **`0.9623` mean cosine similarity** with the true future query $q_t^{(l+2)}$ at zero extra trained parameter cost.
3. **Strict 64-Byte Bias-Centered Sub-Cluster Sketch (`1,024×` Compression):**
   - Deterministically removing `RoPE(k_proj.bias, pos)` (`L2 norm` up to `920.5` in `Qwen2.5-7B`) and packing $G=2$ salience-weighted sub-clusters $\times$ $240$ head-space $W_Q/W_K$ SVD sign bits (`60 B`) + $2 \times \text{fp16}$ sub-cluster norms (`4 B`) fits in **strictly `64 bytes`** per `64 KB` stripe (`0.098%` HBM overhead), co-routed alongside resident Plane-0 anchors.

## 3. Summary Results (`20` Dual-Agent `12`-Turn Trajectories)

| Method | Delayed-Recall Exact Match | Delayed-Recall Field Acc | Steady HBM Reduction | Peak HBM Reduction | Physical NVMe WAF | Cross-Agent SHA-256 Dedup | Cold Stripes Recalled (Turns 3–11) | Cold Stripes Recalled (Turn 12) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `full_kv_flatcake` (Oracle) | **100.0%** | **100.0%** | 0.0% | 0.0% | **1.00×** | **100.0%** | 0.00% | 0.00% |
| `online_evict_actkv` (`H2O`/`ActKV`) | 0.0% | 2.5% | 77.9% | 77.9% | 1.00× | 0.0% | 0.00% | 0.00% |
| `iterative_recompact` (Multi-Turn RMW) | 0.0% | 2.5% | 77.9% | 77.9% | **7.54×** | 0.0% | 0.00% | 0.00% |
| `splitslab_anchor_only` (Ablation) | 0.0% | 2.5% | 77.5% | 77.5% | **1.00×** | **100.0%** | 0.00% | 0.00% |
| **`splitslab_full` (Ours)** | **95.0%** | **95.0%** | **77.5%** | **72.5%** | **1.00×** | **100.0%** | **0.00%** | **6.49%** |

*Physical Apple NVMe SSD 64 KB aligned `pread` median latency: **`5.56 µs`**. Full per-trajectory records are saved in `../results/splitslab_qwen7b_pilot.json`.*
