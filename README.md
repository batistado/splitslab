# SplitSlab: Write-Once Canonical-Hash Residual Tiering for Disaggregated KV Stores

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

**SplitSlab** is a write-once canonical-hash split-plane Key-Value (KV) storage and speculative recall engine for long-horizon Large Language Model (LLM) agent serving. SplitSlab resolves the storage-compute conflict between online GPU High-Bandwidth Memory (HBM) token eviction (`H2O`, `StreamingLLM`, `SnapKV`) and content-addressed NVMe Solid-State Drive (SSD) offloading (`vLLM`, `Mooncake`, `LMCache`).

## Motivation: Three Failures Across the GPU-NVMe Boundary

1. **Prefix-Hash Collapse (`0%` Fleet Deduplication):** Query-driven online KV eviction prunes different token subsets when collaborating agents ask different Turn-2 questions over an identical Turn-1 repository snapshot, shifting sequential slab boundaries and collapsing fleet-wide SHA-256 deduplication to `0.0%` (`5.47x` higher flash write volume across 32 agents).
2. **Flash Write Amplification (`7.5x` to `53.7x` WAF):** Re-compacting KV caches turn-by-turn dirties previously written flash pages, inducing `7.54x` to `20.13x` physical NVMe Write Amplification Factor (WAF) on aligned `64 KB` slabs and `53.67x` on ragged `24 KB` per-head chunks.
3. **Compaction Regret on Delayed Recall:** Early low-saliency details (such as dormant hex error constants) are irreversibly deleted during intermediate turns, causing `0.0%` exact-match accuracy across all 4 evaluated models when a Turn-12 crash dump queries them.

## Core Architecture

* **Universal Zero-Padding `64.0 KB` (`65,536 B`) Split-Plane Geometry (`WAF = 1.00x`):** Maps every standard Grouped-Query Attention (GQA) KV-head count `H_kv in {2, 4, 8}` (`d_h = 128`, FP16, `rho = 0.20`) to logical block lengths `M in {80, 40, 20}` (`(16, 64)`, `(8, 32)`, `(4, 16)`) whose `80%` cold residual slice equals `65,536` bytes per layer with zero padding. Prefix-deterministic splitting preserves `100.0%` cross-agent SHA-256 deduplication.
* **Bias-Centered `64-Byte` Subspace Routing Sketch (`1,024x` Compression):** Subtracts deterministic rotary key-projection bias `RoPE(b_K, pos)` (`||b_K||_2` up to `920.5` in `Qwen2.5`), computes salience-weighted 2-subcluster centroids in static `W_Q/W_K` principal head subspaces, and packs `2 x FP16` norms + `480` sign bits into `64 B` (`0.098%` HBM overhead), co-routed with resident Plane-0 anchors.
* **Parameter-Free Basis-Aligned Two-Layer Speculative Lookahead:** Projects the Layer `l` residual stream through Layer `l+2`'s static `RMSNorm` and `W_Q` (`0.9623` cosine similarity vs. `0.0102` without re-projection), hiding `72.7` to `189.1 us` uncached Darwin `fcntl.F_NOCACHE` NVMe `pread` operations inside `2.03` to `5.17 ms` two-layer GPU compute windows.

## Summary of Multi-Model Direct-NVMe (`fcntl.F_NOCACHE`) Results

Evaluated across `360` multi-turn trajectories on `RecallRegret-Bench` with OS page-cache bypass (`fcntl.F_NOCACHE` on macOS Apple M5 Pro internal NVMe SSD):

| Model (`GQA`, Block `M`) | Method | Exact Match | Field Acc. | Steady HBM Saved | Peak HBM Saved | Physical NVMe WAF | Fleet SHA-256 Dedup | Cold Recall (Dist / T12) | Uncached `pread` p50 / 2-Layer Win. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B** (`16:2`, `M=80`) | Online Evict / Re-Compact | `0.0%` | `0.0%` | `77.9%` | `77.9%` | `7.91x` | `0.0%` | `0.00% / 0.00%` | N/A |
| **Qwen2.5-3B** (`16:2`, `M=80`) | **SplitSlab Full** | **`45.0%`** | **`55.0%`** | **`77.4%`** | **`72.7%`** | **`1.00x`** | **`100.0%`** | **`0.10% / 5.92%`** | **`189.1 / 2,033.8 us`** |
| **Qwen2.5-7B** (`28:4`, `M=40`) | Online Evict / Re-Compact | `0.0%` | `2.5%` | `77.9%` | `77.9%` | `7.54x` | `0.0%` | `0.00% / 0.00%` | N/A |
| **Qwen2.5-7B** (`28:4`, `M=40`) | **SplitSlab Full** | **`95.0%`** | **`95.0%`** | **`77.5%`** | **`72.2%`** | **`1.00x`** | **`100.0%`** | **`0.00% / 6.83%`** | **`84.3 / 4,813.1 us`** |
| **Granite-3.1-8B** (`uP 32:8`, `M=20`) | Online Evict / Re-Compact | `0.0%` | `2.5%` | `78.1%` | `78.1%` | `7.91x` | `0.0%` | `0.00% / 0.00%` | N/A |
| **Granite-3.1-8B** (`uP 32:8`, `M=20`) | **SplitSlab Full** | **`70.0%`** | **`75.0%`** | **`77.7%`** | **`74.3%`** | **`1.00x`** | **`100.0%`** | **`0.24% / 4.08%`** | **`72.7 / 4,087.7 us`** |
| **Qwen2.5-Coder-14B** (`40:8`, `M=20`) | Online Evict / Re-Compact | `0.0%` | `0.0%` | `77.9%` | `77.9%` | `7.53x` | `0.0%` | `0.00% / 0.00%` | N/A |
| **Qwen2.5-Coder-14B** (`40:8`, `M=20`) | **SplitSlab Full** | **`83.3%`** | **`87.5%`** | **`77.6%`** | **`73.5%`** | **`1.00x`** | **`100.0%`** | **`0.38% / 5.21%`** | **`72.9 / 5,167.3 us`** |

## Repository Structure

```text
splitslab/
├── splitslab/                     # Core SplitSlab library (geometry, F_NOCACHE slab store, 64B sketch, engine, theory)
├── recallregret_bench/            # 20-domain multi-agent delayed-recall & long-horizon benchmark generator
├── benchmarks/                    # Multi-model F_NOCACHE evaluation & storage/ablation/scaling suites
├── pilot/                         # Phase 0/1 Qwen2.5-7B standalone pilot script & documentation
├── scripts/                       # Publication figure generator (fig1..fig5 vector PDFs)
├── examples/                      # Fast runnable quickstart demo (<2s, no GPU weights required)
├── tests/                         # 21 pytest unit, storage, bit-exactness, and theoretical bound tests
├── results/                       # Verified hardware JSON artifacts
└── paper/                         # IEEE TPDS (`IEEEtran`) manuscript, validator, figures, and Overleaf bundle
```

## Quickstart & Reproduction

```bash
# 1. Run the <2s quickstart demo (geometry, F_NOCACHE CAS store, 64B sketch, Theorem 1 check)
python3 examples/quickstart_demo.py

# 2. Run the 21-test verification suite
pytest tests/test_splitslab_core.py -v

# 3. Run the 4-model F_NOCACHE end-to-end suite (requires cached HuggingFace weights)
python3 benchmarks/run_multimodel_suite.py

# 4. Run direct-NVMe latency, 24-turn WAF, 32-agent fleet, ablations, Theorem 1 audit & 16.4K scaling
python3 benchmarks/run_storage_and_ablations.py

# 5. Regenerate all 5 paper figures and validate the IEEE TPDS LaTeX manuscript
python3 scripts/generate_paper_figures.py
python3 paper/validate_latex.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
