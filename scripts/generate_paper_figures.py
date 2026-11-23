#!/usr/bin/env python3
"""
Generates 5 Publication-Grade Vector PDF Figures for the SplitSlab IEEE TPDS Manuscript
Directly from Verified Hardware Telemetry (`results/*.json`).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.8,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.6,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "full_kv": "#4C566A",
    "actkv": "#BF616A",
    "recompact": "#D08770",
    "anchor_only": "#EBCB8B",
    "splitslab": "#2E86AB",
    "accent": "#1B998B",
}


def draw_box(ax, x, y, w, h, text, fc="#F0F4F8", ec="#2E3440", lw=1.1, fontsize=7.4, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color="#1D232A")


def generate_fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.95))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Column 1: Canonical Prefix Block Ingestion
    draw_box(
        ax, 0.01, 0.56, 0.225, 0.37,
        "Uncompacted KV Block $B_m$\n($M$ Tokens, Exact RoPE Baked)\n\nGlobal Key: $\\mathrm{SHA256}(T_{1:mM})$\n100% Fleet Prefix Sharing",
        fc="#E8EEF5", ec="#3B5998", lw=1.2, fontsize=7.5, bold=True,
    )
    draw_box(
        ax, 0.01, 0.07, 0.225, 0.37,
        "Universal Zero-Pad Rule\n$H_{kv}\\in\\{2,4,8\\},\\; d_h=128$\n$M \\in \\{80,40,20\\} \\Rightarrow N_{\\mathrm{res}}\\in\\{64,32,16\\}$\n$= 65,536$ B (Exact 64.0 KB)",
        fc="#F7F9FC", ec="#5E81AC", lw=1.0, fontsize=7.2,
    )

    # Column 2: Split-Plane Tiering
    draw_box(
        ax, 0.280, 0.56, 0.295, 0.37,
        "Plane 0: Hot HBM Skeleton (20%)\n• Step/Boundary & Stride Anchors\n• Co-Routed 64-Byte Sketch $\\mathcal{S}_{m,\\ell}$\n  ($G{=}2\\times 240$b SVD $+ 2\\times$FP16)\n  0.098% HBM Overhead (1,024$\\times$)",
        fc="#DCEEF8", ec=COLORS["splitslab"], lw=1.4, fontsize=7.4, bold=True,
    )
    draw_box(
        ax, 0.280, 0.07, 0.295, 0.37,
        "Plane 1: Cold NVMe Residual (80%)\n• Contiguous 64 KB Layer Stripes\n• Strict Write-Once ($\\mathrm{WAF} = 1.00\\times$)\n• Direct DMA / Darwin F_NOCACHE\n  ($p_{50} = 95.7\\,\\mu$s, 0 RMW Wear)",
        fc="#FDF3E7", ec="#C06C2B", lw=1.3, fontsize=7.4, bold=True,
    )

    # Column 3: 2-Layer Speculative Lookahead & Entropy Gate
    draw_box(
        ax, 0.675, 0.56, 0.315, 0.37,
        "Layer $\\ell$ Speculative Lookahead\n$\\tilde{q}_t^{(\\ell \\to \\ell+2)} = \\mathrm{RoPE}(W_Q^{(\\ell+2)}\\mathrm{LN}^{(\\ell+2)}(h_t^{(\\ell)}))$\nCosine Sim $= 0.962$ (vs. $0.010$ Raw)\n\nJoint Gate: $H(p_t) > \\gamma \\;\\wedge\\; \\bar{\\Pi}_m > \\delta$",
        fc="#E5F6F2", ec=COLORS["accent"], lw=1.4, fontsize=7.4, bold=True,
    )
    draw_box(
        ax, 0.675, 0.07, 0.315, 0.37,
        "Layer $\\ell+2$ Exact Attention\n• $>93\\%$ Steps: Plane-0 Speed (0 I/O)\n• Regret Turn: Async Stream $<7\\%$ Stripes\n  Hidden in 2-Layer Compute Window\n  ($\\|O_{\\mathrm{SS}}-O_{\\mathrm{Full}}\\|_2 \\leq 2\\alpha_R \\max_i\\|V_i\\|_2$)",
        fc="#EEFBF3", ec="#2B8A3E", lw=1.3, fontsize=7.2,
    )

    arrow_kw = dict(arrowstyle="->,head_width=0.24,head_length=0.32", lw=1.3, color="#2E3440")
    ax.annotate("", xy=(0.277, 0.745), xytext=(0.238, 0.745), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.277, 0.255), xytext=(0.238, 0.64), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.672, 0.745), xytext=(0.578, 0.745), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.832, 0.445), xytext=(0.832, 0.556), arrowprops=arrow_kw)

    # Clean horizontal corridor arrow for conditional NVMe stripe recall
    ax.annotate(
        "",
        xy=(0.672, 0.245),
        xytext=(0.578, 0.245),
        arrowprops=dict(arrowstyle="->,head_width=0.24,head_length=0.32", lw=1.4, ls="--", color="#C06C2B"),
    )
    ax.text(
        0.625, 0.315, "Async 64KB\npread (Gated)",
        ha="center", va="center", fontsize=6.9, fontweight="bold", color="#A64B00",
    )

    fig.savefig(FIG_DIR / "fig1_architecture_overview.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def generate_fig2_multimodel(mm_data: dict) -> None:
    models_map = [
        ("Qwen/Qwen2.5-3B-Instruct", "Qwen-3B\n(16:2)"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen-7B\n(28:4)"),
        ("ibm-granite/granite-3.1-8b-instruct", "Granite-8B\n(32:8)"),
        ("Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen-14B\n(40:8)"),
    ]
    present = [(k, lbl) for k, lbl in models_map if k in mm_data["models"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.65))
    fig.subplots_adjust(wspace=0.30)
    x = np.arange(len(present))
    w = 0.16

    m_specs = [
        ("full_kv_flatcake", "Full-KV + FlatCake", COLORS["full_kv"]),
        ("online_evict_actkv", "Online Evict (ActKV)", COLORS["actkv"]),
        ("iterative_recompact", "Iterative Re-Compact", COLORS["recompact"]),
        ("splitslab_anchor_only", "SplitSlab (Anchor)", COLORS["anchor_only"]),
        ("splitslab_full", "SplitSlab Full (Ours)", COLORS["splitslab"]),
    ]

    for idx, (m_key, m_lbl, col) in enumerate(m_specs):
        vals = [mm_data["models"][mk]["summary_by_method"][m_key]["exact_match_pct"] for mk, _ in present]
        ax1.bar(x + (idx - 2) * w, vals, width=w, label=m_lbl, color=col, edgecolor="#222222", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in present], fontsize=7.4)
    ax1.set_ylabel("Delayed-Recall Exact Match (%)")
    ax1.set_ylim(0, 144)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("(a) Delayed-Recall Recovery Across Scales")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", ncol=2, frameon=True, fontsize=6.2)

    steady_s = [mm_data["models"][mk]["summary_by_method"]["splitslab_full"]["steady_hbm_reduction_pct"] for mk, _ in present]
    peak_s = [mm_data["models"][mk]["summary_by_method"]["splitslab_full"]["peak_hbm_reduction_pct"] for mk, _ in present]
    rec_t12 = [mm_data["models"][mk]["summary_by_method"]["splitslab_full"]["cold_stripes_recalled_turn12_pct"] for mk, _ in present]

    ax2.bar(x - 0.16, steady_s, width=0.28, label="Steady HBM Saved (%)", color=COLORS["splitslab"], edgecolor="#222222", linewidth=0.5)
    ax2.bar(x + 0.14, peak_s, width=0.28, label="Peak HBM Saved (%)", color="#7FB3D5", edgecolor="#222222", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in present], fontsize=7.4)
    ax2.set_ylabel("GPU HBM Footprint Saved (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("(b) HBM Savings & Cold Recall Fraction")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    ax2_r = ax2.twinx()
    ax2_r.plot(x, rec_t12, color="#B22222", marker="o", lw=1.6, label="Turn-12 Recalled (%)")
    ax2_r.set_ylabel("Cold Stripes Recalled (%)", color="#B22222")
    ax2_r.set_ylim(0, 16)
    ax2_r.tick_params(axis="y", colors="#B22222")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, fontsize=6.5)

    fig.savefig(FIG_DIR / "fig2_multimodel_accuracy_hbm.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def generate_fig3_nvme_and_waf(deep_data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.55))
    fig.subplots_adjust(wspace=0.28)

    sw = deep_data["part_a_nvme_storage"]["stripe_size_sweep"]
    kbs = [f"{r['stripe_kb']} KB" for r in sw]
    buf_p50 = [r["os_buffered"]["pread_qd1_p50_us"] for r in sw]
    nc_qd1 = [r["direct_fnocache"]["pread_qd1_p50_us"] for r in sw]
    nc_qd4 = [r["direct_fnocache"]["pread_qd4_effective_us_per_stripe"] for r in sw]

    x = np.arange(len(kbs))
    w = 0.25
    ax1.bar(x - w, buf_p50, width=w, label="OS Page Cache (Buffered)", color="#95A5A6", edgecolor="#222", lw=0.5)
    ax1.bar(x, nc_qd1, width=w, label="Direct F_NOCACHE (QD=1)", color=COLORS["recompact"], edgecolor="#222", lw=0.5)
    ax1.bar(x + w, nc_qd4, width=w, label="Direct F_NOCACHE (QD=4)", color=COLORS["splitslab"], edgecolor="#222", lw=0.5)
    ax1.axhline(1160.0, color="#2B8A3E", ls="--", lw=1.2, label="Analytical GPU 2-Layer Win. (1.16 ms)")
    ax1.set_yscale("log")
    ax1.set_ylim(0.35, 4500.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(kbs)
    ax1.set_xlabel("NVMe Stripe Granularity")
    ax1.set_ylabel("Read Latency ($\\mu$s, log scale)")
    ax1.set_title("(a) F_NOCACHE Latency vs. Compute Window")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=6.4)

    wc = deep_data["part_a_nvme_storage"]["waf_vs_turns_curve"]
    ks = [r["compaction_turns_k"] for r in wc]
    waf_ss = [r["splitslab_physical_waf"] for r in wc]
    waf_64 = [r["recompact_64kb_slab_physical_waf"] for r in wc]
    waf_sub = [r["recompact_ragged_subpage_physical_waf"] for r in wc]
    waf_th = [r["theorem2_analytical_64kb_waf"] for r in wc]

    ax2.plot(ks, waf_sub, marker="^", lw=1.6, color=COLORS["actkv"], label="Ragged Sub-Page Re-Compact (24KB)")
    ax2.plot(ks, waf_64, marker="s", lw=1.6, color=COLORS["recompact"], label="Aligned 64KB Iterative Re-Compact")
    ax2.plot(ks, waf_th, ls=":", lw=1.3, color="#333333", label="Theorem 2 Analytical WAF Model")
    ax2.plot(ks, waf_ss, marker="o", lw=2.0, color=COLORS["splitslab"], label="SplitSlab Write-Once (WAF = 1.00x)")
    ax2.set_xlabel("Multi-Turn Compaction Epochs ($K$)")
    ax2.set_ylabel("Physical NVMe Write Amplification ($\\times$)")
    ax2.set_title("(b) Physical Flash WAF vs. Turns $K$")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, fontsize=6.4)

    fig.savefig(FIG_DIR / "fig3_nvme_waf_and_fnocache_io.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def generate_fig4_ablations(deep_data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5))
    fig.subplots_adjust(wspace=0.44)

    sk = deep_data["part_c_ablations"]["sketch_byte_sweep"]
    b_lbls = [f"{r['sketch_bytes']} B" for r in sk]
    b_acc = [r["exact_match_pct"] for r in sk]

    x1 = np.arange(len(b_lbls))
    ax1.bar(x1, b_acc, width=0.46, color=COLORS["splitslab"], edgecolor="#222", lw=0.6)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(b_lbls)
    ax1.set_xlabel("HBM Routing Sketch Size per 64 KB Stripe")
    ax1.set_ylabel("Delayed-Recall Exact Match (%)")
    ax1.set_ylim(0, 112)
    ax1.set_title("(a) Sketch Byte Budget Sensitivity")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for xi, v in enumerate(b_acc):
        ax1.text(xi, v + 2.5, f"{v:.1f}%", ha="center", fontsize=7.4, fontweight="bold")

    comp = deep_data["part_c_ablations"]["lookahead_and_component_ablations"]
    label_map = {
        "naive_raw_query_no_reproj": "Raw $q_t^{(\\ell)}$ (No Basis)",
        "no_co_routed_beacon": "No P0 Co-Beacon",
        "lookahead_3_layer": "Lookahead $\\Delta\\ell=3$",
        "lookahead_2_layer_default": "SplitSlab ($\\Delta\\ell=2$)",
        "lookahead_1_layer": "Lookahead $\\Delta\\ell=1$",
        "lookahead_0_sync_stall": "Sync Stall ($\\Delta\\ell=0$)",
    }
    ordered_keys = [
        "naive_raw_query_no_reproj",
        "no_co_routed_beacon",
        "lookahead_3_layer",
        "lookahead_2_layer_default",
        "lookahead_1_layer",
        "lookahead_0_sync_stall",
    ]
    by_k = {r["label"]: r for r in comp}
    y_lbls = [label_map[k] for k in ordered_keys if k in by_k]
    x_vals = [by_k[k]["exact_match_pct"] for k in ordered_keys if k in by_k]
    cols = [
        COLORS["actkv"] if "naive" in k else (COLORS["recompact"] if "beacon" in k else COLORS["splitslab"])
        for k in ordered_keys
        if k in by_k
    ]

    y2 = np.arange(len(y_lbls))
    ax2.barh(y2, x_vals, height=0.55, color=cols, edgecolor="#222", lw=0.5)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(y_lbls)
    ax2.set_xlabel("Delayed-Recall Exact Match (%)")
    ax2.set_xlim(0, 119)
    ax2.set_title("(b) Lookahead & Sketch Component Ablations")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    for yi, v in enumerate(x_vals):
        ax2.text(v + 1.8, yi, f"{v:.1f}%", va="center", fontsize=7.4, fontweight="bold")

    fig.savefig(FIG_DIR / "fig4_lookahead_and_sketch_ablations.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def generate_fig5_fleet_and_horizon(deep_data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.55))
    fig.subplots_adjust(wspace=0.30)

    fl = deep_data["part_b_fleet_dedup"]
    ps = [r["num_agents_p"] for r in fl]
    mb_ss = [r["splitslab_nvme_mb"] for r in fl]
    mb_ev = [r["online_evict_nvme_mb"] for r in fl]
    mb_fk = [r["full_kv_flatcake_nvme_mb"] for r in fl]

    ax1.plot(ps, mb_ev, marker="^", lw=1.7, color=COLORS["actkv"], label="Online Evict (0% Prefix Dedup)")
    ax1.plot(ps, mb_fk, marker="s", lw=1.5, ls="--", color=COLORS["full_kv"], label="Full-KV FlatCake (100% Dedup)")
    ax1.plot(ps, mb_ss, marker="o", lw=2.0, color=COLORS["splitslab"], label="SplitSlab Plane-1 (100% Dedup)")
    ax1.set_xlabel("Collaborating Agents Sharing Repo ($P$)")
    ax1.set_ylabel("Fleet NVMe Storage Written (MB)")
    ax1.set_title("(a) Fleet Prefix-Sharing NVMe Writes")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=6.4)

    lh = deep_data["part_e_long_horizon_scaling"]
    horizons = ["2.3K", "8.2K", "16.4K"]
    ss_lh = {r["horizon_label"]: r for r in lh if r["method"] == "splitslab_full"}
    ev_lh = {r["horizon_label"]: r for r in lh if r["method"] == "online_evict_actkv"}

    x = np.arange(len(horizons))
    w = 0.26
    ss_acc = [ss_lh[h]["exact_match_pct"] for h in horizons if h in ss_lh]
    ev_acc = [ev_lh[h]["exact_match_pct"] for h in horizons if h in ev_lh]
    ss_rec = [ss_lh[h]["cold_stripes_recalled_turn12_pct"] for h in horizons if h in ss_lh]

    ax2.bar(x - w / 2, ev_acc, width=w, color=COLORS["actkv"], edgecolor="#222", lw=0.5, label="ActKV Exact Match (%)")
    ax2.bar(x + w / 2, ss_acc, width=w, color=COLORS["splitslab"], edgecolor="#222", lw=0.5, label="SplitSlab Exact Match (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h} Toks" for h in horizons])
    ax2.set_ylabel("Delayed-Recall Exact Match (%)")
    ax2.set_ylim(0, 114)
    ax2.set_title("(b) Long-Horizon Scaling (Up to 16.4K)")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    ax2_r = ax2.twinx()
    ax2_r.plot(x, ss_rec, color="#B22222", marker="D", lw=1.6, label="Cold Stripes Recalled (%)")
    ax2_r.set_ylabel("Turn-12 Stripes Recalled (%)", color="#B22222")
    ax2_r.set_ylim(0, 10)
    ax2_r.tick_params(axis="y", colors="#B22222")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=6.2)

    fig.savefig(FIG_DIR / "fig5_fleet_dedup_and_long_horizon.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_fig1_architecture()
    print("[Figures] Generated fig1_architecture_overview.pdf")

    mm_path = RESULTS_DIR / "multimodel_fnocache_results.json"
    deep_path = RESULTS_DIR / "storage_ablations_scaling.json"

    if mm_path.exists():
        mm_data = json.loads(mm_path.read_text())
        generate_fig2_multimodel(mm_data)
        print("[Figures] Generated fig2_multimodel_accuracy_hbm.pdf")

    if deep_path.exists():
        deep_data = json.loads(deep_path.read_text())
        generate_fig3_nvme_and_waf(deep_data)
        generate_fig4_ablations(deep_data)
        generate_fig5_fleet_and_horizon(deep_data)
        print("[Figures] Generated fig3, fig4, and fig5 PDFs")


if __name__ == "__main__":
    main()
