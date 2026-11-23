"""
Model-Agnostic Multi-Tier KV Session & 2-Layer Lookahead Recall Engine for SplitSlab.
Supports Qwen2, Granite, Llama, and Mistral GQA Transformer backbones.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from splitslab.config import NVME_STRIPE_BYTES, SplitSlabGeometry
from splitslab.sketch import SketchSpec, pack_subspace_sketch, unpack_subspace_sketches
from splitslab.slab_store import DirectNVMeSlabStore, deserialize_stripe_fp16, serialize_stripe_fp16


@dataclass
class ColdStripeRecord:
    block_idx: int
    canonical_hash: str
    nvme_offsets: list[int]
    sketch_bytes: list[bytes]


@dataclass
class SplitSlabAblationFlags:
    lookahead_layers: int = 2               # 0, 1, 2, 3
    use_basis_aligned_query: bool = True    # False -> naive raw q_t^(l) (Proposition 1 ablation)
    use_bias_centering: bool = True         # False -> skip RoPE(k_proj.bias) removal
    use_co_routed_beacon: bool = True       # False -> Plane-1 sketch only without Plane-0 co-routing
    prominence_delta: float = 0.130
    entropy_gamma: float = 0.32
    max_recall_peaks: int = 1               # >1 enables multi-peak extraction for multi-needle queries
    baseline_topk_blocks: int = 3           # Sync top-k cold blocks for Quest-Box / ArkVale-Centroid baselines


class SplitSlabSession:
    """
    Unified multi-turn KV session supporting all 5 evaluated methods + ablation flags:
      - `full_kv_flatcake`
      - `online_evict_actkv`
      - `iterative_recompact`
      - `splitslab_anchor_only`
      - `splitslab_full`
    """

    def __init__(
        self,
        method: str,
        model: PreTrainedModel,
        geom: SplitSlabGeometry,
        nvme_store: DirectNVMeSlabStore,
        P_subspaces: torch.Tensor,
        sketch_spec: SketchSpec,
        boundary_token_ids: set[int],
        device: str = "mps",
        flags: SplitSlabAblationFlags | None = None,
    ):
        self.method = method
        self.model = model
        self.geom = geom
        self.nvme = nvme_store
        self.P_subspaces = P_subspaces
        self.sketch_spec = sketch_spec
        self.boundary_token_ids = boundary_token_ids
        self.device = device
        self.flags = flags or SplitSlabAblationFlags()

        cfg = model.config
        self.embed_mult = float(getattr(cfg, "embedding_multiplier", 1.0))
        self.logits_div = float(getattr(cfg, "logits_scaling", 1.0))
        self.gqa_groups = geom.num_q_heads // geom.num_kv_heads

        self.logical_seq_len = 0
        self.hbm_keys: list[torch.Tensor | None] = [None] * geom.num_layers
        self.hbm_vals: list[torch.Tensor | None] = [None] * geom.num_layers
        self.hbm_positions: list[int] = []
        self.hbm_token_ids: list[int] = []
        self.all_token_ids: list[int] = []

        # Online eviction trackers
        self.cum_attn_scores: torch.Tensor | None = None
        self.compacted_slab_offsets: dict[int, list[int]] = {}
        self.compacted_slab_sigs: dict[int, bytes] = {}

        # SplitSlab tiering state
        self.tiered_prefix_tokens = 0
        self.cold_stripes: list[ColdStripeRecord] = []
        self._unpacked_sketches_cache: list[torch.Tensor | None] = [None] * geom.num_layers
        # Recallable baseline metadata per layer: Quest [k_min, k_max] (2048 B/stripe) & ArkVale centroid (1024 B/stripe)
        self.quest_kmin: list[list[torch.Tensor]] = [[] for _ in range(geom.num_layers)]
        self.quest_kmax: list[list[torch.Tensor]] = [[] for _ in range(geom.num_layers)]
        self.arkvale_centroids: list[list[torch.Tensor]] = [[] for _ in range(geom.num_layers)]

        # Turn-scoped transient Recall Staging Buffer
        self.staged_recall: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(geom.num_layers)
        ]
        self.enable_recall_now = False
        self.current_step_uncertain = False
        self.current_query_token_ids: list[int] = []
        self.turn_accum_prominence: torch.Tensor | None = None
        self.turn_prominence_steps: int = 0
        self.pinned_turn_blocks: list[int] | None = None

        # Telemetry
        self.distractor_cold_stripe_opportunities = 0
        self.distractor_cold_stripes_recalled = 0
        self.recall_turn_cold_stripe_opportunities = 0
        self.recall_turn_cold_stripes_recalled = 0
        self.peak_staged_bytes = 0
        self.layer_compute_latencies_us: list[float] = []
        self.prefill_ttft_ms_list: list[float] = []
        self.decode_tpot_ms_list: list[float] = []
        self.sketch_scoring_us_list: list[float] = []
        self.entropy_gate_us_list: list[float] = []
        self.nvme_stage_us_list: list[float] = []
        self.exposed_sync_nvme_stall_us: list[float] = []

    def clone_shared_prefix_state(self) -> "SplitSlabSession":
        clone = SplitSlabSession(
            method=self.method,
            model=self.model,
            geom=self.geom,
            nvme_store=self.nvme,
            P_subspaces=self.P_subspaces,
            sketch_spec=self.sketch_spec,
            boundary_token_ids=self.boundary_token_ids,
            device=self.device,
            flags=self.flags,
        )
        clone.logical_seq_len = self.logical_seq_len
        clone.hbm_keys = [k.clone() if k is not None else None for k in self.hbm_keys]
        clone.hbm_vals = [v.clone() if v is not None else None for v in self.hbm_vals]
        clone.hbm_positions = list(self.hbm_positions)
        clone.hbm_token_ids = list(self.hbm_token_ids)
        clone.all_token_ids = list(self.all_token_ids)
        clone.cum_attn_scores = self.cum_attn_scores.clone() if self.cum_attn_scores is not None else None
        clone.tiered_prefix_tokens = self.tiered_prefix_tokens
        clone.cold_stripes = list(self.cold_stripes)
        clone.quest_kmin = [list(x) for x in self.quest_kmin]
        clone.quest_kmax = [list(x) for x in self.quest_kmax]
        clone.arkvale_centroids = [list(x) for x in self.arkvale_centroids]
        return clone

    def clear_staged_recall(self) -> None:
        for d in self.staged_recall:
            d.clear()
        self.turn_accum_prominence = None
        self.turn_prominence_steps = 0
        self.pinned_turn_blocks = None

    def active_hbm_bytes(self) -> tuple[int, int]:
        kv_bytes = 0
        for l in range(self.geom.num_layers):
            if self.hbm_keys[l] is not None:
                kv_bytes += self.hbm_keys[l].numel() * 2 + self.hbm_vals[l].numel() * 2
        if self.method == "quest_box_nvme":
            # 2 FP16 vectors (k_min, k_max) per KV head per stripe -> 2 * H_kv * d_h * 2 bytes (2,048 B for 7B)
            meta_per_stripe = 2 * self.geom.num_kv_heads * self.geom.head_dim * 2
        elif self.method == "arkvale_centroid_nvme":
            # 1 FP16 centroid vector per KV head per stripe -> H_kv * d_h * 2 bytes (1,024 B for 7B)
            meta_per_stripe = self.geom.num_kv_heads * self.geom.head_dim * 2
        else:
            meta_per_stripe = self.sketch_spec.total_bytes
        sketch_bytes = len(self.cold_stripes) * self.geom.num_layers * meta_per_stripe
        steady = kv_bytes + sketch_bytes
        return steady, steady + self.peak_staged_bytes

    def _get_layer_sketches(self, layer_idx: int) -> torch.Tensor | None:
        if not self.cold_stripes:
            return None
        cached = self._unpacked_sketches_cache[layer_idx]
        if cached is None or cached.shape[2] != len(self.cold_stripes):
            raw_list = [cs.sketch_bytes[layer_idx] for cs in self.cold_stripes]
            cached = unpack_subspace_sketches(raw_list, self.sketch_spec, self.device)
            self._unpacked_sketches_cache[layer_idx] = cached
        return cached

    @torch.no_grad()
    def _remove_rope_k_bias(self, k_tensor: torch.Tensor, positions: list[int], layer_idx: int) -> torch.Tensor:
        """
        Deterministically removes `RoPE(k_proj.bias, positions)` when `k_proj.bias` exists
        (e.g. Qwen2.5-3B/7B/14B); exact zero-overhead identity when `k_proj.bias is None` (Granite/Llama).
        """
        k_proj = self.model.model.layers[layer_idx].self_attn.k_proj
        if not self.flags.use_bias_centering or k_proj.bias is None:
            return k_tensor[0].float()

        t_len = len(positions)
        pos_t = torch.tensor([positions], dtype=torch.long, device=self.device)
        dum_emb = torch.zeros((1, t_len, self.geom.head_dim), dtype=torch.float16, device=self.device)
        cos_p, sin_p = self.model.model.rotary_emb(dum_emb, pos_t)
        b_k = (
            k_proj.bias.detach()
            .view(1, 1, self.geom.num_kv_heads, self.geom.head_dim)
            .expand(1, t_len, self.geom.num_kv_heads, self.geom.head_dim)
            .transpose(1, 2)
        )
        q_dum = torch.zeros((1, self.geom.num_q_heads, t_len, self.geom.head_dim), dtype=b_k.dtype, device=self.device)
        _, b_k_rope = apply_rotary_pos_emb(q_dum, b_k, cos_p, sin_p)
        return (k_tensor.float() - b_k_rope.float())[0]

    # ------------------------------------------------------------------------
    # Speculative Lookahead Scoring & Multi-Queue NVMe Recall
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def maybe_lookahead_recall_for_layer(
        self,
        hidden_states: torch.Tensor,
        raw_q_current_layer: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        target_layer: int,
        is_final_recall_turn: bool,
    ) -> None:
        m_blocks = len(self.cold_stripes)
        recency_blocks = max(3, int(round(240.0 / self.geom.block_tokens)))
        if self.method != "splitslab_full" or not self.enable_recall_now or m_blocks <= recency_blocks + 3:
            return
        if target_layer < 3 or target_layer >= self.geom.num_layers:
            return
        if self.pinned_turn_blocks is not None and all(
            m_i in self.staged_recall[target_layer] for m_i in self.pinned_turn_blocks
        ):
            return

        q_total = hidden_states.shape[1]
        if q_total > 1:
            keep_idx = [
                i for i, tid in enumerate(self.current_query_token_ids)
                if tid not in self.boundary_token_ids
            ]
            if not keep_idx:
                return
            keep_t = torch.tensor(keep_idx, dtype=torch.long, device=self.device)
            h_slice = hidden_states[:, keep_t, :]
            cos, sin = position_embeddings
            cos_s, sin_s = cos[:, keep_t, :], sin[:, keep_t, :]
            q_raw_slice = raw_q_current_layer[:, :, keep_t, :]
        else:
            if self.current_query_token_ids and self.current_query_token_ids[0] in self.boundary_token_ids:
                return
            h_slice = hidden_states
            cos_s, sin_s = position_embeddings
            q_raw_slice = raw_q_current_layer

        ql = h_slice.shape[1]
        t_layer_mod = self.model.model.layers[target_layer]
        t_attn = t_layer_mod.self_attn
        attn_scale = float(getattr(t_attn, "scaling", 1.0 / math.sqrt(float(self.geom.head_dim))))

        if self.flags.use_basis_aligned_query or self.flags.lookahead_layers == 0:
            normed = t_layer_mod.input_layernorm(h_slice)
            qp = t_attn.q_proj(normed).view(1, ql, self.geom.num_q_heads, self.geom.head_dim).transpose(1, 2)
            kd = torch.zeros((1, self.geom.num_kv_heads, ql, self.geom.head_dim), dtype=h_slice.dtype, device=self.device)
            q_spec, _ = apply_rotary_pos_emb(qp, kd, cos_s, sin_s)
        else:
            # Proposition 1 ablation: directly reuse raw `q_t^(l)` without basis re-projection!
            q_spec = q_raw_slice

        qf = q_spec[0].float().view(self.geom.num_kv_heads, self.gqa_groups, ql, self.geom.head_dim) * attn_scale

        cutoff_b = max(2, m_blocks - recency_blocks)
        sk_vec = self._get_layer_sketches(target_layer)
        assert sk_vec is not None
        sk_c = sk_vec - sk_vec[:, :, 1:cutoff_b, :].mean(dim=2, keepdim=True)
        q_proj = torch.einsum("grqd,gdb->grqb", qf, self.P_subspaces[target_layer])
        dim_boost = min(1.35, math.sqrt(self.geom.head_dim / max(16.0, float(self.sketch_spec.dims_per_kv_head))) * 0.85)
        est_logits = torch.einsum("grqb,gsmb->grqms", q_proj, sk_c).amax(dim=-1) * dim_boost

        k_hbm_raw = self.hbm_keys[target_layer]
        assert k_hbm_raw is not None
        # Note: if lookahead_layers == 0, `self.hbm_keys[target_layer]` does not yet include `q_total` if called pre-update
        hbm_pos_slice = self.hbm_positions[: k_hbm_raw.shape[2]]
        k_hbm_pure = self._remove_rope_k_bias(k_hbm_raw, hbm_pos_slice, target_layer)
        p0_toks = self.geom.plane0_anchor_tokens
        p0_end = cutoff_b * p0_toks
        kh_mean = k_hbm_pure[:, p0_toks:p0_end, :].mean(dim=1, keepdim=True)
        k_hbm_c = k_hbm_pure - kh_mean

        if self.flags.use_co_routed_beacon:
            k_p0_blocks = k_hbm_c[:, : m_blocks * p0_toks, :].view(
                self.geom.num_kv_heads, m_blocks, p0_toks, self.geom.head_dim
            )
            p0_block_logits = torch.einsum("grqd,gmad->grqma", qf, k_p0_blocks).amax(dim=-1)
            p0_w = 0.72 if self.geom.block_tokens <= 20 else 0.86
            joint_block_logits = torch.maximum(est_logits, p0_w * p0_block_logits)
            k_ref = torch.cat([k_hbm_c[:, :p0_toks, :], k_hbm_c[:, m_blocks * p0_toks :, :]], dim=1)
        else:
            joint_block_logits = est_logits
            k_ref = k_hbm_c

        lse_ref = torch.logsumexp(torch.matmul(qf, k_ref.unsqueeze(1).transpose(-1, -2)), dim=-1, keepdim=True)
        denom = torch.logaddexp(lse_ref, torch.logsumexp(joint_block_logits, dim=-1, keepdim=True))
        alpha_m = torch.exp(joint_block_logits - denom)
        raw_m = alpha_m.amax(dim=(0, 1, 2))
        raw_m[0] = 0.0

        nbr_rad = max(4, int(round(200.0 / self.geom.block_tokens)))
        excl_rad = max(2, int(round(60.0 / self.geom.block_tokens)))
        valid_hi = max(2, m_blocks - 1)
        detrended = torch.zeros_like(raw_m)
        for m in range(1, cutoff_b):
            nbrs = [
                raw_m[j].item()
                for j in range(max(1, m - nbr_rad), min(valid_hi, m + nbr_rad + 1))
                if abs(j - m) >= excl_rad
            ]
            detrended[m] = max(0.0, float(raw_m[m].item()) - (float(np.median(nbrs)) if nbrs else 0.0))
        # Scale-invariant block-density normalization across GQA stripe sizes and 2K-16K horizons
        density_scale = (float(cutoff_b) / 38.0) ** 0.62
        detrended = detrended * density_scale
        detrended[cutoff_b:] = 0.0

        if self.turn_accum_prominence is None or self.turn_accum_prominence.shape[0] != m_blocks:
            self.turn_accum_prominence = detrended.clone()
            self.turn_prominence_steps = 1
        else:
            self.turn_accum_prominence = self.turn_accum_prominence + detrended
            self.turn_prominence_steps += 1

        min_consensus_layers = max(12, int(round(0.60 * self.geom.num_layers)))
        if self.turn_prominence_steps < min_consensus_layers or not self.current_step_uncertain:
            return

        mean_prom = self.turn_accum_prominence / float(self.turn_prominence_steps)

        if self.pinned_turn_blocks is not None:
            chosen = self.pinned_turn_blocks
        else:
            peak_val = float(mean_prom.max().item())
            if peak_val < self.flags.prominence_delta:
                return

            if self.flags.max_recall_peaks <= 1:
                best_m = int(torch.argmax(mean_prom).item())
                # Token-normalized ~80-100 token micro-window (< 8% of cold stripes across all GQA configs!)
                if self.geom.block_tokens >= 80:
                    left_v = float(mean_prom[best_m - 1].item()) if best_m - 1 >= 1 else -1.0
                    right_v = float(mean_prom[best_m + 1].item()) if best_m + 1 < cutoff_b else -1.0
                    nbr_m = (best_m + 1) if right_v >= left_v else (best_m - 1)
                    chosen = [best_m] + ([nbr_m] if (1 <= nbr_m < cutoff_b and max(left_v, right_v) > 0.04) else [])
                else:
                    half_w = 1 if self.geom.block_tokens >= 40 else 2
                    chosen = [m_i for m_i in range(best_m - half_w, best_m + half_w + 1) if 1 <= m_i < cutoff_b]
            else:
                # Multi-peak extraction (Section 7(2)) for multi-needle queries: up to `max_recall_peaks`
                # non-overlapping local peaks above `prominence_delta` (or second peak above 0.55*delta).
                half_w = 1 if self.geom.block_tokens >= 40 else 2
                min_sep = 2 * half_w + 1
                work = mean_prom.clone()
                chosen_set: set[int] = set()
                for pk_i in range(self.flags.max_recall_peaks):
                    pk_m = int(torch.argmax(work).item())
                    pk_v = float(work[pk_m].item())
                    req_thr = self.flags.prominence_delta if pk_i == 0 else (0.55 * self.flags.prominence_delta)
                    if pk_v < req_thr or pk_m < 1 or pk_m >= cutoff_b:
                        break
                    for m_i in range(pk_m - half_w, pk_m + half_w + 1):
                        if 1 <= m_i < cutoff_b:
                            chosen_set.add(m_i)
                    lo_s = max(0, pk_m - min_sep)
                    hi_s = min(m_blocks, pk_m + min_sep + 1)
                    work[lo_s:hi_s] = 0.0
                if not chosen_set:
                    return
                chosen = sorted(chosen_set)
            self.pinned_turn_blocks = chosen

            import time as _t_rec
            t_io0 = _t_rec.perf_counter()
            for early_l in range(0, target_layer):
                missing_m = [m_i for m_i in chosen if m_i not in self.staged_recall[early_l]]
                if missing_m:
                    offs = [self.cold_stripes[m_i].nvme_offsets[early_l] for m_i in missing_m]
                    blobs = self.nvme.pread_stripes_parallel(offs, NVME_STRIPE_BYTES)
                    for m_i, b_e in zip(missing_m, blobs):
                        self.staged_recall[early_l][m_i] = deserialize_stripe_fp16(
                            b_e, self.geom.num_kv_heads, self.geom.plane1_residual_tokens, self.geom.head_dim, self.device
                        )
                    if is_final_recall_turn:
                        self.recall_turn_cold_stripes_recalled += len(missing_m)
                    else:
                        self.distractor_cold_stripes_recalled += len(missing_m)
            self.nvme_stage_us_list.append((_t_rec.perf_counter() - t_io0) * 1e6)

        missing_tl = [m_i for m_i in chosen if m_i not in self.staged_recall[target_layer]]
        if missing_tl:
            import time as _t_rec2
            t_io1 = _t_rec2.perf_counter()
            offs_tl = [self.cold_stripes[m_i].nvme_offsets[target_layer] for m_i in missing_tl]
            blobs_tl = self.nvme.pread_stripes_parallel(offs_tl, NVME_STRIPE_BYTES)
            for m_i, b_t in zip(missing_tl, blobs_tl):
                self.staged_recall[target_layer][m_i] = deserialize_stripe_fp16(
                    b_t, self.geom.num_kv_heads, self.geom.plane1_residual_tokens, self.geom.head_dim, self.device
                )
            self.nvme_stage_us_list.append((_t_rec2.perf_counter() - t_io1) * 1e6)
            if is_final_recall_turn:
                self.recall_turn_cold_stripes_recalled += len(missing_tl)
            else:
                self.distractor_cold_stripes_recalled += len(missing_tl)

        cur_staged = sum(len(d) * NVME_STRIPE_BYTES for d in self.staged_recall)
        if cur_staged > self.peak_staged_bytes:
            self.peak_staged_bytes = cur_staged

    @torch.no_grad()
    def _sync_baseline_recall_for_layer(
        self,
        q_states: torch.Tensor,  # [1, H_q, q_len, d_h]
        layer_idx: int,
        is_final_recall_turn: bool,
    ) -> None:
        """
        Synchronous Layer-l recall for Quest-style `[k_min, k_max]` bounding-box routing
        (`quest_box_nvme`, 2048 B/stripe) and ArkVale-style uncentered single-centroid
        routing (`arkvale_centroid_nvme`, 1024 B/stripe) over the identical F_NOCACHE NVMe store.
        """
        import time as _time_b

        m_blocks = len(self.cold_stripes)
        if m_blocks <= 2:
            return
        q_len = q_states.shape[2]
        # Score using the last non-delimiter token(s) of the turn (or current decode token)
        if q_len > 1:
            keep_idx = [
                i for i, tid in enumerate(self.current_query_token_ids)
                if tid not in self.boundary_token_ids
            ]
            if not keep_idx:
                return
            q_use = q_states[0, :, keep_idx[-min(16, len(keep_idx)) :], :].float()
        else:
            q_use = q_states[0].float()

        qf = q_use.view(self.geom.num_kv_heads, self.gqa_groups, -1, self.geom.head_dim)
        t_sc0 = _time_b.perf_counter()
        if self.method == "quest_box_nvme":
            # Quest coordinate-wise upper bound: sum_d max(q_d * k_min_d, q_d * k_max_d)
            kmin_all = torch.stack(self.quest_kmin[layer_idx], dim=1).float()  # [H_kv, M, d_h]
            kmax_all = torch.stack(self.quest_kmax[layer_idx], dim=1).float()  # [H_kv, M, d_h]
            # [H_kv, G, q, M, d_h]
            prod_min = qf.unsqueeze(3) * kmin_all.unsqueeze(1).unsqueeze(2)
            prod_max = qf.unsqueeze(3) * kmax_all.unsqueeze(1).unsqueeze(2)
            box_scores = torch.maximum(prod_min, prod_max).sum(dim=-1)  # [H_kv, G, q, M]
            blk_scores = box_scores.amax(dim=(0, 1, 2))  # [M]
        else:
            # ArkVale-style uncentered single-centroid dot product
            cent_all = torch.stack(self.arkvale_centroids[layer_idx], dim=1).float()  # [H_kv, M, d_h]
            cent_scores = torch.einsum("grqd,gmd->grqm", qf, cent_all)
            blk_scores = cent_scores.amax(dim=(0, 1, 2))  # [M]

        blk_scores[0] = -1e9
        k_sel = min(self.flags.baseline_topk_blocks, m_blocks - 1)
        top_m = torch.topk(blk_scores, k=k_sel).indices.tolist()
        self.sketch_scoring_us_list.append((_time_b.perf_counter() - t_sc0) * 1e6)

        missing = [m_i for m_i in top_m if m_i not in self.staged_recall[layer_idx]]
        if missing:
            t_io = _time_b.perf_counter()
            offs = [self.cold_stripes[m_i].nvme_offsets[layer_idx] for m_i in missing]
            blobs = self.nvme.pread_stripes_parallel(offs, NVME_STRIPE_BYTES)
            for m_i, b_raw in zip(missing, blobs):
                self.staged_recall[layer_idx][m_i] = deserialize_stripe_fp16(
                    b_raw, self.geom.num_kv_heads, self.geom.plane1_residual_tokens, self.geom.head_dim, self.device
                )
            stall_us = (_time_b.perf_counter() - t_io) * 1e6
            self.exposed_sync_nvme_stall_us.append(stall_us)
            self.nvme_stage_us_list.append(stall_us)
            if is_final_recall_turn:
                self.recall_turn_cold_stripes_recalled += len(missing)
            else:
                self.distractor_cold_stripes_recalled += len(missing)

        cur_staged = sum(len(d) * NVME_STRIPE_BYTES for d in self.staged_recall)
        if cur_staged > self.peak_staged_bytes:
            self.peak_staged_bytes = cur_staged

    # ------------------------------------------------------------------------
    # Model-Agnostic Forward Pass (Qwen2, Granite, Llama, Mistral)
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def forward_tokens(
        self,
        token_ids: list[int],
        is_final_recall_turn: bool = False,
        allow_recall: bool = False,
        track_online_attn: bool = True,
        record_layer_timing: bool = False,
    ) -> torch.Tensor:
        import time as _time

        if record_layer_timing and self.device == "mps":
            torch.mps.synchronize()
        t_step0 = _time.perf_counter()

        q_len = len(token_ids)
        self.current_query_token_ids = token_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        pos_ids = torch.arange(
            self.logical_seq_len, self.logical_seq_len + q_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        self.enable_recall_now = allow_recall and (
            self.method in ("splitslab_full", "quest_box_nvme", "arkvale_centroid_nvme")
        )
        if self.enable_recall_now and len(self.cold_stripes) > 0 and q_len > 1:
            total_cold = len(self.cold_stripes) * self.geom.num_layers
            if is_final_recall_turn:
                self.recall_turn_cold_stripe_opportunities += total_cold
            else:
                self.distractor_cold_stripe_opportunities += total_cold

        backbone = self.model.model
        hidden_states = backbone.embed_tokens(input_ids)
        if self.embed_mult != 1.0:
            hidden_states = hidden_states * self.embed_mult
        position_embeddings = backbone.rotary_emb(hidden_states, pos_ids)
        cos, sin = position_embeddings

        per_layer_new_attn_mass: list[torch.Tensor] = []
        la_delta = self.flags.lookahead_layers
        entropy_check_layer = max(3, int(round(0.75 * self.geom.num_layers)))

        for l in range(self.geom.num_layers):
            if record_layer_timing and q_len == 1:
                if self.device == "mps":
                    torch.mps.synchronize()
                t_l0 = _time.perf_counter()

            layer_mod = backbone.layers[l]
            res_mult = float(getattr(layer_mod, "residual_multiplier", 1.0))

            # Zero-lookahead (`la_delta == 0`) synchronous check before Layer `l` updates its KV!
            if self.enable_recall_now and la_delta == 0 and l >= 3:
                if l == entropy_check_layer and not self.current_step_uncertain:
                    mid_logits = self.model.lm_head(backbone.norm(hidden_states[:, -1:, :]))[0, -1].float() / self.logits_div
                    mid_p = torch.softmax(mid_logits, dim=-1)
                    mid_h = float(-(mid_p * torch.log(mid_p.clamp(min=1e-12))).sum().item())
                    if mid_h > self.flags.entropy_gamma:
                        self.current_step_uncertain = True
                dum_q = torch.zeros((1, self.geom.num_q_heads, q_len, self.geom.head_dim), dtype=hidden_states.dtype, device=self.device)
                self.maybe_lookahead_recall_for_layer(hidden_states, dum_q, position_embeddings, l, is_final_recall_turn)

            residual = hidden_states
            normed = layer_mod.input_layernorm(hidden_states)

            attn = layer_mod.self_attn
            attn_scale = float(getattr(attn, "scaling", 1.0 / math.sqrt(float(self.geom.head_dim))))
            q_states = attn.q_proj(normed).view(1, q_len, self.geom.num_q_heads, self.geom.head_dim).transpose(1, 2)
            k_new = attn.k_proj(normed).view(1, q_len, self.geom.num_kv_heads, self.geom.head_dim).transpose(1, 2)
            v_new = attn.v_proj(normed).view(1, q_len, self.geom.num_kv_heads, self.geom.head_dim).transpose(1, 2)
            q_states, k_new = apply_rotary_pos_emb(q_states, k_new, cos, sin)

            if self.hbm_keys[l] is None:
                k_hbm, v_hbm = k_new, v_new
            else:
                k_hbm = torch.cat([self.hbm_keys[l], k_new], dim=2)
                v_hbm = torch.cat([self.hbm_vals[l], v_new], dim=2)
            self.hbm_keys[l] = k_hbm
            self.hbm_vals[l] = v_hbm

            if self.enable_recall_now and self.method in ("quest_box_nvme", "arkvale_centroid_nvme"):
                self._sync_baseline_recall_for_layer(q_states, l, is_final_recall_turn)

            if self.staged_recall[l]:
                rec_ks = [pair[0] for _, pair in sorted(self.staged_recall[l].items())]
                rec_vs = [pair[1] for _, pair in sorted(self.staged_recall[l].items())]
                past_hbm_len = k_hbm.shape[2] - q_len
                k_eff = torch.cat([k_hbm[:, :, :past_hbm_len, :], *rec_ks, k_hbm[:, :, past_hbm_len:, :]], dim=2)
                v_eff = torch.cat([v_hbm[:, :, :past_hbm_len, :], *rec_vs, v_hbm[:, :, past_hbm_len:, :]], dim=2)
            else:
                k_eff, v_eff = k_hbm, v_hbm

            k_rep = repeat_kv(k_eff, self.gqa_groups)
            v_rep = repeat_kv(v_eff, self.gqa_groups)

            kv_len = k_eff.shape[2]
            if q_len == 1:
                sdpa_mask = None
            else:
                past_len = kv_len - q_len
                sdpa_mask = torch.ones((1, 1, q_len, kv_len), dtype=torch.bool, device=self.device)
                sdpa_mask[:, :, :, past_len:] = torch.tril(
                    torch.ones((q_len, q_len), dtype=torch.bool, device=self.device)
                )

            attn_out = F.scaled_dot_product_attention(
                q_states, k_rep, v_rep, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False, scale=attn_scale
            )
            attn_out = attn_out.transpose(1, 2).reshape(1, q_len, -1).contiguous()
            hidden_states = residual + attn.o_proj(attn_out) * res_mult

            if track_online_attn and self.method in ("online_evict_actkv", "iterative_recompact"):
                q_sub_cnt = min(q_len, 12)
                q_tail_t = q_states[0, :, -q_sub_cnt:, :].float() * attn_scale
                k_hbm_rep = repeat_kv(k_hbm, self.gqa_groups)[0].float()
                scores = torch.matmul(q_tail_t, k_hbm_rep.transpose(-1, -2))
                if q_sub_cnt > 1:
                    p_l = k_hbm.shape[2] - q_sub_cnt
                    c_mask = torch.ones((q_sub_cnt, k_hbm.shape[2]), dtype=torch.bool, device=self.device)
                    c_mask[:, p_l:] = torch.tril(torch.ones((q_sub_cnt, q_sub_cnt), dtype=torch.bool, device=self.device))
                    scores = scores.masked_fill(~c_mask.unsqueeze(0), float("-inf"))
                probs = torch.softmax(scores, dim=-1).sum(dim=(0, 1))
                per_layer_new_attn_mass.append(probs)

            residual = hidden_states
            hidden_states = residual + layer_mod.mlp(layer_mod.post_attention_layernorm(hidden_states)) * res_mult

            if record_layer_timing and q_len == 1:
                if self.device == "mps":
                    torch.mps.synchronize()
                self.layer_compute_latencies_us.append((_time.perf_counter() - t_l0) * 1e6)

            # Asynchronous `la_delta`-layer lookahead trigger!
            if (
                self.enable_recall_now
                and self.method == "splitslab_full"
                and la_delta > 0
                and (l + la_delta) < self.geom.num_layers
            ):
                if l == entropy_check_layer and not self.current_step_uncertain:
                    t_eg0 = _time.perf_counter()
                    mid_logits = self.model.lm_head(backbone.norm(hidden_states[:, -1:, :]))[0, -1].float() / self.logits_div
                    mid_p = torch.softmax(mid_logits, dim=-1)
                    mid_h = float(-(mid_p * torch.log(mid_p.clamp(min=1e-12))).sum().item())
                    if mid_h > self.flags.entropy_gamma:
                        self.current_step_uncertain = True
                    self.entropy_gate_us_list.append((_time.perf_counter() - t_eg0) * 1e6)
                t_sk0 = _time.perf_counter()
                self.maybe_lookahead_recall_for_layer(
                    hidden_states, q_states, position_embeddings, l + la_delta, is_final_recall_turn
                )
                self.sketch_scoring_us_list.append((_time.perf_counter() - t_sk0) * 1e6)

        for idx, tid in enumerate(token_ids):
            self.hbm_positions.append(self.logical_seq_len + idx)
            self.hbm_token_ids.append(tid)
        self.logical_seq_len += q_len
        self.all_token_ids.extend(token_ids)

        if per_layer_new_attn_mass:
            step_mass = torch.stack(per_layer_new_attn_mass, dim=0).mean(dim=0)
            if self.cum_attn_scores is None:
                self.cum_attn_scores = step_mass
            else:
                padded = F.pad(self.cum_attn_scores * 0.92, (0, q_len), value=0.0)
                self.cum_attn_scores = padded + step_mass

        final_norm = backbone.norm(hidden_states[:, -1:, :])
        logits = (self.model.lm_head(final_norm)[:, -1, :].float() / self.logits_div)

        probs_out = torch.softmax(logits[0], dim=-1)
        entropy_val = float(-(probs_out * torch.log(probs_out.clamp(min=1e-12))).sum().item())
        self.current_step_uncertain = entropy_val > self.flags.entropy_gamma

        if record_layer_timing:
            if self.device == "mps":
                torch.mps.synchronize()
            dt_ms = (_time.perf_counter() - t_step0) * 1e3
            if q_len > 1:
                self.prefill_ttft_ms_list.append(dt_ms)
            else:
                self.decode_tpot_ms_list.append(dt_ms)
        return logits

    # ------------------------------------------------------------------------
    # Post-Turn Compaction & Physical NVMe Tiering Policies
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def apply_post_turn_tiering(
        self, shared_prefix_token_limit: int | None = None, is_dedup_probe: bool = False
    ) -> None:
        p0_toks = self.geom.plane0_anchor_tokens
        p1_toks = self.geom.plane1_residual_tokens
        blk_toks = self.geom.block_tokens
        loc_win = self.geom.local_window_tokens

        if self.method == "full_kv_flatcake":
            slab_toks = p1_toks  # Exact 64 KB/layer uncompacted slab across any GQA config!
            total_hbm = len(self.hbm_positions)
            safe_limit = max(0, total_hbm - loc_win)
            if is_dedup_probe and shared_prefix_token_limit is not None:
                safe_limit = min(safe_limit, shared_prefix_token_limit)
            while self.tiered_prefix_tokens + slab_toks <= safe_limit:
                b_start = self.tiered_prefix_tokens
                b_end = b_start + slab_toks
                prefix_bytes = struct.pack(f"<{b_end}I", *self.all_token_ids[:b_end])
                for l in range(self.geom.num_layers):
                    h_hex = hashlib.sha256(prefix_bytes + struct.pack("<H", l)).hexdigest()
                    blob = serialize_stripe_fp16(
                        self.hbm_keys[l][0, :, b_start:b_end, :],
                        self.hbm_vals[l][0, :, b_start:b_end, :],
                    )
                    self.nvme.put_immutable_stripe(
                        h_hex, blob, count_dedup=is_dedup_probe, count_waf_logical=not is_dedup_probe
                    )
                self.tiered_prefix_tokens += slab_toks
            self.nvme.sync_batch()
            return

        if self.method in ("splitslab_anchor_only", "splitslab_full", "quest_box_nvme", "arkvale_centroid_nvme"):
            stride_step = max(2, blk_toks // p0_toks)
            init_sinks = max(1, p0_toks // 4)
            while True:
                cur_hbm_len = len(self.hbm_positions)
                num_tiered_blocks = len(self.cold_stripes)
                hbm_untiered_start = num_tiered_blocks * p0_toks
                untiered_available = cur_hbm_len - hbm_untiered_start - loc_win
                if untiered_available < blk_toks:
                    break

                b_hbm_start = hbm_untiered_start
                b_hbm_end = b_hbm_start + blk_toks
                logical_end_pos = self.hbm_positions[b_hbm_end - 1] + 1
                if is_dedup_probe and shared_prefix_token_limit is not None and logical_end_pos > shared_prefix_token_limit:
                    break

                prefix_token_bytes = struct.pack(f"<{logical_end_pos}I", *self.all_token_ids[:logical_end_pos])
                block_canon_hash = hashlib.sha256(prefix_token_bytes).hexdigest()

                cand_scores: list[tuple[float, int, int]] = []
                for rel_i in range(blk_toks):
                    tid = self.hbm_token_ids[b_hbm_start + rel_i]
                    if rel_i < init_sinks:
                        sc = 100.0
                    elif tid in self.boundary_token_ids:
                        sc = 50.0
                    elif rel_i % stride_step == 0:
                        sc = 10.0
                    else:
                        sc = 0.0
                    cand_scores.append((sc, -rel_i, rel_i))
                cand_scores.sort(reverse=True)
                anchor_list = sorted([x[2] for x in cand_scores[:p0_toks]])
                anchor_rel = torch.tensor(anchor_list, dtype=torch.long, device=self.device)
                keep_mask = torch.zeros(blk_toks, dtype=torch.bool, device=self.device)
                keep_mask[anchor_rel] = True
                res_rel = torch.nonzero(~keep_mask, as_tuple=False).flatten()

                res_positions = [self.hbm_positions[b_hbm_start + int(i)] for i in res_rel.tolist()]
                layer_offsets: list[int] = []
                layer_sketches: list[bytes] = []

                for l in range(self.geom.num_layers):
                    k_full = self.hbm_keys[l]
                    v_full = self.hbm_vals[l]
                    k_blk = k_full[:, :, b_hbm_start:b_hbm_end, :]
                    v_blk = v_full[:, :, b_hbm_start:b_hbm_end, :]

                    k_anc = k_blk[:, :, anchor_rel, :]
                    v_anc = v_blk[:, :, anchor_rel, :]
                    k_res_4d = k_blk[:, :, res_rel, :]
                    v_res = v_blk[0, :, res_rel, :]

                    stripe_hash = hashlib.sha256(
                        block_canon_hash.encode("ascii") + struct.pack("<H", l)
                    ).hexdigest()
                    stripe_blob = serialize_stripe_fp16(k_res_4d[0], v_res)
                    off, _ = self.nvme.put_immutable_stripe(
                        stripe_hash, stripe_blob, count_dedup=is_dedup_probe, count_waf_logical=not is_dedup_probe
                    )
                    layer_offsets.append(off)

                    if self.method == "quest_box_nvme":
                        self.quest_kmin[l].append(k_res_4d[0].amin(dim=1).half())
                        self.quest_kmax[l].append(k_res_4d[0].amax(dim=1).half())
                        layer_sketches.append(b"")
                    elif self.method == "arkvale_centroid_nvme":
                        self.arkvale_centroids[l].append(k_res_4d[0].float().mean(dim=1).half())
                        layer_sketches.append(b"")
                    else:
                        k_res_pure = self._remove_rope_k_bias(k_res_4d, res_positions, l)
                        sk_bytes = pack_subspace_sketch(k_res_pure, self.P_subspaces[l], self.sketch_spec)
                        layer_sketches.append(sk_bytes)

                    self.hbm_keys[l] = torch.cat(
                        [k_full[:, :, :b_hbm_start, :], k_anc, k_full[:, :, b_hbm_end:, :]], dim=2
                    )
                    self.hbm_vals[l] = torch.cat(
                        [v_full[:, :, :b_hbm_start, :], v_anc, v_full[:, :, b_hbm_end:, :]], dim=2
                    )

                anc_pos = [self.hbm_positions[b_hbm_start + int(i)] for i in anchor_rel.tolist()]
                anc_tids = [self.hbm_token_ids[b_hbm_start + int(i)] for i in anchor_rel.tolist()]
                self.hbm_positions = self.hbm_positions[:b_hbm_start] + anc_pos + self.hbm_positions[b_hbm_end:]
                self.hbm_token_ids = self.hbm_token_ids[:b_hbm_start] + anc_tids + self.hbm_token_ids[b_hbm_end:]

                self.cold_stripes.append(
                    ColdStripeRecord(
                        block_idx=num_tiered_blocks,
                        canonical_hash=block_canon_hash,
                        nvme_offsets=layer_offsets,
                        sketch_bytes=layer_sketches,
                    )
                )
            self.nvme.sync_batch()
            return

        if self.method in ("online_evict_actkv", "iterative_recompact"):
            cur_len = len(self.hbm_positions)
            if cur_len <= loc_win + 16:
                return
            evictable_len = cur_len - loc_win
            hist_logical = max(0, self.logical_seq_len - loc_win)
            target_hist_keep = max(8, int(round(0.20 * hist_logical)))

            if evictable_len > target_hist_keep:
                scores_evictable = self.cum_attn_scores[:evictable_len].clone()
                scores_evictable[:4] = 1e9
                keep_hist_idx = torch.topk(scores_evictable, k=target_hist_keep).indices.sort().values
                local_idx = torch.arange(evictable_len, cur_len, device=self.device)
                full_keep_idx = torch.cat([keep_hist_idx, local_idx], dim=0)

                for l in range(self.geom.num_layers):
                    self.hbm_keys[l] = self.hbm_keys[l][:, :, full_keep_idx, :].contiguous()
                    self.hbm_vals[l] = self.hbm_vals[l][:, :, full_keep_idx, :].contiguous()

                self.cum_attn_scores = self.cum_attn_scores[full_keep_idx].contiguous()
                self.hbm_positions = [self.hbm_positions[int(i)] for i in full_keep_idx.tolist()]
                self.hbm_token_ids = [self.hbm_token_ids[int(i)] for i in full_keep_idx.tolist()]

            slab_toks = p1_toks  # Exact 64 KB slab size for this model!
            hist_kept_len = len(self.hbm_positions) - loc_win
            if is_dedup_probe and shared_prefix_token_limit is not None:
                hist_kept_len = sum(1 for p in self.hbm_positions[:hist_kept_len] if p < shared_prefix_token_limit)

            num_slabs = hist_kept_len // slab_toks
            for s_idx in range(num_slabs):
                s_lo = s_idx * slab_toks
                s_hi = s_lo + slab_toks
                pos_tuple = self.hbm_positions[s_lo:s_hi]
                pos_sig = struct.pack(f"<{slab_toks}I", *pos_tuple)

                if self.method == "online_evict_actkv":
                    if is_dedup_probe or s_idx not in self.compacted_slab_sigs:
                        for l in range(self.geom.num_layers):
                            blob = serialize_stripe_fp16(
                                self.hbm_keys[l][0, :, s_lo:s_hi, :],
                                self.hbm_vals[l][0, :, s_lo:s_hi, :],
                            )
                            slab_hash = hashlib.sha256(pos_sig + struct.pack("<H", l) + blob[:128]).hexdigest()
                            self.nvme.put_immutable_stripe(
                                slab_hash, blob, count_dedup=is_dedup_probe, count_waf_logical=not is_dedup_probe
                            )
                        if not is_dedup_probe:
                            self.compacted_slab_sigs[s_idx] = pos_sig

                elif self.method == "iterative_recompact":
                    if is_dedup_probe:
                        for l in range(self.geom.num_layers):
                            blob = serialize_stripe_fp16(
                                self.hbm_keys[l][0, :, s_lo:s_hi, :],
                                self.hbm_vals[l][0, :, s_lo:s_hi, :],
                            )
                            slab_hash = hashlib.sha256(pos_sig + struct.pack("<H", l) + blob[:128]).hexdigest()
                            self.nvme.put_immutable_stripe(
                                slab_hash, blob, count_dedup=True, count_waf_logical=False
                            )
                    else:
                        prev_sig = self.compacted_slab_sigs.get(s_idx)
                        if prev_sig is None:
                            offs = []
                            for l in range(self.geom.num_layers):
                                blob = serialize_stripe_fp16(
                                    self.hbm_keys[l][0, :, s_lo:s_hi, :],
                                    self.hbm_vals[l][0, :, s_lo:s_hi, :],
                                )
                                slab_hash = hashlib.sha256(pos_sig + struct.pack("<H", l)).hexdigest()
                                off, _ = self.nvme.put_immutable_stripe(
                                    slab_hash, blob, count_dedup=False, count_waf_logical=True
                                )
                                offs.append(off)
                            self.compacted_slab_offsets[s_idx] = offs
                            self.compacted_slab_sigs[s_idx] = pos_sig
                        elif prev_sig != pos_sig:
                            new_offs = []
                            for l in range(self.geom.num_layers):
                                blob = serialize_stripe_fp16(
                                    self.hbm_keys[l][0, :, s_lo:s_hi, :],
                                    self.hbm_vals[l][0, :, s_lo:s_hi, :],
                                )
                                new_off = self.nvme.rewrite_mutable_page(self.compacted_slab_offsets[s_idx][l], blob)
                                new_offs.append(new_off)
                            self.compacted_slab_offsets[s_idx] = new_offs
                            self.compacted_slab_sigs[s_idx] = pos_sig
            self.nvme.sync_batch()

    @torch.no_grad()
    def rollback_last_tokens(self, n_toks: int) -> None:
        if n_toks <= 0:
            return
        for l in range(self.geom.num_layers):
            if self.hbm_keys[l] is not None:
                self.hbm_keys[l] = self.hbm_keys[l][:, :, :-n_toks, :].contiguous()
                self.hbm_vals[l] = self.hbm_vals[l][:, :, :-n_toks, :].contiguous()
        del self.hbm_positions[-n_toks:]
        del self.hbm_token_ids[-n_toks:]
        del self.all_token_ids[-n_toks:]
        self.logical_seq_len -= n_toks

    @torch.no_grad()
    def greedy_generate(
        self,
        prompt_token_ids: list[int],
        tokenizer: AutoTokenizer,
        stop_token_ids: set[int],
        max_new_tokens: int = 52,
        is_final_recall_turn: bool = True,
        record_layer_timing: bool = False,
        use_live_entropy_gate_on_prefill: bool = False,
    ) -> str:
        import time as _t_gen

        self.clear_staged_recall()
        self.current_step_uncertain = not use_live_entropy_gate_on_prefill
        if record_layer_timing and self.device == "mps":
            torch.mps.synchronize()
        t_pf0 = _t_gen.perf_counter()
        logits = self.forward_tokens(
            prompt_token_ids,
            is_final_recall_turn=is_final_recall_turn,
            allow_recall=True,
            track_online_attn=False,
        )
        if self.method == "splitslab_full" and self.pinned_turn_blocks is not None:
            self.rollback_last_tokens(len(prompt_token_ids))
            logits = self.forward_tokens(
                prompt_token_ids,
                is_final_recall_turn=is_final_recall_turn,
                allow_recall=False,
                track_online_attn=False,
            )
        if record_layer_timing:
            if self.device == "mps":
                torch.mps.synchronize()
            self.prefill_ttft_ms_list.append((_t_gen.perf_counter() - t_pf0) * 1e3)

        gen_ids: list[int] = []
        for step_idx in range(max_new_tokens):
            next_id = int(torch.argmax(logits[0]).item())
            if next_id in stop_token_ids:
                break
            gen_ids.append(next_id)
            logits = self.forward_tokens(
                [next_id],
                is_final_recall_turn=is_final_recall_turn,
                allow_recall=(self.pinned_turn_blocks is None) or (self.method in ("quest_box_nvme", "arkvale_centroid_nvme")),
                track_online_attn=False,
                record_layer_timing=record_layer_timing,
            )

        self.clear_staged_recall()
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
