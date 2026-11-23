#!/usr/bin/env python3
"""
SplitSlab Phase 0/1 End-to-End Pilot (`RecallRegret-Pilot`)
===========================================================
Evaluates SplitSlab on local Apple Silicon (`mps`, `float16`) + internal Apple NVMe SSD
using cached `Qwen/Qwen2.5-7B-Instruct` across 20 multi-turn dual-agent trajectories.

Benchmarks 5 KV-cache management & offloading architectures head-to-head:
  1. `full_kv_flatcake`      : 100% HBM KV + Write-Once Uncompacted 64KB NVMe Offload (Oracle)
  2. `online_evict_actkv`    : Irreversible Online Observation-Window Eviction (@ ~22% HBM)
  3. `iterative_recompact`   : Iterative Multi-Turn Re-Compaction + 64KB NVMe Slab RMW Updates
  4. `splitslab_anchor_only` : SplitSlab Canonical Plane-0 Skeleton Only (No Plane-1 Recall)
  5. `splitslab_full`        : SplitSlab Full (Canonical Two-Plane WAF=1.0 NVMe Slabs +
                               Strict 64-Byte Bias-Centered W_Q/W_K SVD Sub-Cluster Sketch +
                               2-Layer Lookahead Entropy-Gated Residual Recall)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

# ============================================================================
# Storage & Block Geometry Constants (Exact 64 KB Stripe Alignment on Qwen2.5-7B)
# ============================================================================
# Qwen2.5-7B-Instruct: 28 layers, 28 Q heads, 4 KV heads, head_dim = 128, fp16 (2B).
# 1 token KV per layer = 2 * 4 * 128 * 2 bytes = 2,048 bytes (2 KB).
# Logical Block size M = 40 tokens:
#   - Plane 0 (Hot HBM Anchor Skeleton, 20%) =  8 tokens / block
#   - Plane 1 (Cold NVMe Residual Slab,  80%) = 32 tokens / block
#     32 tokens * 2,048 B/token = 65,536 bytes = EXACT 64.0 KB per layer stripe!
BLOCK_TOKENS = 40
PLANE0_ANCHOR_TOKENS = 8
PLANE1_RESIDUAL_TOKENS = 32
LOCAL_WINDOW_TOKENS = 48  # Uncompacted active local tail kept in HBM across all compacted methods
NVME_PAGE_BYTES = 65536   # 64 KB NVMe GDS / flash alignment unit

# Strict 64-byte binary Subspace Routing Sketch layout per 64 KB Cold Stripe:
#   - 2 x float16 sub-cluster subspace L2 norms (4 bytes)
#   - G = 2 sub-clusters (16 tokens each) x 4 KV heads x 60 W_Q/W_K SVD dims = 480 sign bits
#     bit-packed via `np.packbits` into 60 uint8 bytes (60 bytes)
#   Total = 4 + 60 = 64 bytes EXACT (1,024x compression of 65,536-byte NVMe stripe!).
SKETCH_HEADER_STRUCT = struct.Struct("<2e")
SKETCH_BYTES = 64
assert SKETCH_HEADER_STRUCT.size + 60 == SKETCH_BYTES


# ============================================================================
# 20 Realistic Multi-Turn Delayed-Recall / Compaction-Regret Agent Trajectories
# ============================================================================
@dataclass
class DelayedRecallSpec:
    traj_id: str
    domain: str
    target_module_idx: int
    module_header: str
    const_name: str
    hex_code: str
    param_name: str
    param_val: str
    extra_clause: str
    incident_symptom: str


TRAJECTORY_SPECS: list[DelayedRecallSpec] = [
    DelayedRecallSpec(
        "RR-01", "tls_handshake", 3,
        "module tls_compat_table", "LEGACY_RENEG_ABORT_HEX", "0x7F3A",
        "teardown_grace_ms", "4350", "cipher_fallback=ECDHE_RSA_AES256",
        "client renegotiation handshake teardown panic",
    ),
    DelayedRecallSpec(
        "RR-02", "raft_wal", 4,
        "module raft_wal_integrity", "WAL_CRC_MISMATCH_TRAP", "0x91B4",
        "quorum_lease_ms", "2780", "segment_prealloc_mb=64",
        "follower log append checksum verification fault",
    ),
    DelayedRecallSpec(
        "RR-03", "nvme_rdma", 5,
        "module nvmeof_rdma_transport", "QP_DRAIN_STALL_CODE", "0x4E2C",
        "cq_moderation_us", "185", "max_sge_segments=16",
        "RDMA queue-pair completion drain timeout",
    ),
    DelayedRecallSpec(
        "RR-04", "grpc_http2", 3,
        "module h2_flow_control", "PING_STRIKE_VIOLATION_HEX", "0x3C09",
        "goaway_drain_ms", "6120", "initial_stream_win_kb=512",
        "HTTP/2 keepalive ping flood disconnect",
    ),
    DelayedRecallSpec(
        "RR-05", "lsm_compaction", 6,
        "module sst_tombstone_gc", "RANGE_DEL_OVERFLOW_HEX", "0x8A5D",
        "subcompaction_cap", "14", "bloom_bits_per_key=10",
        "L2 SSTable range-deletion tombstone surge",
    ),
    DelayedRecallSpec(
        "RR-06", "oauth_jwks", 4,
        "module jwks_key_rotator", "STALE_KID_SIGNATURE_ERR", "0x5B71",
        "clock_leeway_sec", "415", "jwks_refresh_backoff_ms=900",
        "JWT bearer signature verification failure after key rollover",
    ),
    DelayedRecallSpec(
        "RR-07", "cni_vxlan", 5,
        "module overlay_vxlan_fdb", "FDB_ENCAP_MISS_HEX", "0x6D18",
        "arp_proxy_hold_ms", "3290", "vxlan_udp_dst_port=8472",
        "cross-subnet pod VXLAN tunnel neighbor miss",
    ),
    DelayedRecallSpec(
        "RR-08", "kafka_rebalance", 3,
        "module consumer_coordinator", "EPOCH_FENCE_REJECT_HEX", "0x2F84",
        "rebalance_grace_ms", "7450", "offset_commit_codec=zstd",
        "partition assignment generation fence rejection",
    ),
    DelayedRecallSpec(
        "RR-09", "jemalloc_arena", 6,
        "module allocator_dirty_decay", "ARENA_PURGE_ABORT_HEX", "0xB319",
        "muzzy_decay_ms", "5640", "tcache_max_class_kb=32",
        "background madvise page purge contention stall",
    ),
    DelayedRecallSpec(
        "RR-10", "postgres_ssi", 4,
        "module serializable_ssi_lock", "RW_ANTIDEP_CYCLE_HEX", "0x74C6",
        "predicate_lock_slots", "384", "tuple_xmin_horizon_sec=90",
        "serializable snapshot isolation rw-antidependency abort",
    ),
    DelayedRecallSpec(
        "RR-11", "ebpf_verifier", 5,
        "module bpf_insn_verifier", "STATE_PRUNE_LIMIT_HEX", "0x9E42",
        "stack_frame_cap_bytes", "448", "jit_harden_mode=2",
        "XDP packet filter eBPF verifier state explosion",
    ),
    DelayedRecallSpec(
        "RR-12", "redis_cluster", 3,
        "module cluster_slot_migrator", "ASK_REDIRECT_STALL_HEX", "0x1D6F",
        "migrate_socket_ms", "2910", "gossip_ping_interval_ms=1000",
        "live hash-slot resharding ASK redirect hang",
    ),
    DelayedRecallSpec(
        "RR-13", "pcie_p2p_dma", 6,
        "module pcie_acs_aperture", "BAR_APERTURE_FAULT_HEX", "0xC83A",
        "tlp_payload_bytes", "512", "relaxed_ordering_enable=1",
        "peer-to-peer GPU-NVMe DMA BAR aperture fault",
    ),
    DelayedRecallSpec(
        "RR-14", "envoy_outlier", 4,
        "module upstream_circuit_breaker", "CONSEC_5XX_EJECT_HEX", "0x47E5",
        "base_eject_ms", "8350", "panic_threshold_pct=50",
        "upstream canary host consecutive-5xx outlier ejection",
    ),
    DelayedRecallSpec(
        "RR-15", "zstd_dict", 5,
        "module frame_dict_decoder", "DICT_MAGIC_CORRUPT_HEX", "0xA62B",
        "window_log_cap", "23", "checksum_verify_flag=true",
        "compressed payload dictionary header mismatch",
    ),
    DelayedRecallSpec(
        "RR-16", "chrony_ntp", 3,
        "module ptp_clock_discipline", "MAX_SLEW_CLAMP_HEX", "0x53D9",
        "poll_exponent_max", "9", "stratum_weight_ppm=25",
        "monotonic clock discipline slew clamp fault",
    ),
    DelayedRecallSpec(
        "RR-17", "bgp_reflector", 6,
        "module bgp_route_reflector", "ORIGINATOR_LOOP_HEX", "0x8F17",
        "hold_timer_sec", "165", "graceful_restart_sec=120",
        "iBGP route-reflector cluster-list loop drop",
    ),
    DelayedRecallSpec(
        "RR-18", "io_uring_sqpoll", 4,
        "module uring_sqpoll_engine", "CQE_RING_OVERFLOW_HEX", "0x62A4",
        "sq_thread_idle_ms", "1920", "fixed_buffers_count=128",
        "io_uring completion queue overflow under burst IOPS",
    ),
    DelayedRecallSpec(
        "RR-19", "nccl_allreduce", 5,
        "module nccl_ring_watchdog", "LL128_PROTO_ABORT_HEX", "0xD458",
        "watchdog_abort_sec", "235", "buffsize_mb=8",
        "multi-rank ring AllReduce LL128 protocol stall",
    ),
    DelayedRecallSpec(
        "RR-20", "ceph_bluestore", 3,
        "module bluestore_deferred_io", "BLOB_CSUM_VERIFY_HEX", "0x39F1",
        "min_alloc_kb", "96", "rocksdb_cache_mb=2048",
        "OSD BlueStore deferred write extent checksum fault",
    ),
]


DISTRACTOR_QA_TURNS: list[tuple[str, str]] = [
    (
        "Turn 3 Task: Write a compact formula for exponential retry backoff with capped delay and symmetric 10% jitter.",
        "Use `delay_ms = min(cap_ms, base_ms * (2 ** attempt)) * (0.9 + 0.2 * rand_u01)` so retries spread evenly within +/-10% without exceeding `cap_ms`.",
    ),
    (
        "Turn 4 Task: How should our cursor pagination encoder bind `page_size` and `last_id` to prevent parameter tampering?",
        "Serialize `payload = f'{last_id}:{page_size}:{exp_ts}'` in URL-safe Base64 and append a 16-byte truncated `HMAC-SHA256(secret, payload)` tag verified in constant time.",
    ),
    (
        "Turn 5 Task: Recommend Prometheus histogram bucket boundaries in milliseconds for an internal RPC service targeting p99 < 50ms.",
        "Configure `[0.5, 1, 2, 5, 10, 20, 35, 50, 75, 100, 250, 500]` ms so the 20ms-75ms SLO region has dense resolution while bounding cardinality to 12 buckets.",
    ),
    (
        "Turn 6 Task: What connection pool invariants prevent idle socket leaks behind an L4 TCP load balancer with a 300s idle reset?",
        "Set `conn_max_idle_time = 180s` (< 300s LB timeout), enable `TCP_KEEPINTVL = 30s`, and cap `min_idle_conns` to 25% of `max_open_conns`.",
    ),
    (
        "Turn 7 Task: Give a lock-free token-bucket refill rule using integer nanoseconds to avoid floating-point drift.",
        "Track `tokens_nano` up to `burst * 1_000_000_000`; on arrival at `now_ns`, add `(now_ns - last_ns) * rate_per_sec` via atomic CAS before subtracting `1_000_000_000`.",
    ),
    (
        "Turn 8 Task: How should `/readyz` differ from `/healthz` during graceful SIGTERM container drain in Kubernetes?",
        "`/readyz` must immediately return HTTP 503 on SIGTERM so kube-proxy removes the pod from Endpoints, whereas `/healthz` stays HTTP 200 until in-flight requests finish.",
    ),
    (
        "Turn 9 Task: Specify a compact JSON structured log schema for request tracing without high-cardinality label explosion.",
        "Emit `{'ts_ns', 'lvl', 'trace_id', 'route_template', 'status', 'dur_ms'}`—always logging `route_template` (`/v1/items/:id`) rather than raw URL paths in metric tags.",
    ),
    (
        "Turn 10 Task: What bounds check is needed before slicing a length-prefixed frame `[u32_be_len | body]` from a socket ring buffer?",
        "Reject frames where `u32_be_len > MAX_FRAME_BYTES` prior to allocation, and wait until `readable_bytes >= 4 + u32_be_len` before advancing the consumer tail.",
    ),
]


def build_dual_agent_trajectory(spec: DelayedRecallSpec) -> dict[str, Any]:
    modules: list[str] = []
    for m_idx in range(16):
        if m_idx == spec.target_module_idx:
            mod_text = (
                f"[{spec.module_header}] subsystem={spec.domain} "
                f"error_trap: {spec.const_name} = {spec.hex_code}, "
                f"{spec.param_name} = {spec.param_val}, {spec.extra_clause}, "
                f"audit_level=warn, auto_recover=enabled.\n"
            )
        elif m_idx == 0:
            mod_text = (
                "[module service_router_00] ingress_port=8080, worker_threads=32, "
                "header_read_timeout_ms=2500, keepalive_reqs=1000, tls_min_ver=1.3, "
                "compression_level=4, access_log=async_ring.\n"
            )
        elif m_idx == 15:
            mod_text = (
                "[module telemetry_sink_15] otlp_grpc_port=4317, batch_export_sz=512, "
                "flush_interval_ms=5000, queue_capacity=8192, sampler_ratio=0.05, "
                "metric_temporality=delta, drop_on_full=true.\n"
            )
        else:
            mod_text = (
                f"[module core_worker_{m_idx:02d}] shard_group={m_idx % 4}, "
                f"queue_depth={128 + m_idx * 16}, batch_timeout_ms={40 + m_idx * 5}, "
                f"checksum_mode=xxhash64, max_inflight={32 + m_idx * 2}, "
                f"buffer_slab_kb={64 * ((m_idx % 3) + 1)}, status=nominal.\n"
            )
        modules.append(mod_text)

    shared_repo_turn1 = (
        "<|im_start|>system\nYou are a principal systems debugging assistant analyzing a repository configuration snapshot. "
        "Answer concisely and accurately.<|im_end|>\n"
        "<|im_start|>user\n[Turn 1 — Shared Repository Snapshot]\n"
        + "".join(modules)
        + "Acknowledge receipt of the 16-module repository snapshot.<|im_end|>\n"
        "<|im_start|>assistant\nRepository snapshot (16 modules) indexed and ready.<|im_end|>\n"
    )

    turn2_agent_a = (
        "<|im_start|>user\n[Turn 2] From `core_worker_01`, what are `queue_depth` and `batch_timeout_ms`?<|im_end|>\n"
        "<|im_start|>assistant\nIn `core_worker_01`, `queue_depth=144` and `batch_timeout_ms=45`.<|im_end|>\n"
    )
    turn2_agent_b = (
        "<|im_start|>user\n[Turn 2] From `telemetry_sink_15`, what are `otlp_grpc_port` and `batch_export_sz`?<|im_end|>\n"
        "<|im_start|>assistant\nIn `telemetry_sink_15`, `otlp_grpc_port=4317` and `batch_export_sz=512`.<|im_end|>\n"
    )

    distractor_turns = [
        f"<|im_start|>user\n{q_str}<|im_end|>\n<|im_start|>assistant\n{a_str}<|im_end|>\n"
        for q_str, a_str in DISTRACTOR_QA_TURNS
    ]

    turn11 = (
        "<|im_start|>user\n[Turn 11] All helper utilities passed unit tests. Ready for production canary check.<|im_end|>\n"
        "<|im_start|>assistant\nUnderstood. Standing by for canary telemetry and incident logs.<|im_end|>\n"
    )

    turn12_prompt = (
        f"<|im_start|>user\n[Turn 12 — URGENT PRODUCTION INCIDENT]\n"
        f"Canary crash dump reports `{spec.incident_symptom}` with raw trap code `{spec.hex_code}`.\n"
        f"Look back at the Turn 1 repository snapshot and report the EXACT error constant name "
        f"and the EXACT `{spec.param_name}` numeric value mapped to `{spec.hex_code}`.\n"
        f"Format strictly as: `<CONSTANT_NAME> | <{spec.param_name.upper()}_VALUE>`<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return {
        "spec": spec,
        "turn1_shared": shared_repo_turn1,
        "turn2_agent_a": turn2_agent_a,
        "turn2_agent_b": turn2_agent_b,
        "distractor_turns": distractor_turns,
        "turn11": turn11,
        "turn12_prompt": turn12_prompt,
    }


# ============================================================================
# Physical Apple NVMe SSD Binary Slab Store (Measures Exact Physical Bytes & WAF)
# ============================================================================
class PhysicalNVMeSlabStore:
    def __init__(self, root_dir: Path, sync_writes: bool = True):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.sync_writes = sync_writes

        self.slab_path = self.root_dir / "plane1_stripes.slab"
        self.fd = os.open(str(self.slab_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        self.write_cursor = 0

        self.physical_bytes_written = 0
        self.logical_unique_bytes_stored = 0
        self.physical_bytes_read = 0
        self.pread_calls = 0
        self.write_time_ms = 0.0
        self.read_time_us_samples: list[float] = []

        self.cas_index: dict[str, int] = {}
        self.dedup_queries = 0
        self.dedup_hits = 0

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    @staticmethod
    def _align_up(n: int, align: int = NVME_PAGE_BYTES) -> int:
        return ((n + align - 1) // align) * align

    def put_immutable_stripe(
        self, canonical_hash: str, payload: bytes, count_dedup: bool = False, count_waf_logical: bool = True
    ) -> tuple[int, bool]:
        if count_dedup:
            self.dedup_queries += 1
        if canonical_hash in self.cas_index:
            if count_dedup:
                self.dedup_hits += 1
            return self.cas_index[canonical_hash], True

        if count_dedup and not count_waf_logical:
            return -1, False

        aligned_len = self._align_up(len(payload), NVME_PAGE_BYTES)
        buf = payload if len(payload) == aligned_len else (payload + b"\x00" * (aligned_len - len(payload)))

        offset = self.write_cursor
        t0 = time.perf_counter()
        os.pwrite(self.fd, buf, offset)
        if self.sync_writes:
            os.fsync(self.fd)
        self.write_time_ms += (time.perf_counter() - t0) * 1000.0

        self.write_cursor += aligned_len
        self.physical_bytes_written += aligned_len
        if count_waf_logical:
            self.logical_unique_bytes_stored += len(payload)
        self.cas_index[canonical_hash] = offset
        return offset, False

    def rewrite_mutable_page(self, offset: int, new_payload: bytes) -> int:
        aligned_len = self._align_up(len(new_payload), NVME_PAGE_BYTES)
        t0 = time.perf_counter()
        if 0 <= offset < self.write_cursor:
            _ = os.pread(self.fd, aligned_len, offset)
            self.physical_bytes_read += aligned_len
        buf = new_payload if len(new_payload) == aligned_len else (new_payload + b"\x00" * (aligned_len - len(new_payload)))
        new_off = self.write_cursor
        os.pwrite(self.fd, buf, new_off)
        if self.sync_writes:
            os.fsync(self.fd)
        self.write_time_ms += (time.perf_counter() - t0) * 1000.0
        self.write_cursor += aligned_len
        self.physical_bytes_written += aligned_len
        return new_off

    def pread_stripe(self, offset: int, nbytes: int = NVME_PAGE_BYTES) -> bytes:
        t0 = time.perf_counter()
        raw = os.pread(self.fd, nbytes, offset)
        dt_us = (time.perf_counter() - t0) * 1e6
        self.read_time_us_samples.append(dt_us)
        self.physical_bytes_read += len(raw)
        self.pread_calls += 1
        return raw

    @property
    def waf(self) -> float:
        if self.logical_unique_bytes_stored == 0:
            return 1.0
        return self.physical_bytes_written / float(self.logical_unique_bytes_stored)


# ============================================================================
# Static Head-Space W_Q/W_K SVD Subspace + Strict 64-Byte Sub-Cluster Sketch
# ============================================================================
def build_static_subspaces(model: AutoModelForCausalLM, device: str) -> torch.Tensor:
    """
    Computes static, model-only top-60 head-space query-key interaction singular directions
    P[28, 4, 128, 60] weighted by 16-token RoPE phase coherence (100% deterministic across fleet!).
    G = 2 sub-clusters x (4 KV heads * 60 bits = 240 bits = 30B) = 60B + 2 x fp16 norms (4B) = 64B!
    """
    inv_freq = model.model.rotary_emb.inv_freq.float().cpu()
    t_pos = torch.arange(16, dtype=torch.float32).unsqueeze(1) * inv_freq.unsqueeze(0)
    coh = torch.sqrt(torch.cos(t_pos).mean(0) ** 2 + torch.sin(t_pos).mean(0) ** 2)
    coh_128 = torch.cat([coh, coh], dim=0)

    num_layers = model.config.num_hidden_layers
    P60 = torch.zeros((num_layers, 4, 128, 60), dtype=torch.float32)
    for l in range(num_layers):
        attn = model.model.layers[l].self_attn
        Wq = attn.q_proj.weight.detach().float().cpu().view(4, 7, 128, -1)
        Wk = attn.k_proj.weight.detach().float().cpu().view(4, 128, -1)
        for g in range(4):
            cov_k = torch.matmul(Wk[g], Wk[g].T)
            cov_q = torch.einsum("rqh,rkh->qk", Wq[g], Wq[g]) / 7.0
            energy = (coh_128.unsqueeze(1) * (cov_k @ cov_q + cov_q @ cov_k)) * coh_128.unsqueeze(0)
            _, vecs = torch.linalg.eigh(0.5 * (energy + energy.T))
            P60[l, g] = vecs[:, -60:].flip(-1)
    return P60.to(device=device, dtype=torch.float32)


def pack_64byte_sketch(
    k_res_pure: torch.Tensor,  # [4, 32, 128] fp32 (post-RoPE(k_proj.bias) removal!)
    P_layer: torch.Tensor,     # [4, 128, 60] fp32
) -> bytes:
    """
    Packs a 32-token bias-centered Cold Residual stripe into a STRICT 64-byte binary sketch:
      - G = 2 sub-clusters (16 tokens each) x 240 sign bits (60 bytes) + 2 x fp16 norms (4 bytes) = 64 bytes!
    """
    kc = k_res_pure.view(4, 2, 16, 128)
    tok_w = torch.softmax(kc.norm(dim=-1) * 0.35, dim=-1).unsqueeze(-1)
    mu = (kc * tok_w).sum(dim=2)  # [4_kv, 2_clust, 128]
    proj = torch.einsum("gsc,gcb->gsb", mu, P_layer)  # [4_kv, 2_clust, 60]
    proj_c = proj.permute(1, 0, 2).reshape(2, 240)
    norms_fp16 = proj_c.norm(dim=-1).half().cpu().tolist()
    packed_bits = np.packbits((proj_c >= 0).to(torch.uint8).cpu().numpy().reshape(-1)).tobytes()
    raw = SKETCH_HEADER_STRUCT.pack(*norms_fp16) + packed_bits
    assert len(raw) == SKETCH_BYTES
    return raw


def unpack_64byte_sketches(sketch_bytes_list: list[bytes], device: str) -> torch.Tensor:
    """
    Unpacks M strict 64-byte binary sketches into `[4_kv, 2_clust, M, 60]` on GPU HBM.
    """
    m_count = len(sketch_bytes_list)
    norms = np.zeros((m_count, 2), dtype=np.float32)
    bits_u8 = np.zeros((m_count, 60), dtype=np.uint8)
    for i, raw in enumerate(sketch_bytes_list):
        norms[i] = SKETCH_HEADER_STRUCT.unpack(raw[:4])
        bits_u8[i] = np.frombuffer(raw[4:], dtype=np.uint8)

    signs = np.unpackbits(bits_u8, axis=1).astype(np.float32) * 2.0 - 1.0  # [M, 480]
    signs_t = torch.from_numpy(signs).to(device=device).view(m_count, 2, 4, 60)
    norms_t = torch.from_numpy(norms).to(device=device)
    scale = (norms_t / math.sqrt(240.0) * math.sqrt(math.pi / 2.0)).unsqueeze(-1).unsqueeze(-1)
    return (signs_t * scale).permute(2, 1, 0, 3).contiguous()  # [4_kv, 2_clust, M, 60]


def serialize_plane1_stripe(k_res: torch.Tensor, v_res: torch.Tensor) -> bytes:
    k_np = k_res.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    v_np = v_res.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    blob = k_np + v_np
    assert len(blob) == NVME_PAGE_BYTES
    return blob


def deserialize_plane1_stripe(blob: bytes, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    half = NVME_PAGE_BYTES // 2
    k_arr = np.frombuffer(blob[:half], dtype=np.float16).reshape(1, 4, PLANE1_RESIDUAL_TOKENS, 128)
    v_arr = np.frombuffer(blob[half:], dtype=np.float16).reshape(1, 4, PLANE1_RESIDUAL_TOKENS, 128)
    return (
        torch.from_numpy(k_arr.copy()).to(device=device, dtype=torch.float16),
        torch.from_numpy(v_arr.copy()).to(device=device, dtype=torch.float16),
    )


# ============================================================================
# Multi-Tier KV Session Engine Supporting All 5 Architectures
# ============================================================================
@dataclass
class ColdStripeMeta:
    block_idx: int
    canonical_hash: str
    nvme_offsets: list[int]
    sketch_bytes: list[bytes]


class MultiTierKVSession:
    def __init__(
        self,
        method: str,
        model: AutoModelForCausalLM,
        nvme_store: PhysicalNVMeSlabStore,
        P_subspaces: torch.Tensor,
        boundary_token_ids: set[int],
        device: str = "mps",
        prominence_delta: float = 0.170,
        entropy_gamma: float = 0.30,
    ):
        self.method = method
        self.model = model
        self.nvme = nvme_store
        self.P_subspaces = P_subspaces
        self.boundary_token_ids = boundary_token_ids
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.prominence_delta = prominence_delta
        self.entropy_gamma = entropy_gamma

        self.logical_seq_len = 0
        self.hbm_keys: list[torch.Tensor | None] = [None] * self.num_layers
        self.hbm_vals: list[torch.Tensor | None] = [None] * self.num_layers
        self.hbm_positions: list[int] = []
        self.hbm_token_ids: list[int] = []
        self.all_token_ids: list[int] = []

        self.cum_attn_scores: torch.Tensor | None = None
        self.compacted_slab_offsets: dict[int, list[int]] = {}
        self.compacted_slab_sigs: dict[int, bytes] = {}

        self.tiered_prefix_tokens = 0
        self.cold_stripes: list[ColdStripeMeta] = []
        self._unpacked_sketches_cache: list[torch.Tensor | None] = [None] * self.num_layers

        self.staged_recall: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(self.num_layers)
        ]
        self.enable_recall_now = False
        self.current_step_uncertain = False
        self.current_query_token_ids: list[int] = []
        self.turn_accum_prominence: torch.Tensor | None = None
        self.turn_prominence_steps: int = 0
        self.pinned_turn_blocks: list[int] | None = None

        self.distractor_cold_stripe_opportunities = 0
        self.distractor_cold_stripes_recalled = 0
        self.recall_turn_cold_stripe_opportunities = 0
        self.recall_turn_cold_stripes_recalled = 0
        self.peak_staged_bytes = 0

    def clone_shared_prefix_state(self) -> "MultiTierKVSession":
        clone = MultiTierKVSession(
            method=self.method,
            model=self.model,
            nvme_store=self.nvme,
            P_subspaces=self.P_subspaces,
            boundary_token_ids=self.boundary_token_ids,
            device=self.device,
            prominence_delta=self.prominence_delta,
            entropy_gamma=self.entropy_gamma,
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
        return clone

    def clear_staged_recall(self) -> None:
        for d in self.staged_recall:
            d.clear()
        self.turn_accum_prominence = None
        self.turn_prominence_steps = 0
        self.pinned_turn_blocks = None

    def active_hbm_bytes(self) -> tuple[int, int]:
        kv_bytes = 0
        for l in range(self.num_layers):
            if self.hbm_keys[l] is not None:
                kv_bytes += self.hbm_keys[l].numel() * 2 + self.hbm_vals[l].numel() * 2
        sketch_bytes = len(self.cold_stripes) * self.num_layers * SKETCH_BYTES
        steady = kv_bytes + sketch_bytes
        return steady, steady + self.peak_staged_bytes

    def _get_layer_sketches(self, layer_idx: int) -> torch.Tensor | None:
        if not self.cold_stripes:
            return None
        cached = self._unpacked_sketches_cache[layer_idx]
        if cached is None or cached.shape[2] != len(self.cold_stripes):
            raw_list = [cs.sketch_bytes[layer_idx] for cs in self.cold_stripes]
            cached = unpack_64byte_sketches(raw_list, self.device)
            self._unpacked_sketches_cache[layer_idx] = cached
        return cached

    @torch.no_grad()
    def _remove_rope_k_bias(self, k_tensor: torch.Tensor, positions: list[int], layer_idx: int) -> torch.Tensor:
        """
        Deterministically subtracts `RoPE(k_proj.bias, positions)` from `k_tensor` ([1, 4, T, 128]),
        eliminating Qwen2.5's static ~920-norm rotating key bias with zero prompt/query dependence!
        """
        t_len = len(positions)
        pos_t = torch.tensor([positions], dtype=torch.long, device=self.device)
        dum_emb = torch.zeros((1, t_len, 128), dtype=torch.float16, device=self.device)
        cos_p, sin_p = self.model.model.rotary_emb(dum_emb, pos_t)
        b_k = (
            self.model.model.layers[layer_idx]
            .self_attn.k_proj.bias.detach()
            .view(1, 1, 4, 128)
            .expand(1, t_len, 4, 128)
            .transpose(1, 2)
        )
        q_dum = torch.zeros((1, 28, t_len, 128), dtype=b_k.dtype, device=self.device)
        _, b_k_rope = apply_rotary_pos_emb(q_dum, b_k, cos_p, sin_p)
        return (k_tensor.float() - b_k_rope.float())[0]  # [4, T, 128]

    # ------------------------------------------------------------------------
    # 2-Layer Lookahead Entropy-Gated Residual Recall
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def maybe_lookahead_recall_for_layer(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        target_layer: int,
        is_final_recall_turn: bool,
    ) -> None:
        m_blocks = len(self.cold_stripes)
        if self.method != "splitslab_full" or not self.enable_recall_now or m_blocks < 8:
            return
        if target_layer < 4 or target_layer >= self.num_layers:
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
            cos_s = cos[:, keep_t, :]
            sin_s = sin[:, keep_t, :]
        else:
            if self.current_query_token_ids and self.current_query_token_ids[0] in self.boundary_token_ids:
                return
            h_slice = hidden_states
            cos_s, sin_s = position_embeddings

        ql = h_slice.shape[1]
        t_layer_mod = self.model.model.layers[target_layer]
        normed = t_layer_mod.input_layernorm(h_slice)
        qp = t_layer_mod.self_attn.q_proj(normed).view(1, ql, 28, 128).transpose(1, 2)
        kd = torch.zeros((1, 4, ql, 128), dtype=h_slice.dtype, device=self.device)
        q_spec, _ = apply_rotary_pos_emb(qp, kd, cos_s, sin_s)
        qf = q_spec[0].float().view(4, 7, ql, 128) / math.sqrt(128.0)

        cutoff_b = max(2, m_blocks - 6)

        # Unpack strict 64-byte Cold Residual sketches [4_kv, 2_clust, M, 60] and zero-center across cold stripes
        sk_vec = self._get_layer_sketches(target_layer)
        assert sk_vec is not None
        sk_c = sk_vec - sk_vec[:, :, 1:cutoff_b, :].mean(dim=2, keepdim=True)
        q_proj = torch.einsum("grqd,gdb->grqb", qf, self.P_subspaces[target_layer])
        est_logits = torch.einsum("grqb,gsmb->grqms", q_proj, sk_c).amax(dim=-1) * 1.15  # [4, 7, ql, M]

        # Co-Routed Plane-0 Beacon + 64-Byte Plane-1 Sketch Scoring:
        # Each cold block m has 8 Plane-0 anchors ALREADY in HBM (`k_hbm_pure[:, m*8:(m+1)*8]`) plus its
        # 64-byte Plane-1 sketch `sk_c[:, :, m, :]`. Combining both ensures that if a queried literal
        # straddles Plane 0 and Plane 1 (e.g. half of a hex token in Plane 0, constant name in Plane 1),
        # Plane 0 acts as an exact routing beacon for its sibling cold stripe rather than suppressing it!
        k_hbm_raw = self.hbm_keys[target_layer]
        assert k_hbm_raw is not None
        k_hbm_pure = self._remove_rope_k_bias(k_hbm_raw, self.hbm_positions, target_layer)  # [4, T_hbm, 128]
        p0_end = cutoff_b * PLANE0_ANCHOR_TOKENS
        kh_mean = k_hbm_pure[:, PLANE0_ANCHOR_TOKENS:p0_end, :].mean(dim=1, keepdim=True)
        k_hbm_c = k_hbm_pure - kh_mean

        # Per-block Plane-0 beacon max logit: [4, 7, ql, m_blocks]
        k_p0_blocks = k_hbm_c[:, : m_blocks * PLANE0_ANCHOR_TOKENS, :].view(4, m_blocks, PLANE0_ANCHOR_TOKENS, 128)
        p0_block_logits = torch.einsum("grqd,gmad->grqma", qf, k_p0_blocks).amax(dim=-1)  # [4, 7, ql, M]
        joint_block_logits = torch.maximum(est_logits, 0.88 * p0_block_logits)

        # Reference partition function: Global Sink (Block 0) + Active Local Window (uncompacted recent tail)
        k_sink_and_local = torch.cat(
            [k_hbm_c[:, :PLANE0_ANCHOR_TOKENS, :], k_hbm_c[:, m_blocks * PLANE0_ANCHOR_TOKENS :, :]], dim=1
        )
        lse_ref = torch.logsumexp(
            torch.matmul(qf, k_sink_and_local.unsqueeze(1).transpose(-1, -2)), dim=-1, keepdim=True
        )

        denom = torch.logaddexp(lse_ref, torch.logsumexp(joint_block_logits, dim=-1, keepdim=True))
        alpha_m = torch.exp(joint_block_logits - denom)
        raw_m = alpha_m.amax(dim=(0, 1, 2))
        raw_m[0] = 0.0

        # Compute neighborhood median over true unmasked `raw_m` BEFORE applying `cutoff_b`
        # (prevents one-sided boundary cliff artifact at `cutoff_b - 1`!)
        valid_hi = max(2, m_blocks - 1)
        detrended = torch.zeros_like(raw_m)
        for m in range(1, cutoff_b):
            nbrs = [
                raw_m[j].item()
                for j in range(max(1, m - 5), min(valid_hi, m + 6))
                if abs(j - m) >= 2
            ]
            detrended[m] = max(0.0, float(raw_m[m].item()) - (float(np.median(nbrs)) if nbrs else 0.0))
        detrended[cutoff_b:] = 0.0

        if self.turn_accum_prominence is None or self.turn_accum_prominence.shape[0] != m_blocks:
            self.turn_accum_prominence = detrended.clone()
            self.turn_prominence_steps = 1
        else:
            self.turn_accum_prominence = self.turn_accum_prominence + detrended
            self.turn_prominence_steps += 1

        # Require 18-layer HBM sketch consensus (target_layer >= 21) AND predictive uncertainty!
        if self.turn_prominence_steps < 18 or not self.current_step_uncertain:
            return

        mean_prom = self.turn_accum_prominence / float(self.turn_prominence_steps)

        if self.pinned_turn_blocks is not None:
            chosen = self.pinned_turn_blocks
        else:
            peak_val = float(mean_prom.max().item())
            if peak_val < self.prominence_delta:
                return

            best_m = int(torch.argmax(mean_prom).item())
            # Stage contiguous micro-window [best_m - 1, best_m, best_m + 1] (<= 3 cold blocks = ~6.8% < 8% cold stripes!)
            chosen = [m_i for m_i in (best_m - 1, best_m, best_m + 1) if 1 <= m_i < cutoff_b]
            self.pinned_turn_blocks = chosen
            for early_l in range(0, target_layer):
                for m_idx in chosen:
                    if m_idx not in self.staged_recall[early_l]:
                        off_e = self.cold_stripes[m_idx].nvme_offsets[early_l]
                        b_e = self.nvme.pread_stripe(off_e, NVME_PAGE_BYTES)
                        self.staged_recall[early_l][m_idx] = deserialize_plane1_stripe(b_e, self.device)
                        if is_final_recall_turn:
                            self.recall_turn_cold_stripes_recalled += 1
                        else:
                            self.distractor_cold_stripes_recalled += 1

        for m_idx in chosen:
            if m_idx not in self.staged_recall[target_layer]:
                offset = self.cold_stripes[m_idx].nvme_offsets[target_layer]
                raw_blob = self.nvme.pread_stripe(offset, NVME_PAGE_BYTES)
                self.staged_recall[target_layer][m_idx] = deserialize_plane1_stripe(raw_blob, self.device)
                if is_final_recall_turn:
                    self.recall_turn_cold_stripes_recalled += 1
                else:
                    self.distractor_cold_stripes_recalled += 1

        cur_staged = sum(len(d) * NVME_PAGE_BYTES for d in self.staged_recall)
        if cur_staged > self.peak_staged_bytes:
            self.peak_staged_bytes = cur_staged

    # ------------------------------------------------------------------------
    # Forward Step With Exact Uncompacted RoPE + Custom Tiered KV Splicing
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def forward_tokens(
        self,
        token_ids: list[int],
        is_final_recall_turn: bool = False,
        allow_recall: bool = False,
        track_online_attn: bool = True,
    ) -> torch.Tensor:
        q_len = len(token_ids)
        self.current_query_token_ids = token_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        pos_ids = torch.arange(
            self.logical_seq_len, self.logical_seq_len + q_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        self.enable_recall_now = allow_recall and (self.method == "splitslab_full")
        if self.enable_recall_now and len(self.cold_stripes) > 0 and q_len > 1:
            total_cold_stripes = len(self.cold_stripes) * self.num_layers
            if is_final_recall_turn:
                self.recall_turn_cold_stripe_opportunities += total_cold_stripes
            else:
                self.distractor_cold_stripe_opportunities += total_cold_stripes

        qwen_model = self.model.model
        hidden_states = qwen_model.embed_tokens(input_ids)
        position_embeddings = qwen_model.rotary_emb(hidden_states, pos_ids)
        cos, sin = position_embeddings

        per_layer_new_attn_mass: list[torch.Tensor] = []

        for l in range(self.num_layers):
            layer_mod = qwen_model.layers[l]
            residual = hidden_states
            normed = layer_mod.input_layernorm(hidden_states)

            attn = layer_mod.self_attn
            q_states = attn.q_proj(normed).view(1, q_len, 28, 128).transpose(1, 2)
            k_new = attn.k_proj(normed).view(1, q_len, 4, 128).transpose(1, 2)
            v_new = attn.v_proj(normed).view(1, q_len, 4, 128).transpose(1, 2)
            q_states, k_new = apply_rotary_pos_emb(q_states, k_new, cos, sin)

            if self.hbm_keys[l] is None:
                k_hbm, v_hbm = k_new, v_new
            else:
                k_hbm = torch.cat([self.hbm_keys[l], k_new], dim=2)
                v_hbm = torch.cat([self.hbm_vals[l], v_new], dim=2)
            self.hbm_keys[l] = k_hbm
            self.hbm_vals[l] = v_hbm

            if self.staged_recall[l]:
                rec_ks = [pair[0] for _, pair in sorted(self.staged_recall[l].items())]
                rec_vs = [pair[1] for _, pair in sorted(self.staged_recall[l].items())]
                past_hbm_len = k_hbm.shape[2] - q_len
                k_eff = torch.cat([k_hbm[:, :, :past_hbm_len, :], *rec_ks, k_hbm[:, :, past_hbm_len:, :]], dim=2)
                v_eff = torch.cat([v_hbm[:, :, :past_hbm_len, :], *rec_vs, v_hbm[:, :, past_hbm_len:, :]], dim=2)
            else:
                k_eff, v_eff = k_hbm, v_hbm

            k_rep = repeat_kv(k_eff, 7)
            v_rep = repeat_kv(v_eff, 7)

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
                q_states, k_rep, v_rep, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).reshape(1, q_len, 3584).contiguous()
            hidden_states = residual + attn.o_proj(attn_out)

            if track_online_attn and self.method in ("online_evict_actkv", "iterative_recompact"):
                q_sub_cnt = min(q_len, 12)
                q_tail_t = q_states[0, :, -q_sub_cnt:, :].float() / math.sqrt(128.0)
                k_hbm_rep = repeat_kv(k_hbm, 7)[0].float()
                scores = torch.matmul(q_tail_t, k_hbm_rep.transpose(-1, -2))
                if q_sub_cnt > 1:
                    p_l = k_hbm.shape[2] - q_sub_cnt
                    c_mask = torch.ones((q_sub_cnt, k_hbm.shape[2]), dtype=torch.bool, device=self.device)
                    c_mask[:, p_l:] = torch.tril(torch.ones((q_sub_cnt, q_sub_cnt), dtype=torch.bool, device=self.device))
                    scores = scores.masked_fill(~c_mask.unsqueeze(0), float("-inf"))
                probs = torch.softmax(scores, dim=-1).sum(dim=(0, 1))
                per_layer_new_attn_mass.append(probs)

            residual = hidden_states
            hidden_states = residual + layer_mod.mlp(layer_mod.post_attention_layernorm(hidden_states))

            if self.enable_recall_now and (l + 2) < self.num_layers:
                if l == 21 and not self.current_step_uncertain:
                    mid_logits = self.model.lm_head(qwen_model.norm(hidden_states[:, -1:, :]))[0, -1].float()
                    mid_p = torch.softmax(mid_logits, dim=-1)
                    mid_h = float(-(mid_p * torch.log(mid_p.clamp(min=1e-12))).sum().item())
                    if mid_h > self.entropy_gamma:
                        self.current_step_uncertain = True
                self.maybe_lookahead_recall_for_layer(hidden_states, position_embeddings, l + 2, is_final_recall_turn)

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

        final_norm = qwen_model.norm(hidden_states[:, -1:, :])
        logits = self.model.lm_head(final_norm)[:, -1, :]

        probs_out = torch.softmax(logits[0].float(), dim=-1)
        entropy_val = float(-(probs_out * torch.log(probs_out.clamp(min=1e-12))).sum().item())
        self.current_step_uncertain = entropy_val > self.entropy_gamma
        return logits

    # ------------------------------------------------------------------------
    # Post-Turn Compaction & Physical NVMe Tiering Policies
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def apply_post_turn_tiering(
        self, shared_prefix_token_limit: int | None = None, is_dedup_probe: bool = False
    ) -> None:
        if self.method == "full_kv_flatcake":
            slab_toks = 32
            total_hbm = len(self.hbm_positions)
            safe_limit = max(0, total_hbm - LOCAL_WINDOW_TOKENS)
            if is_dedup_probe and shared_prefix_token_limit is not None:
                safe_limit = min(safe_limit, shared_prefix_token_limit)
            while self.tiered_prefix_tokens + slab_toks <= safe_limit:
                b_start = self.tiered_prefix_tokens
                b_end = b_start + slab_toks
                prefix_bytes = struct.pack(f"<{b_end}I", *self.all_token_ids[:b_end])
                for l in range(self.num_layers):
                    h_hex = hashlib.sha256(prefix_bytes + struct.pack("<H", l)).hexdigest()
                    k_blk = self.hbm_keys[l][0, :, b_start:b_end, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    v_blk = self.hbm_vals[l][0, :, b_start:b_end, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    self.nvme.put_immutable_stripe(
                        h_hex, k_blk + v_blk, count_dedup=is_dedup_probe, count_waf_logical=not is_dedup_probe
                    )
                self.tiered_prefix_tokens += slab_toks
            return

        if self.method in ("splitslab_anchor_only", "splitslab_full"):
            while True:
                cur_hbm_len = len(self.hbm_positions)
                num_tiered_blocks = len(self.cold_stripes)
                hbm_untiered_start = num_tiered_blocks * PLANE0_ANCHOR_TOKENS
                untiered_available = cur_hbm_len - hbm_untiered_start - LOCAL_WINDOW_TOKENS
                if untiered_available < BLOCK_TOKENS:
                    break

                b_hbm_start = hbm_untiered_start
                b_hbm_end = b_hbm_start + BLOCK_TOKENS
                logical_end_pos = self.hbm_positions[b_hbm_end - 1] + 1
                if is_dedup_probe and shared_prefix_token_limit is not None and logical_end_pos > shared_prefix_token_limit:
                    break

                prefix_token_bytes = struct.pack(f"<{logical_end_pos}I", *self.all_token_ids[:logical_end_pos])
                block_canon_hash = hashlib.sha256(prefix_token_bytes).hexdigest()

                # Canonical Prefix-Only Plane-0 Selection (8 tokens = 20%):
                # Deterministic O(1) structural-boundary + stride skeleton (preserves step/chat boundaries
                # and module headers in Plane 0 while keeping interior key-value clauses contiguous in Plane 1,
                # avoiding k_proj.bias distortion and guaranteeing 100% cross-agent SHA-256 isomorphism!)
                cand_scores: list[tuple[float, int, int]] = []
                for rel_i in range(BLOCK_TOKENS):
                    tid = self.hbm_token_ids[b_hbm_start + rel_i]
                    if rel_i < 2:
                        sc = 100.0
                    elif tid in self.boundary_token_ids:
                        sc = 50.0
                    elif rel_i % 5 == 0:
                        sc = 10.0
                    else:
                        sc = 0.0
                    cand_scores.append((sc, -rel_i, rel_i))
                cand_scores.sort(reverse=True)
                anchor_list = sorted([x[2] for x in cand_scores[:PLANE0_ANCHOR_TOKENS]])
                anchor_rel = torch.tensor(anchor_list, dtype=torch.long, device=self.device)
                keep_mask = torch.zeros(BLOCK_TOKENS, dtype=torch.bool, device=self.device)
                keep_mask[anchor_rel] = True
                res_rel = torch.nonzero(~keep_mask, as_tuple=False).flatten()

                res_positions = [self.hbm_positions[b_hbm_start + int(i)] for i in res_rel.tolist()]
                layer_offsets: list[int] = []
                layer_sketches: list[bytes] = []

                for l in range(self.num_layers):
                    k_full = self.hbm_keys[l]
                    v_full = self.hbm_vals[l]
                    k_blk = k_full[:, :, b_hbm_start:b_hbm_end, :]
                    v_blk = v_full[:, :, b_hbm_start:b_hbm_end, :]

                    k_anc = k_blk[:, :, anchor_rel, :]
                    v_anc = v_blk[:, :, anchor_rel, :]
                    k_res_4d = k_blk[:, :, res_rel, :]  # [1, 4, 32, 128]
                    v_res = v_blk[0, :, res_rel, :]     # [4, 32, 128]

                    stripe_hash = hashlib.sha256(
                        block_canon_hash.encode("ascii") + struct.pack("<H", l)
                    ).hexdigest()
                    stripe_blob = serialize_plane1_stripe(k_res_4d[0], v_res)
                    off, _ = self.nvme.put_immutable_stripe(
                        stripe_hash, stripe_blob, count_dedup=is_dedup_probe, count_waf_logical=not is_dedup_probe
                    )
                    layer_offsets.append(off)

                    # Exact deterministic RoPE(k_proj.bias, res_positions) removal before packing 64B sketch!
                    k_res_pure = self._remove_rope_k_bias(k_res_4d, res_positions, l)
                    sk_bytes = pack_64byte_sketch(k_res_pure, self.P_subspaces[l])
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
                    ColdStripeMeta(
                        block_idx=num_tiered_blocks,
                        canonical_hash=block_canon_hash,
                        nvme_offsets=layer_offsets,
                        sketch_bytes=layer_sketches,
                    )
                )
            return

        if self.method in ("online_evict_actkv", "iterative_recompact"):
            cur_len = len(self.hbm_positions)
            if cur_len <= LOCAL_WINDOW_TOKENS + 16:
                return
            evictable_len = cur_len - LOCAL_WINDOW_TOKENS
            hist_logical = max(0, self.logical_seq_len - LOCAL_WINDOW_TOKENS)
            target_hist_keep = max(8, int(round(0.20 * hist_logical)))

            if evictable_len > target_hist_keep:
                scores_evictable = self.cum_attn_scores[:evictable_len].clone()
                scores_evictable[:4] = 1e9
                keep_hist_idx = torch.topk(scores_evictable, k=target_hist_keep).indices.sort().values
                local_idx = torch.arange(evictable_len, cur_len, device=self.device)
                full_keep_idx = torch.cat([keep_hist_idx, local_idx], dim=0)

                for l in range(self.num_layers):
                    self.hbm_keys[l] = self.hbm_keys[l][:, :, full_keep_idx, :].contiguous()
                    self.hbm_vals[l] = self.hbm_vals[l][:, :, full_keep_idx, :].contiguous()

                self.cum_attn_scores = self.cum_attn_scores[full_keep_idx].contiguous()
                self.hbm_positions = [self.hbm_positions[int(i)] for i in full_keep_idx.tolist()]
                self.hbm_token_ids = [self.hbm_token_ids[int(i)] for i in full_keep_idx.tolist()]

            hist_kept_len = len(self.hbm_positions) - LOCAL_WINDOW_TOKENS
            if is_dedup_probe and shared_prefix_token_limit is not None:
                hist_kept_len = sum(1 for p in self.hbm_positions[:hist_kept_len] if p < shared_prefix_token_limit)

            num_slabs = hist_kept_len // 32
            for s_idx in range(num_slabs):
                s_lo = s_idx * 32
                s_hi = s_lo + 32
                pos_tuple = self.hbm_positions[s_lo:s_hi]
                pos_sig = struct.pack("<32I", *pos_tuple)

                if self.method == "online_evict_actkv":
                    if is_dedup_probe or s_idx not in self.compacted_slab_sigs:
                        for l in range(self.num_layers):
                            kb = self.hbm_keys[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                            vb = self.hbm_vals[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                            slab_hash = hashlib.sha256(pos_sig + struct.pack("<H", l) + kb[:128]).hexdigest()
                            self.nvme.put_immutable_stripe(
                                slab_hash, kb + vb, count_dedup=is_dedup_probe, count_waf_logical=not is_dedup_probe
                            )
                        if not is_dedup_probe:
                            self.compacted_slab_sigs[s_idx] = pos_sig

                elif self.method == "iterative_recompact":
                    if is_dedup_probe:
                        for l in range(self.num_layers):
                            kb = self.hbm_keys[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                            vb = self.hbm_vals[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                            slab_hash = hashlib.sha256(pos_sig + struct.pack("<H", l) + kb[:128]).hexdigest()
                            self.nvme.put_immutable_stripe(
                                slab_hash, kb + vb, count_dedup=True, count_waf_logical=False
                            )
                    else:
                        prev_sig = self.compacted_slab_sigs.get(s_idx)
                        if prev_sig is None:
                            offs = []
                            for l in range(self.num_layers):
                                kb = self.hbm_keys[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                vb = self.hbm_vals[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                slab_hash = hashlib.sha256(pos_sig + struct.pack("<H", l)).hexdigest()
                                off, _ = self.nvme.put_immutable_stripe(
                                    slab_hash, kb + vb, count_dedup=False, count_waf_logical=True
                                )
                                offs.append(off)
                            self.compacted_slab_offsets[s_idx] = offs
                            self.compacted_slab_sigs[s_idx] = pos_sig
                        elif prev_sig != pos_sig:
                            new_offs = []
                            for l in range(self.num_layers):
                                kb = self.hbm_keys[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                vb = self.hbm_vals[l][0, :, s_lo:s_hi, :].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                new_off = self.nvme.rewrite_mutable_page(self.compacted_slab_offsets[s_idx][l], kb + vb)
                                new_offs.append(new_off)
                            self.compacted_slab_offsets[s_idx] = new_offs
                            self.compacted_slab_sigs[s_idx] = pos_sig

    # ------------------------------------------------------------------------
    # Greedy Decoding With Live Entropy-Gated Lookahead Recall
    # ------------------------------------------------------------------------
    @torch.no_grad()
    def rollback_last_tokens(self, n_toks: int) -> None:
        if n_toks <= 0:
            return
        for l in range(self.num_layers):
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
        max_new_tokens: int = 36,
        is_final_recall_turn: bool = True,
    ) -> str:
        self.clear_staged_recall()
        self.current_step_uncertain = True
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

        gen_ids: list[int] = []
        eos_id = tokenizer.eos_token_id
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        for _ in range(max_new_tokens):
            next_id = int(torch.argmax(logits[0]).item())
            if next_id in (eos_id, im_end_id):
                break
            gen_ids.append(next_id)
            logits = self.forward_tokens(
                [next_id],
                is_final_recall_turn=is_final_recall_turn,
                allow_recall=(self.pinned_turn_blocks is None),
                track_online_attn=False,
            )

        self.clear_staged_recall()
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ============================================================================
# Evaluation Scoring & Full 5-Method Benchmark Runner
# ============================================================================
def score_delayed_recall_response(resp: str, spec: DelayedRecallSpec) -> dict[str, Any]:
    norm = resp.upper()
    has_const = spec.const_name.upper() in norm
    has_val = spec.param_val in resp
    partial = 0.5 * float(has_const) + 0.5 * float(has_val)
    exact_both = bool(has_const and has_val)
    return {
        "has_const": has_const,
        "has_val": has_val,
        "partial_score": partial,
        "exact_match": exact_both,
    }


def run_single_trajectory_for_method(
    method: str,
    traj: dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    P_subspaces: torch.Tensor,
    boundary_token_ids: set[int],
    nvme_dir: Path,
    device: str,
) -> dict[str, Any]:
    spec: DelayedRecallSpec = traj["spec"]
    m_dir = nvme_dir / f"{method}_{spec.traj_id}"
    if m_dir.exists():
        shutil.rmtree(m_dir)
    nvme_store = PhysicalNVMeSlabStore(m_dir, sync_writes=True)

    t_start = time.perf_counter()
    try:
        sess_a = MultiTierKVSession(
            method=method,
            model=model,
            nvme_store=nvme_store,
            P_subspaces=P_subspaces,
            boundary_token_ids=boundary_token_ids,
            device=device,
        )

        # Turn 1: Shared repository snapshot
        t1_ids = tokenizer.encode(traj["turn1_shared"], add_special_tokens=False)
        shared_prefix_len = len(t1_ids)
        sess_a.forward_tokens(t1_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=False)
        sess_a.cum_attn_scores = torch.zeros(len(t1_ids), dtype=torch.float32, device=device)

        # Branch Agent B from shared Turn-1 state prior to divergent Turn-2 queries
        sess_b = sess_a.clone_shared_prefix_state()

        # Agent A executes Turn 2 and offloads slabs
        t2a_ids = tokenizer.encode(traj["turn2_agent_a"], add_special_tokens=False)
        sess_a.forward_tokens(t2a_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=True)
        sess_a.apply_post_turn_tiering(shared_prefix_token_limit=shared_prefix_len, is_dedup_probe=False)

        # Agent B executes its own Turn 2 and probes L2 SHA-256 deduplication on shared Turn-1 slabs
        t2b_ids = tokenizer.encode(traj["turn2_agent_b"], add_special_tokens=False)
        sess_b.forward_tokens(t2b_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=True)
        sess_b.apply_post_turn_tiering(shared_prefix_token_limit=shared_prefix_len, is_dedup_probe=True)
        del sess_b

        # Agent A executes Distractor Turns 3..10 + Turn 11 incrementally with live Entropy + Sketch Gate!
        split_marker = "<|im_start|>assistant\n"
        for d_turn in [*traj["distractor_turns"], traj["turn11"]]:
            d_user, d_asst = d_turn.split(split_marker, 1)
            d_prompt_ids = tokenizer.encode(d_user + split_marker, add_special_tokens=False)
            d_asst_ids = tokenizer.encode(d_asst, add_special_tokens=False)
            sess_a.clear_staged_recall()
            sess_a.current_step_uncertain = False
            sess_a.forward_tokens(d_prompt_ids, is_final_recall_turn=False, allow_recall=True, track_online_attn=True)
            sess_a.forward_tokens(d_asst_ids, is_final_recall_turn=False, allow_recall=False, track_online_attn=True)
            sess_a.clear_staged_recall()
            sess_a.apply_post_turn_tiering(is_dedup_probe=False)

        steady_hbm_b, _ = sess_a.active_hbm_bytes()
        full_equiv_b = sess_a.logical_seq_len * model.config.num_hidden_layers * 2048

        # Turn 12: Production Stack-Trace Delayed-Recall Trigger!
        t12_ids = tokenizer.encode(traj["turn12_prompt"], add_special_tokens=False)
        response = sess_a.greedy_generate(
            t12_ids, tokenizer=tokenizer, max_new_tokens=52, is_final_recall_turn=True
        )
        peak_hbm_with_stage = steady_hbm_b + sess_a.peak_staged_bytes

        acc = score_delayed_recall_response(response, spec)
        elapsed_s = time.perf_counter() - t_start

        dedup_rate = (
            nvme_store.dedup_hits / float(nvme_store.dedup_queries)
            if nvme_store.dedup_queries > 0
            else 0.0
        )
        recall_turn_rate = (
            sess_a.recall_turn_cold_stripes_recalled / float(sess_a.recall_turn_cold_stripe_opportunities)
            if sess_a.recall_turn_cold_stripe_opportunities > 0
            else 0.0
        )
        distractor_recall_rate = (
            sess_a.distractor_cold_stripes_recalled / float(sess_a.distractor_cold_stripe_opportunities)
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
            "steady_hbm_bytes": steady_hbm_b,
            "peak_hbm_bytes": peak_hbm_with_stage,
            "full_kv_equiv_bytes": full_equiv_b,
            "steady_hbm_reduction_pct": 100.0 * (1.0 - steady_hbm_b / float(full_equiv_b)),
            "peak_hbm_reduction_pct": 100.0 * (1.0 - peak_hbm_with_stage / float(full_equiv_b)),
            "nvme_physical_bytes_written": nvme_store.physical_bytes_written,
            "nvme_logical_unique_bytes": nvme_store.logical_unique_bytes_stored,
            "nvme_waf": nvme_store.waf,
            "cross_agent_dedup_pct": 100.0 * dedup_rate,
            "cold_stripes_recalled_turn12_pct": 100.0 * recall_turn_rate,
            "cold_stripes_recalled_distractor_pct": 100.0 * distractor_recall_rate,
            "nvme_pread_calls": nvme_store.pread_calls,
            "nvme_pread_median_us": (
                float(np.median(nvme_store.read_time_us_samples))
                if nvme_store.read_time_us_samples
                else 0.0
            ),
            "elapsed_sec": elapsed_s,
        }
    finally:
        nvme_store.close()
        if m_dir.exists():
            shutil.rmtree(m_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="SplitSlab Phase 0/1 Qwen2.5-7B Pilot")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-trajectories", type=int, default=20)
    parser.add_argument(
        "--methods",
        type=str,
        default="full_kv_flatcake,online_evict_actkv,iterative_recompact,splitslab_anchor_only,splitslab_full",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/splitslab_qwen7b_pilot.json",
    )
    parser.add_argument("--nvme-dir", type=str, default="/tmp/splitslab_nvme_pilot")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[SplitSlab Pilot] Device={device} | Model={args.model} | Trajectories={args.num_trajectories}")

    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    model.eval()
    print(f"[SplitSlab Pilot] Loaded {args.model} in {time.perf_counter() - t_load:.2f}s")

    P_subspaces = build_static_subspaces(model, device=device)
    boundary_token_ids = set(
        tokenizer.encode("<|im_start|><|im_end|>system\nuser\nassistant\n", add_special_tokens=False)
    )

    specs = TRAJECTORY_SPECS[: args.num_trajectories]
    trajectories = [build_dual_agent_trajectory(s) for s in specs]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    nvme_root = Path(args.nvme_dir)
    nvme_root.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "metadata": {
            "project": "SplitSlab",
            "benchmark": "RecallRegret-Pilot",
            "model": args.model,
            "device": device,
            "dtype": "float16",
            "num_trajectories": len(trajectories),
            "block_tokens": BLOCK_TOKENS,
            "plane0_anchor_tokens": PLANE0_ANCHOR_TOKENS,
            "plane1_residual_tokens": PLANE1_RESIDUAL_TOKENS,
            "local_window_tokens": LOCAL_WINDOW_TOKENS,
            "nvme_stripe_bytes": NVME_PAGE_BYTES,
            "sketch_bytes": SKETCH_BYTES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "summary_by_method": {},
        "per_trajectory": {m: [] for m in methods},
    }

    for method in methods:
        print(f"\n==================== Running Method: {method} ====================")
        m_runs: list[dict[str, Any]] = []
        for idx, traj in enumerate(trajectories):
            res = run_single_trajectory_for_method(
                method=method,
                traj=traj,
                model=model,
                tokenizer=tokenizer,
                P_subspaces=P_subspaces,
                boundary_token_ids=boundary_token_ids,
                nvme_dir=nvme_root,
                device=device,
            )
            m_runs.append(res)
            print(
                f"  [{idx+1:02d}/{len(trajectories):02d}] {res['traj_id']} ({res['domain']:16s}) | "
                f"Exact={int(res['exact_match'])} Partial={res['partial_score']:.1f} | "
                f"HBM_Save={res['steady_hbm_reduction_pct']:5.1f}% | "
                f"WAF={res['nvme_waf']:5.2f}x | Dedup={res['cross_agent_dedup_pct']:5.1f}% | "
                f"RecallT12={res['cold_stripes_recalled_turn12_pct']:4.2f}% | "
                f"RecallDist={res['cold_stripes_recalled_distractor_pct']:4.2f}% | "
                f"Resp='{res['response'][:42]}'"
            )
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

        all_results["per_trajectory"][method] = m_runs
        summary = {
            "delayed_recall_exact_match_pct": 100.0 * float(np.mean([r["exact_match"] for r in m_runs])),
            "delayed_recall_field_accuracy_pct": 100.0 * float(np.mean([r["partial_score"] for r in m_runs])),
            "const_name_accuracy_pct": 100.0 * float(np.mean([r["has_const"] for r in m_runs])),
            "param_val_accuracy_pct": 100.0 * float(np.mean([r["has_val"] for r in m_runs])),
            "steady_hbm_reduction_pct_mean": float(np.mean([r["steady_hbm_reduction_pct"] for r in m_runs])),
            "peak_hbm_reduction_pct_mean": float(np.mean([r["peak_hbm_reduction_pct"] for r in m_runs])),
            "nvme_waf_mean": float(np.mean([r["nvme_waf"] for r in m_runs])),
            "nvme_physical_mb_written_total": float(
                np.sum([r["nvme_physical_bytes_written"] for r in m_runs]) / (1024 * 1024)
            ),
            "cross_agent_dedup_pct_mean": float(np.mean([r["cross_agent_dedup_pct"] for r in m_runs])),
            "cold_stripes_recalled_turn12_pct_mean": float(
                np.mean([r["cold_stripes_recalled_turn12_pct"] for r in m_runs])
            ),
            "cold_stripes_recalled_distractor_pct_mean": float(
                np.mean([r["cold_stripes_recalled_distractor_pct"] for r in m_runs])
            ),
            "nvme_pread_median_us": float(
                np.median([r["nvme_pread_median_us"] for r in m_runs if r["nvme_pread_median_us"] > 0] or [0.0])
            ),
            "mean_trajectory_sec": float(np.mean([r["elapsed_sec"] for r in m_runs])),
        }
        all_results["summary_by_method"][method] = summary
        print(
            f"--> [{method}] ExactMatch={summary['delayed_recall_exact_match_pct']:.1f}% | "
            f"FieldAcc={summary['delayed_recall_field_accuracy_pct']:.1f}% | "
            f"HBM_Save={summary['steady_hbm_reduction_pct_mean']:.1f}% (Peak {summary['peak_hbm_reduction_pct_mean']:.1f}%) | "
            f"WAF={summary['nvme_waf_mean']:.2f}x | Dedup={summary['cross_agent_dedup_pct_mean']:.1f}% | "
            f"Recall(T12)={summary['cold_stripes_recalled_turn12_pct_mean']:.2f}% | "
            f"Recall(Dist)={summary['cold_stripes_recalled_distractor_pct_mean']:.2f}%"
        )

    if nvme_root.exists():
        shutil.rmtree(nvme_root)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[SplitSlab Pilot] Saved complete results to {out_path}")


if __name__ == "__main__":
    main()
