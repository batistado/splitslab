"""
Model-Template-Aware Multi-Turn Delayed-Recall & Long-Horizon Benchmark (`RecallRegret-Bench`).
Supports standard 12-turn (~2.3K token) and long-horizon (~8.2K and ~16.4K token) trajectories
across Qwen2, Granite, Llama, and Mistral chat templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from transformers import AutoTokenizer


@dataclass(frozen=True)
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


def get_model_chat_delimiters(tokenizer: AutoTokenizer, model_type: str) -> dict[str, Any]:
    """
    Returns clean chat turn wrappers, boundary token IDs, and stop token IDs for
    `qwen2` (`<|im_start|>`/`<|im_end|>`) and `granite` (`<|start_of_role|>`/`<|end_of_text|>`).
    """
    stop_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    if model_type == "granite":
        sor = "<|start_of_role|>"
        eor = "<|end_of_role|>"
        eot = "<|end_of_text|>"
        b_ids = set(tokenizer.encode("system\nuser\nassistant\n\n", add_special_tokens=False))
        for tok_s in (eot, sor, eor):
            tid = tokenizer.convert_tokens_to_ids(tok_s)
            if isinstance(tid, int) and tid >= 0:
                b_ids.add(tid)
                if tok_s == eot:
                    stop_ids.add(tid)
        return {
            "sys_wrap": lambda s: f"{sor}system{eor}{s}{eot}\n",
            "user_prefix": lambda u: f"{sor}user{eor}{u}{eot}\n{sor}assistant{eor}",
            "asst_suffix": lambda a: f"{a}{eot}\n",
            "boundary_ids": b_ids,
            "stop_ids": stop_ids,
        }

    # Default ChatML (`qwen2`, `SmolLM2` llama)
    im_s = "<|im_start|>"
    im_e = "<|im_end|>"
    b_ids = set(tokenizer.encode("system\nuser\nassistant\n\n", add_special_tokens=False))
    for tok_s in (im_s, im_e):
        tid = tokenizer.convert_tokens_to_ids(tok_s)
        if isinstance(tid, int) and tid >= 0:
            b_ids.add(tid)
            if tok_s == im_e:
                stop_ids.add(tid)
    return {
        "sys_wrap": lambda s: f"{im_s}system\n{s}{im_e}\n",
        "user_prefix": lambda u: f"{im_s}user\n{u}{im_e}\n{im_s}assistant\n",
        "asst_suffix": lambda a: f"{a}{im_e}\n",
        "boundary_ids": b_ids,
        "stop_ids": stop_ids,
    }


def build_model_formatted_trajectory(
    spec: DelayedRecallSpec,
    tokenizer: AutoTokenizer,
    model_type: str,
    num_repo_modules: int = 16,
    distractor_repeats: int = 1,
) -> dict[str, Any]:
    """
    Constructs a multi-turn dual-agent trajectory formatted for `model_type`.
    Setting `num_repo_modules=64, distractor_repeats=4` produces ~8.2K tokens;
    setting `num_repo_modules=128, distractor_repeats=8` produces ~16.4K tokens!
    """
    delim = get_model_chat_delimiters(tokenizer, model_type)
    sys_w = delim["sys_wrap"]
    u_pre = delim["user_prefix"]
    a_suf = delim["asst_suffix"]

    modules: list[str] = []
    for m_idx in range(num_repo_modules):
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
        elif m_idx == num_repo_modules - 1:
            mod_text = (
                f"[module telemetry_sink_{m_idx:02d}] otlp_grpc_port=4317, batch_export_sz=512, "
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

    turn1_shared = (
        sys_w(
            "You are a principal systems debugging assistant analyzing a repository configuration snapshot. "
            "Answer concisely and accurately."
        )
        + u_pre(
            "[Turn 1 — Shared Repository Snapshot]\n"
            + "".join(modules)
            + f"Acknowledge receipt of the {num_repo_modules}-module repository snapshot."
        )
        + a_suf(f"Repository snapshot ({num_repo_modules} modules) indexed and ready.")
    )

    last_m = num_repo_modules - 1
    turn2_agent_a = (
        u_pre("[Turn 2] From `core_worker_01`, what are `queue_depth` and `batch_timeout_ms`?")
        + a_suf("In `core_worker_01`, `queue_depth=144` and `batch_timeout_ms=45`.")
    )
    turn2_agent_b = (
        u_pre(f"[Turn 2] From `telemetry_sink_{last_m:02d}`, what are `otlp_grpc_port` and `batch_export_sz`?")
        + a_suf(f"In `telemetry_sink_{last_m:02d}`, `otlp_grpc_port=4317` and `batch_export_sz=512`.")
    )

    distractor_pairs: list[tuple[str, str]] = []
    for rep in range(distractor_repeats):
        for q_str, a_str in DISTRACTOR_QA_TURNS:
            tag = f"[Pass {rep + 1}] " if distractor_repeats > 1 else ""
            distractor_pairs.append((u_pre(tag + q_str), a_suf(a_str)))

    distractor_pairs.append(
        (
            u_pre("[Pre-Deploy Check] All helper utilities passed unit tests. Ready for production canary check."),
            a_suf("Understood. Standing by for canary telemetry and incident logs."),
        )
    )

    turn12_prompt = u_pre(
        f"[URGENT PRODUCTION INCIDENT]\n"
        f"Canary crash dump reports `{spec.incident_symptom}` with raw trap code `{spec.hex_code}`.\n"
        f"Look back at the Turn 1 repository snapshot and report the EXACT error constant name "
        f"and the EXACT `{spec.param_name}` numeric value mapped to `{spec.hex_code}`.\n"
        f"Format strictly as: `<CONSTANT_NAME> | <{spec.param_name.upper()}_VALUE>`"
    )

    return {
        "spec": spec,
        "turn1_shared": turn1_shared,
        "turn2_agent_a": turn2_agent_a,
        "turn2_agent_b": turn2_agent_b,
        "distractor_pairs": distractor_pairs,
        "turn12_prompt": turn12_prompt,
        "boundary_ids": delim["boundary_ids"],
        "stop_ids": delim["stop_ids"],
    }


def score_delayed_recall_response(resp: str, spec: DelayedRecallSpec) -> dict[str, Any]:
    norm = resp.upper()
    has_const = spec.const_name.upper() in norm
    has_val = spec.param_val in resp
    return {
        "has_const": has_const,
        "has_val": has_val,
        "partial_score": 0.5 * float(has_const) + 0.5 * float(has_val),
        "exact_match": bool(has_const and has_val),
    }
