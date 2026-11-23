"""
Direct-NVMe Split-Plane Content-Addressed Slab Store (`fcntl.F_NOCACHE` + `F_FULLFSYNC`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from splitslab.config import NVME_STRIPE_BYTES

# Darwin/macOS kernel flags for direct page-cache bypass (equivalent to Linux O_DIRECT)
# and controller-level NAND flush.
DARWIN_F_NOCACHE: int = getattr(fcntl, "F_NOCACHE", 48)
DARWIN_F_FULLFSYNC: int = getattr(fcntl, "F_FULLFSYNC", 51)


class DirectNVMeSlabStore:
    """
    64KB-aligned binary KV slab manager backed by physical NVMe files.
    When `direct_nocache=True` on macOS, sets `fcntl(fd, F_NOCACHE, 1)` so `os.pread`
    and `os.pwrite` bypass the unified OS page cache and hit the NVMe controller directly.
    """

    def __init__(
        self,
        root_dir: Path,
        direct_nocache: bool = True,
        sync_mode: str = "fsync",  # "none" | "fsync" | "fullfsync"
        io_workers: int = 4,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.direct_nocache = direct_nocache
        self.sync_mode = sync_mode

        self.slab_path = self.root_dir / "plane1_stripes.slab"
        self.fd = os.open(str(self.slab_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        self.nocache_enabled = False
        if self.direct_nocache and sys.platform == "darwin":
            fcntl.fcntl(self.fd, DARWIN_F_NOCACHE, 1)
            self.nocache_enabled = True

        self.pool = ThreadPoolExecutor(max_workers=max(1, io_workers))
        self.write_cursor = 0

        # Physical telemetry
        self.physical_bytes_written = 0
        self.logical_unique_bytes_stored = 0
        self.physical_bytes_read = 0
        self.pread_calls = 0
        self.pwrite_calls = 0
        self.write_latencies_us: list[float] = []
        self.read_latencies_us: list[float] = []

        # Content-addressed SHA-256 index: digest -> aligned byte offset
        self.cas_index: dict[str, int] = {}
        self.dedup_queries = 0
        self.dedup_hits = 0

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def _sync_fd(self) -> None:
        if self.sync_mode == "fullfsync" and sys.platform == "darwin":
            fcntl.fcntl(self.fd, DARWIN_F_FULLFSYNC)
        elif self.sync_mode == "fsync":
            os.fsync(self.fd)

    def sync_batch(self) -> None:
        """Flushes pending writes once per turn-tiering batch."""
        if self.fd >= 0 and self.sync_mode == "batch_fsync":
            os.fsync(self.fd)

    @staticmethod
    def align_up(n: int, align: int = NVME_STRIPE_BYTES) -> int:
        return ((n + align - 1) // align) * align

    def put_immutable_stripe(
        self,
        canonical_hash: str,
        payload: bytes,
        count_dedup: bool = False,
        count_waf_logical: bool = True,
    ) -> tuple[int, bool]:
        if count_dedup:
            self.dedup_queries += 1
        if canonical_hash in self.cas_index:
            if count_dedup:
                self.dedup_hits += 1
            return self.cas_index[canonical_hash], True

        if count_dedup and not count_waf_logical:
            return -1, False

        aligned_len = self.align_up(len(payload), NVME_STRIPE_BYTES)
        buf = payload if len(payload) == aligned_len else (payload + b"\x00" * (aligned_len - len(payload)))

        offset = self.write_cursor
        t0 = time.perf_counter()
        os.pwrite(self.fd, buf, offset)
        self._sync_fd()
        self.write_latencies_us.append((time.perf_counter() - t0) * 1e6)

        self.write_cursor += aligned_len
        self.physical_bytes_written += aligned_len
        self.pwrite_calls += 1
        if count_waf_logical:
            self.logical_unique_bytes_stored += len(payload)
        self.cas_index[canonical_hash] = offset
        return offset, False

    def rewrite_mutable_page(self, offset: int, new_payload: bytes) -> int:
        """
        Models copy-on-write / read-modify-write of a 64KB flash page when iterative
        token eviction mutates a previously offloaded slab.
        """
        aligned_len = self.align_up(len(new_payload), NVME_STRIPE_BYTES)
        t0 = time.perf_counter()
        if 0 <= offset < self.write_cursor:
            _ = os.pread(self.fd, aligned_len, offset)
            self.physical_bytes_read += aligned_len
            self.pread_calls += 1
        buf = (
            new_payload
            if len(new_payload) == aligned_len
            else (new_payload + b"\x00" * (aligned_len - len(new_payload)))
        )
        new_off = self.write_cursor
        os.pwrite(self.fd, buf, new_off)
        self._sync_fd()
        self.write_latencies_us.append((time.perf_counter() - t0) * 1e6)
        self.write_cursor += aligned_len
        self.physical_bytes_written += aligned_len
        self.pwrite_calls += 1
        return new_off

    def pread_stripe(self, offset: int, nbytes: int = NVME_STRIPE_BYTES) -> bytes:
        t0 = time.perf_counter()
        raw = os.pread(self.fd, nbytes, offset)
        self.read_latencies_us.append((time.perf_counter() - t0) * 1e6)
        self.physical_bytes_read += len(raw)
        self.pread_calls += 1
        return raw

    def pread_stripes_parallel(self, offsets: list[int], nbytes: int = NVME_STRIPE_BYTES) -> list[bytes]:
        """
        Issues concurrent `os.pread` requests across `io_workers` threads to exploit
        NVMe multi-queue parallelism when recalling a micro-window of cold stripes.
        """
        if len(offsets) <= 1:
            return [self.pread_stripe(off, nbytes) for off in offsets]
        t0 = time.perf_counter()
        blobs = list(self.pool.map(lambda o: os.pread(self.fd, nbytes, o), offsets))
        dt_per_call = (time.perf_counter() - t0) * 1e6 / len(offsets)
        self.read_latencies_us.extend([dt_per_call] * len(offsets))
        self.physical_bytes_read += sum(len(b) for b in blobs)
        self.pread_calls += len(offsets)
        return blobs

    @property
    def waf(self) -> float:
        if self.logical_unique_bytes_stored == 0:
            return 1.0
        return self.physical_bytes_written / float(self.logical_unique_bytes_stored)


def serialize_stripe_fp16(k_res: torch.Tensor, v_res: torch.Tensor) -> bytes:
    """
    Packs `k_res` `[H_kv, N_res, D]` and `v_res` `[H_kv, N_res, D]` (`fp16`) into an
    exact 65,536-byte (`64 KB`) contiguous binary stripe.
    """
    k_b = k_res.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    v_b = v_res.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    blob = k_b + v_b
    if len(blob) != NVME_STRIPE_BYTES:
        raise ValueError(f"Expected {NVME_STRIPE_BYTES}-byte stripe, got {len(blob)} bytes")
    return blob


def deserialize_stripe_fp16(
    blob: bytes, num_kv_heads: int, num_res_tokens: int, head_dim: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    half = NVME_STRIPE_BYTES // 2
    k_arr = np.frombuffer(blob[:half], dtype=np.float16).reshape(1, num_kv_heads, num_res_tokens, head_dim)
    v_arr = np.frombuffer(blob[half:], dtype=np.float16).reshape(1, num_kv_heads, num_res_tokens, head_dim)
    return (
        torch.from_numpy(k_arr.copy()).to(device=device, dtype=torch.float16),
        torch.from_numpy(v_arr.copy()).to(device=device, dtype=torch.float16),
    )
