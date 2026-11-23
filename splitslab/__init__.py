"""
SplitSlab: Write-Once Canonical-Hash Residual Tiering to Eliminate Write Amplification
and Compaction Regret in Disaggregated KV Stores.
"""

from splitslab.config import NVME_STRIPE_BYTES, SplitSlabGeometry
from splitslab.engine import ColdStripeRecord, SplitSlabAblationFlags, SplitSlabSession
from splitslab.sketch import SketchSpec, build_model_subspaces, pack_subspace_sketch, unpack_subspace_sketches
from splitslab.slab_store import (
    DARWIN_F_FULLFSYNC,
    DARWIN_F_NOCACHE,
    DirectNVMeSlabStore,
    deserialize_stripe_fp16,
    serialize_stripe_fp16,
)
from splitslab.theory import (
    analytical_recompaction_waf,
    async_lookahead_overlap_ratio,
    attention_output_perturbation_bound,
    greedy_margin_equivalence_condition,
    verify_empirical_perturbation_bound,
)

__version__ = "0.1.0"

__all__ = [
    "DARWIN_F_FULLFSYNC",
    "DARWIN_F_NOCACHE",
    "NVME_STRIPE_BYTES",
    "ColdStripeRecord",
    "DirectNVMeSlabStore",
    "SketchSpec",
    "SplitSlabAblationFlags",
    "SplitSlabGeometry",
    "SplitSlabSession",
    "analytical_recompaction_waf",
    "async_lookahead_overlap_ratio",
    "attention_output_perturbation_bound",
    "build_model_subspaces",
    "deserialize_stripe_fp16",
    "greedy_margin_equivalence_condition",
    "pack_subspace_sketch",
    "serialize_stripe_fp16",
    "unpack_subspace_sketches",
    "verify_empirical_perturbation_bound",
]
