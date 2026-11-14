"""Host-side bucket-SAH BVH construction.

The FCPW-style NumPy builder and its JAX-array payload are shared by the
Warp query backends and static factories. This module only builds trees
and is not a query backend.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.bvh.payload import BVHPayload
from surfaces.explicit.primitives import Primitives


_INDEX_DTYPE = jnp.int32
_NDIM = 3

# FCPW defaults (bvh.h Bvh<DIM>::buildLBvhCpu / SAH constants).
_N_BUCKETS = 8
_LEAF_SIZE_DEFAULT = 4
_FCPW_BVH_MAX_DEPTH = 64
_TRAV_COST = np.float32(1.0)
_ISECT_COST = np.float32(1.0)


# =====================================================================
# Host-side build helpers (Note section "BVH build", FCPW bvh.inl 1-217).
# =====================================================================


def _surface_area_np(lo: np.ndarray, hi: np.ndarray, dimension: int = 3) -> float:
    """Box area in 3D, perimeter in 2D (zero-padded bounds)."""
    e = np.maximum(hi - lo, 0.0)
    if dimension == 2:
        return float(2.0 * e.sum())
    return float(2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]))


class _BuildState:
    """Growing storage for nodes + flat prim-id permutation. The FCPW
    builder is in-place on a `primitives[]` vector; we mirror that with a
    `prim_perm` integer array that holds the permuted primitive index
    sequence (no duplication — FCPW BVH does object-only splits).
    """

    def __init__(self, n_prims: int, dimension: int = 3):
        self.dimension = dimension
        cap = max(8, 2 * n_prims + 1)
        self.node_lower = np.zeros((cap, 3), dtype=np.float32)
        self.node_upper = np.zeros((cap, 3), dtype=np.float32)
        self.node_left = np.full((cap,), -1, dtype=np.int32)
        self.node_right = np.full((cap,), -1, dtype=np.int32)
        self.node_leaf_offset = np.zeros((cap,), dtype=np.int32)
        self.node_leaf_count = np.zeros((cap,), dtype=np.int32)
        self.next_free = 0

    def alloc(self) -> int:
        if self.next_free >= self.node_lower.shape[0]:
            self._grow(self.node_lower.shape[0] * 2)
        idx = self.next_free
        self.next_free += 1
        return idx

    def _grow(self, new_cap: int) -> None:
        for name in ("node_left", "node_right",
                     "node_leaf_offset", "node_leaf_count"):
            arr = getattr(self, name)
            new_arr = np.zeros((new_cap,), dtype=arr.dtype)
            if name in ("node_left", "node_right"):
                new_arr.fill(-1)
            new_arr[:arr.shape[0]] = arr
            setattr(self, name, new_arr)
        for name in ("node_lower", "node_upper"):
            arr = getattr(self, name)
            new_arr = np.zeros((new_cap, 3), dtype=arr.dtype)
            new_arr[:arr.shape[0]] = arr
            setattr(self, name, new_arr)


def _compute_object_split(node_lo: np.ndarray, node_hi: np.ndarray,
                          centroid_lo: np.ndarray, centroid_hi: np.ndarray,
                          ref_lo: np.ndarray, ref_hi: np.ndarray,
                          ref_cent: np.ndarray,
                          prim_idx: np.ndarray, dimension: int = 3) -> Tuple[int, float, float]:
    """Bucket-SAH split selection over all 3 axes (Note step build:7-8;
    Source: bvh.inl:computeObjectSplit, lines 47-119).

    Returns `(best_axis, best_split_coord, best_cost)` with `best_axis=-1`
    if no valid bucketed split was found (caller falls back to longest-
    axis-centre on the centroid bbox).
    """
    n = prim_idx.shape[0]
    extent = node_hi - node_lo
    total_sa = _surface_area_np(node_lo, node_hi, dimension)
    if total_sa <= 0.0 and dimension == 3:
        dimension = 2  # Length-based costs for collinear 3D geometry.
        total_sa = _surface_area_np(node_lo, node_hi, dimension)
    if total_sa <= 0.0:
        return -1, 0.0, float(np.inf)
    inv_total_sa = 1.0 / total_sa

    best_axis = -1
    best_coord = 0.0
    best_cost = float(np.inf)

    for axis in range(_NDIM):
        ext_a = float(extent[axis])
        if ext_a <= 0.0:
            continue
        bucket_width = ext_a / _N_BUCKETS
        # Bin centroids; bucketIndex = clamp((c - pMin) / bw, 0, N-1).
        cent_a = ref_cent[prim_idx, axis]
        bidx = np.floor((cent_a - node_lo[axis]) / bucket_width).astype(np.int32)
        bidx = np.clip(bidx, 0, _N_BUCKETS - 1)

        # Per-bucket bounds + counts.
        bucket_lo = np.full((_N_BUCKETS, 3), np.inf, dtype=np.float32)
        bucket_hi = np.full((_N_BUCKETS, 3), -np.inf, dtype=np.float32)
        bucket_n = np.zeros((_N_BUCKETS,), dtype=np.int32)
        for b in range(_N_BUCKETS):
            mask = bidx == b
            if not np.any(mask):
                continue
            sel = prim_idx[mask]
            bucket_lo[b] = ref_lo[sel].min(axis=0)
            bucket_hi[b] = ref_hi[sel].max(axis=0)
            bucket_n[b] = int(mask.sum())

        # Right-to-left sweep: cumulative right boxes + counts (Source:
        # bvh.inl:81-88).
        right_lo = np.full((_N_BUCKETS + 1, 3), np.inf, dtype=np.float32)
        right_hi = np.full((_N_BUCKETS + 1, 3), -np.inf, dtype=np.float32)
        right_n = np.zeros((_N_BUCKETS + 1,), dtype=np.int32)
        for b in range(_N_BUCKETS - 1, -1, -1):
            right_lo[b] = np.minimum(right_lo[b + 1], bucket_lo[b])
            right_hi[b] = np.maximum(right_hi[b + 1], bucket_hi[b])
            right_n[b] = right_n[b + 1] + bucket_n[b]

        # Left-to-right sweep: evaluate cost at each split boundary
        # (Source: bvh.inl:91-108).
        left_lo = np.full((3,), np.inf, dtype=np.float32)
        left_hi = np.full((3,), -np.inf, dtype=np.float32)
        n_left = 0
        for b in range(1, _N_BUCKETS):
            left_lo = np.minimum(left_lo, bucket_lo[b - 1])
            left_hi = np.maximum(left_hi, bucket_hi[b - 1])
            n_left += int(bucket_n[b - 1])
            n_right = int(right_n[b])
            if n_left > 0 and n_right > 0:
                sa_l = _surface_area_np(left_lo, left_hi, dimension)
                sa_r = _surface_area_np(right_lo[b], right_hi[b], dimension)
                # SAH unit-cost form (Note: trav=1, isect=1).
                cost = (float(_TRAV_COST)
                        + float(_ISECT_COST) * inv_total_sa
                        * (sa_l * n_left + sa_r * n_right))
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_coord = float(node_lo[axis] + b * bucket_width)

    return best_axis, best_coord, best_cost


def _partition_centroid(prim_idx: np.ndarray, ref_cent: np.ndarray,
                        axis: int, split_coord: float) -> Tuple[np.ndarray, np.ndarray]:
    """Partition `prim_idx` by `centroid[axis] < split_coord` (Source:
    bvh.inl:performObjectSplit, 121-148). Falls back to median on
    degenerate splits (all-left or all-right).
    """
    cent_a = ref_cent[prim_idx, axis]
    left_mask = cent_a < split_coord
    left = prim_idx[left_mask]
    right = prim_idx[~left_mask]
    if left.shape[0] == 0 or right.shape[0] == 0:
        # Median fallback (FCPW: midpoint partition on the index range).
        n = prim_idx.shape[0]
        half = n // 2
        left = prim_idx[:half]
        right = prim_idx[half:]
    return left, right


def _build_recursive(state: _BuildState, node_id: int,
                     prim_idx: np.ndarray,
                     ref_lo: np.ndarray, ref_hi: np.ndarray,
                     ref_cent: np.ndarray,
                     leaf_size: int, depth: int,
                     prim_perm_out: list) -> None:
    """Recursive bucket-SAH BVH build (Source: bvh.inl:buildRecursive,
    150-217). `prim_perm_out` is appended in-place with leaf-order
    primitive ids — equivalent to FCPW's in-place `primitives[]`
    reordering.
    """
    n_refs = prim_idx.shape[0]

    # Tight per-node AABB (Source: bvh.inl:167-172).
    node_lo = ref_lo[prim_idx].min(axis=0)
    node_hi = ref_hi[prim_idx].max(axis=0)
    state.node_lower[node_id] = node_lo
    state.node_upper[node_id] = node_hi

    # Note step build:5 — leaf if at-or-below threshold, or near max depth.
    if n_refs <= leaf_size or depth >= _FCPW_BVH_MAX_DEPTH - 2:
        offset = len(prim_perm_out)
        prim_perm_out.extend(int(p) for p in prim_idx)
        state.node_leaf_offset[node_id] = offset
        state.node_leaf_count[node_id] = n_refs
        state.node_left[node_id] = -1
        state.node_right[node_id] = -1
        return

    # Centroid bbox (Source: bvh.inl:167-172).
    cent_lo = ref_cent[prim_idx].min(axis=0)
    cent_hi = ref_cent[prim_idx].max(axis=0)

    # Note step build:7-8 — bucket-SAH search.
    axis, coord, best_cost = _compute_object_split(
        node_lo, node_hi, cent_lo, cent_hi,
        ref_lo, ref_hi, ref_cent, prim_idx, state.dimension)

    # Longest-axis-centre fallback when no bucketed split was valid
    # (Source: bvh.inl:113-116).
    if axis == -1:
        cent_extent = cent_hi - cent_lo
        axis = int(np.argmax(cent_extent))
        coord = 0.5 * float(cent_lo[axis] + cent_hi[axis])
        best_cost = float(np.inf)  # treated as "no SAH win" below

    # Note step build:5 (cont.) — additional FCPW-spec leaf rule:
    # accept the leaf if best SAH cost exceeds the trivial leaf cost
    # `isect_cost * n_prims`. This is the per-brief deviation from FCPW's
    # original (which only checks size/depth); it implements the standard
    # SAH "leaf vs split" comparison.
    leaf_cost = float(_ISECT_COST) * n_refs
    if best_cost > leaf_cost:
        offset = len(prim_perm_out)
        prim_perm_out.extend(int(p) for p in prim_idx)
        state.node_leaf_offset[node_id] = offset
        state.node_leaf_count[node_id] = n_refs
        state.node_left[node_id] = -1
        state.node_right[node_id] = -1
        return

    # Note step build:9 — partition primitives by centroid side.
    left_idx, right_idx = _partition_centroid(prim_idx, ref_cent, axis, coord)

    # Note step build:10 — allocate children, recurse left then right.
    left_id = state.alloc()
    right_id = state.alloc()
    state.node_left[node_id] = left_id
    state.node_right[node_id] = right_id
    state.node_leaf_offset[node_id] = 0
    state.node_leaf_count[node_id] = 0

    _build_recursive(state, left_id, left_idx, ref_lo, ref_hi, ref_cent,
                     leaf_size, depth + 1, prim_perm_out)
    _build_recursive(state, right_id, right_idx, ref_lo, ref_hi, ref_cent,
                     leaf_size, depth + 1, prim_perm_out)


def _build_payload(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray,
                   leaf_size: int, *, dimension: int = 3) -> BVHPayload:
    """Host-side bucket-SAH BVH build (Note section "BVH build", FCPW
    bvh.inl 1-242). Returns a fully-populated shared `BVHPayload`.
    """
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    # Per-primitive AABBs already provided; centroids are midpoints.
    ref_lo = prim_lo_np.astype(np.float32)
    ref_hi = prim_hi_np.astype(np.float32)
    ref_cent = (0.5 * (ref_lo + ref_hi)).astype(np.float32)

    state = _BuildState(n, dimension)
    root = state.alloc()
    assert root == 0
    prim_perm_out: list = []
    _build_recursive(state, root,
                     np.arange(n, dtype=np.int32),
                     ref_lo, ref_hi, ref_cent,
                     leaf_size=max(1, leaf_size),
                     depth=0,
                     prim_perm_out=prim_perm_out)

    n_nodes = state.next_free
    prim_ids = np.asarray(prim_perm_out, dtype=np.int32)
    assert prim_ids.shape[0] == n, (
        f"FCPW BVH does object-only splits (no duplication): "
        f"prim_ids length {prim_ids.shape[0]} != n_primitives {n}")

    return BVHPayload(
        node_lower=jnp.asarray(state.node_lower[:n_nodes]),
        node_upper=jnp.asarray(state.node_upper[:n_nodes]),
        node_left=jnp.asarray(state.node_left[:n_nodes]),
        node_right=jnp.asarray(state.node_right[:n_nodes]),
        node_leaf_offset=jnp.asarray(state.node_leaf_offset[:n_nodes]),
        node_leaf_count=jnp.asarray(state.node_leaf_count[:n_nodes]),
        prim_ids=jnp.asarray(prim_ids),
        root=jnp.asarray(np.int32(0)),
    )


def build(primitives: Primitives, config: Config | None = None) -> BVHPayload:
    """Build a SAH payload with the original host-builder leaf settings."""
    config = config if config is not None else Config()
    n = int(primitives.n_primitives)
    idx = jnp.arange(n, dtype=_INDEX_DTYPE)
    lo, hi = primitives.primitive_aabb(idx)
    prim_lo_np = np.asarray(lo, dtype=np.float32)
    prim_hi_np = np.asarray(hi, dtype=np.float32)
    leaf_size = max(1, int(config.leaf_size_threshold)
                    if config.leaf_size_threshold else _LEAF_SIZE_DEFAULT)
    if leaf_size <= 1:
        leaf_size = _LEAF_SIZE_DEFAULT
    return _build_payload(prim_lo_np, prim_hi_np, leaf_size)
