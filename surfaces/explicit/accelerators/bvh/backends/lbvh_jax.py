"""Host-side Karras LBVH construction.

Morton encoding, radix-tree construction and refitting run in NumPy;
results retain the original JAX-array payload layout. This module only
builds trees and is not a query backend.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.bvh.payload import BVHPayload
from surfaces.explicit.primitives import Primitives


# We use the shared BVHPayload directly: Karras LBVH produces (node
# bounds, two-child topology, leaf prim ranges, prim_ids) which is
# exactly the BVHPayload SoA. No backend-private extras needed.


_INDEX_DTYPE = jnp.int32

# 21-bit-per-axis Morton magic constants (yggdrax/morton.py:16-21).
# Held as NumPy uint64 since the build is host-side.
_MASK_21 = np.uint64(0x1FFFFF)
_M1 = np.uint64(0x1F00000000FFFF)
_M2 = np.uint64(0x1F0000FF0000FF)
_M3 = np.uint64(0x100F00F00F00F00F)
_M4 = np.uint64(0x10C30C30C30C30C3)
_M5 = np.uint64(0x1249249249249249)


# =====================================================================
# Morton encoding (build step 1; Source: yggdrax/morton.py:26-61).
# Implemented in NumPy uint64.
# =====================================================================


def _spread3_u64(x: np.ndarray) -> np.ndarray:
    x = np.bitwise_and(x, _MASK_21)
    x = (x | (x << np.uint64(32))) & _M1
    x = (x | (x << np.uint64(16))) & _M2
    x = (x | (x << np.uint64(8))) & _M3
    x = (x | (x << np.uint64(4))) & _M4
    x = (x | (x << np.uint64(2))) & _M5
    return x


def _morton_encode_np(positions: np.ndarray, bounds_min: np.ndarray,
                      bounds_max: np.ndarray) -> np.ndarray:
    """63-bit Morton encode for `(N, 3)` positions inside [bmin, bmax]."""
    # Map a collapsed axis to the same mid-grid bucket for every primitive.
    extent = bounds_max - bounds_min
    degenerate = extent == 0
    safe_extent = np.where(degenerate, np.float32(1.0), extent)
    norm = (positions - bounds_min) / safe_extent
    norm = np.where(degenerate, np.float32(0.5), norm)
    # `nextafter(1.0, 0.0)` clip — yggdrax/morton.py:50 quirk.
    upper = np.nextafter(np.float32(1.0), np.float32(0.0))
    norm = np.clip(norm, 0.0, upper)
    max_val = (1 << 21) - 1
    coords = np.rint(norm * max_val).astype(np.uint64)
    x = _spread3_u64(coords[:, 0])
    y = _spread3_u64(coords[:, 1])
    z = _spread3_u64(coords[:, 2])
    return x | (y << np.uint64(1)) | (z << np.uint64(2))


# =====================================================================
# Count leading zeros for radix tree prefix comparisons.
# =====================================================================


def _clz_u64_np(x: np.ndarray) -> np.ndarray:
    """Branchless CLZ for uint64 NumPy arrays.

    Mirrors yggdrax/_tree_impl.py:90-128 but vectorised in NumPy.
    """
    x = x.astype(np.uint64).copy()
    n = np.zeros_like(x, dtype=np.int32)
    # (test_shift, update_shift, add) — update_shift halves each step so
    # the surviving low bits move into the next test window. Mirrors
    # yggdrax/_tree_impl_build.py:99-127 (`_clz_u64`).
    for test_shift, upd_shift, add in (
        (np.uint64(32), np.uint64(32), 32),
        (np.uint64(48), np.uint64(16), 16),
        (np.uint64(56), np.uint64(8), 8),
        (np.uint64(60), np.uint64(4), 4),
        (np.uint64(62), np.uint64(2), 2),
        (np.uint64(63), np.uint64(1), 1),
    ):
        y = x >> test_shift
        zero = (y == 0)
        n = n + np.where(zero, add, 0).astype(np.int32)
        x = np.where(zero, x << upd_shift, x)
    return np.where(x == 0, np.int32(64), n)


# =====================================================================
# Karras-2012 internal-node solve (build steps 4-9), NumPy edition.
# Source: yggdrax/_tree_impl.py:191-303 (`_build_tree_from_leaf_partitions`).
# =====================================================================


def _karras_internal_solve_np(leaf_codes: np.ndarray, num_leaves: int):
    """Vectorised delta/range/split solve over all internal nodes.

    Returns `(left_child, right_child, left_is_leaf, right_is_leaf,
              first, last)` int32 arrays of shape `(num_internal,)`.
    """
    num_internal = num_leaves - 1
    indices = np.arange(num_internal, dtype=np.int32)

    def _delta_vec(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        cond = (j >= 0) & (j < num_leaves)
        safe_j = np.where(cond, j, 0)
        ci = leaf_codes[i]
        cj = leaf_codes[safe_j]
        same = ci == cj
        clz_code = _clz_u64_np(np.bitwise_xor(ci, cj))
        clz_idx = _clz_u64_np(np.bitwise_xor(
            i.astype(np.uint64), safe_j.astype(np.uint64)))
        val = np.where(same, np.int32(64) + clz_idx, clz_code)
        return np.where(cond, val, np.int32(-1))

    delta_plus = _delta_vec(indices, indices + 1)
    delta_minus = _delta_vec(indices, indices - 1)
    d = np.sign(delta_plus - delta_minus).astype(np.int32)
    d = np.where(indices == 0, np.int32(1), d)
    d = np.where(d == 0, np.int32(1), d)

    delta_min = _delta_vec(indices, indices - d)

    # Note step build:6 — exponential-search range upper bound (31 steps).
    span = np.zeros_like(indices)
    for k in range(30, -1, -1):
        step = np.int32(1 << k)
        candidate = indices + (span + step) * d
        delta_candidate = _delta_vec(indices, candidate)
        cond = ((candidate >= 0) & (candidate < num_leaves)
                & (delta_candidate > delta_min))
        span = np.where(cond, span + step, span)

    end_idx = indices + span * d
    first = np.minimum(indices, end_idx)
    last = np.maximum(indices, end_idx)

    # Use the same leaf-index suffix for the split as for the range.
    # Morton codes can coincide even for spatially distinct primitives.
    common_prefix = _delta_vec(first, last)

    # Note step build:8 — binary-search the split (31 steps).
    split = first.copy()
    for k in range(30, -1, -1):
        step = np.int32(1 << k)
        mid = split + step
        safe_mid = np.where(mid < last, mid, last)
        cmp = _delta_vec(first, safe_mid)
        cond = (mid < last) & (cmp > common_prefix)
        split = np.where(cond, mid, split)

    left_is_leaf = (split == first)
    right_is_leaf = (split + 1 == last)

    leaf_offset = np.int32(num_internal)
    left_child = np.where(left_is_leaf, leaf_offset + first, split).astype(np.int32)
    right_child = np.where(right_is_leaf, leaf_offset + last, split + 1).astype(np.int32)

    return (left_child, right_child,
            left_is_leaf.astype(np.bool_), right_is_leaf.astype(np.bool_),
            first.astype(np.int32), last.astype(np.int32))


# =====================================================================
# AABB refit (host-side NumPy bottom-up walk).
# Equivalent to cuBQL's `refit.h` rendezvous refit.
# =====================================================================


def _refit_aabbs_np(node_left: np.ndarray, node_right: np.ndarray,
                    node_leaf_offset: np.ndarray, node_leaf_count: np.ndarray,
                    prim_ids: np.ndarray,
                    prim_lo: np.ndarray, prim_hi: np.ndarray,
                    num_internal: int, num_leaves: int):
    total = num_internal + num_leaves
    INF = np.float32(np.inf)
    node_lo = np.full((total, 3), INF, dtype=np.float32)
    node_hi = np.full((total, 3), -INF, dtype=np.float32)

    # Initialise leaf bounds.
    for li in range(num_leaves):
        node_id = num_internal + li
        start = int(node_leaf_offset[node_id])
        count = int(node_leaf_count[node_id])
        if count <= 0:
            continue
        ids = prim_ids[start:start + count]
        node_lo[node_id] = prim_lo[ids].min(axis=0)
        node_hi[node_id] = prim_hi[ids].max(axis=0)

    if num_internal == 0:
        return node_lo, node_hi

    # Compute depth via parent-pointer iteration. We have child arrays;
    # build parent by scatter then walk leaves up to the root, settling
    # internal nodes in topological order.
    parent = np.full((total,), -1, dtype=np.int32)
    parent[node_left[:num_internal]] = np.arange(num_internal, dtype=np.int32)
    parent[node_right[:num_internal]] = np.arange(num_internal, dtype=np.int32)

    # ready[i] = number of children settled. Internal node settles when both
    # children are settled. Process via BFS up from leaves.
    ready = np.zeros((num_internal,), dtype=np.int32)
    queue = list(range(num_internal, total))  # leaves
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        p = parent[node]
        if p < 0:
            continue
        ready[p] += 1
        if ready[p] == 2:
            l = int(node_left[p])
            r = int(node_right[p])
            node_lo[p] = np.minimum(node_lo[l], node_lo[r])
            node_hi[p] = np.maximum(node_hi[l], node_hi[r])
            queue.append(p)

    return node_lo, node_hi


# =====================================================================
# Build entry point.
# =====================================================================


def _build_payload(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray,
                   leaf_size: int) -> BVHPayload:
    """Host-side build: Morton sort, Karras solve, refit. Returns BVHPayload.
    """
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    # Preserve the original Morton quantization for the host builder.
    with np.errstate(over="ignore", invalid="ignore"):
        centroids = (0.5 * (prim_lo_np + prim_hi_np)).astype(np.float32)
        bmin = centroids.min(axis=0)
        bmax = bmin + np.maximum(centroids.max(axis=0) - bmin, 1e-20)
        finite_extent = np.isfinite(bmax - bmin).all()
    if not finite_extent:
        # The full float32 coordinate range needs wider midpoint and extent arithmetic.
        centroids = 0.5 * (prim_lo_np.astype(np.float64) + prim_hi_np)
        bmin = centroids.min(axis=0)
        bmax = bmin + np.maximum(centroids.max(axis=0) - bmin, 1e-20)

    # Note step build:1 — encode 63-bit Morton codes.
    codes = _morton_encode_np(centroids, bmin, bmax)
    # Note step build:2 — stable Morton order with original-index tie-break
    # via lexsort (last key is primary).
    sorted_indices = np.lexsort((np.arange(n, dtype=np.int64), codes))
    sorted_indices = sorted_indices.astype(np.int32)
    sorted_codes = codes[sorted_indices].astype(np.uint64)

    # Note step build:3 — partition sorted stream into Morton leaves.
    leaf_starts = np.arange(0, n, leaf_size, dtype=np.int32)
    leaf_ends = np.minimum(leaf_starts.astype(np.int64) + leaf_size, n).astype(np.int32)
    num_leaves = int(leaf_starts.shape[0])
    num_internal = num_leaves - 1
    total = num_internal + num_leaves

    node_left = np.full((total,), -1, dtype=np.int32)
    node_right = np.full((total,), -1, dtype=np.int32)
    node_leaf_offset = np.zeros((total,), dtype=np.int32)
    node_leaf_count = np.zeros((total,), dtype=np.int32)
    node_leaf_offset[num_internal:] = leaf_starts
    node_leaf_count[num_internal:] = leaf_ends - leaf_starts

    if num_internal > 0:
        leaf_codes = sorted_codes[leaf_starts]
        (lc, rc, _llf, _rlf, _first, _last) = _karras_internal_solve_np(
            leaf_codes, num_leaves)
        node_left[:num_internal] = lc
        node_right[:num_internal] = rc
        # Internal nodes have leaf_count = 0 (already zero-initialised).

    # Refit.
    node_lo, node_hi = _refit_aabbs_np(
        node_left, node_right, node_leaf_offset, node_leaf_count,
        sorted_indices, prim_lo_np.astype(np.float32),
        prim_hi_np.astype(np.float32),
        num_internal=num_internal, num_leaves=num_leaves,
    )

    return BVHPayload(
        node_lower=jnp.asarray(node_lo),
        node_upper=jnp.asarray(node_hi),
        node_left=jnp.asarray(node_left),
        node_right=jnp.asarray(node_right),
        node_leaf_offset=jnp.asarray(node_leaf_offset),
        node_leaf_count=jnp.asarray(node_leaf_count),
        prim_ids=jnp.asarray(sorted_indices),
        root=jnp.asarray(np.int32(0)),
    )


def build(primitives: Primitives, config: Config | None = None) -> BVHPayload:
    """Build an LBVH payload without registering a query backend."""
    config = config if config is not None else Config()
    n = int(primitives.n_primitives)
    idx = jnp.arange(n, dtype=_INDEX_DTYPE)
    lo, hi = primitives.primitive_aabb(idx)
    prim_lo_np = np.asarray(lo, dtype=np.float32)
    prim_hi_np = np.asarray(hi, dtype=np.float32)
    leaf_size = max(1, int(config.leaf_size_threshold))
    return _build_payload(prim_lo_np, prim_hi_np, leaf_size)
