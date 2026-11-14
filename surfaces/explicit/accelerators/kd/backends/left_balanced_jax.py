"""Host-side left-balanced kd-tree construction.

The original round-robin median builder and subtree AABB refit operate
on primitive centroids in NumPy. This module only builds trees and is
not a query backend.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.kd.payload import KdPayload
from surfaces.explicit.primitives import Primitives


_INDEX_DTYPE = jnp.int32
_NDIM = 3


# =====================================================================
# Host-side build helpers.
# =====================================================================


def _left_subtree_size(n: int) -> int:
    """Number of nodes in the LEFT subtree of a left-balanced tree with `n`
    nodes, root at 0. Mirrors cudaKDTree's `ArbitraryBinaryTree` math
    (helpers.h:162-178). For `n` total nodes, the last level holds
    `n - (2^(L-1) - 1)` nodes (where `L = ceil(log2(n+1))` levels). The
    full left subtree size is `2^(L-1) - 1`; the lowest level adds up to
    `2^(L-2)` extras, capped by what is actually present.
    """
    if n <= 1:
        return 0
    levels = int(math.floor(math.log2(n))) + 1   # numLevelsFor
    last_level_capacity = 1 << (levels - 1)
    full_above = (1 << (levels - 1)) - 1         # everything above the last
    last_level_count = n - full_above
    left_full_above = (full_above - 1) // 2 if full_above > 0 else 0
    left_last_capacity = last_level_capacity // 2
    left_last_actual = min(last_level_count, left_last_capacity)
    return left_full_above + left_last_actual


def _build_left_balanced_np(centroids: np.ndarray) -> Tuple[np.ndarray,
                                                            np.ndarray]:
    """Permute primitive indices into a left-balanced kd-tree layout.

    Mirrors the *output* of cudaKDTree's tag-update cascade
    (note Build pipeline, builder_thrust.h): node `i` holds the median
    primitive of the i-th subtree along the level's split axis. Round-
    robin axis = `level(i) % 3`.

    Returns:
      prim_ids (N,)    : permutation of [0, N) — `prim_ids[i]` is the
                         primitive seated at node `i`.
      split_axis (N,)  : `level(i) % 3` for each node.
    """
    n = int(centroids.shape[0])
    prim_ids = np.empty((n,), dtype=np.int32)
    split_axis = np.empty((n,), dtype=np.int32)

    # Iterative recursion over (segment of original-id slice, dest node id,
    # depth). Each step picks the median along the level's axis, writes
    # it at `node_id`, recurses on left/right slices.
    work_idx = np.arange(n, dtype=np.int32)
    stack = [(work_idx, 0, 0)]
    while stack:
        ids, node_id, depth = stack.pop()
        m = ids.shape[0]
        if m == 0:
            continue
        axis = depth % _NDIM
        # Pivot position inside this segment so the LEFT child gets exactly
        # `_left_subtree_size(m)` items — reproduces cudaKDTree's
        # `ArrayLayoutInStep::pivotPosOf` choice.
        left_size = _left_subtree_size(m)
        coords = centroids[ids, axis]
        # `np.argpartition` is O(N); the K-th smallest goes to position K,
        # everything below is < pivot, everything above is >= pivot.
        order = np.argpartition(coords, left_size)
        left_part = ids[order[:left_size]]
        pivot = ids[order[left_size]]
        right_part = ids[order[left_size + 1:]]

        prim_ids[node_id] = pivot
        split_axis[node_id] = axis

        # Children of node `i`: 2i+1 (left), 2i+2 (right).
        left_child = 2 * node_id + 1
        right_child = 2 * node_id + 2
        if left_child < n:
            stack.append((left_part, left_child, depth + 1))
        if right_child < n:
            stack.append((right_part, right_child, depth + 1))

    return prim_ids, split_axis


def _refit_aabbs_np(prim_lo: np.ndarray, prim_hi: np.ndarray,
                    prim_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bottom-up AABB propagation over the implicit left-balanced tree.

    Each node owns one primitive (its own AABB) and the union of its
    children's bounds. Iterates nodes in reverse index order — for a
    left-balanced layout that visits all descendants of node `i` before
    `i` itself (children `2i+1`, `2i+2` always have larger indices).
    """
    n = int(prim_ids.shape[0])
    INF = np.float32(np.inf)
    node_lo = np.full((n, 3), INF, dtype=np.float32)
    node_hi = np.full((n, 3), -INF, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        pid = int(prim_ids[i])
        lo = prim_lo[pid]
        hi = prim_hi[pid]
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n:
            lo = np.minimum(lo, node_lo[l])
            hi = np.maximum(hi, node_hi[l])
        if r < n:
            lo = np.minimum(lo, node_lo[r])
            hi = np.maximum(hi, node_hi[r])
        node_lo[i] = lo
        node_hi[i] = hi
    return node_lo, node_hi


def _build_payload(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray
                   ) -> KdPayload:
    """Host-side kd-tree build. Returns a `KdPayload` with jnp arrays."""
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    # Build over centroids (note: adaptation step 1).
    centroids = (0.5 * (prim_lo_np + prim_hi_np)).astype(np.float32)

    # Permute primitives into left-balanced order (note: adaptation step 1).
    prim_ids, split_axis = _build_left_balanced_np(centroids)

    # Per-node split coordinate is the centroid component on the chosen axis
    # of the primitive seated there (note: adaptation step 4).
    node_centroids = centroids[prim_ids]  # (N, 3)
    node_arange = np.arange(n, dtype=np.int32)
    split_pos = node_centroids[node_arange, split_axis].astype(np.float32)

    # Implicit topology: children of i are 2i+1, 2i+2; -1 if out of range
    # (note: adaptation step 2).
    left = 2 * node_arange + 1
    right = 2 * node_arange + 2
    node_left = np.where(left < n, left, -1).astype(np.int32)
    node_right = np.where(right < n, right, -1).astype(np.int32)

    # Each node owns exactly one primitive (note: adaptation step 4).
    node_leaf_offset = node_arange.astype(np.int32)
    node_leaf_count = np.ones((n,), dtype=np.int32)

    # Subtree AABBs (note: adaptation step 3).
    node_lo, node_hi = _refit_aabbs_np(
        prim_lo_np.astype(np.float32),
        prim_hi_np.astype(np.float32),
        prim_ids,
    )

    return KdPayload(
        node_split_axis=jnp.asarray(split_axis),
        node_split_pos=jnp.asarray(split_pos),
        node_left=jnp.asarray(node_left),
        node_right=jnp.asarray(node_right),
        node_lower=jnp.asarray(node_lo),
        node_upper=jnp.asarray(node_hi),
        node_leaf_offset=jnp.asarray(node_leaf_offset),
        node_leaf_count=jnp.asarray(node_leaf_count),
        prim_ids=jnp.asarray(prim_ids),
        root=jnp.asarray(np.int32(0)),
    )


def build(primitives: Primitives, config: Config | None = None) -> KdPayload:
    """Build a left-balanced kd payload; Config does not affect this builder."""
    n = int(primitives.n_primitives)
    idx = jnp.arange(n, dtype=_INDEX_DTYPE)
    lo, hi = primitives.primitive_aabb(idx)
    prim_lo_np = np.asarray(lo, dtype=np.float32)
    prim_hi_np = np.asarray(hi, dtype=np.float32)
    return _build_payload(prim_lo_np, prim_hi_np)
