"""Host-side Morton explicit-octree construction.

The original NumPy Morton/LCA builder retains its ordering, eight-way
layout and JAX-array payload. This module only builds trees and is not
a query backend.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.octree.payload import OctreePayload
from surfaces.explicit.primitives import Primitives


_INDEX_DTYPE = jnp.int32
_NDIM = 3
_OCTREE_FANOUT = 8
# Yggdrax morton.py:13 — 21 bits/axis * 3 axes = 63 morton bits.
_MORTON_BITS = 63
# Yggdrax octree.py:_MAX_MORTON_LEVEL — limit is 21 (bits/axis).
_MAX_MORTON_LEVEL = 21

# 21-bit-per-axis Morton magic constants (yggdrax/morton.py:16-21).
_MASK_21 = np.uint64(0x1FFFFF)
_M1 = np.uint64(0x1F00000000FFFF)
_M2 = np.uint64(0x1F0000FF0000FF)
_M3 = np.uint64(0x100F00F00F00F00F)
_M4 = np.uint64(0x10C30C30C30C30C3)
_M5 = np.uint64(0x1249249249249249)


# =====================================================================
# Morton encoding (Note step build:1; Source: yggdrax/morton.py:26-61).
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
    """63-bit Morton encode for `(N, 3)` positions inside `[bmin, bmax]`."""
    norm = (positions - bounds_min) / (bounds_max - bounds_min)
    upper = np.nextafter(np.float32(1.0), np.float32(0.0))
    norm = np.clip(norm, 0.0, upper)
    max_val = (1 << _MAX_MORTON_LEVEL) - 1
    coords = np.rint(norm * max_val).astype(np.uint64)
    x = _spread3_u64(coords[:, 0])
    y = _spread3_u64(coords[:, 1])
    z = _spread3_u64(coords[:, 2])
    return x | (y << np.uint64(1)) | (z << np.uint64(2))


def _compact1_u64(x: np.ndarray) -> np.ndarray:
    """Inverse of `_spread3_u64`: pulls every 3rd bit back together."""
    x = x & _M5
    x = (x ^ (x >> np.uint64(2))) & _M4
    x = (x ^ (x >> np.uint64(4))) & _M3
    x = (x ^ (x >> np.uint64(8))) & _M2
    x = (x ^ (x >> np.uint64(16))) & _M1
    x = (x ^ (x >> np.uint64(32))) & _MASK_21
    return x


# =====================================================================
# CLZ for uint64 NumPy arrays (yggdrax/_tree_impl.py:90-128).
# =====================================================================


def _clz_u64_np(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(x).astype(np.uint64).copy()
    n = np.zeros_like(x, dtype=np.int32)
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
# Yggdrax octree helpers (Source: yggdrax/octree.py:107-148).
# =====================================================================


def _oct_address(code: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """`(8^d - 1)/7 + (code >> (63 - 3*d))` — sortable address per node.

    Codes from different depths cannot collide because the leading term
    is the count of all nodes above `depth` in a complete octree.
    Source: yggdrax/octree.py:107-122.
    """
    d = depth.astype(np.int64)
    # Avoid huge shifts when depth=0: `>> 63` is well-defined for uint64.
    shift = np.uint64(63) - (np.uint64(3) * d.astype(np.uint64))
    # Level offset: sum_{k=0}^{d-1} 8^k = (8^d - 1) // 7
    eight_pow_d = np.where(d <= 21,
                           np.left_shift(np.uint64(1), np.uint64(3) * d.astype(np.uint64)),
                           np.uint64(0))
    level_offset = (eight_pow_d - np.uint64(1)) // np.uint64(7)
    prefix = np.where(d > 0,
                      np.right_shift(code.astype(np.uint64), shift),
                      np.uint64(0))
    return level_offset + prefix


def _prefix_code(code: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Truncate `code` to its top `3*depth` bits (note quirk #4)."""
    d = depth.astype(np.int64)
    shift = np.uint64(63) - (np.uint64(3) * d.astype(np.uint64))
    truncated = np.left_shift(np.right_shift(code.astype(np.uint64), shift), shift)
    return np.where(d > 0, truncated, np.uint64(0))


def _common_depth_from_codes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """LCA depth between two Morton codes (yggdrax/octree.py:138-148).

    Morton codes occupy bits [0, 62] of a uint64 (bit 63 is the padding
    slot — yggdrax uses 63-bit morton in a 64-bit word). The common-
    prefix length over morton bits is therefore `clz_of_xor - 1`
    (subtract 1 to skip the always-zero padding bit). Each octree level
    consumes 3 morton bits, so common depth = `(clz - 1) // 3`. When
    `xor == 0` the codes are identical, `clz = 64` gives `(63) // 3 = 21
    = _MAX_MORTON_LEVEL` — exactly the desired clamp.
    """
    xor = np.bitwise_xor(a.astype(np.uint64), b.astype(np.uint64))
    clz = _clz_u64_np(xor)
    depth = (clz - np.int32(1)) // np.int32(3)
    return np.clip(depth, 0, _MAX_MORTON_LEVEL).astype(np.int32)


def _octant_at_depth(code: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Recover the 0..7 octant the node occupies inside its parent.

    Octant is the 3 bits at position `64 - 3*depth` (i.e. the depth'th
    triplet from the MSB). Source: yggdrax/octree.py (octant_at_depth).
    """
    d = depth.astype(np.int64)
    shift = np.uint64(63) - (np.uint64(3) * d.astype(np.uint64))
    bits = np.right_shift(code.astype(np.uint64), shift) & np.uint64(0x7)
    return bits.astype(np.int32)


# =====================================================================
# Build entry point (Note pipeline steps 1-9).
# =====================================================================


def _build_payload(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray,
                   leaf_size: int) -> Tuple[OctreePayload, int]:
    """Host-side build: Morton sort, leaf bucketize, LCA enumerate,
    dedup, parent/child scatter, refit. Returns `(OctreePayload,
    tree_depth)`.
    """
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    centroids = (0.5 * (prim_lo_np + prim_hi_np)).astype(np.float32)
    bmin = centroids.min(axis=0)
    bmax = centroids.max(axis=0)
    # Tight scene enclosure plus a tiny epsilon to avoid zero-extent axes.
    span = np.maximum(bmax - bmin, np.float32(1e-20))
    bmax = bmin + span

    # --- Note step build:1 — encode 63-bit Morton codes for centroids.
    codes = _morton_encode_np(centroids, bmin, bmax)
    # --- Note step build:2 — stable Morton order with original-index tie-break.
    sorted_indices = np.lexsort((np.arange(n, dtype=np.int64), codes)).astype(np.int32)
    sorted_codes = codes[sorted_indices].astype(np.uint64)

    # --- Note step build:3 — partition Morton-sorted stream into leaves.
    # Leaves are buckets of `leaf_size` Morton-sorted primitives. Each
    # leaf gets a representative Morton code = first primitive's code.
    leaf_size = max(1, int(leaf_size))
    leaf_starts = np.arange(0, n, leaf_size, dtype=np.int32)
    leaf_ends = np.minimum(leaf_starts + leaf_size, n).astype(np.int32)
    num_leaves = int(leaf_starts.shape[0])
    leaf_codes = sorted_codes[leaf_starts]

    # Choose a uniform leaf depth: deepest depth where all leaves remain
    # distinguishable. Practically we use _MAX_MORTON_LEVEL (the radix
    # leaves "live at the bottom"); this is faithful to yggdrax's
    # `build_fixed_depth_tree` upstream (note line 149-151).
    leaf_depths = np.full((num_leaves,), _MAX_MORTON_LEVEL, dtype=np.int32)

    # Single-leaf shortcut: only the root exists, holding all primitives.
    if num_leaves == 1:
        return _payload_single_leaf(prim_lo_np, prim_hi_np, sorted_indices), 1

    # --- Note step build:2 (LCA enumeration) — for each adjacent pair of
    # Morton-sorted leaves, compute their lowest-common-ancestor depth +
    # truncated prefix code. This yields N-1 internal-node candidates.
    a = leaf_codes[:-1]
    b = leaf_codes[1:]
    lca_depths = _common_depth_from_codes(a, b)
    lca_codes = _prefix_code(a, lca_depths)

    # --- Note step build:3 — concat leaves + LCAs.
    all_codes = np.concatenate([leaf_codes, lca_codes]).astype(np.uint64)
    all_depths = np.concatenate([leaf_depths, lca_depths]).astype(np.int32)
    # Mark which entries originate as leaves (so we can map leaf->slot
    # later for the range scatter).
    is_leaf_origin = np.concatenate([
        np.ones((num_leaves,), dtype=np.bool_),
        np.zeros((lca_depths.shape[0],), dtype=np.bool_),
    ])
    leaf_origin_idx = np.concatenate([
        np.arange(num_leaves, dtype=np.int32),
        np.full((lca_depths.shape[0],), -1, dtype=np.int32),
    ])

    # --- Note step build:4 — sortable address per (code, depth).
    addr = _oct_address(all_codes, all_depths)

    # --- Note step build:5 — stable argsort + dedup.
    order = np.argsort(addr, kind="stable")
    addr_s = addr[order]
    codes_s = all_codes[order]
    depths_s = all_depths[order]
    is_leaf_s = is_leaf_origin[order]
    leaf_origin_s = leaf_origin_idx[order]

    is_new = np.concatenate([
        np.array([True]),
        addr_s[1:] != addr_s[:-1],
    ])
    unique_index = np.cumsum(is_new.astype(np.int32)) - np.int32(1)
    num_unique = int(is_new.sum())

    # Compact arrays — one entry per unique octree node.
    first_idx = np.flatnonzero(is_new).astype(np.int32)
    oct_node_codes = codes_s[first_idx].astype(np.uint64)
    oct_node_depths = depths_s[first_idx].astype(np.int32)
    oct_addr = addr_s[first_idx].astype(np.uint64)

    # Map sorted-unique slot -> originating-leaf slot (-1 if internal).
    oct_leaf_origin = np.full((num_unique,), -1, dtype=np.int32)
    # Walk through entries; if an entry is a leaf-origin and its slot's
    # leaf_origin not yet set, assign it. Multiple identical leaf entries
    # cannot occur (leaves have unique codes by Morton-sort); but a leaf
    # may collide with an LCA at the same address if a parent "happens
    # to be" a leaf (won't here since leaf depth=21 is the deepest).
    for k in range(addr_s.shape[0]):
        if is_leaf_s[k]:
            slot = int(unique_index[k])
            if oct_leaf_origin[slot] < 0:
                oct_leaf_origin[slot] = int(leaf_origin_s[k])

    # --- Note step build:6 — parent links via deepest-ancestor lookup.
    # The LCA-compressed tree does NOT materialise every intermediate
    # depth: a depth-21 leaf's parent may be a depth-2 LCA node if no
    # depth-3..20 ancestors are needed. So parent linking cannot assume
    # `parent_depth = depth - 1`. Instead, for each non-root node, scan
    # candidate ancestor depths from `depth-1` down to 0 and pick the
    # deepest one whose (prefix_code, depth) address is present in the
    # sorted unique-address table.
    valid_internal = oct_node_depths > 0
    oct_parent = np.full((num_unique,), -1, dtype=np.int32)
    parent_at_depth = np.full((num_unique,), -1, dtype=np.int32)
    # Walk from deepest possible parent (depth-1) down to 0.
    for pd in range(_MAX_MORTON_LEVEL, -1, -1):
        # Only nodes whose own depth is > pd can have a parent at depth pd.
        candidate_mask = valid_internal & (oct_parent < 0) & (oct_node_depths > pd)
        if not np.any(candidate_mask):
            continue
        cand_idx = np.flatnonzero(candidate_mask)
        cand_codes = oct_node_codes[cand_idx]
        cand_pd = np.full((cand_idx.shape[0],), pd, dtype=np.int32)
        cand_pcode = _prefix_code(cand_codes, cand_pd)
        cand_paddr = _oct_address(cand_pcode, cand_pd)
        lookup = np.searchsorted(oct_addr, cand_paddr).astype(np.int32)
        lookup_safe = np.clip(lookup, 0, num_unique - 1)
        matched = oct_addr[lookup_safe] == cand_paddr
        # Record the parent and the depth at which it was found, for use
        # in the children scatter (octant must be read at parent_depth+1
        # of the child's code, not at the child's own depth).
        for i, slot in enumerate(cand_idx):
            if matched[i]:
                oct_parent[slot] = lookup_safe[i]
                parent_at_depth[slot] = pd
    oct_parent = oct_parent.astype(np.int32)

    # --- Note step build:7 — children scatter via per-node octant.
    # Octant index within the parent = the 3 bits of the child's Morton
    # code at depth (parent_depth + 1).
    child_octant_depth = np.where(oct_parent >= 0,
                                  parent_at_depth + 1,
                                  oct_node_depths)
    octant = _octant_at_depth(oct_node_codes, child_octant_depth)
    node_children = np.full((num_unique, _OCTREE_FANOUT), -1, dtype=np.int32)
    has_parent = oct_parent >= 0
    if np.any(has_parent):
        parents_for_scatter = oct_parent[has_parent]
        octants_for_scatter = octant[has_parent]
        slots_for_scatter = np.arange(num_unique, dtype=np.int32)[has_parent]
        node_children[parents_for_scatter, octants_for_scatter] = slots_for_scatter

    has_any_child = (node_children >= 0).any(axis=1)
    node_is_leaf = ~has_any_child

    # --- Note step build:8 — particle ranges.
    # Octree leaf = no children; assign particle range from originating
    # leaf bucket. Then bottom-up fold for internal nodes (level-major).
    node_leaf_offset = np.zeros((num_unique,), dtype=np.int32)
    node_leaf_count = np.zeros((num_unique,), dtype=np.int32)
    range_lo = np.full((num_unique,), n, dtype=np.int32)  # min start
    range_hi = np.full((num_unique,), -1, dtype=np.int32)  # max end-exclusive

    # Initialise leaf ranges from the originating leaf bucket.
    leaf_slots_idx = np.flatnonzero(node_is_leaf)
    for slot in leaf_slots_idx:
        origin = oct_leaf_origin[slot]
        if origin >= 0:
            s = int(leaf_starts[origin])
            e = int(leaf_ends[origin])
            range_lo[slot] = s
            range_hi[slot] = e
        else:
            # Internal node demoted to leaf because it has no children
            # in our enumeration (degenerate; shouldn't occur for
            # well-formed input). Skip range; leaf will be empty.
            range_lo[slot] = 0
            range_hi[slot] = 0

    # Bottom-up fold: process by decreasing depth.
    depth_order = np.argsort(-oct_node_depths, kind="stable")
    for slot in depth_order:
        if not node_is_leaf[slot]:
            cs = node_children[slot]
            valid_cs = cs[cs >= 0]
            if valid_cs.shape[0] > 0:
                range_lo[slot] = range_lo[valid_cs].min()
                range_hi[slot] = range_hi[valid_cs].max()

    node_leaf_offset[node_is_leaf] = range_lo[node_is_leaf]
    node_leaf_count[node_is_leaf] = np.maximum(
        range_hi[node_is_leaf] - range_lo[node_is_leaf], 0)

    # --- Note step build:9 — refit AABBs bottom-up.
    INF = np.float32(np.inf)
    node_lower = np.full((num_unique, 3), INF, dtype=np.float32)
    node_upper = np.full((num_unique, 3), -INF, dtype=np.float32)

    # Leaves: tight enclosure of contained primitives' AABBs.
    for slot in leaf_slots_idx:
        cnt = int(node_leaf_count[slot])
        if cnt <= 0:
            # Empty leaf — degenerate AABB equal to a point at scene min.
            node_lower[slot] = bmin
            node_upper[slot] = bmin
            continue
        offset = int(node_leaf_offset[slot])
        ids = sorted_indices[offset:offset + cnt]
        node_lower[slot] = prim_lo_np[ids].min(axis=0)
        node_upper[slot] = prim_hi_np[ids].max(axis=0)

    # Internal nodes: union of populated children's AABBs (bottom-up).
    for slot in depth_order:
        if node_is_leaf[slot]:
            continue
        cs = node_children[slot]
        valid_cs = cs[cs >= 0]
        if valid_cs.shape[0] == 0:
            node_lower[slot] = bmin
            node_upper[slot] = bmin
            continue
        node_lower[slot] = node_lower[valid_cs].min(axis=0)
        node_upper[slot] = node_upper[valid_cs].max(axis=0)

    # The root sits at slot 0 because (depth=0, code=0) yields address 0
    # and sorts first (note quirk #8). Verify.
    assert oct_node_depths[0] == 0, "root must be at slot 0 (yggdrax invariant)"

    payload = OctreePayload(
        node_lower=jnp.asarray(node_lower),
        node_upper=jnp.asarray(node_upper),
        node_children=jnp.asarray(node_children),
        node_is_leaf=jnp.asarray(node_is_leaf),
        node_leaf_offset=jnp.asarray(node_leaf_offset),
        node_leaf_count=jnp.asarray(node_leaf_count),
        prim_ids=jnp.asarray(sorted_indices),
        root=jnp.asarray(np.int32(0)),
    )
    tree_depth = int(oct_node_depths.max() + 1)
    return payload, tree_depth


def _payload_single_leaf(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray,
                         sorted_indices: np.ndarray) -> OctreePayload:
    """Single-node tree: root is a leaf containing every primitive."""
    n = int(sorted_indices.shape[0])
    node_lower = prim_lo_np.min(axis=0, keepdims=True).astype(np.float32)
    node_upper = prim_hi_np.max(axis=0, keepdims=True).astype(np.float32)
    node_children = np.full((1, _OCTREE_FANOUT), -1, dtype=np.int32)
    node_is_leaf = np.array([True], dtype=np.bool_)
    node_leaf_offset = np.array([0], dtype=np.int32)
    node_leaf_count = np.array([n], dtype=np.int32)
    return OctreePayload(
        node_lower=jnp.asarray(node_lower),
        node_upper=jnp.asarray(node_upper),
        node_children=jnp.asarray(node_children),
        node_is_leaf=jnp.asarray(node_is_leaf),
        node_leaf_offset=jnp.asarray(node_leaf_offset),
        node_leaf_count=jnp.asarray(node_leaf_count),
        prim_ids=jnp.asarray(sorted_indices),
        root=jnp.asarray(np.int32(0)),
    )


def build(primitives: Primitives, config: Config | None = None) -> OctreePayload:
    """Build a Morton octree payload without registering a query backend."""
    config = config if config is not None else Config()
    n = int(primitives.n_primitives)
    idx = jnp.arange(n, dtype=_INDEX_DTYPE)
    lo, hi = primitives.primitive_aabb(idx)
    prim_lo_np = np.asarray(lo, dtype=np.float32)
    prim_hi_np = np.asarray(hi, dtype=np.float32)
    leaf_size = max(1, int(config.leaf_size_threshold)
                    if config.leaf_size_threshold else 8)
    payload, _ = _build_payload(prim_lo_np, prim_hi_np, leaf_size)
    return payload
