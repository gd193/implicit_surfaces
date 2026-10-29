"""Yggdrax-style explicit octree backend, pure JAX traversal.

Build follows the yggdrax explicit-octree note:
  algo_infos_crawled/octree.morton_explicit.tobibu_yggdrax.md
  algo_infos_crawled/octree.morton_explicit.tobibu_yggdrax.code/

Runtime: pure JAX (jax.jit / lax.while_loop / lax.fori_loop) for the
hot-path traversal kernels. The build is host-side NumPy, matching
`bvh.lbvh.jax` and `kd.sah.jax`: (a) `n_primitives` is static at
construction (fixed mesh), (b) Morton codes need uint64 which JAX only
exposes under `jax_enable_x64`, (c) the LCA-derivation/dedup is a
one-shot O(N log N) sort. Single-thread-per-query traversal; jax.vmap
fans queries across batches.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
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


# =====================================================================
# Traversal helpers.
# =====================================================================


def _sqr_dist_point_to_box(p: jnp.ndarray, lo: jnp.ndarray,
                           hi: jnp.ndarray) -> jnp.ndarray:
    cl = jnp.clip(p, lo, hi)
    d = p - cl
    return jnp.sum(d * d)


def _ray_box_t(origin: jnp.ndarray, rcp_dir: jnp.ndarray,
               lo: jnp.ndarray, hi: jnp.ndarray,
               tmin: jnp.ndarray, tmax: jnp.ndarray
               ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t1 = (lo - origin) * rcp_dir
    t2 = (hi - origin) * rcp_dir
    t_lo = jnp.minimum(t1, t2)
    t_hi = jnp.maximum(t1, t2)
    t_enter = jnp.maximum(jnp.max(t_lo), tmin)
    t_exit = jnp.minimum(jnp.min(t_hi), tmax)
    hit = t_enter <= t_exit
    return t_enter, hit


def _box_overlap(a_lo, a_hi, b_lo, b_hi) -> jnp.ndarray:
    return jnp.all(a_lo <= b_hi) & jnp.all(b_lo <= a_hi)


def _box_box_dist_bounds(a_lo: jnp.ndarray, a_hi: jnp.ndarray,
                         b_lo: jnp.ndarray, b_hi: jnp.ndarray
                         ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Unsigned distance bounds between two AABBs.

    Lower bound: per-axis non-negative separation, Euclidean-norm'd.
        sep_i = max(0, max(a_lo_i - b_hi_i, b_lo_i - a_hi_i))
        d_lo = sqrt(sum(sep_i^2))
    Upper bound: maximum corner-to-corner distance — bounded above by
        d_hi = sqrt(sum(max((a_hi_i - b_lo_i)^2, (b_hi_i - a_lo_i)^2)))
    This is a valid (sound) upper bound on any pair of points (xa in A,
    xb in B): for each axis, |xa_i - xb_i| <= max(a_hi_i - b_lo_i,
    b_hi_i - a_lo_i). Tighter forms exist (the FCPW point-vs-box analog
    uses min(u,v) per-axis), but for box-vs-box this max-extents form is
    also tight when A or B degenerates to a point and remains a valid
    upper bound otherwise.
    """
    sep = jnp.maximum(jnp.maximum(a_lo - b_hi, b_lo - a_hi),
                      jnp.float32(0.0))
    d_lo2 = jnp.sum(sep * sep)
    span = jnp.maximum((a_hi - b_lo) ** 2, (b_hi - a_lo) ** 2)
    d_hi2 = jnp.sum(span)
    eps = jnp.float32(0.0)
    return jnp.sqrt(jnp.maximum(d_lo2, eps)), jnp.sqrt(jnp.maximum(d_hi2, eps))


# =====================================================================
# closest_point — shrinking-radius octree descent.
# At each interior node, push every overlapping child onto the stack
# with its (squared) distance as the priority key. Pop entries whose
# saved distance no longer beats `best_d`.
# =====================================================================


def _closest_point_single(payload: OctreePayload,
                          primitives: Primitives,
                          x: jnp.ndarray,
                          max_stack_depth: int,
                          max_leaf_size: int):
    INF = jnp.float32(jnp.inf)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_d_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    init = (payload.root.astype(_INDEX_DTYPE),
            stack_idx_init, stack_d_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            INF,
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False))

    def cond_fn(state):
        return jnp.logical_not(state[7])

    def _pop(si, sd, sp, best_d, best_cp, best_pid):
        def pop_cond(s):
            _ssp, _ssi, _ssd, found, empty = s
            return jnp.logical_not(found | empty)

        def pop_body(s):
            ssp, ssi, ssd, _f, _e = s
            empty = ssp == 0
            new_sp = jnp.where(empty, ssp, ssp - 1)
            top_d = ssd[new_sp]
            in_range = (top_d < best_d) & jnp.logical_not(empty)
            return new_sp, ssi, ssd, in_range, empty

        new_sp, _, _, found, _empty = lax.while_loop(
            pop_cond, pop_body,
            (sp, si, sd, jnp.bool_(False), jnp.bool_(False)))
        next_node = jnp.where(found, si[new_sp], jnp.asarray(0, _INDEX_DTYPE))
        done = jnp.logical_not(found)
        return next_node, si, sd, new_sp, best_d, best_cp, best_pid, done

    def body_fn(state):
        cur, si, sd, sp, best_d, best_cp, best_pid, _done = state
        is_leaf = payload.node_is_leaf[cur]

        def inner_branch(_):
            # Fold over 8 child slots: push each populated child whose
            # AABB-distance still beats best_d.
            children = payload.node_children[cur]

            def child_step(k, carry):
                ssi, ssd, ssp = carry
                child = children[k]
                has_child = child >= 0
                # Guard out-of-range indexing under -1 sentinels.
                safe_child = jnp.where(has_child, child,
                                       jnp.asarray(0, _INDEX_DTYPE))
                d2 = _sqr_dist_point_to_box(
                    x, payload.node_lower[safe_child],
                    payload.node_upper[safe_child])
                # Stack is sized to 7*tree_depth + 1 (exact fanout-8
                # worst case), so overflow is provably impossible — no
                # `ssp < max_stack_depth` guard needed.
                push = has_child & (d2 < best_d)
                nsi = jnp.where(push, ssi.at[ssp].set(child), ssi)
                nsd = jnp.where(push, ssd.at[ssp].set(d2), ssd)
                nsp = jnp.where(push, ssp + 1, ssp)
                return nsi, nsd, nsp

            new_si, new_sd, new_sp = lax.fori_loop(
                0, _OCTREE_FANOUT, child_step, (si, sd, sp))
            # Pop best-priority node next.
            return _pop(new_si, new_sd, new_sp, best_d, best_cp, best_pid)

        def leaf_branch(_):
            offset = payload.node_leaf_offset[cur]
            count = payload.node_leaf_count[cur]
            prim_ids_len = payload.prim_ids.shape[0]

            def leaf_step(k, lc):
                bd, bcp, bp = lc
                valid = k < count
                pid = payload.prim_ids[jnp.minimum(offset + k,
                                                   prim_ids_len - 1)]
                cp_k, dist_k = primitives.closest_point_on_primitive(x, pid)
                d2_k = dist_k * dist_k
                better = valid & (d2_k < bd)
                bd = jnp.where(better, d2_k, bd)
                bcp = jnp.where(better, cp_k, bcp)
                bp = jnp.where(better, pid, bp)
                return bd, bcp, bp

            new_bd, new_bcp, new_bp = lax.fori_loop(
                0, max_leaf_size, leaf_step, (best_d, best_cp, best_pid))
            return _pop(si, sd, sp, new_bd, new_bcp, new_bp)

        return lax.cond(is_leaf, leaf_branch, inner_branch, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _sd, _sp, best_d, best_cp, best_pid, _done = final
    best_dist = jnp.sqrt(jnp.maximum(best_d, jnp.float32(0.0)))
    return best_cp, best_dist, best_pid


# =====================================================================
# first_hit — ray-AABB descent.
# At each interior node, push every overlapping child onto the stack
# with its t_near as priority. Sign-preserving inv_dir guard same as
# `lbvh_jax.py`.
# =====================================================================


def _first_hit_single(payload: OctreePayload,
                      primitives: Primitives,
                      ray_origin: jnp.ndarray,
                      ray_direction: jnp.ndarray,
                      max_t: jnp.ndarray,
                      max_stack_depth: int,
                      max_leaf_size: int):
    INF = jnp.float32(jnp.inf)
    safe_dir = jnp.where(jnp.abs(ray_direction) < 1e-20,
                         jnp.copysign(jnp.float32(1e-20), ray_direction),
                         ray_direction)
    rcp_dir = jnp.float32(1.0) / safe_dir
    tmin = jnp.float32(0.0)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_t_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    init = (payload.root.astype(_INDEX_DTYPE),
            stack_idx_init, stack_t_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            max_t.astype(jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False),
            jnp.bool_(False))

    def cond_fn(s):
        return jnp.logical_not(s[7])

    def _pop_ray(si, st, sp, cur_tmax, best_pid, hit):
        def pop_cond(s):
            _ssp, _ssi, _sst, found, empty = s
            return jnp.logical_not(found | empty)

        def pop_body(s):
            ssp, ssi, sst, _f, _e = s
            empty = ssp == 0
            new_sp = jnp.where(empty, ssp, ssp - 1)
            top_t = sst[new_sp]
            in_range = (top_t < cur_tmax) & jnp.logical_not(empty)
            return new_sp, ssi, sst, in_range, empty

        new_sp, _, _, found, _empty = lax.while_loop(
            pop_cond, pop_body,
            (sp, si, st, jnp.bool_(False), jnp.bool_(False)))
        next_node = jnp.where(found, si[new_sp], jnp.asarray(0, _INDEX_DTYPE))
        done = jnp.logical_not(found)
        return next_node, si, st, new_sp, cur_tmax, best_pid, hit, done

    def body_fn(s):
        cur, si, st, sp, cur_tmax, best_pid, hit, _done = s
        is_leaf = payload.node_is_leaf[cur]

        def inner(_):
            children = payload.node_children[cur]

            def child_step(k, carry):
                ssi, sst, ssp = carry
                child = children[k]
                has_child = child >= 0
                safe_child = jnp.where(has_child, child,
                                       jnp.asarray(0, _INDEX_DTYPE))
                t_enter, hit_box = _ray_box_t(
                    ray_origin, rcp_dir,
                    payload.node_lower[safe_child],
                    payload.node_upper[safe_child],
                    tmin, cur_tmax)
                # Stack sized to 7*tree_depth + 1; overflow impossible.
                push = has_child & hit_box
                nsi = jnp.where(push, ssi.at[ssp].set(child), ssi)
                nst = jnp.where(push, sst.at[ssp].set(t_enter), sst)
                nsp = jnp.where(push, ssp + 1, ssp)
                return nsi, nst, nsp

            new_si, new_st, new_sp = lax.fori_loop(
                0, _OCTREE_FANOUT, child_step, (si, st, sp))
            return _pop_ray(new_si, new_st, new_sp, cur_tmax, best_pid, hit)

        def leaf(_):
            offset = payload.node_leaf_offset[cur]
            count = payload.node_leaf_count[cur]
            prim_ids_len = payload.prim_ids.shape[0]

            def leaf_step(k, lc):
                ctm, bp, h = lc
                valid = k < count
                pid = payload.prim_ids[jnp.minimum(offset + k,
                                                   prim_ids_len - 1)]
                t_k, hit_k = primitives.ray_primitive_intersect(
                    ray_origin, ray_direction, ctm, pid)
                accept = valid & hit_k & (t_k < ctm) & (t_k >= tmin)
                ctm = jnp.where(accept, t_k, ctm)
                bp = jnp.where(accept, pid, bp)
                h = h | accept
                return ctm, bp, h

            new_tm, new_bp, new_h = lax.fori_loop(
                0, max_leaf_size, leaf_step, (cur_tmax, best_pid, hit))
            return _pop_ray(si, st, sp, new_tm, new_bp, new_h)

        return lax.cond(is_leaf, leaf, inner, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _st, _sp, cur_tmax, best_pid, hit, _done = final
    return cur_tmax, best_pid, hit


# =====================================================================
# aabb_overlap — early-out box-overlap descent over 8-way fanout.
# =====================================================================


def _aabb_overlap_single(payload: OctreePayload, primitives: Primitives,
                         q_lo: jnp.ndarray, q_hi: jnp.ndarray,
                         max_stack_depth: int,
                         max_leaf_size: int) -> jnp.ndarray:
    stack_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)

    init = (payload.root.astype(_INDEX_DTYPE),
            stack_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            jnp.bool_(False),
            jnp.bool_(False))

    def cond_fn(s):
        return jnp.logical_not(s[3] | s[4])

    def _pop_box(st, sp):
        empty = sp == 0
        new_sp = jnp.where(empty, sp, sp - 1)
        next_node = jnp.where(empty, jnp.asarray(0, _INDEX_DTYPE),
                              st[new_sp])
        return next_node, st, new_sp, jnp.bool_(False), empty

    def body_fn(s):
        cur, st, sp, _found, _done = s
        is_leaf = payload.node_is_leaf[cur]

        def inner(_):
            children = payload.node_children[cur]

            def child_step(k, carry):
                sst, ssp = carry
                child = children[k]
                has_child = child >= 0
                safe_child = jnp.where(has_child, child,
                                       jnp.asarray(0, _INDEX_DTYPE))
                overlap = has_child & _box_overlap(
                    payload.node_lower[safe_child],
                    payload.node_upper[safe_child],
                    q_lo, q_hi)
                # Stack sized to 7*tree_depth + 1; overflow impossible.
                push = overlap
                nst = jnp.where(push, sst.at[ssp].set(child), sst)
                nsp = jnp.where(push, ssp + 1, ssp)
                return nst, nsp

            new_st, new_sp = lax.fori_loop(
                0, _OCTREE_FANOUT, child_step, (st, sp))
            return _pop_box(new_st, new_sp)

        def leaf(_):
            offset = payload.node_leaf_offset[cur]
            count = payload.node_leaf_count[cur]
            prim_ids_len = payload.prim_ids.shape[0]

            def leaf_step(k, f):
                valid = k < count
                pid = payload.prim_ids[jnp.minimum(offset + k,
                                                   prim_ids_len - 1)]
                p_lo, p_hi = primitives.primitive_aabb(pid)
                hit = valid & _box_overlap(p_lo, p_hi, q_lo, q_hi)
                return f | hit

            new_f = lax.fori_loop(0, max_leaf_size, leaf_step,
                                  jnp.bool_(False))

            def short_done(_):
                return (jnp.asarray(0, _INDEX_DTYPE), st, sp,
                        jnp.bool_(True), jnp.bool_(True))

            def normal_pop(_):
                return _pop_box(st, sp)

            return lax.cond(new_f, short_done, normal_pop, operand=None)

        return lax.cond(is_leaf, leaf, inner, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    return final[3]


# =====================================================================
# distance_bounds — query-AABB to mesh distance bounds via stack-descent.
# Ported from `bvh.sah.jax::_distance_bounds_single` with fanout-8 child
# iteration and the kd.sah.jax empty-leaf treatment: octree leaves whose
# `leaf_count == 0` carry Morton-bucket cell bounds (not a tight prim
# enclosure), so their `d_hi` cannot soundly tighten `best_hi`. Same
# guard applied during interior-descent child-side tightening (push +inf
# for empty children's `d_hi`).
# =====================================================================


def _distance_bounds_single(payload: OctreePayload, primitives: Primitives,
                            q_lo: jnp.ndarray, q_hi: jnp.ndarray,
                            max_stack_depth: int,
                            max_leaf_size: int
                            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    INF = jnp.float32(jnp.inf)
    stack_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)

    # Initial best_hi from root box; best_lo is INF (running MIN of
    # leaf-prim lower bounds).
    root = payload.root.astype(_INDEX_DTYPE)
    _root_lo, root_hi = _box_box_dist_bounds(
        q_lo, q_hi, payload.node_lower[root], payload.node_upper[root])

    # state: (cur, stack, sp, best_lo, best_hi, done)
    init = (root,
            stack_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            INF,
            root_hi,
            jnp.bool_(False))

    def cond_fn(s):
        return jnp.logical_not(s[5])

    def _pop(st, sp, best_lo, best_hi):
        empty = sp == 0
        new_sp = jnp.where(empty, sp, sp - 1)
        next_node = jnp.where(empty, jnp.asarray(0, _INDEX_DTYPE),
                              st[new_sp])
        done = empty
        return next_node, st, new_sp, best_lo, best_hi, done

    def body_fn(s):
        cur, st, sp, best_lo, best_hi, _done = s
        is_leaf = payload.node_is_leaf[cur]
        leaf_count = payload.node_leaf_count[cur]
        # Octree empty-leaf guard (kd.sah.jax pattern, lines 916-936):
        # Morton-bucket leaves with no centroid in the bucket carry cell
        # bounds (not a tight primitive enclosure); using their d_hi to
        # tighten best_hi is unsound w.r.t. the brute reference.
        is_empty_leaf = is_leaf & (leaf_count == 0)
        # Compute bounds for current node — prune if its lo bound is
        # already worse than our best upper bound.
        d_lo_cur, d_hi_cur = _box_box_dist_bounds(
            q_lo, q_hi, payload.node_lower[cur], payload.node_upper[cur])
        # Tighten best_hi using current node's hi bound — but only for
        # nodes that actually enclose primitives (interior or non-empty
        # leaf).
        best_hi = jnp.where(is_empty_leaf, best_hi,
                            jnp.minimum(best_hi, d_hi_cur))
        skip = is_empty_leaf | (d_lo_cur > best_hi)

        def skip_branch(_):
            return _pop(st, sp, best_lo, best_hi)

        def descend(_):
            def inner(_):
                # Fanout-8: iterate over 8 child slots. Pass 1 — compute
                # each child's (d_lo, d_hi); use d_hi (with +inf for
                # empty/missing children) to tighten best_hi. Pass 2 —
                # push children whose d_lo <= new_best_hi onto the stack.
                # Stack sized to 7*tree_depth + 1; overflow impossible.
                children = payload.node_children[cur]

                def hi_step(k, bh):
                    child = children[k]
                    has_child = child >= 0
                    safe_child = jnp.where(has_child, child,
                                           jnp.asarray(0, _INDEX_DTYPE))
                    _d_lo_c, d_hi_c = _box_box_dist_bounds(
                        q_lo, q_hi,
                        payload.node_lower[safe_child],
                        payload.node_upper[safe_child])
                    # Treat absent child slot OR empty-leaf child as +inf
                    # for upper-bound tightening.
                    c_is_leaf = payload.node_is_leaf[safe_child]
                    c_count = payload.node_leaf_count[safe_child]
                    c_empty_leaf = c_is_leaf & (c_count == 0)
                    skip_hi = jnp.logical_not(has_child) | c_empty_leaf
                    d_hi_safe = jnp.where(skip_hi, INF, d_hi_c)
                    return jnp.minimum(bh, d_hi_safe)

                new_best_hi = lax.fori_loop(
                    0, _OCTREE_FANOUT, hi_step, best_hi)

                def push_step(k, carry):
                    sst, ssp = carry
                    child = children[k]
                    has_child = child >= 0
                    safe_child = jnp.where(has_child, child,
                                           jnp.asarray(0, _INDEX_DTYPE))
                    d_lo_c, _d_hi_c = _box_box_dist_bounds(
                        q_lo, q_hi,
                        payload.node_lower[safe_child],
                        payload.node_upper[safe_child])
                    push = has_child & (d_lo_c <= new_best_hi)
                    nst = jnp.where(push, sst.at[ssp].set(child), sst)
                    nsp = jnp.where(push, ssp + 1, ssp)
                    return nst, nsp

                new_st, new_sp = lax.fori_loop(
                    0, _OCTREE_FANOUT, push_step, (st, sp))
                # Pop the next node to visit.
                return _pop(new_st, new_sp, best_lo, new_best_hi)

            def leaf(_):
                offset = payload.node_leaf_offset[cur]
                count = leaf_count

                def leaf_step(k, lc):
                    bl, bh = lc
                    valid = k < count
                    pid = payload.prim_ids[jnp.minimum(
                        offset + k, payload.prim_ids.shape[0] - 1)]
                    p_lo, p_hi = primitives.primitive_aabb(pid)
                    p_d_lo, p_d_hi = _box_box_dist_bounds(
                        q_lo, q_hi, p_lo, p_hi)
                    new_bl = jnp.where(valid, jnp.minimum(bl, p_d_lo), bl)
                    new_bh = jnp.where(valid, jnp.minimum(bh, p_d_hi), bh)
                    return new_bl, new_bh

                new_bl, new_bh = lax.fori_loop(
                    0, max_leaf_size, leaf_step, (best_lo, best_hi))
                return _pop(st, sp, new_bl, new_bh)

            return lax.cond(is_leaf, leaf, inner, operand=None)

        return lax.cond(skip, skip_branch, descend, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _st, _sp, best_lo, best_hi, _done = final
    # `best_lo` is the running MIN of leaf-prim lower bounds. If we
    # never reached a leaf, best_lo stays INF; clamp to a safe small
    # value (matches `bvh.sah.jax` / `kd.sah.jax`).
    best_lo = jnp.where(jnp.isinf(best_lo), jnp.float32(0.0), best_lo)
    return best_lo.astype(jnp.float32), best_hi.astype(jnp.float32)


# =====================================================================
# Backend class.
# =====================================================================


@register_backend
class OctreeMortonExplicitJaxBackend(Backend[OctreePayload]):
    """Yggdrax explicit-octree (Morton-LCA derivation) host-side build,
    pure-JAX traversal kernels (closest_point / first_hit / aabb_overlap).
    8-way fanout traversal via fori_loop over the 8 child slots; fixed-
    size stack `max(7*_MAX_MORTON_LEVEL + 1, 7*tree_depth + 1)`. The 7x
    factor is exact for fanout-8: each interior node visited descends
    one child directly and pushes the other 7, so a path of length
    `tree_depth` produces at most `7*tree_depth + 1` queued entries.
    """

    name = "octree.morton_explicit.jax"
    flavor = "octree"
    algorithm = "morton_explicit"
    runtime = "jax"
    payload_type = OctreePayload
    is_differentiable = False
    has_distance_bounds = True

    def __init__(self):
        self._primitives: Primitives | None = None
        self._max_stack_depth: int = 64
        self._max_leaf_size: int = 8

    def build(self, primitives: Primitives, config: Config) -> OctreePayload:
        self._primitives = primitives
        n = int(primitives.n_primitives)
        idx = jnp.arange(n, dtype=_INDEX_DTYPE)
        lo, hi = primitives.primitive_aabb(idx)
        prim_lo_np = np.asarray(lo, dtype=np.float32)
        prim_hi_np = np.asarray(hi, dtype=np.float32)
        leaf_size = max(1, int(config.leaf_size_threshold)
                        if config.leaf_size_threshold else 8)

        payload, tree_depth = _build_payload(
            prim_lo_np, prim_hi_np, leaf_size)

        # Fanout-8 worst case: each interior node descended pushes the
        # other 7 children onto the stack, so a path of length
        # `tree_depth` produces at most `7*tree_depth + 1` queued entries.
        # The lower clamp `7*_MAX_MORTON_LEVEL + 1 = 148` covers the
        # deepest octree the Morton build can produce, so the push helpers
        # cannot overflow and need no overflow guard.
        needed = max(7 * _MAX_MORTON_LEVEL + 1, 7 * tree_depth + 1)
        self._max_stack_depth = max(int(config.max_stack_depth), needed)
        leaf_counts = np.asarray(payload.node_leaf_count, dtype=np.int32)
        self._max_leaf_size = max(1, int(leaf_counts.max()))
        return payload

    def closest_point(self, payload: OctreePayload, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth
        max_leaf = self._max_leaf_size

        def single(xq):
            return _closest_point_single(payload, prims, xq, max_stack,
                                         max_leaf)

        if x.ndim == 1:
            return single(x)
        return jax.vmap(single)(x)

    def first_hit(self, payload: OctreePayload, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth
        max_leaf = self._max_leaf_size

        def single(o, d, mt):
            return _first_hit_single(payload, prims, o, d, mt,
                                     max_stack, max_leaf)

        if ray_origin.ndim == 1:
            return single(ray_origin, ray_direction, max_t)
        return jax.vmap(single)(ray_origin, ray_direction, max_t)

    def aabb_overlap(self, payload: OctreePayload,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        prims = self._primitives
        max_stack = self._max_stack_depth
        max_leaf = self._max_leaf_size

        def single(lo, hi):
            return _aabb_overlap_single(payload, prims, lo, hi,
                                        max_stack, max_leaf)

        if lower.ndim == 1:
            return single(lower, upper)
        return jax.vmap(single)(lower, upper)

    def distance_bounds(self, payload: OctreePayload,
                        lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth
        max_leaf = self._max_leaf_size

        def single(lo, hi):
            return _distance_bounds_single(payload, prims, lo, hi,
                                           max_stack, max_leaf)

        if lower.ndim == 1:
            return single(lower, upper)
        return jax.vmap(single)(lower, upper)
