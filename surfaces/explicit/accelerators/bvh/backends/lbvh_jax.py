"""LBVH (Karras 2012) BVH backend, pure JAX.

Build follows the yggdrax LBVH note:
  algo_infos_crawled/bvh.lbvh.tobibu_yggdrax.md

Traversal kernels translate the cuBQL CUDA reference templates:
  algo_infos_crawled/bvh.lbvh.wald_cubql.md
  - shrinkingRadiusQuery::forEachLeaf  -> closest_point
  - shrinkingRayQuery::forEachLeaf     -> first_hit
  - fixedBoxQuery::forEachLeaf         -> aabb_overlap

Runtime: pure JAX (jax.jit / lax.while_loop / lax.fori_loop) for the
hot-path traversal kernels. The build is done host-side in NumPy
because (a) `n_primitives` is static at construction (fixed mesh),
(b) Morton codes need uint64, which JAX only exposes under
`jax_enable_x64` and the rest of the project runs in 32-bit mode, and
(c) the Karras solve is a one-shot O(N) computation. Single-thread-
per-query traversal; jax.vmap fans queries across batches.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
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
    # Degenerate-axis guard: when bmax==bmin along an axis (single primitive,
    # planar/line/point clouds), or when caller's epsilon was absorbed by
    # float32 rounding in `bmax = bmin + eps`, the divide produces NaN. All
    # primitives collapse to the same coordinate along that axis, so map to
    # mid-grid (0.5) deterministically — Morton order along a zero-extent
    # axis is undefined but all primitives share the bucket, which is correct.
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
# CLZ (uint64) and LCP via Python int (built-in bit_length) — only
# used for occasional checks. The Karras solve below uses NumPy CLZ.
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


def _lcp_codes_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a.astype(np.uint64), b.astype(np.uint64))
    return np.where(xor == np.uint64(0), np.int32(64), _clz_u64_np(xor))


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

    first_code = leaf_codes[first]
    last_code = leaf_codes[last]
    common_prefix = _lcp_codes_np(first_code, last_code)

    # Note step build:8 — binary-search the split (31 steps).
    split = first.copy()
    for k in range(30, -1, -1):
        step = np.int32(1 << k)
        mid = split + step
        safe_mid = np.where(mid < last, mid, last)
        cmp = _lcp_codes_np(first_code, leaf_codes[safe_mid])
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

    centroids = (0.5 * (prim_lo_np + prim_hi_np)).astype(np.float32)
    bmin = centroids.min(axis=0)
    bmax = centroids.max(axis=0)
    span = np.maximum(bmax - bmin, 1e-20)
    bmax = bmin + span

    # Note step build:1 — encode 63-bit Morton codes.
    codes = _morton_encode_np(centroids, bmin, bmax)
    # Note step build:2 — stable Morton order with original-index tie-break
    # via lexsort (last key is primary).
    sorted_indices = np.lexsort((np.arange(n, dtype=np.int64), codes))
    sorted_indices = sorted_indices.astype(np.int32)
    sorted_codes = codes[sorted_indices].astype(np.uint64)

    # Note step build:3 — partition sorted stream into Morton leaves.
    leaf_starts = np.arange(0, n, leaf_size, dtype=np.int32)
    leaf_ends = np.minimum(leaf_starts + leaf_size, n).astype(np.int32)
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
# closest_point — translated from cuBQL shrinkingRadiusQuery::forEachLeaf
# (Source: bvh.lbvh.wald_cubql.code/shrinkingRadiusQuery.h:48-124).
# =====================================================================


def _closest_point_single(payload: BVHPayload,
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

    def _pop(si, sd, sp, best_d, best_cp, best_prim):
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
        return next_node, si, sd, new_sp, best_d, best_cp, best_prim, done

    def body_fn(state):
        cur, si, sd, sp, best_d, best_cp, best_prim, _done = state
        leaf_count = payload.node_leaf_count[cur]
        is_leaf = leaf_count > 0

        def inner_branch(_):
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            d0 = _sqr_dist_point_to_box(x, payload.node_lower[l],
                                        payload.node_upper[l])
            d1 = _sqr_dist_point_to_box(x, payload.node_lower[r],
                                        payload.node_upper[r])
            both_far = jnp.minimum(d0, d1) >= best_d
            near = jnp.where(d0 < d1, l, r)
            far = jnp.where(d0 < d1, r, l)
            far_d = jnp.maximum(d0, d1)
            push_far = (far_d < best_d) & jnp.logical_not(both_far)

            def descend(_):
                def do_push(_):
                    overflow = sp >= max_stack_depth
                    nsi = jnp.where(overflow, si, si.at[sp].set(far))
                    nsd = jnp.where(overflow, sd, sd.at[sp].set(far_d))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return near, nsi, nsd, nsp, best_d, best_cp, best_prim, jnp.bool_(overflow)

                def no_push(_):
                    return near, si, sd, sp, best_d, best_cp, best_prim, jnp.bool_(False)

                return lax.cond(push_far, do_push, no_push, operand=None)

            def pop_path(_):
                return _pop(si, sd, sp, best_d, best_cp, best_prim)

            return lax.cond(both_far, pop_path, descend, operand=None)

        def leaf_branch(_):
            offset = payload.node_leaf_offset[cur]
            count = leaf_count

            def leaf_step(k, lc):
                bd, bcp, bp = lc
                valid = k < count
                pid = payload.prim_ids[jnp.minimum(offset + k,
                                                   payload.prim_ids.shape[0] - 1)]
                cp_k, dist_k = primitives.closest_point_on_primitive(x, pid)
                d2_k = dist_k * dist_k
                better = valid & (d2_k < bd)
                bd = jnp.where(better, d2_k, bd)
                bcp = jnp.where(better, cp_k, bcp)
                bp = jnp.where(better, pid, bp)
                return bd, bcp, bp

            new_bd, new_bcp, new_bp = lax.fori_loop(
                0, max_leaf_size, leaf_step, (best_d, best_cp, best_prim))
            return _pop(si, sd, sp, new_bd, new_bcp, new_bp)

        return lax.cond(is_leaf, leaf_branch, inner_branch, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _sd, _sp, best_d, best_cp, best_prim, _done = final
    best_dist = jnp.sqrt(jnp.maximum(best_d, jnp.float32(0.0)))
    return best_cp, best_dist, best_prim


# =====================================================================
# first_hit — translated from cuBQL shrinkingRayQuery::forEachLeaf
# (Source: bvh.lbvh.wald_cubql.code/rayQueries.h:50-130).
# =====================================================================


def _first_hit_single(payload: BVHPayload,
                      primitives: Primitives,
                      ray_origin: jnp.ndarray,
                      ray_direction: jnp.ndarray,
                      max_t: jnp.ndarray,
                      max_stack_depth: int,
                      max_leaf_size: int):
    INF = jnp.float32(jnp.inf)
    # Preserve sign on near-zero components to avoid flipping the ray direction.
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

    def _pop_ray(si, st, sp, cur_tmax, best_prim, hit):
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
        return next_node, si, st, new_sp, cur_tmax, best_prim, hit, done

    def body_fn(s):
        cur, si, st, sp, cur_tmax, best_prim, hit, _done = s
        leaf_count = payload.node_leaf_count[cur]
        is_leaf = leaf_count > 0

        def inner(_):
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            t0, o0 = _ray_box_t(ray_origin, rcp_dir,
                                payload.node_lower[l], payload.node_upper[l],
                                tmin, cur_tmax)
            t1, o1 = _ray_box_t(ray_origin, rcp_dir,
                                payload.node_lower[r], payload.node_upper[r],
                                tmin, cur_tmax)

            def both_miss(_):
                return _pop_ray(si, st, sp, cur_tmax, best_prim, hit)

            def at_least_one(_):
                near = jnp.where(o0 & jnp.logical_not(o1), l,
                                 jnp.where(o1 & jnp.logical_not(o0), r,
                                           jnp.where(t0 <= t1, l, r)))
                far = jnp.where(t0 <= t1, r, l)
                far_t = jnp.where(t0 <= t1, t1, t0)
                push_far = o0 & o1

                def do_push(_):
                    overflow = sp >= max_stack_depth
                    nsi = jnp.where(overflow, si, si.at[sp].set(far))
                    nst = jnp.where(overflow, st, st.at[sp].set(far_t))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return near, nsi, nst, nsp, cur_tmax, best_prim, hit, jnp.bool_(overflow)

                def no_push(_):
                    return near, si, st, sp, cur_tmax, best_prim, hit, jnp.bool_(False)

                return lax.cond(push_far, do_push, no_push, operand=None)

            return lax.cond(o0 | o1, at_least_one, both_miss, operand=None)

        def leaf(_):
            offset = payload.node_leaf_offset[cur]
            count = leaf_count

            def leaf_step(k, lc):
                ctm, bp, h = lc
                valid = k < count
                pid = payload.prim_ids[jnp.minimum(offset + k,
                                                   payload.prim_ids.shape[0] - 1)]
                t_k, hit_k = primitives.ray_primitive_intersect(
                    ray_origin, ray_direction, ctm, pid)
                accept = valid & hit_k & (t_k < ctm) & (t_k >= tmin)
                ctm = jnp.where(accept, t_k, ctm)
                bp = jnp.where(accept, pid, bp)
                h = h | accept
                return ctm, bp, h

            new_tm, new_bp, new_h = lax.fori_loop(
                0, max_leaf_size, leaf_step, (cur_tmax, best_prim, hit))
            return _pop_ray(si, st, sp, new_tm, new_bp, new_h)

        return lax.cond(is_leaf, leaf, inner, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _st, _sp, cur_tmax, best_prim, hit, _done = final
    return cur_tmax, best_prim, hit


# =====================================================================
# aabb_overlap — translated from cuBQL fixedBoxQuery::forEachLeaf
# (Source: bvh.lbvh.wald_cubql.code/fixedBoxQuery.h:62-140).
# =====================================================================


def _aabb_overlap_single(payload: BVHPayload, primitives: Primitives,
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
        leaf_count = payload.node_leaf_count[cur]
        is_leaf = leaf_count > 0

        def inner(_):
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            o0 = _box_overlap(payload.node_lower[l], payload.node_upper[l],
                              q_lo, q_hi)
            o1 = _box_overlap(payload.node_lower[r], payload.node_upper[r],
                              q_lo, q_hi)

            def both_miss(_):
                return _pop_box(st, sp)

            def at_least(_):
                next_node = jnp.where(o0, l, r)

                def push_r(_):
                    overflow = sp >= max_stack_depth
                    nst = jnp.where(overflow, st, st.at[sp].set(r))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return next_node, nst, nsp, jnp.bool_(False), jnp.bool_(overflow)

                def no_push(_):
                    return next_node, st, sp, jnp.bool_(False), jnp.bool_(False)

                return lax.cond(o0 & o1, push_r, no_push, operand=None)

            return lax.cond(o0 | o1, at_least, both_miss, operand=None)

        def leaf(_):
            offset = payload.node_leaf_offset[cur]
            count = leaf_count

            def leaf_step(k, f):
                valid = k < count
                pid = payload.prim_ids[jnp.minimum(offset + k,
                                                   payload.prim_ids.shape[0] - 1)]
                p_lo, p_hi = primitives.primitive_aabb(pid)
                hit = valid & _box_overlap(p_lo, p_hi, q_lo, q_hi)
                return f | hit

            new_f = lax.fori_loop(0, max_leaf_size, leaf_step, jnp.bool_(False))

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
# distance_bounds — query AABB to mesh distance bounds.
# Algorithm: stack-descent, at each node compute (d_lo, d_hi) box-vs-box
# bounds; tighten running globals; prune subtree if d_lo_node > best_hi.
# At leaf, exact box-vs-primitive-AABB bounds shrink globals.
# =====================================================================


def _distance_bounds_single(payload: BVHPayload, primitives: Primitives,
                            q_lo: jnp.ndarray, q_hi: jnp.ndarray,
                            max_stack_depth: int,
                            max_leaf_size: int
                            ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    INF = jnp.float32(jnp.inf)
    stack_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)

    # Initial best_hi from root box; best_lo is INF (max-of-min reduces
    # to MIN of leaf lower bounds).
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
        # Compute bounds for current node — prune if its lo bound is
        # already worse than our best upper bound.
        d_lo_cur, d_hi_cur = _box_box_dist_bounds(
            q_lo, q_hi, payload.node_lower[cur], payload.node_upper[cur])
        # Tighten best_hi using current node's hi bound.
        best_hi = jnp.minimum(best_hi, d_hi_cur)
        skip = d_lo_cur > best_hi

        def skip_branch(_):
            return _pop(st, sp, best_lo, best_hi)

        leaf_count = payload.node_leaf_count[cur]
        is_leaf = leaf_count > 0

        def descend(_):
            def inner(_):
                l = payload.node_left[cur]
                r = payload.node_right[cur]
                d0_lo, d0_hi = _box_box_dist_bounds(
                    q_lo, q_hi,
                    payload.node_lower[l], payload.node_upper[l])
                d1_lo, d1_hi = _box_box_dist_bounds(
                    q_lo, q_hi,
                    payload.node_lower[r], payload.node_upper[r])
                # Tighten best_hi using each child's far-corner upper.
                new_best_hi = jnp.minimum(jnp.minimum(best_hi, d0_hi), d1_hi)
                hit0 = d0_lo <= new_best_hi
                hit1 = d1_lo <= new_best_hi

                def both(_):
                    left_first = d0_lo <= d1_lo
                    near = jnp.where(left_first, l, r)
                    far = jnp.where(left_first, r, l)
                    overflow = sp >= max_stack_depth
                    nst = jnp.where(overflow, st, st.at[sp].set(far))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return (near, nst, nsp, best_lo, new_best_hi,
                            jnp.bool_(overflow))

                def only_l(_):
                    return (l, st, sp, best_lo, new_best_hi, jnp.bool_(False))

                def only_r(_):
                    return (r, st, sp, best_lo, new_best_hi, jnp.bool_(False))

                def none(_):
                    return _pop(st, sp, best_lo, new_best_hi)

                return lax.cond(
                    hit0 & hit1, both,
                    lambda _: lax.cond(
                        hit0, only_l,
                        lambda __: lax.cond(hit1, only_r, none,
                                            operand=None),
                        operand=None),
                    operand=None)

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
    # `best_lo` is the running MIN of leaf-prim lower bounds (so a true
    # lower bound on any prim distance). If we never reached a leaf,
    # best_lo stays INF; clamp to a safe small value.
    best_lo = jnp.where(jnp.isinf(best_lo), jnp.float32(0.0), best_lo)
    return best_lo.astype(jnp.float32), best_hi.astype(jnp.float32)


# =====================================================================
# Backend class.
# =====================================================================


@register_backend
class LBVHJaxBackend(Backend[BVHPayload]):
    """Karras-2012 LBVH (Morton-sort + radix-tree) built host-side, with
    pure-JAX traversal kernels translated from cuBQL CUDA reference
    templates. Single Backend per file.
    """

    name = "bvh.lbvh.jax"
    flavor = "bvh"
    algorithm = "lbvh"
    runtime = "jax"
    payload_type = BVHPayload
    is_differentiable = False
    has_distance_bounds = True

    def __init__(self):
        self._primitives: Primitives | None = None
        self._max_stack_depth: int = 32
        self._max_leaf_size: int = 8

    def build(self, primitives: Primitives, config: Config) -> BVHPayload:
        self._primitives = primitives
        n = int(primitives.n_primitives)
        idx = jnp.arange(n, dtype=_INDEX_DTYPE)
        lo, hi = primitives.primitive_aabb(idx)
        prim_lo_np = np.asarray(lo, dtype=np.float32)
        prim_hi_np = np.asarray(hi, dtype=np.float32)
        leaf_size = max(1, int(config.leaf_size_threshold))
        self._max_stack_depth = int(config.max_stack_depth)
        self._max_leaf_size = leaf_size
        return _build_payload(prim_lo_np, prim_hi_np, leaf_size)

    def closest_point(self, payload: BVHPayload, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth
        max_leaf = self._max_leaf_size

        def single(xq):
            return _closest_point_single(payload, prims, xq, max_stack, max_leaf)

        if x.ndim == 1:
            return single(x)
        return jax.vmap(single)(x)

    def first_hit(self, payload: BVHPayload, ray_origin: jnp.ndarray,
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

    def aabb_overlap(self, payload: BVHPayload,
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

    def distance_bounds(self, payload: BVHPayload,
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
