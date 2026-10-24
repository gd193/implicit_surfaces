"""SAH binary BVH backend (FCPW-style bucket-SAH), pure JAX traversal.

Build follows the FCPW Bvh<DIM=3> note:
  algo_infos_crawled/bvh.sah_mbvh.fcpw_sawhney.md
  algo_infos_crawled/bvh.sah_mbvh.fcpw_sawhney.code/bvh_build.inl
Traversal kernels mirror FCPW `processSubtreeForIntersection` (ray) and
`processSubtreeForClosestPoint` (CP):
  algo_infos_crawled/bvh.sah_mbvh.fcpw_sawhney.code/bvh_traverse.inl

Runtime: pure JAX (jax.jit / lax.while_loop / lax.fori_loop). Build is
host-side NumPy because the FCPW bucket-SAH builder is sequential and
recursive (n_primitives is static at construction). Single-thread-per-
query traversal; jax.vmap fans queries across batches.

This backend is the first to expose `has_distance_bounds = True`,
implementing query-AABB to mesh distance-bound queries via the same
shrinking-bound branch-and-bound stack pattern.
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


def _surface_area_np(lo: np.ndarray, hi: np.ndarray) -> float:
    """Surface area of an axis-aligned box from (3,) lo/hi vectors."""
    e = np.maximum(hi - lo, 0.0)
    return float(2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]))


class _BuildState:
    """Growing storage for nodes + flat prim-id permutation. The FCPW
    builder is in-place on a `primitives[]` vector; we mirror that with a
    `prim_perm` integer array that holds the permuted primitive index
    sequence (no duplication — FCPW BVH does object-only splits).
    """

    def __init__(self, n_prims: int):
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
                          prim_idx: np.ndarray) -> Tuple[int, float, float]:
    """Bucket-SAH split selection over all 3 axes (Note step build:7-8;
    Source: bvh.inl:computeObjectSplit, lines 47-119).

    Returns `(best_axis, best_split_coord, best_cost)` with `best_axis=-1`
    if no valid bucketed split was found (caller falls back to longest-
    axis-centre on the centroid bbox).
    """
    n = prim_idx.shape[0]
    extent = node_hi - node_lo
    total_sa = _surface_area_np(node_lo, node_hi)
    if total_sa <= 0.0:
        return -1, 0.0, float(np.inf)
    inv_total_sa = 1.0 / total_sa

    best_axis = -1
    best_coord = 0.0
    best_cost = float(np.inf)

    for axis in range(_NDIM):
        ext_a = float(extent[axis])
        if ext_a < 1e-6:
            continue  # FCPW skips flat axes (bvh.inl:64).
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
                sa_l = _surface_area_np(left_lo, left_hi)
                sa_r = _surface_area_np(right_lo[b], right_hi[b])
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
        ref_lo, ref_hi, ref_cent, prim_idx)

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
                   leaf_size: int) -> BVHPayload:
    """Host-side bucket-SAH BVH build (Note section "BVH build", FCPW
    bvh.inl 1-242). Returns a fully-populated shared `BVHPayload`.
    """
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    # Per-primitive AABBs already provided; centroids are midpoints.
    ref_lo = prim_lo_np.astype(np.float32)
    ref_hi = prim_hi_np.astype(np.float32)
    ref_cent = (0.5 * (ref_lo + ref_hi)).astype(np.float32)

    state = _BuildState(n)
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


# =====================================================================
# Traversal helpers.
# =====================================================================


def _sqr_dist_point_to_box(p: jnp.ndarray, lo: jnp.ndarray,
                           hi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns `(d2_min, d2_max)` from point `p` to AABB `[lo, hi]`.

    Mirrors FCPW `BoundingBox::computeSquaredDistance` (Source:
    bounding_volumes_overlap.h:41-46):
        u = pMin - p; v = p - pMax;
        d2Min = ||max(max(u,v), 0)||^2
        d2Max = ||min(u,v)||^2     # squared distance to farthest corner
    """
    u = lo - p
    v = p - hi
    d2_min = jnp.sum(jnp.maximum(jnp.maximum(u, v), jnp.float32(0.0)) ** 2)
    d2_max = jnp.sum(jnp.minimum(u, v) ** 2)
    return d2_min, d2_max


def _ray_box_t(origin: jnp.ndarray, rcp_dir: jnp.ndarray,
               lo: jnp.ndarray, hi: jnp.ndarray,
               tmin: jnp.ndarray, tmax: jnp.ndarray
               ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Standard slab test; returns `(t_enter, hit)`."""
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
# closest_point — port of FCPW processSubtreeForClosestPoint
# (Source: bvh.inl:processSubtreeForClosestPoint:136-233 +
#          bvh.inl:findClosestPointFromNode:236-263).
# Implements the d2Max-shrink (FCPW lines 185, 188): after each child
# box overlap test, tighten s.r2 using d2_max of the visited child box,
# even before descending. THIS IS THE KEY SAH-BVH IMPROVEMENT OVER LBVH.
# =====================================================================


def _closest_point_single(payload: BVHPayload,
                          primitives: Primitives,
                          x: jnp.ndarray,
                          max_stack_depth: int,
                          max_leaf_size: int):
    INF = jnp.float32(jnp.inf)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_d_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    # Initial root tightening: equivalent to FCPW lines 245-246
    # (s.r2 = min(s.r2, boxHits[1]) at root entry).
    root = payload.root.astype(_INDEX_DTYPE)
    d2_root_min, d2_root_max = _sqr_dist_point_to_box(
        x, payload.node_lower[root], payload.node_upper[root])
    init_best_d = d2_root_max
    # state: (cur, si, sd, sp, best_d, best_cp, best_pid, done)
    init = (root,
            stack_idx_init, stack_d_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            init_best_d,
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False))

    def cond_fn(state):
        return jnp.logical_not(state[7])

    def _pop(si, sd, sp, best_d, best_cp, best_pid):
        """Pop entries while their saved d2_min > best_d (FCPW line 149)."""
        def pop_cond(s):
            _ssp, _ssi, _ssd, found, empty = s
            return jnp.logical_not(found | empty)

        def pop_body(s):
            ssp, ssi, ssd, _f, _e = s
            empty = ssp == 0
            new_sp = jnp.where(empty, ssp, ssp - 1)
            top_d = ssd[new_sp]
            in_range = (top_d <= best_d) & jnp.logical_not(empty)
            return new_sp, ssi, ssd, in_range, empty

        new_sp, _, _, found, _empty = lax.while_loop(
            pop_cond, pop_body,
            (sp, si, sd, jnp.bool_(False), jnp.bool_(False)))
        next_node = jnp.where(found, si[new_sp], jnp.asarray(0, _INDEX_DTYPE))
        done = jnp.logical_not(found)
        return next_node, si, sd, new_sp, best_d, best_cp, best_pid, done

    def body_fn(state):
        cur, si, sd, sp, best_d, best_cp, best_pid, _done = state
        leaf_count = payload.node_leaf_count[cur]
        is_leaf = leaf_count > 0

        def inner_branch(_):
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            d0_min, d0_max = _sqr_dist_point_to_box(
                x, payload.node_lower[l], payload.node_upper[l])
            d1_min, d1_max = _sqr_dist_point_to_box(
                x, payload.node_lower[r], payload.node_upper[r])
            # *** d2Max CP shrink — FCPW bvh.inl:185, 188 ***
            # Tighten best_d by each child's max-corner squared distance
            # (upper bound on the closest point in that subtree). This is
            # the algorithmic improvement over a naive BVH CP and the
            # primary reason this backend outperforms `bvh.lbvh.jax`.
            tightened_d = jnp.minimum(jnp.minimum(best_d, d0_max), d1_max)
            hit0 = d0_min <= tightened_d
            hit1 = d1_min <= tightened_d

            def both(_):
                # FCPW special-case for "query inside both boxes"
                # (boxHits[0]==0 && boxHits[2]==0): order by d2_max
                # (smaller far-corner = more pruning) instead of d2_min.
                inside_both = (d0_min == 0.0) & (d1_min == 0.0)
                left_closer_normal = d0_min <= d1_min
                left_closer_inside = d0_max <= d1_max
                left_first = jnp.where(inside_both, left_closer_inside,
                                       left_closer_normal)
                near = jnp.where(left_first, l, r)
                far = jnp.where(left_first, r, l)
                # Push farther first with its d2_min as guard distance
                # (FCPW line 213).
                far_min = jnp.where(left_first, d1_min, d0_min)
                overflow = sp >= max_stack_depth
                nsi = jnp.where(overflow, si, si.at[sp].set(far))
                nsd = jnp.where(overflow, sd, sd.at[sp].set(far_min))
                nsp = jnp.where(overflow, sp, sp + 1)
                return (near, nsi, nsd, nsp, tightened_d, best_cp, best_pid,
                        jnp.bool_(overflow))

            def only_left(_):
                return (l, si, sd, sp, tightened_d, best_cp, best_pid,
                        jnp.bool_(False))

            def only_right(_):
                return (r, si, sd, sp, tightened_d, best_cp, best_pid,
                        jnp.bool_(False))

            def neither(_):
                return _pop(si, sd, sp, tightened_d, best_cp, best_pid)

            return lax.cond(
                hit0 & hit1, both,
                lambda _: lax.cond(
                    hit0, only_left,
                    lambda __: lax.cond(hit1, only_right, neither,
                                        operand=None),
                    operand=None),
                operand=None)

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
                0, max_leaf_size, leaf_step, (best_d, best_cp, best_pid))
            return _pop(si, sd, sp, new_bd, new_bcp, new_bp)

        # Quick prune: if this node entry is already worse than best_d,
        # skip directly to pop.
        node_d2_min, _ = _sqr_dist_point_to_box(
            x, payload.node_lower[cur], payload.node_upper[cur])
        skip = node_d2_min > best_d

        def skip_branch(_):
            return _pop(si, sd, sp, best_d, best_cp, best_pid)

        def descend_branch(_):
            return lax.cond(is_leaf, leaf_branch, inner_branch, operand=None)

        return lax.cond(skip, skip_branch, descend_branch, operand=None)

    # If the root box is already farther than best_d (impossible at init,
    # since best_d=d2_root_max), we still enter the loop to compute leaf
    # results normally.
    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _sd, _sp, best_d, best_cp, best_pid, _done = final
    best_dist = jnp.sqrt(jnp.maximum(best_d, jnp.float32(0.0)))
    return best_cp, best_dist, best_pid


# =====================================================================
# first_hit — port of FCPW processSubtreeForIntersection (ray)
# (Source: bvh.inl:processSubtreeForIntersection:15-110).
# =====================================================================


def _first_hit_single(payload: BVHPayload,
                      primitives: Primitives,
                      ray_origin: jnp.ndarray,
                      ray_direction: jnp.ndarray,
                      max_t: jnp.ndarray,
                      max_stack_depth: int,
                      max_leaf_size: int):
    INF = jnp.float32(jnp.inf)
    # Sign-preserving inv_dir guard (matches lbvh_jax.py).
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
                return _pop_ray(si, st, sp, cur_tmax, best_pid, hit)

            def at_least_one(_):
                # If only one hits, descend it; otherwise sort by t.
                left_first = jnp.where(o0 & jnp.logical_not(o1), True,
                                       jnp.where(o1 & jnp.logical_not(o0),
                                                 False, t0 <= t1))
                near = jnp.where(left_first, l, r)
                far = jnp.where(left_first, r, l)
                far_t = jnp.where(left_first, t1, t0)
                push_far = o0 & o1

                def do_push(_):
                    overflow = sp >= max_stack_depth
                    nsi = jnp.where(overflow, si, si.at[sp].set(far))
                    nst = jnp.where(overflow, st, st.at[sp].set(far_t))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return (near, nsi, nst, nsp, cur_tmax, best_pid, hit,
                            jnp.bool_(overflow))

                def no_push(_):
                    return (near, si, st, sp, cur_tmax, best_pid, hit,
                            jnp.bool_(False))

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
                0, max_leaf_size, leaf_step, (cur_tmax, best_pid, hit))
            return _pop_ray(si, st, sp, new_tm, new_bp, new_h)

        return lax.cond(is_leaf, leaf, inner, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _st, _sp, cur_tmax, best_pid, hit, _done = final
    return cur_tmax, best_pid, hit


# =====================================================================
# aabb_overlap — same pattern as lbvh_jax.
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
                    return (next_node, nst, nsp, jnp.bool_(False),
                            jnp.bool_(overflow))

                def no_push(_):
                    return (next_node, st, sp, jnp.bool_(False),
                            jnp.bool_(False))

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
class BVHSahJaxBackend(Backend[BVHPayload]):
    """FCPW-style bucket-SAH binary BVH built host-side, with pure-JAX
    traversal. First backend to expose `has_distance_bounds = True`,
    enabling query-AABB to mesh distance-bound queries through a
    dedicated stack-descent kernel.
    """

    name = "bvh.sah.jax"
    flavor = "bvh"
    algorithm = "sah"
    runtime = "jax"
    payload_type = BVHPayload
    is_differentiable = False
    has_distance_bounds = True

    def __init__(self):
        self._primitives: Primitives | None = None
        self._max_stack_depth: int = _FCPW_BVH_MAX_DEPTH
        self._max_leaf_size: int = _LEAF_SIZE_DEFAULT

    def build(self, primitives: Primitives, config: Config) -> BVHPayload:
        self._primitives = primitives
        n = int(primitives.n_primitives)
        idx = jnp.arange(n, dtype=_INDEX_DTYPE)
        lo, hi = primitives.primitive_aabb(idx)
        prim_lo_np = np.asarray(lo, dtype=np.float32)
        prim_hi_np = np.asarray(hi, dtype=np.float32)
        leaf_size = max(1, int(config.leaf_size_threshold)
                        if config.leaf_size_threshold else _LEAF_SIZE_DEFAULT)
        # FCPW default leaf size is 4; honour caller's larger override
        # via Config.leaf_size_threshold but treat very small values as
        # the FCPW default to keep tree shape sensible.
        if leaf_size <= 1:
            leaf_size = _LEAF_SIZE_DEFAULT

        payload = _build_payload(prim_lo_np, prim_hi_np, leaf_size)

        # Stack depth: FCPW uses 64 unconditionally. Honour any caller-
        # supplied larger value.
        self._max_stack_depth = max(int(config.max_stack_depth),
                                    _FCPW_BVH_MAX_DEPTH)
        leaf_counts = np.asarray(payload.node_leaf_count, dtype=np.int32)
        self._max_leaf_size = max(1, int(leaf_counts.max()))
        return payload

    def closest_point(self, payload: BVHPayload, x: jnp.ndarray
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
