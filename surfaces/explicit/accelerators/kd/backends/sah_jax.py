"""SAH primitive-bucketing kd-tree backend, pure JAX.

Build follows the PBRT-v4 KdTreeAggregate note:
  algo_infos_crawled/kd.sah.pbrt_v4.md
  algo_infos_crawled/kd.sah.pbrt_v4.code/aggregates_buildTree.cpp

The build is host-side NumPy: SAH edge-event scan is sequential and
recursive, so we do not JIT it. Build is one-shot setup, not on the
hot path. Traversal is pure JAX (jax.jit / lax.while_loop) using a
fixed-depth explicit stack, vmapped over the query batch — same idiom
as `bvh.lbvh.jax` and `kd.left_balanced.jax`. The PBRT-v4 traversal
(`Intersect`) is ported faithfully for `first_hit`; closest-point and
AABB-overlap are grafted on (PBRT does not provide them) using the
shrinking-radius / box-overlap descents from `kd.left_balanced.jax`,
adapted for primitive-bucketing leaves.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.accelerators.kd.payload import KdPayload
from surfaces.explicit.primitives import Primitives


_INDEX_DTYPE = jnp.int32
_NDIM = 3

# PBRT-v4 SAH defaults (aggregates.h ~99-141, aggregates.cpp ~1084-1091).
_TRAVERSAL_COST = np.float32(1.0)
_ISECT_COST = np.float32(80.0)
_EMPTY_BONUS = np.float32(0.5)
_MAX_PRIMS_PER_LEAF = 1
_BAD_REFINES_CAP = 3

# Edge-event types — Start < End at coincident position so a primitive
# whose pMin equals another's pMax is counted as "below" before split.
# (Note quirk #4; aggregates.cpp ~1186-1192.)
_EVENT_START = np.int8(0)
_EVENT_END = np.int8(1)


# =====================================================================
# Host-side build helpers (Note step build:1-14, mirrors PBRT-v4
# `KdTreeAggregate::buildTree`).
# =====================================================================


def _surface_area(extent: np.ndarray) -> float:
    """Surface area of an axis-aligned box from its (3,) extent."""
    dx, dy, dz = float(extent[0]), float(extent[1]), float(extent[2])
    return 2.0 * (dx * dy + dy * dz + dz * dx)


def _max_dimension(extent: np.ndarray) -> int:
    return int(np.argmax(extent))


class _BuildState:
    """Scratch state for the recursive SAH build.

    Owns the growing `nodes`/`prim_ids` arrays and the per-axis edge-
    event buffers. Mirrors PBRT-v4's class-member scratch arrays
    (aggregates.cpp ~1101-1130).
    """

    def __init__(self, n_prims: int, max_depth: int):
        # Initial node capacity 512 (matches PBRT-v4 first-allocation).
        cap = max(512, 2 * n_prims + 1)
        self.node_split_axis = np.full((cap,), -1, dtype=np.int32)
        self.node_split_pos = np.zeros((cap,), dtype=np.float32)
        self.node_left = np.full((cap,), -1, dtype=np.int32)
        self.node_right = np.full((cap,), -1, dtype=np.int32)
        self.node_lower = np.zeros((cap, 3), dtype=np.float32)
        self.node_upper = np.zeros((cap, 3), dtype=np.float32)
        self.node_leaf_offset = np.zeros((cap,), dtype=np.int32)
        self.node_leaf_count = np.zeros((cap,), dtype=np.int32)
        self.next_free = 0

        # Flat prim_ids store grown by `extend`. Initial cap = N (typical
        # case lower-bound); grows on demand for SAH duplication.
        self.prim_ids = np.empty((max(n_prims, 16),), dtype=np.int32)
        self.prim_ids_len = 0

        # Per-axis edge-event buffers (max 2N each); reused across
        # recursion levels (PBRT-v4 aggregates.cpp ~1101-1108).
        self._edge_t = [np.empty((2 * n_prims,), dtype=np.float32)
                        for _ in range(_NDIM)]
        self._edge_type = [np.empty((2 * n_prims,), dtype=np.int8)
                           for _ in range(_NDIM)]
        self._edge_prim = [np.empty((2 * n_prims,), dtype=np.int32)
                           for _ in range(_NDIM)]

    def alloc_node(self) -> int:
        """Return a fresh node index, doubling node arrays if at cap."""
        if self.next_free >= self.node_split_axis.shape[0]:
            new_cap = self.node_split_axis.shape[0] * 2
            self._grow_nodes(new_cap)
        idx = self.next_free
        self.next_free += 1
        return idx

    def _grow_nodes(self, new_cap: int) -> None:
        for name in ("node_split_axis", "node_split_pos",
                     "node_left", "node_right",
                     "node_leaf_offset", "node_leaf_count"):
            arr = getattr(self, name)
            new_arr = np.zeros((new_cap,), dtype=arr.dtype)
            if name == "node_split_axis":
                new_arr.fill(-1)
            elif name in ("node_left", "node_right"):
                new_arr.fill(-1)
            new_arr[:arr.shape[0]] = arr
            setattr(self, name, new_arr)
        for name in ("node_lower", "node_upper"):
            arr = getattr(self, name)
            new_arr = np.zeros((new_cap, 3), dtype=arr.dtype)
            new_arr[:arr.shape[0]] = arr
            setattr(self, name, new_arr)

    def append_prim_ids(self, ids: np.ndarray) -> int:
        """Append `ids` to `prim_ids`, growing if needed. Returns the
        offset where `ids` starts.
        """
        n = ids.shape[0]
        offset = self.prim_ids_len
        needed = self.prim_ids_len + n
        if needed > self.prim_ids.shape[0]:
            new_cap = max(self.prim_ids.shape[0] * 2, needed)
            new_arr = np.empty((new_cap,), dtype=np.int32)
            new_arr[:self.prim_ids_len] = self.prim_ids[:self.prim_ids_len]
            self.prim_ids = new_arr
        self.prim_ids[self.prim_ids_len:needed] = ids
        self.prim_ids_len = needed
        return offset


def _make_leaf(state: _BuildState, node_id: int,
               node_lo: np.ndarray, node_hi: np.ndarray,
               prim_nums: np.ndarray,
               prim_lo: np.ndarray, prim_hi: np.ndarray) -> None:
    """Write a leaf node (PBRT InitLeaf, but with explicit per-node
    AABB enclosing this leaf's primitives).
    """
    state.node_split_axis[node_id] = -1
    state.node_split_pos[node_id] = 0.0
    state.node_left[node_id] = -1
    state.node_right[node_id] = -1
    if prim_nums.shape[0] == 0:
        # Empty leaf: AABB = degenerate, pMin = pMax (we use the
        # incoming node bounds; behaviour analogous to PBRT's empty
        # InitLeaf).
        state.node_lower[node_id] = node_lo
        state.node_upper[node_id] = node_hi
        state.node_leaf_offset[node_id] = 0
        state.node_leaf_count[node_id] = 0
        return
    # Tight leaf AABB = union of contained primitives' AABBs.
    leaf_lo = prim_lo[prim_nums].min(axis=0)
    leaf_hi = prim_hi[prim_nums].max(axis=0)
    state.node_lower[node_id] = leaf_lo
    state.node_upper[node_id] = leaf_hi
    offset = state.append_prim_ids(prim_nums.astype(np.int32))
    state.node_leaf_offset[node_id] = offset
    state.node_leaf_count[node_id] = prim_nums.shape[0]


def _sah_split(node_lo: np.ndarray, node_hi: np.ndarray,
               prim_nums: np.ndarray,
               prim_lo: np.ndarray, prim_hi: np.ndarray,
               state: _BuildState) -> Tuple[int, int, float, np.ndarray]:
    """SAH cost sweep with axis retry (Note step build:7-12).

    Returns `(best_axis, best_offset, best_cost, sorted_indices)` where
    `sorted_indices` is the permutation that, applied to the per-axis
    edge buffer, yields the sorted edge order. `best_axis = -1` means no
    valid split was found across all three axes.
    """
    n_prims = prim_nums.shape[0]
    extent = node_hi - node_lo
    total_sa = _surface_area(extent)
    inv_total_sa = np.float32(1.0 / total_sa) if total_sa > 0 else np.float32(0.0)

    best_axis = -1
    best_offset = -1
    best_cost = np.float32(np.inf)
    best_sorted = None

    axis = _max_dimension(extent)  # Note step build:7 — start with longest.
    for retries in range(_NDIM):
        # Note step build:8 — emit 2N edge events for this axis.
        edge_t = state._edge_t[axis]
        edge_type = state._edge_type[axis]
        edge_prim = state._edge_prim[axis]

        lo_a = prim_lo[prim_nums, axis]
        hi_a = prim_hi[prim_nums, axis]
        edge_t[0:2 * n_prims:2] = lo_a
        edge_t[1:2 * n_prims:2] = hi_a
        edge_type[0:2 * n_prims:2] = _EVENT_START
        edge_type[1:2 * n_prims:2] = _EVENT_END
        edge_prim[0:2 * n_prims:2] = prim_nums
        edge_prim[1:2 * n_prims:2] = prim_nums

        # Note step build:9 — sort by (t, type). Start (0) < End (1) at
        # coincident positions; np.lexsort uses the LAST key as primary,
        # so we pass (type, t). This is PBRT-v4 convention
        # (aggregates.cpp:~1186-1192): with the END decrement before
        # cost-eval and the START increment after, a primitive ending at
        # t_split lands in n_below at the cost evaluation, and a
        # primitive starting at t_split lands in n_above. Cost evaluated
        # at every interior edge position thus reflects the standard
        # boundary convention.
        sort_idx = np.lexsort((edge_type[:2 * n_prims], edge_t[:2 * n_prims]))
        s_t = edge_t[sort_idx]
        s_type = edge_type[sort_idx]

        # Note step build:10 — left-to-right sweep.
        n_below = 0
        n_above = n_prims
        node_min_a = float(node_lo[axis])
        node_max_a = float(node_hi[axis])
        a0 = (axis + 1) % 3
        a1 = (axis + 2) % 3
        d0 = float(extent[a0])
        d1 = float(extent[a1])
        d_axis = float(extent[axis])

        for i in range(2 * n_prims):
            if s_type[i] == _EVENT_END:
                n_above -= 1
            t_edge = float(s_t[i])
            if node_min_a < t_edge < node_max_a:
                # Slab surface areas (note quirk #12).
                below_sa = 2.0 * (d0 * d1 + (t_edge - node_min_a) * (d0 + d1))
                above_sa = 2.0 * (d0 * d1 + (node_max_a - t_edge) * (d0 + d1))
                p_below = below_sa * float(inv_total_sa)
                p_above = above_sa * float(inv_total_sa)
                eb = float(_EMPTY_BONUS) if (n_above == 0 or n_below == 0) else 0.0
                cost = (float(_TRAVERSAL_COST)
                        + float(_ISECT_COST) * (1.0 - eb)
                        * (p_below * n_below + p_above * n_above))
                if cost < best_cost:
                    best_cost = np.float32(cost)
                    best_axis = axis
                    best_offset = i
                    best_sorted = sort_idx.copy()
            if s_type[i] == _EVENT_START:
                n_below += 1

        # Note step build:11 — axis retry if no valid split this axis
        # AND we haven't tried all three axes yet.
        if best_axis != -1:
            break
        axis = (axis + 1) % _NDIM

    return best_axis, best_offset, float(best_cost), best_sorted


def _classify_primitives(best_axis: int, best_offset: int,
                         sorted_idx: np.ndarray,
                         state: _BuildState,
                         n_prims: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Walk sorted edges to partition primitives into below/above lists
    (Note step build:13). Primitives whose AABBs straddle the split
    appear in BOTH lists (PBRT-v4 spatial-kd duplication; quirk #1).
    """
    edge_type = state._edge_type[best_axis]
    edge_prim = state._edge_prim[best_axis]
    s_type = edge_type[sorted_idx]
    s_prim = edge_prim[sorted_idx]
    s_t = state._edge_t[best_axis][sorted_idx]

    prims_below_mask = (s_type[:best_offset] == _EVENT_START)
    prims0 = s_prim[:best_offset][prims_below_mask]
    prims_above_mask = (s_type[best_offset + 1:2 * n_prims] == _EVENT_END)
    prims1 = s_prim[best_offset + 1:2 * n_prims][prims_above_mask]
    t_split = float(s_t[best_offset])
    return prims0, prims1, t_split


def _build_recursive(state: _BuildState, node_id: int,
                     node_lo: np.ndarray, node_hi: np.ndarray,
                     prim_nums: np.ndarray, depth: int,
                     bad_refines: int,
                     prim_lo: np.ndarray, prim_hi: np.ndarray,
                     max_depth: int) -> None:
    """Recursive SAH kd-tree build mirroring PBRT-v4's `buildTree`
    (aggregates.cpp ~1131-1264).
    """
    n_prims = prim_nums.shape[0]

    # Note step build:6 — leaf termination.
    if n_prims <= _MAX_PRIMS_PER_LEAF or depth >= max_depth:
        _make_leaf(state, node_id, node_lo, node_hi,
                   prim_nums, prim_lo, prim_hi)
        return

    # Note step build:7-11 — SAH split selection with axis retry.
    best_axis, best_offset, best_cost, sorted_idx = _sah_split(
        node_lo, node_hi, prim_nums, prim_lo, prim_hi, state)

    leaf_cost = float(_ISECT_COST) * n_prims

    # Note step build:12 — bad-refine bookkeeping & forced leaf.
    if best_cost > leaf_cost:
        bad_refines += 1
    if (best_axis == -1
            or (best_cost > 4.0 * leaf_cost and n_prims < 16)
            or bad_refines >= _BAD_REFINES_CAP):
        _make_leaf(state, node_id, node_lo, node_hi,
                   prim_nums, prim_lo, prim_hi)
        return

    # Note step build:13 — classify primitives (duplicate straddlers).
    prims0, prims1, t_split = _classify_primitives(
        best_axis, best_offset, sorted_idx, state, n_prims)

    # Note step build:14 — recurse below child first (PBRT places the
    # below child at `nodeNum + 1`), record above-child index, write
    # interior node, recurse above.
    bounds0_hi = node_hi.copy()
    bounds0_hi[best_axis] = t_split
    bounds1_lo = node_lo.copy()
    bounds1_lo[best_axis] = t_split

    left_id = state.alloc_node()
    _build_recursive(state, left_id, node_lo, bounds0_hi,
                     prims0, depth + 1, bad_refines,
                     prim_lo, prim_hi, max_depth)

    right_id = state.alloc_node()
    _build_recursive(state, right_id, bounds1_lo, node_hi,
                     prims1, depth + 1, bad_refines,
                     prim_lo, prim_hi, max_depth)

    # Interior node: tight AABB = union of children's AABBs (which are
    # themselves tight after their recursive builds finished). This
    # gives interior nodes meaningful per-node bounds for the grafted
    # CP / AABB-overlap traversals below.
    state.node_split_axis[node_id] = best_axis
    state.node_split_pos[node_id] = np.float32(t_split)
    state.node_left[node_id] = left_id
    state.node_right[node_id] = right_id
    state.node_lower[node_id] = np.minimum(state.node_lower[left_id],
                                           state.node_lower[right_id])
    state.node_upper[node_id] = np.maximum(state.node_upper[left_id],
                                           state.node_upper[right_id])
    state.node_leaf_offset[node_id] = 0
    state.node_leaf_count[node_id] = 0


def _build_payload(prim_lo_np: np.ndarray, prim_hi_np: np.ndarray
                   ) -> Tuple[KdPayload, int]:
    """Host-side SAH kd-tree build (Note steps build:1-14). Returns
    `(KdPayload, max_depth_used)` so the caller can size the traversal
    stack.
    """
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    # Note step build:1 — default max depth (PBRT-v4 default formula).
    max_depth = int(round(8 + 1.3 * math.log2(max(n, 1))))
    max_depth = max(max_depth, 1)

    # Note step build:2 — scene AABB (enclosure of all primitive AABBs).
    scene_lo = prim_lo_np.min(axis=0).astype(np.float32)
    scene_hi = prim_hi_np.max(axis=0).astype(np.float32)
    # Guard degenerate (zero-extent) axes.
    span = np.maximum(scene_hi - scene_lo, np.float32(1e-20))
    scene_hi = scene_lo + span

    # Note step build:3 — allocate scratch.
    state = _BuildState(n, max_depth)

    # Note step build:4 — seed the recursion at root (node 0).
    root_id = state.alloc_node()
    assert root_id == 0
    prim_nums = np.arange(n, dtype=np.int32)
    _build_recursive(state, root_id, scene_lo, scene_hi,
                     prim_nums, depth=0, bad_refines=0,
                     prim_lo=prim_lo_np.astype(np.float32),
                     prim_hi=prim_hi_np.astype(np.float32),
                     max_depth=max_depth)

    # Slice the grown arrays to actual size and convert to jnp.
    n_nodes = state.next_free
    prim_ids_final = state.prim_ids[:state.prim_ids_len].astype(np.int32)

    payload = KdPayload(
        node_split_axis=jnp.asarray(state.node_split_axis[:n_nodes]),
        node_split_pos=jnp.asarray(state.node_split_pos[:n_nodes]),
        node_left=jnp.asarray(state.node_left[:n_nodes]),
        node_right=jnp.asarray(state.node_right[:n_nodes]),
        node_lower=jnp.asarray(state.node_lower[:n_nodes]),
        node_upper=jnp.asarray(state.node_upper[:n_nodes]),
        node_leaf_offset=jnp.asarray(state.node_leaf_offset[:n_nodes]),
        node_leaf_count=jnp.asarray(state.node_leaf_count[:n_nodes]),
        prim_ids=jnp.asarray(prim_ids_final),
        root=jnp.asarray(np.int32(0)),
    )
    return payload, max_depth


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
               ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns `(t_enter, t_exit, hit)` for a ray vs AABB."""
    t1 = (lo - origin) * rcp_dir
    t2 = (hi - origin) * rcp_dir
    t_lo = jnp.minimum(t1, t2)
    t_hi = jnp.maximum(t1, t2)
    t_enter = jnp.maximum(jnp.max(t_lo), tmin)
    t_exit = jnp.minimum(jnp.min(t_hi), tmax)
    hit = t_enter <= t_exit
    return t_enter, t_exit, hit


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
    Mirrors `bvh.sah.jax::_box_box_dist_bounds`; same Euclidean
    formulation (FCPW BoundingBox::computeSquaredDistance pattern).
    """
    sep = jnp.maximum(jnp.maximum(a_lo - b_hi, b_lo - a_hi),
                      jnp.float32(0.0))
    d_lo2 = jnp.sum(sep * sep)
    span = jnp.maximum((a_hi - b_lo) ** 2, (b_hi - a_lo) ** 2)
    d_hi2 = jnp.sum(span)
    eps = jnp.float32(0.0)
    return jnp.sqrt(jnp.maximum(d_lo2, eps)), jnp.sqrt(jnp.maximum(d_hi2, eps))


# =====================================================================
# first_hit — port of PBRT-v4 KdTreeAggregate::Intersect
# (aggregates.cpp ~1237-1330; Note Traversal pipeline).
# =====================================================================


def _first_hit_single(payload: KdPayload, primitives: Primitives,
                      ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                      max_t: jnp.ndarray, max_stack_depth: int,
                      max_leaf_size: int):
    INF = jnp.float32(jnp.inf)

    # Sign-preserving guard: inv_dir must keep the sign of d, otherwise
    # the ray-AABB slab test flips. Same fix as `bvh.lbvh.jax`.
    safe_dir = jnp.where(jnp.abs(ray_direction) < 1e-20,
                         jnp.copysign(jnp.float32(1e-20), ray_direction),
                         ray_direction)
    rcp_dir = jnp.float32(1.0) / safe_dir

    # Note traversal step 1 — outer ray-AABB slab vs root.
    root = payload.root.astype(_INDEX_DTYPE)
    t_enter, t_exit, root_hit = _ray_box_t(
        ray_origin, rcp_dir,
        payload.node_lower[root], payload.node_upper[root],
        jnp.float32(0.0), max_t.astype(jnp.float32))

    # Stack entries store (node_id, tmin, tmax). PBRT uses 64; we obey
    # the caller-provided bound.
    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_tmin_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)
    stack_tmax_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    # If the root miss, short-circuit by marking done immediately.
    init_tmin = jnp.where(root_hit, t_enter, INF)
    init_tmax = jnp.where(root_hit, t_exit, jnp.float32(0.0))
    init_done = jnp.logical_not(root_hit)

    # state: (cur, stack_idx, stack_tmin, stack_tmax, sp,
    #         tmin, tmax, cur_tmax, best_pid, hit, done)
    init = (root,
            stack_idx_init, stack_tmin_init, stack_tmax_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            init_tmin, init_tmax,
            max_t.astype(jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False),
            init_done)

    def cond_fn(s):
        return jnp.logical_not(s[10])

    def _pop(si, stmin, stmax, sp, cur_tmax, best_pid, hit):
        """Pop next node from stack; mark done if empty. Note traversal
        step 8.
        """
        empty = sp == 0
        new_sp = jnp.where(empty, sp, sp - 1)
        node = jnp.where(empty, jnp.asarray(0, _INDEX_DTYPE), si[new_sp])
        tmin = jnp.where(empty, INF, stmin[new_sp])
        tmax = jnp.where(empty, jnp.float32(0.0), stmax[new_sp])
        return (node, si, stmin, stmax, new_sp, tmin, tmax,
                cur_tmax, best_pid, hit, empty)

    def body_fn(s):
        (cur, si, stmin, stmax, sp, tmin, tmax,
         cur_tmax, best_pid, hit, _done) = s

        # Note traversal step 3 — early exit if best hit closer than
        # current node's near plane.
        early = cur_tmax < tmin

        def early_exit(_):
            return (cur, si, stmin, stmax, sp, tmin, tmax,
                    cur_tmax, best_pid, hit, jnp.bool_(True))

        def visit(_):
            leaf_count = payload.node_leaf_count[cur]
            is_leaf = (payload.node_split_axis[cur] < 0) | (leaf_count > 0)
            # Treat any node with split_axis == -1 as leaf (covers empty
            # leaves with leaf_count == 0 as well).

            def interior(_):
                # Note traversal step 4-6 — split-plane distance,
                # near/far ordering, three-case slab skip.
                axis = payload.node_split_axis[cur]
                split_pos = payload.node_split_pos[cur]
                ro_a = ray_origin[axis]
                rd_a = ray_direction[axis]
                inv_a = rcp_dir[axis]
                t_split = (split_pos - ro_a) * inv_a

                # Note traversal step 5 — origin-on-plane tie-break
                # (quirk #6): below-first if ro<split, OR ro==split and
                # rd<=0.
                below_first = (ro_a < split_pos) | (
                    (ro_a == split_pos) & (rd_a <= 0.0))
                left_id = payload.node_left[cur]
                right_id = payload.node_right[cur]
                first = jnp.where(below_first, left_id, right_id)
                second = jnp.where(below_first, right_id, left_id)

                # Note traversal step 6 — three-case logic.
                # Case A: tSplit > tmax || tSplit <= 0 -> only first.
                # Case B: tSplit < tmin                -> only second.
                # Case C: otherwise                    -> push second
                #         with [tSplit, tmax], descend first with
                #         [tmin, tSplit].
                case_a = (t_split > tmax) | (t_split <= 0.0)
                case_b = jnp.logical_and(jnp.logical_not(case_a),
                                         t_split < tmin)
                case_c = jnp.logical_and(jnp.logical_not(case_a),
                                         jnp.logical_not(case_b))

                # Case A path: visit first, keep [tmin, tmax].
                def a_path(_):
                    return (first, si, stmin, stmax, sp, tmin, tmax,
                            cur_tmax, best_pid, hit, jnp.bool_(False))

                # Case B path: visit second, keep [tmin, tmax].
                def b_path(_):
                    return (second, si, stmin, stmax, sp, tmin, tmax,
                            cur_tmax, best_pid, hit, jnp.bool_(False))

                # Case C path: push second with [tSplit, tmax]; descend
                # first with [tmin, tSplit].
                def c_path(_):
                    overflow = sp >= max_stack_depth
                    nsi = jnp.where(overflow, si, si.at[sp].set(second))
                    nstmin = jnp.where(overflow, stmin,
                                       stmin.at[sp].set(t_split))
                    nstmax = jnp.where(overflow, stmax,
                                       stmax.at[sp].set(tmax))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return (first, nsi, nstmin, nstmax, nsp,
                            tmin, t_split,
                            cur_tmax, best_pid, hit,
                            jnp.bool_(overflow))

                return lax.cond(
                    case_a, a_path,
                    lambda _: lax.cond(case_b, b_path, c_path, operand=None),
                    operand=None)

            def leaf(_):
                # Note traversal step 7 — primitive tests in leaf.
                offset = payload.node_leaf_offset[cur]
                count = leaf_count
                prim_ids_len = payload.prim_ids.shape[0]

                def leaf_step(k, lc):
                    ctm, bp, h = lc
                    valid = k < count
                    pid = payload.prim_ids[jnp.minimum(offset + k,
                                                       prim_ids_len - 1)]
                    t_k, hit_k = primitives.ray_primitive_intersect(
                        ray_origin, ray_direction, ctm, pid)
                    accept = valid & hit_k & (t_k < ctm) & (t_k >= 0.0)
                    ctm = jnp.where(accept, t_k, ctm)
                    bp = jnp.where(accept, pid, bp)
                    h = h | accept
                    return ctm, bp, h

                new_tm, new_bp, new_h = lax.fori_loop(
                    0, max_leaf_size, leaf_step, (cur_tmax, best_pid, hit))

                # Note traversal step 8 — pop next node.
                return _pop(si, stmin, stmax, sp, new_tm, new_bp, new_h)

            return lax.cond(is_leaf, leaf, interior, operand=None)

        return lax.cond(early, early_exit, visit, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    (_cur, _si, _stmin, _stmax, _sp, _tmin, _tmax,
     cur_tmax, best_pid, hit, _done) = final
    out_t = jnp.where(hit, cur_tmax, max_t.astype(jnp.float32))
    return out_t, best_pid, hit


# =====================================================================
# closest_point — grafted shrinking-radius descent (PBRT does NOT
# implement this). Adapted from `kd.left_balanced.jax` for primitive-
# bucketing leaves.
# =====================================================================


def _closest_point_single(payload: KdPayload, primitives: Primitives,
                          x: jnp.ndarray,
                          max_stack_depth: int,
                          max_leaf_size: int):
    INF = jnp.float32(jnp.inf)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_d_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    # state: (cur, stack_idx, stack_d, sp, best_d2, best_cp, best_pid, done)
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
        # Pop entries whose AABB-distance bound no longer beats best_d.
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
        leaf_count = payload.node_leaf_count[cur]
        is_leaf = (payload.node_split_axis[cur] < 0) | (leaf_count > 0)

        def interior(_):
            # Pure-kd descent (no per-node primitive seated here).
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            has_l = l >= 0
            has_r = r >= 0
            safe_l = jnp.where(has_l, l, cur)
            safe_r = jnp.where(has_r, r, cur)
            d_l = _sqr_dist_point_to_box(x, payload.node_lower[safe_l],
                                         payload.node_upper[safe_l])
            d_r = _sqr_dist_point_to_box(x, payload.node_lower[safe_r],
                                         payload.node_upper[safe_r])
            d_l = jnp.where(has_l & (d_l < best_d), d_l, INF)
            d_r = jnp.where(has_r & (d_r < best_d), d_r, INF)
            any_child = (d_l < INF) | (d_r < INF)
            l_first = d_l <= d_r
            near = jnp.where(l_first, safe_l, safe_r)
            far = jnp.where(l_first, safe_r, safe_l)
            d_near = jnp.where(l_first, d_l, d_r)
            d_far = jnp.where(l_first, d_r, d_l)
            push_far = d_far < INF
            do_descend = d_near < INF

            def descend_path(_):
                def push_then_go(_):
                    overflow = sp >= max_stack_depth
                    nsi = jnp.where(overflow, si, si.at[sp].set(far))
                    nsd = jnp.where(overflow, sd, sd.at[sp].set(d_far))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return (near, nsi, nsd, nsp, best_d, best_cp, best_pid,
                            jnp.bool_(overflow))

                def no_push_go(_):
                    return (near, si, sd, sp, best_d, best_cp, best_pid,
                            jnp.bool_(False))

                return lax.cond(push_far, push_then_go, no_push_go,
                                operand=None)

            def pop_path(_):
                return _pop(si, sd, sp, best_d, best_cp, best_pid)

            return lax.cond(do_descend & any_child, descend_path, pop_path,
                            operand=None)

        def leaf(_):
            offset = payload.node_leaf_offset[cur]
            count = leaf_count
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

        return lax.cond(is_leaf, leaf, interior, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _sd, _sp, best_d2, best_cp, best_pid, _done = final
    best_dist = jnp.sqrt(jnp.maximum(best_d2, jnp.float32(0.0)))
    return best_cp, best_dist, best_pid


# =====================================================================
# aabb_overlap — grafted box-overlap descent (PBRT does NOT implement
# this). Pattern adapted from `kd.left_balanced.jax`.
# =====================================================================


def _aabb_overlap_single(payload: KdPayload, primitives: Primitives,
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
        is_leaf = (payload.node_split_axis[cur] < 0) | (leaf_count > 0)

        def interior(_):
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            has_l = l >= 0
            has_r = r >= 0
            safe_l = jnp.where(has_l, l, cur)
            safe_r = jnp.where(has_r, r, cur)
            o_l = has_l & _box_overlap(payload.node_lower[safe_l],
                                       payload.node_upper[safe_l],
                                       q_lo, q_hi)
            o_r = has_r & _box_overlap(payload.node_lower[safe_r],
                                       payload.node_upper[safe_r],
                                       q_lo, q_hi)

            def both_miss(_):
                return _pop_box(st, sp)

            def at_least(_):
                next_node = jnp.where(o_l, safe_l, safe_r)

                def push_r(_):
                    overflow = sp >= max_stack_depth
                    nst = jnp.where(overflow, st, st.at[sp].set(safe_r))
                    nsp = jnp.where(overflow, sp, sp + 1)
                    return (next_node, nst, nsp, jnp.bool_(False),
                            jnp.bool_(overflow))

                def no_push(_):
                    return (next_node, st, sp, jnp.bool_(False),
                            jnp.bool_(False))

                return lax.cond(o_l & o_r, push_r, no_push, operand=None)

            return lax.cond(o_l | o_r, at_least, both_miss, operand=None)

        def leaf(_):
            offset = payload.node_leaf_offset[cur]
            count = leaf_count
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

        return lax.cond(is_leaf, leaf, interior, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    return final[3]


# =====================================================================
# distance_bounds — query-AABB to mesh distance bounds via stack-descent.
# Ported from `bvh.sah.jax::_distance_bounds_single`. The kd interior
# node split plane is irrelevant here — only the per-node AABB matters,
# so the traversal is structurally identical to the BVH version.
# =====================================================================


def _distance_bounds_single(payload: KdPayload, primitives: Primitives,
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
        leaf_count = payload.node_leaf_count[cur]
        # kd convention: empty leaves have leaf_count == 0 but split_axis
        # < 0; mirror the same disambiguation as _first_hit/_closest_point/
        # _aabb_overlap_single in this file.
        is_leaf = (payload.node_split_axis[cur] < 0) | (leaf_count > 0)
        # Empty leaves carry their cell bounds (not a tight primitive
        # union) — those bounds may not enclose any actual primitive, so
        # using their d_hi to tighten best_hi can produce values smaller
        # than any primitive's d_hi (unsound w.r.t. the brute reference).
        # Skip such empty-leaf nodes entirely (they contribute no
        # primitives anyway).
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
                l = payload.node_left[cur]
                r = payload.node_right[cur]
                d0_lo, d0_hi = _box_box_dist_bounds(
                    q_lo, q_hi,
                    payload.node_lower[l], payload.node_upper[l])
                d1_lo, d1_hi = _box_box_dist_bounds(
                    q_lo, q_hi,
                    payload.node_lower[r], payload.node_upper[r])
                # Empty-leaf children carry cell bounds that may not
                # enclose primitives — exclude them from best_hi
                # tightening (the per-node guard above will skip them
                # when popped, so they still get visited but contribute
                # nothing).
                l_empty = ((payload.node_split_axis[l] < 0)
                           & (payload.node_leaf_count[l] == 0))
                r_empty = ((payload.node_split_axis[r] < 0)
                           & (payload.node_leaf_count[r] == 0))
                d0_hi_safe = jnp.where(l_empty, jnp.float32(jnp.inf), d0_hi)
                d1_hi_safe = jnp.where(r_empty, jnp.float32(jnp.inf), d1_hi)
                # Tighten best_hi using each non-empty child's far-corner
                # upper.
                new_best_hi = jnp.minimum(
                    jnp.minimum(best_hi, d0_hi_safe), d1_hi_safe)
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
    # `best_lo` is the running MIN of leaf-prim lower bounds. If we
    # never reached a leaf, best_lo stays INF; clamp to a safe small
    # value (matches `bvh.sah.jax`).
    best_lo = jnp.where(jnp.isinf(best_lo), jnp.float32(0.0), best_lo)
    return best_lo.astype(jnp.float32), best_hi.astype(jnp.float32)


# =====================================================================
# Backend class.
# =====================================================================


@register_backend
class KdSahJaxBackend(Backend[KdPayload]):
    """PBRT-v4 SAH primitive-bucketing kd-tree, host-side build, pure-
    JAX traversal. Build mirrors `KdTreeAggregate::buildTree`; first-hit
    ports `KdTreeAggregate::Intersect`. Closest-point and aabb-overlap
    are grafted on (PBRT does not provide them) using the same patterns
    as `kd.left_balanced.jax`.
    """

    name = "kd.sah.jax"
    flavor = "kd"
    algorithm = "sah"
    runtime = "jax"
    payload_type = KdPayload
    is_differentiable = False
    has_distance_bounds = True

    def __init__(self):
        self._primitives: Primitives | None = None
        # PBRT-v4 hard-codes 64 (aggregates.cpp ~1245); we honour that
        # as the lower bound and grow if the user requests more.
        self._max_stack_depth: int = 64
        self._max_leaf_size: int = 1

    def build(self, primitives: Primitives, config: Config) -> KdPayload:
        self._primitives = primitives
        n = int(primitives.n_primitives)
        idx = jnp.arange(n, dtype=_INDEX_DTYPE)
        lo, hi = primitives.primitive_aabb(idx)
        prim_lo_np = np.asarray(lo, dtype=np.float32)
        prim_hi_np = np.asarray(hi, dtype=np.float32)

        payload, max_depth = _build_payload(prim_lo_np, prim_hi_np)

        # Stack depth bound: PBRT uses 64 explicitly; we ensure at least
        # `2 * max_depth + 4` to cover both children-pushed-and-popped
        # cycles plus slack.
        needed = max(64, 2 * max_depth + 4)
        self._max_stack_depth = max(int(config.max_stack_depth), needed)
        # Largest leaf size — bounds the per-leaf fori_loop. Read from
        # the payload (host-side construction means we know it now).
        leaf_counts = np.asarray(payload.node_leaf_count, dtype=np.int32)
        self._max_leaf_size = max(1, int(leaf_counts.max()))
        return payload

    def closest_point(self, payload: KdPayload, x: jnp.ndarray
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

    def first_hit(self, payload: KdPayload, ray_origin: jnp.ndarray,
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

    def aabb_overlap(self, payload: KdPayload,
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

    def distance_bounds(self, payload: KdPayload,
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
