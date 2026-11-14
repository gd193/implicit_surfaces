"""Host-side PBRT-style SAH kd-tree construction.

The original NumPy edge-event scan and recursive primitive-bucketing
builder retain their settings and payload layout. This module only
builds trees and is not a query backend.
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


def build(primitives: Primitives, config: Config | None = None) -> KdPayload:
    """Build a SAH kd payload; Config does not affect this builder."""
    n = int(primitives.n_primitives)
    idx = jnp.arange(n, dtype=_INDEX_DTYPE)
    lo, hi = primitives.primitive_aabb(idx)
    prim_lo_np = np.asarray(lo, dtype=np.float32)
    prim_hi_np = np.asarray(hi, dtype=np.float32)
    payload, _ = _build_payload(prim_lo_np, prim_hi_np)
    return payload
