"""SNCH (Silhouette Normal Cone Hierarchy) BVH backend, pure JAX traversal.

Builds a binary SAH BVH over triangle faces (mirroring `bvh.sah.jax`)
and augments each node with a per-subtree silhouette-edge cone
(`axis`, `half_angle`, `radius`) plus a flat list of silhouette-edge
references. First backend to expose `has_silhouette_query = True`
(closest-silhouette-edge query, used by Walk-on-Spheres / signed
distance pipelines).

Source notes:
  algo_infos_crawled/bvh.sah_mbvh.fcpw_sawhney.md  ("SnchNode" /
  "Closest silhouette-point query (SNCH only)" sections)
  algo_infos_crawled/bvh.sah_mbvh.fcpw_sawhney.code/bvh_traverse.inl
  (processSubtreeForClosestSilhouettePoint:268-392)

Runtime: pure JAX (jax.jit / lax.while_loop / lax.fori_loop). Build is
host-side NumPy because the FCPW bucket-SAH builder is recursive and
the silhouette cone propagation is a sequential post-order pass.
Triangle-mesh specific (mirrors FCPW's SNCH = DIM=3-only design):
build raises `TypeError` on non-`TriMeshPrimitives` input.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives, TriMeshPrimitives


_INDEX_DTYPE = jnp.int32
_NDIM = 3

# FCPW defaults — same constants as `bvh.sah.jax` so the binary tree
# matches its sibling backend's shape.
_N_BUCKETS = 8
_LEAF_SIZE_DEFAULT = 4
_FCPW_BVH_MAX_DEPTH = 64
_TRAV_COST = np.float32(1.0)
_ISECT_COST = np.float32(1.0)


# =====================================================================
# Payload — all BVHPayload fields + SNCH cone/silhouette extensions.
# typing.NamedTuple does not support inheritance-with-extra-fields,
# so we declare a fresh tuple here per the project convention
# (see `surfaces/explicit/accelerators/bvh/payload.py` docstring).
# =====================================================================


class BVHSnchPayload(NamedTuple):
    # All BVHPayload fields:
    node_lower: jnp.ndarray         # (N_nodes, 3)
    node_upper: jnp.ndarray         # (N_nodes, 3)
    node_left: jnp.ndarray          # (N_nodes,)
    node_right: jnp.ndarray         # (N_nodes,)
    node_leaf_offset: jnp.ndarray   # (N_nodes,) — face refs
    node_leaf_count: jnp.ndarray    # (N_nodes,)
    prim_ids: jnp.ndarray           # (N_face_refs,)
    root: jnp.ndarray               # ()
    # SNCH additions — per-node silhouette cone summary:
    node_cone_axis: jnp.ndarray         # (N_nodes, 3)
    node_cone_half_angle: jnp.ndarray   # (N_nodes,)
    node_cone_radius: jnp.ndarray       # (N_nodes,)
    # SNCH additions — per-node silhouette refs:
    node_silhouette_offset: jnp.ndarray # (N_nodes,)
    node_silhouette_count: jnp.ndarray  # (N_nodes,)
    silhouette_refs: jnp.ndarray        # (N_silh_refs,)
    # Silhouette edge geometry (one entry per edge):
    silhouette_v0: jnp.ndarray          # (E, 3)
    silhouette_v1: jnp.ndarray          # (E, 3)
    silhouette_n0: jnp.ndarray          # (E, 3)
    silhouette_n1: jnp.ndarray          # (E, 3)


# =====================================================================
# SAH bucket-SAH binary BVH build (mirrors `bvh.sah.jax`).
# =====================================================================


def _surface_area_np(lo: np.ndarray, hi: np.ndarray) -> float:
    e = np.maximum(hi - lo, 0.0)
    return float(2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]))


class _BuildState:
    """Growing storage mirroring the in-place FCPW builder."""

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


def _compute_object_split(node_lo, node_hi, ref_lo, ref_hi, ref_cent,
                          prim_idx) -> Tuple[int, float, float]:
    """Bucket-SAH split selection (mirrors `sah_jax._compute_object_split`,
    Source: bvh.inl:computeObjectSplit:47-119)."""
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
            continue
        bucket_width = ext_a / _N_BUCKETS
        cent_a = ref_cent[prim_idx, axis]
        bidx = np.floor((cent_a - node_lo[axis]) / bucket_width).astype(np.int32)
        bidx = np.clip(bidx, 0, _N_BUCKETS - 1)

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

        right_lo = np.full((_N_BUCKETS + 1, 3), np.inf, dtype=np.float32)
        right_hi = np.full((_N_BUCKETS + 1, 3), -np.inf, dtype=np.float32)
        right_n = np.zeros((_N_BUCKETS + 1,), dtype=np.int32)
        for b in range(_N_BUCKETS - 1, -1, -1):
            right_lo[b] = np.minimum(right_lo[b + 1], bucket_lo[b])
            right_hi[b] = np.maximum(right_hi[b + 1], bucket_hi[b])
            right_n[b] = right_n[b + 1] + bucket_n[b]

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
                cost = (float(_TRAV_COST)
                        + float(_ISECT_COST) * inv_total_sa
                        * (sa_l * n_left + sa_r * n_right))
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    best_coord = float(node_lo[axis] + b * bucket_width)

    return best_axis, best_coord, best_cost


def _partition_centroid(prim_idx, ref_cent, axis, split_coord):
    cent_a = ref_cent[prim_idx, axis]
    left_mask = cent_a < split_coord
    left = prim_idx[left_mask]
    right = prim_idx[~left_mask]
    if left.shape[0] == 0 or right.shape[0] == 0:
        n = prim_idx.shape[0]
        half = n // 2
        left = prim_idx[:half]
        right = prim_idx[half:]
    return left, right


def _build_recursive(state, node_id, prim_idx,
                     ref_lo, ref_hi, ref_cent,
                     leaf_size, depth, prim_perm_out, leaf_buckets):
    """Recursive bucket-SAH BVH build (mirrors `sah_jax._build_recursive`).

    `leaf_buckets[node_id]` is set to the list of face indices held by
    that leaf (used downstream for silhouette-ref assignment in step
    "Step 3: Assign silhouette refs to leaves" of the brief). Internal
    nodes have an empty list there.
    """
    n_refs = prim_idx.shape[0]
    node_lo = ref_lo[prim_idx].min(axis=0)
    node_hi = ref_hi[prim_idx].max(axis=0)
    state.node_lower[node_id] = node_lo
    state.node_upper[node_id] = node_hi

    if n_refs <= leaf_size or depth >= _FCPW_BVH_MAX_DEPTH - 2:
        offset = len(prim_perm_out)
        prim_perm_out.extend(int(p) for p in prim_idx)
        state.node_leaf_offset[node_id] = offset
        state.node_leaf_count[node_id] = n_refs
        state.node_left[node_id] = -1
        state.node_right[node_id] = -1
        leaf_buckets[node_id] = list(int(p) for p in prim_idx)
        return

    axis, coord, best_cost = _compute_object_split(
        node_lo, node_hi, ref_lo, ref_hi, ref_cent, prim_idx)

    if axis == -1:
        cent_lo = ref_cent[prim_idx].min(axis=0)
        cent_hi = ref_cent[prim_idx].max(axis=0)
        cent_extent = cent_hi - cent_lo
        axis = int(np.argmax(cent_extent))
        coord = 0.5 * float(cent_lo[axis] + cent_hi[axis])
        best_cost = float(np.inf)

    leaf_cost = float(_ISECT_COST) * n_refs
    if best_cost > leaf_cost:
        offset = len(prim_perm_out)
        prim_perm_out.extend(int(p) for p in prim_idx)
        state.node_leaf_offset[node_id] = offset
        state.node_leaf_count[node_id] = n_refs
        state.node_left[node_id] = -1
        state.node_right[node_id] = -1
        leaf_buckets[node_id] = list(int(p) for p in prim_idx)
        return

    left_idx, right_idx = _partition_centroid(prim_idx, ref_cent, axis, coord)
    left_id = state.alloc()
    right_id = state.alloc()
    state.node_left[node_id] = left_id
    state.node_right[node_id] = right_id
    state.node_leaf_offset[node_id] = 0
    state.node_leaf_count[node_id] = 0
    leaf_buckets[node_id] = []

    # Ensure children index space exists (alloc may grow state arrays).
    while len(leaf_buckets) <= max(left_id, right_id):
        leaf_buckets.append([])

    _build_recursive(state, left_id, left_idx, ref_lo, ref_hi, ref_cent,
                     leaf_size, depth + 1, prim_perm_out, leaf_buckets)
    _build_recursive(state, right_id, right_idx, ref_lo, ref_hi, ref_cent,
                     leaf_size, depth + 1, prim_perm_out, leaf_buckets)


# =====================================================================
# SNCH-specific build steps (Note Steps 2-4).
# =====================================================================


def _extract_silhouette_edges(faces_np: np.ndarray, vertices_np: np.ndarray):
    """Step 2 of the brief: extract silhouette-edge candidates from a
    triangle mesh.

    Manifold edges (exactly 2 incident faces) emit `(n0, n1)` from the
    two adjacent face normals. Boundary edges (exactly 1 incident face)
    emit `(n0, -n0)` as a sentinel encoding: the sign-product validity
    test then collapses to `dot(n0, x-v0) * dot(-n0, x-v0) =
    -dot(n0, x-v0)^2 <= 0`, i.e. the boundary edge is always treated
    as a silhouette (a boundary edge separates the surface from
    "nothing", which is the canonical silhouette geometry). Non-
    manifold edges (3+ faces) are skipped (ill-defined).

    Returns `(v0_pos, v1_pos, n0, n1, edge_face_a, edge_face_b)` where
    each row is one silhouette edge. `edge_face_a/b` give the indices
    of the incident face(s) (used for silhouette-ref leaf assignment in
    step 3); for boundary edges `edge_face_b == edge_face_a`.
    """
    F = int(faces_np.shape[0])
    # Build an edge -> [(face_idx, opposite_vertex_idx), ...] dictionary.
    edge_map: dict = {}
    for f in range(F):
        a, b, c = int(faces_np[f, 0]), int(faces_np[f, 1]), int(faces_np[f, 2])
        for (u, v) in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(f)

    v0_pos_list, v1_pos_list = [], []
    n0_list, n1_list = [], []
    edge_face_a, edge_face_b = [], []

    for (u, v), face_list in edge_map.items():
        n_faces = len(face_list)
        if n_faces == 2:
            fa, fb = face_list[0], face_list[1]
            # Per-face unit normal via cross product of edges.
            tri_a = vertices_np[faces_np[fa]]
            n_a = np.cross(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
            n_a_norm = np.linalg.norm(n_a) + 1e-30
            n_a = n_a / n_a_norm
            tri_b = vertices_np[faces_np[fb]]
            n_b = np.cross(tri_b[1] - tri_b[0], tri_b[2] - tri_b[0])
            n_b_norm = np.linalg.norm(n_b) + 1e-30
            n_b = n_b / n_b_norm

            v0_pos_list.append(vertices_np[u])
            v1_pos_list.append(vertices_np[v])
            n0_list.append(n_a)
            n1_list.append(n_b)
            edge_face_a.append(fa)
            edge_face_b.append(fb)
        elif n_faces == 1:
            # Boundary edge sentinel: encode n1 = -n0 so the leaf-step
            # sign-product test `s0 * s1 = -dot(n0, x-v0)^2 <= 0` is
            # satisfied for every query (boundary edges are silhouettes
            # by definition; only the on-plane case yields equality).
            fa = face_list[0]
            tri_a = vertices_np[faces_np[fa]]
            n_a = np.cross(tri_a[1] - tri_a[0], tri_a[2] - tri_a[0])
            n_a_norm = np.linalg.norm(n_a) + 1e-30
            n_a = n_a / n_a_norm

            v0_pos_list.append(vertices_np[u])
            v1_pos_list.append(vertices_np[v])
            n0_list.append(n_a)
            n1_list.append(-n_a)
            edge_face_a.append(fa)
            edge_face_b.append(fa)
        else:
            # Non-manifold (>=3 faces): skip (ill-defined).
            continue

    if len(v0_pos_list) == 0:
        # Closed manifold with no silhouette edges is impossible (any
        # edge is a silhouette candidate); but for safety we emit
        # zero-length arrays and the traversal will trivially miss.
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(v0_pos_list, dtype=np.float32),
        np.asarray(v1_pos_list, dtype=np.float32),
        np.asarray(n0_list, dtype=np.float32),
        np.asarray(n1_list, dtype=np.float32),
        np.asarray(edge_face_a, dtype=np.int32),
        np.asarray(edge_face_b, dtype=np.int32),
    )


def _assign_silhouette_refs_and_cones(state, n_nodes,
                                      leaf_buckets,
                                      silh_v0, silh_v1, silh_n0, silh_n1,
                                      silh_face_a, silh_face_b):
    """Assign leaf edges and bound subtree normals and edge midpoints."""
    E = int(silh_v0.shape[0])

    face_to_silhs: dict = {}
    for e in range(E):
        face_to_silhs.setdefault(int(silh_face_a[e]), []).append(e)
        face_to_silhs.setdefault(int(silh_face_b[e]), []).append(e)

    node_silh_set: list = [set() for _ in range(n_nodes)]
    node_silh_offset = np.zeros((n_nodes,), dtype=np.int32)
    node_silh_count = np.zeros((n_nodes,), dtype=np.int32)
    silh_ref_out: list = []

    post: list = []
    visit_stack = [(0, False)]
    while visit_stack:
        node, visited = visit_stack.pop()
        if visited:
            post.append(node)
            continue
        visit_stack.append((node, True))
        if state.node_leaf_count[node] == 0:
            l = int(state.node_left[node])
            r = int(state.node_right[node])
            if r >= 0:
                visit_stack.append((r, False))
            if l >= 0:
                visit_stack.append((l, False))

    node_centroid = 0.5 * (state.node_lower[:n_nodes] + state.node_upper[:n_nodes])

    silh_mid = 0.5 * (silh_v0 + silh_v1) if E > 0 else np.zeros((0, 3),
                                                                 dtype=np.float32)

    cone_axis_arr = np.zeros((n_nodes, 3), dtype=np.float32)
    cone_half_angle_arr = np.full((n_nodes,), float(np.pi), dtype=np.float32)
    cone_radius_arr = np.zeros((n_nodes,), dtype=np.float32)

    for node in post:
        is_leaf = int(state.node_leaf_count[node]) > 0
        if is_leaf:
            faces_here = leaf_buckets[node]
            silh_set = set()
            for f in faces_here:
                if f in face_to_silhs:
                    silh_set.update(face_to_silhs[f])
            node_silh_set[node] = silh_set
            if len(silh_set) > 0:
                offset = len(silh_ref_out)
                refs_sorted = sorted(silh_set)
                silh_ref_out.extend(refs_sorted)
                node_silh_offset[node] = offset
                node_silh_count[node] = len(refs_sorted)
        else:
            l = int(state.node_left[node])
            r = int(state.node_right[node])
            silh_set = node_silh_set[l] | node_silh_set[r]
            node_silh_set[node] = silh_set
            node_silh_offset[node] = 0
            node_silh_count[node] = 0

        if len(silh_set) == 0:
            cone_axis_arr[node] = np.zeros(3, dtype=np.float32)
            cone_half_angle_arr[node] = float(np.pi)
            cone_radius_arr[node] = 0.0
        else:
            idx = np.fromiter(silh_set, dtype=np.int64, count=len(silh_set))
            normals = np.concatenate([silh_n0[idx], silh_n1[idx]], axis=0)
            axis_sum = normals.sum(axis=0)
            axis_norm = float(np.linalg.norm(axis_sum))
            if axis_norm < 1e-12:
                # Opposing normals need a wide valid cone, not an empty cone.
                cone_axis_arr[node] = np.zeros(3, dtype=np.float32)
                cone_half_angle_arr[node] = float(np.pi) * np.float32(
                    0.5)
            else:
                axis = axis_sum / axis_norm
                # Float32 acos can collapse narrow cones to zero.
                normals64 = normals.astype(np.float64)
                axis64 = axis.astype(np.float64)
                angles = np.arctan2(np.linalg.norm(np.cross(normals64, axis64), axis=1),
                                    normals64 @ axis64)
                half_angle = float(np.nextafter(np.float32(angles.max()), np.float32(np.inf)))
                cone_axis_arr[node] = axis.astype(np.float32)
                cone_half_angle_arr[node] = min(half_angle, float(np.pi))
            # JAX cones bound edge midpoints; Warp expands radii during packing.
            mids = silh_mid[idx]
            d = np.linalg.norm(mids - node_centroid[node][None, :], axis=-1)
            cone_radius_arr[node] = float(d.max()) if d.size > 0 else 0.0

    return (node_silh_offset, node_silh_count,
            np.asarray(silh_ref_out, dtype=np.int32),
            cone_axis_arr, cone_half_angle_arr, cone_radius_arr)


def _build_payload_snch(prim_lo_np, prim_hi_np, leaf_size,
                        faces_np, vertices_np) -> BVHSnchPayload:
    n = int(prim_lo_np.shape[0])
    assert n >= 1, "Need at least one primitive"

    ref_lo = prim_lo_np.astype(np.float32)
    ref_hi = prim_hi_np.astype(np.float32)
    ref_cent = (0.5 * (ref_lo + ref_hi)).astype(np.float32)

    state = _BuildState(n)
    root = state.alloc()
    assert root == 0
    prim_perm_out: list = []
    leaf_buckets: list = [[]]  # grows alongside state allocations
    _build_recursive(state, root, np.arange(n, dtype=np.int32),
                     ref_lo, ref_hi, ref_cent,
                     leaf_size=max(1, leaf_size), depth=0,
                     prim_perm_out=prim_perm_out, leaf_buckets=leaf_buckets)
    n_nodes = state.next_free
    # Pad leaf_buckets to n_nodes (alloc can outpace appends in mid-recurse).
    while len(leaf_buckets) < n_nodes:
        leaf_buckets.append([])

    prim_ids = np.asarray(prim_perm_out, dtype=np.int32)
    assert prim_ids.shape[0] == n

    # Step 2.
    (silh_v0, silh_v1, silh_n0, silh_n1,
     silh_face_a, silh_face_b) = _extract_silhouette_edges(faces_np,
                                                           vertices_np)

    # Steps 3 + 4.
    (node_silh_offset, node_silh_count, silh_refs,
     cone_axis, cone_half_angle, cone_radius) = (
        _assign_silhouette_refs_and_cones(
            state, n_nodes, leaf_buckets,
            silh_v0, silh_v1, silh_n0, silh_n1,
            silh_face_a, silh_face_b))

    return BVHSnchPayload(
        node_lower=jnp.asarray(state.node_lower[:n_nodes]),
        node_upper=jnp.asarray(state.node_upper[:n_nodes]),
        node_left=jnp.asarray(state.node_left[:n_nodes]),
        node_right=jnp.asarray(state.node_right[:n_nodes]),
        node_leaf_offset=jnp.asarray(state.node_leaf_offset[:n_nodes]),
        node_leaf_count=jnp.asarray(state.node_leaf_count[:n_nodes]),
        prim_ids=jnp.asarray(prim_ids),
        root=jnp.asarray(np.int32(0)),
        node_cone_axis=jnp.asarray(cone_axis),
        node_cone_half_angle=jnp.asarray(cone_half_angle),
        node_cone_radius=jnp.asarray(cone_radius),
        node_silhouette_offset=jnp.asarray(node_silh_offset),
        node_silhouette_count=jnp.asarray(node_silh_count),
        silhouette_refs=jnp.asarray(silh_refs if silh_refs.size > 0
                                    else np.zeros((1,), dtype=np.int32)),
        silhouette_v0=jnp.asarray(silh_v0 if silh_v0.shape[0] > 0
                                  else np.zeros((1, 3), dtype=np.float32)),
        silhouette_v1=jnp.asarray(silh_v1 if silh_v1.shape[0] > 0
                                  else np.zeros((1, 3), dtype=np.float32)),
        silhouette_n0=jnp.asarray(silh_n0 if silh_n0.shape[0] > 0
                                  else np.zeros((1, 3), dtype=np.float32)),
        silhouette_n1=jnp.asarray(silh_n1 if silh_n1.shape[0] > 0
                                  else np.zeros((1, 3), dtype=np.float32)),
    )


# =====================================================================
# Traversal helpers (shared with sah_jax pattern; copied here to avoid
# importing private helpers across backend files).
# =====================================================================


def _sqr_dist_point_to_box(p, lo, hi):
    """FCPW BoundingBox::computeSquaredDistance — returns (d2_min, d2_max)."""
    u = lo - p
    v = p - hi
    d2_min = jnp.sum(jnp.maximum(jnp.maximum(u, v), jnp.float32(0.0)) ** 2)
    d2_max = jnp.sum(jnp.minimum(u, v) ** 2)
    return d2_min, d2_max


def _ray_box_t(origin, rcp_dir, lo, hi, tmin, tmax):
    t1 = (lo - origin) * rcp_dir
    t2 = (hi - origin) * rcp_dir
    t_lo = jnp.minimum(t1, t2)
    t_hi = jnp.maximum(t1, t2)
    t_enter = jnp.maximum(jnp.max(t_lo), tmin)
    t_exit = jnp.minimum(jnp.min(t_hi), tmax)
    hit = t_enter <= t_exit
    return t_enter, hit


def _box_overlap(a_lo, a_hi, b_lo, b_hi):
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
# closest_point — adapted from sah_jax (same algorithm, payload type
# differs).
# =====================================================================


def _closest_point_single(payload: BVHSnchPayload, primitives: Primitives,
                          x: jnp.ndarray, max_stack_depth: int,
                          max_leaf_size: int):
    INF = jnp.float32(jnp.inf)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_d_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    root = payload.root.astype(_INDEX_DTYPE)
    _d2_root_min, d2_root_max = _sqr_dist_point_to_box(
        x, payload.node_lower[root], payload.node_upper[root])
    init_best_d = d2_root_max
    init = (root, stack_idx_init, stack_d_init,
            jnp.asarray(0, dtype=_INDEX_DTYPE),
            init_best_d,
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
            tightened_d = jnp.minimum(jnp.minimum(best_d, d0_max), d1_max)
            hit0 = d0_min <= tightened_d
            hit1 = d1_min <= tightened_d

            def both(_):
                inside_both = (d0_min == 0.0) & (d1_min == 0.0)
                left_closer_normal = d0_min <= d1_min
                left_closer_inside = d0_max <= d1_max
                left_first = jnp.where(inside_both, left_closer_inside,
                                       left_closer_normal)
                near = jnp.where(left_first, l, r)
                far = jnp.where(left_first, r, l)
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

        node_d2_min, _ = _sqr_dist_point_to_box(
            x, payload.node_lower[cur], payload.node_upper[cur])
        skip = node_d2_min > best_d

        def skip_branch(_):
            return _pop(si, sd, sp, best_d, best_cp, best_pid)

        def descend_branch(_):
            return lax.cond(is_leaf, leaf_branch, inner_branch, operand=None)

        return lax.cond(skip, skip_branch, descend_branch, operand=None)

    final = lax.while_loop(cond_fn, body_fn, init)
    _cur, _si, _sd, _sp, best_d, best_cp, best_pid, _done = final
    best_dist = jnp.sqrt(jnp.maximum(best_d, jnp.float32(0.0)))
    return best_cp, best_dist, best_pid


# =====================================================================
# first_hit — adapted from sah_jax.
# =====================================================================


def _first_hit_single(payload: BVHSnchPayload, primitives: Primitives,
                      ray_origin, ray_direction, max_t,
                      max_stack_depth, max_leaf_size):
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
# aabb_overlap — adapted from sah_jax.
# =====================================================================


def _aabb_overlap_single(payload: BVHSnchPayload, primitives: Primitives,
                         q_lo, q_hi, max_stack_depth, max_leaf_size):
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


def _distance_bounds_single(payload: BVHSnchPayload, primitives: Primitives,
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
# closest_silhouette — port of FCPW processSubtreeForClosestSilhouettePoint
# (Source: bvh_traverse.inl:268-392). Uses a fixed-size traversal stack;
# at each interior node the cone-validity prune is applied (sentinel
# `half_angle >= pi` means "no silhouettes in subtree" -> skip). The
# tighter cone-vs-query overlap test (FCPW BoundingCone::overlap) is
# ported in `_cone_overlap_box` below.
# =====================================================================


def _compute_orthonormal_basis(n):
    """Pixar 2017 branchless orthonormal-basis construction.

    Source: bounding_cone.h:29-38 (computeOrthonormalBasis).
    Given unit vector ``n``, returns two vectors b1, b2 forming an
    orthonormal frame with ``n``. Implemented branchlessly so it is
    safe under jit + lax.while_loop tracing.
    """
    # std::copysignf(1.0f, n[2]) — sign of n[2], with +1 for n[2] == 0.
    sign = jnp.where(n[2] >= 0.0, jnp.float32(1.0), jnp.float32(-1.0))
    a = jnp.float32(-1.0) / (sign + n[2])
    b = n[0] * n[1] * a
    b1 = jnp.stack([
        jnp.float32(1.0) + sign * n[0] * n[0] * a,
        sign * b,
        -sign * n[0],
    ])
    b2 = jnp.stack([
        b,
        sign + n[1] * n[1] * a,
        -n[1],
    ])
    return b1, b2


def _project_to_plane3(view_axis, e):
    """FCPW projectToPlane<3> — bounding_cone.h:64-75.

    Maximum projection radius of a box (with half-extents ``e``,
    centred at the centroid) onto the plane orthogonal to
    ``view_axis``. Uses an orthonormal basis (b1, b2) of the plane and
    sums the per-axis cwiseAbs projections in quadrature.
    """
    b1, b2 = _compute_orthonormal_basis(view_axis)
    r1 = jnp.dot(e, jnp.abs(b1))
    r2 = jnp.dot(e, jnp.abs(b2))
    return jnp.sqrt(r1 * r1 + r2 * r2)


def _cone_overlap_box(cone_axis, cone_half_angle, cone_radius,
                      x, box_lo, box_hi, dist_to_box):
    """Port of FCPW ``BoundingCone<3>::overlap`` (bounding_cone.h:95-141).

    Returns True iff the per-subtree silhouette cone (axis,
    half_angle, radius) and the "view cone" rooted at ``x`` toward the
    bounding box ``[box_lo, box_hi]`` overlap — i.e. there exist two
    vectors, one in each cone, that are orthogonal to each other.
    Conservative: when uncertain, returns True (no over-culling).

    The two stub outputs ``minAngleRange`` / ``maxAngleRange`` from
    FCPW are dropped (never read by the traversal — see brief).
    """
    eps = jnp.float32(1e-6)
    PI = jnp.float32(np.pi)
    PI_2 = jnp.float32(np.pi * 0.5)

    # Step 1: half_angle >= pi/2 OR query inside box -> overlap (FCPW:103).
    early_pass = (cone_half_angle >= PI_2) | (dist_to_box < eps)

    # Step 2: view cone axis (FCPW:106-109).
    centroid = jnp.float32(0.5) * (box_lo + box_hi)
    view = centroid - x
    l_sq = jnp.dot(view, view)
    l = jnp.sqrt(jnp.maximum(l_sq, jnp.float32(0.0)))
    # Degenerate: x == centroid -> conservative pass.
    l_safe = jnp.maximum(l, eps)
    view_axis = jnp.where(l > eps, view / l_safe, jnp.zeros_like(view))
    l_degenerate = l <= eps

    # Step 3: angle between cone axis and view axis (FCPW:112).
    dot_av = jnp.clip(jnp.dot(cone_axis, view_axis),
                      jnp.float32(-1.0), jnp.float32(1.0))
    d_axis_angle = jnp.arccos(dot_av)

    # Step 4: pre-far/near inRange test using only cone half_angle
    # (FCPW:113). inRange(pi/2, lo, hi) := lo <= pi/2 <= hi.
    in_range_cone_only = (
        ((d_axis_angle - cone_half_angle) <= PI_2)
        & (PI_2 <= (d_axis_angle + cone_half_angle))
    )

    # Step 5a: FAR path — l > cone_radius (FCPW:117-123).
    asin_arg = jnp.clip(cone_radius / jnp.maximum(l, eps),
                        jnp.float32(-1.0), jnp.float32(1.0))
    far_view_half = jnp.arcsin(asin_arg)
    far_half_sum = cone_half_angle + far_view_half
    far_lo = d_axis_angle - far_half_sum
    far_hi = d_axis_angle + far_half_sum
    far_pass = (far_half_sum >= PI_2) | ((far_lo <= PI_2) & (PI_2 <= far_hi))

    # Step 5b: NEAR path — l <= cone_radius (FCPW:128-140).
    # e = b.pMax - centroid (box half-extents).
    e = box_hi - centroid
    # max projection of box onto view axis.
    d_axis = jnp.dot(e, jnp.abs(view_axis))
    s = l - d_axis
    # If s <= 0 the plane through x orthogonal to view_axis cuts the
    # box — view cone is "wider than 90 deg" — overlap (FCPW:131).
    near_plane_hits_box = s <= jnp.float32(0.0)
    # Otherwise project max extents to plane orthogonal to view_axis.
    d_plane = _project_to_plane3(view_axis, e)
    s_safe = jnp.where(near_plane_hits_box, jnp.float32(1.0), s)
    near_view_half = jnp.arctan2(d_plane, s_safe)
    near_half_sum = cone_half_angle + near_view_half
    near_lo = d_axis_angle - near_half_sum
    near_hi = d_axis_angle + near_half_sum
    near_in_range = (near_half_sum >= PI_2) | (
        (near_lo <= PI_2) & (PI_2 <= near_hi)
    )
    near_pass = near_plane_hits_box | near_in_range

    # Step 6: select far vs near branch by l vs cone_radius (FCPW:117).
    far_or_near_pass = jnp.where(l > cone_radius, far_pass, near_pass)

    # Final: overlap iff any of the early-out conditions hold OR the
    # in-range test on the cone half_angle alone passes OR the
    # far/near path passes. Also keep the half_angle < pi validity
    # gate so sentinel cones never overlap.
    half_angle_invalid = cone_half_angle >= PI
    overlap = (
        early_pass
        | l_degenerate
        | in_range_cone_only
        | far_or_near_pass
    )
    return overlap & jnp.logical_not(half_angle_invalid)


def _closest_segment_point(x, v0, v1):
    """Closest point on segment [v0, v1] to point x. Returns (cp, dist)."""
    eps = jnp.float32(1e-20)
    d = v1 - v0
    denom = jnp.dot(d, d) + eps
    t = jnp.clip(jnp.dot(x - v0, d) / denom, 0.0, 1.0)
    cp = v0 + t * d
    dist = jnp.linalg.norm(x - cp)
    return cp, dist


def _closest_silhouette_single(payload: BVHSnchPayload,
                               x: jnp.ndarray,
                               max_stack_depth: int,
                               max_silh_per_leaf: int):
    INF = jnp.float32(jnp.inf)
    PI = jnp.float32(np.pi)

    stack_idx_init = jnp.full((max_stack_depth,), -1, dtype=_INDEX_DTYPE)
    stack_d_init = jnp.full((max_stack_depth,), INF, dtype=jnp.float32)

    root = payload.root.astype(_INDEX_DTYPE)
    d2_root_min, _d2_root_max = _sqr_dist_point_to_box(
        x, payload.node_lower[root], payload.node_upper[root])

    # state: (cur, si, sd, sp, best_d2, best_cp, best_eid, done)
    init = (root,
            stack_idx_init.at[0].set(root),
            stack_d_init.at[0].set(d2_root_min),
            jnp.asarray(1, dtype=_INDEX_DTYPE),
            INF,
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.asarray(-1, dtype=_INDEX_DTYPE),
            jnp.bool_(False))

    def cond_fn(state):
        return jnp.logical_not(state[7])

    def _pop_silh(si, sd, sp, best_d2, best_cp, best_eid):
        """Pop entries while their saved d2_min > best_d2 (FCPW line 287)."""
        def pop_cond(s):
            _ssp, _ssi, _ssd, found, empty = s
            return jnp.logical_not(found | empty)

        def pop_body(s):
            ssp, ssi, ssd, _f, _e = s
            empty = ssp == 0
            new_sp = jnp.where(empty, ssp, ssp - 1)
            top_d = ssd[new_sp]
            in_range = (top_d <= best_d2) & jnp.logical_not(empty)
            return new_sp, ssi, ssd, in_range, empty

        new_sp, _, _, found, _empty = lax.while_loop(
            pop_cond, pop_body,
            (sp, si, sd, jnp.bool_(False), jnp.bool_(False)))
        next_node = jnp.where(found, si[new_sp], jnp.asarray(0, _INDEX_DTYPE))
        done = jnp.logical_not(found)
        return next_node, si, sd, new_sp, best_d2, best_cp, best_eid, done

    def body_fn(state):
        cur, si, sd, sp, best_d2, best_cp, best_eid, _done = state
        leaf_count = payload.node_leaf_count[cur]
        is_leaf = leaf_count > 0

        def leaf_branch(_):
            offset = payload.node_silhouette_offset[cur]
            count = payload.node_silhouette_count[cur]

            def silh_step(k, lc):
                bd, bcp, beid = lc
                valid = k < count
                # Defensive index clamp so out-of-bounds reads when
                # max_silh_per_leaf > count return safe values.
                ref_idx = payload.silhouette_refs[
                    jnp.minimum(offset + k,
                                payload.silhouette_refs.shape[0] - 1)]
                v0 = payload.silhouette_v0[ref_idx]
                v1 = payload.silhouette_v1[ref_idx]
                n0 = payload.silhouette_n0[ref_idx]
                n1 = payload.silhouette_n1[ref_idx]
                # Silhouette validity test (sign-product test):
                # the edge separates a forward-facing face from a
                # back-facing face from x iff dot(n0, x-v0) and
                # dot(n1, x-v0) have opposite signs. `<= 0.0` rather
                # than `< 0.0` accepts the boundary-edge sentinel
                # encoding (n1 = -n0 -> s0*s1 = -s0^2 <= 0) and the
                # on-plane equality case by convention.
                s0 = jnp.dot(n0, x - v0)
                s1 = jnp.dot(n1, x - v0)
                is_silh = (s0 * s1) <= 0.0
                cp_k, dist_k = _closest_segment_point(x, v0, v1)
                d2_k = dist_k * dist_k
                better = valid & is_silh & (d2_k < bd)
                bd = jnp.where(better, d2_k, bd)
                bcp = jnp.where(better, cp_k, bcp)
                beid = jnp.where(better, ref_idx, beid)
                return bd, bcp, beid

            new_bd, new_bcp, new_be = lax.fori_loop(
                0, max_silh_per_leaf, silh_step, (best_d2, best_cp, best_eid))
            return _pop_silh(si, sd, sp, new_bd, new_bcp, new_be)

        def inner_branch(_):
            l = payload.node_left[cur]
            r = payload.node_right[cur]
            l_lo = payload.node_lower[l]
            l_hi = payload.node_upper[l]
            r_lo = payload.node_lower[r]
            r_hi = payload.node_upper[r]
            d0_min, _ = _sqr_dist_point_to_box(x, l_lo, l_hi)
            d1_min, _ = _sqr_dist_point_to_box(x, r_lo, r_hi)
            # Cone validity prune: half_angle >= pi -> sentinel
            # "no silhouettes in subtree" -> never push.
            cone_valid_l = payload.node_cone_half_angle[l] < PI
            cone_valid_r = payload.node_cone_half_angle[r] < PI
            # Box prune.
            box_hit_l = d0_min <= best_d2
            box_hit_r = d1_min <= best_d2
            # Tight FCPW BoundingCone<3>::overlap test (bounding_cone.h:95-141)
            # — see _cone_overlap_box. Uses sqrt of the squared box
            # distance so the eps test matches FCPW's distToBox.
            d0_dist = jnp.sqrt(jnp.maximum(d0_min, jnp.float32(0.0)))
            d1_dist = jnp.sqrt(jnp.maximum(d1_min, jnp.float32(0.0)))
            cone_hit_l = _cone_overlap_box(
                payload.node_cone_axis[l],
                payload.node_cone_half_angle[l],
                payload.node_cone_radius[l],
                x, l_lo, l_hi, d0_dist)
            cone_hit_r = _cone_overlap_box(
                payload.node_cone_axis[r],
                payload.node_cone_half_angle[r],
                payload.node_cone_radius[r],
                x, r_lo, r_hi, d1_dist)

            hit_l = box_hit_l & cone_valid_l & cone_hit_l
            hit_r = box_hit_r & cone_valid_r & cone_hit_r

            def both(_):
                left_first = d0_min <= d1_min
                near = jnp.where(left_first, l, r)
                far = jnp.where(left_first, r, l)
                near_d = jnp.where(left_first, d0_min, d1_min)
                far_d = jnp.where(left_first, d1_min, d0_min)
                # Push farther first (FCPW line 369-376).
                overflow = sp >= max_stack_depth
                nsi = jnp.where(overflow, si, si.at[sp].set(far))
                nsd = jnp.where(overflow, sd, sd.at[sp].set(far_d))
                nsp = jnp.where(overflow, sp, sp + 1)
                # Then push closer.
                overflow2 = nsp >= max_stack_depth
                nsi2 = jnp.where(overflow2, nsi, nsi.at[nsp].set(near))
                nsd2 = jnp.where(overflow2, nsd, nsd.at[nsp].set(near_d))
                nsp2 = jnp.where(overflow2, nsp, nsp + 1)
                return _pop_silh(nsi2, nsd2, nsp2, best_d2, best_cp, best_eid)

            def only_left(_):
                overflow = sp >= max_stack_depth
                nsi = jnp.where(overflow, si, si.at[sp].set(l))
                nsd = jnp.where(overflow, sd, sd.at[sp].set(d0_min))
                nsp = jnp.where(overflow, sp, sp + 1)
                return _pop_silh(nsi, nsd, nsp, best_d2, best_cp, best_eid)

            def only_right(_):
                overflow = sp >= max_stack_depth
                nsi = jnp.where(overflow, si, si.at[sp].set(r))
                nsd = jnp.where(overflow, sd, sd.at[sp].set(d1_min))
                nsp = jnp.where(overflow, sp, sp + 1)
                return _pop_silh(nsi, nsd, nsp, best_d2, best_cp, best_eid)

            def neither(_):
                return _pop_silh(si, sd, sp, best_d2, best_cp, best_eid)

            return lax.cond(
                hit_l & hit_r, both,
                lambda _: lax.cond(
                    hit_l, only_left,
                    lambda __: lax.cond(hit_r, only_right, neither,
                                        operand=None),
                    operand=None),
                operand=None)

        return lax.cond(is_leaf, leaf_branch, inner_branch, operand=None)

    # Pop the seeded root entry first.
    def _seeded_pop(state):
        cur, si, sd, sp, best_d2, best_cp, best_eid, _done = state
        return _pop_silh(si, sd, sp, best_d2, best_cp, best_eid)

    init_after_pop = _seeded_pop(init)
    final = lax.while_loop(cond_fn, body_fn, init_after_pop)
    _cur, _si, _sd, _sp, best_d2, best_cp, best_eid, _done = final
    best_dist = jnp.where(jnp.isinf(best_d2), INF,
                          jnp.sqrt(jnp.maximum(best_d2, jnp.float32(0.0))))
    return best_cp, best_dist, best_eid


# =====================================================================
# Backend class.
# =====================================================================


@register_backend
class BVHSnchJaxBackend(Backend[BVHSnchPayload]):
    """SNCH-augmented SAH BVH with pure-JAX traversal.

    First backend to expose `has_silhouette_query = True`. Triangle-
    mesh specific (mirrors FCPW's SnchNode<DIM=3> design); raises
    `TypeError` on non-`TriMeshPrimitives` input at build time.
    """

    name = "bvh.snch.jax"
    flavor = "bvh"
    algorithm = "snch"
    runtime = "jax"
    payload_type = BVHSnchPayload
    is_differentiable = False
    has_distance_bounds = True
    has_silhouette_query = True

    def __init__(self):
        self._primitives: Primitives | None = None
        self._max_stack_depth: int = _FCPW_BVH_MAX_DEPTH
        self._max_leaf_size: int = _LEAF_SIZE_DEFAULT
        self._max_silh_per_leaf: int = 1

    def build(self, primitives: Primitives, config: Config) -> BVHSnchPayload:
        if not isinstance(primitives, TriMeshPrimitives):
            raise TypeError(
                f"bvh.snch.jax requires TriMeshPrimitives "
                f"(SNCH is triangle-mesh specific); got "
                f"{type(primitives).__name__}"
            )
        self._primitives = primitives
        n = int(primitives.n_primitives)
        idx = jnp.arange(n, dtype=_INDEX_DTYPE)
        lo, hi = primitives.primitive_aabb(idx)
        prim_lo_np = np.asarray(lo, dtype=np.float32)
        prim_hi_np = np.asarray(hi, dtype=np.float32)
        leaf_size = max(1, int(config.leaf_size_threshold)
                        if config.leaf_size_threshold else _LEAF_SIZE_DEFAULT)
        if leaf_size <= 1:
            leaf_size = _LEAF_SIZE_DEFAULT

        faces_np = np.asarray(primitives.mesh.faces, dtype=np.int32)
        vertices_np = np.asarray(primitives.mesh.vertices, dtype=np.float32)
        payload = _build_payload_snch(prim_lo_np, prim_hi_np, leaf_size,
                                      faces_np, vertices_np)

        self._max_stack_depth = max(int(config.max_stack_depth),
                                    _FCPW_BVH_MAX_DEPTH)
        leaf_counts = np.asarray(payload.node_leaf_count, dtype=np.int32)
        self._max_leaf_size = max(1, int(leaf_counts.max()))
        silh_counts = np.asarray(payload.node_silhouette_count, dtype=np.int32)
        self._max_silh_per_leaf = max(1, int(silh_counts.max()))
        return payload

    def closest_point(self, payload: BVHSnchPayload, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        prims = self._primitives
        max_stack = self._max_stack_depth
        max_leaf = self._max_leaf_size

        def single(xq):
            return _closest_point_single(payload, prims, xq, max_stack, max_leaf)

        if x.ndim == 1:
            return single(x)
        return jax.vmap(single)(x)

    def first_hit(self, payload: BVHSnchPayload, ray_origin: jnp.ndarray,
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

    def aabb_overlap(self, payload: BVHSnchPayload,
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

    def distance_bounds(self, payload: BVHSnchPayload,
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

    def closest_silhouette(self, payload: BVHSnchPayload, x: jnp.ndarray
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        max_stack = self._max_stack_depth
        max_silh = self._max_silh_per_leaf

        def single(xq):
            return _closest_silhouette_single(payload, xq, max_stack, max_silh)

        if x.ndim == 1:
            return single(x)
        return jax.vmap(single)(x)
