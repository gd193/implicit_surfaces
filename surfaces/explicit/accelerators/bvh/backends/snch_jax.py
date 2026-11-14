"""Host-side silhouette normal cone hierarchy construction.

Builds the original bucket-SAH BVH and propagates silhouette references
and normal cones in NumPy. The Warp SNCH backend consumes this payload;
this module only builds trees and is not a query backend.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Config
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


def build(primitives: Primitives, config: Config | None = None) -> BVHSnchPayload:
    """Build an SNCH payload with the original host-builder leaf settings."""
    config = config if config is not None else Config()
    if not isinstance(primitives, TriMeshPrimitives):
        raise TypeError(
            "SNCH construction requires TriMeshPrimitives "
            f"(SNCH is triangle-mesh specific); got {type(primitives).__name__}"
        )
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
    return _build_payload_snch(prim_lo_np, prim_hi_np, leaf_size,
                               faces_np, vertices_np)
