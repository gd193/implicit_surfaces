"""Array-only JAX pytree for Warp BVHs."""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


_PACKED_PAYLOAD_LIMIT = 1 << 31


class WarpBVHPayload(NamedTuple):
    # Root=0; prim_ids maps leaf slots to original face IDs.
    node_lower: jnp.ndarray
    node_upper: jnp.ndarray
    node_left: jnp.ndarray
    node_right: jnp.ndarray
    node_leaf_offset: jnp.ndarray
    node_leaf_count: jnp.ndarray
    prim_ids: jnp.ndarray
    root: jnp.ndarray
    # SNCH data; empty for ordinary SAH.
    node_cone_axis: jnp.ndarray
    node_cone_sin_cos: jnp.ndarray
    node_cone_radius: jnp.ndarray
    node_silhouette_offset: jnp.ndarray
    node_silhouette_count: jnp.ndarray
    silhouette_refs: jnp.ndarray
    silhouette_v0: jnp.ndarray
    silhouette_v1: jnp.ndarray
    silhouette_n0: jnp.ndarray
    silhouette_n1: jnp.ndarray
    # Packed triangles in leaf order.
    tri_v0: jnp.ndarray
    tri_e1: jnp.ndarray
    tri_e2: jnp.ndarray
    tri_normal: jnp.ndarray
    tri_area: jnp.ndarray
    node_area: jnp.ndarray
    # Ray nodes: float4 xyz=bounds, w=uint32 metadata.
    node_ray_lower: jnp.ndarray
    node_ray_upper: jnp.ndarray
    # Shared vertices avoid cracks from reconstructing packed triangle edges.
    mesh_vertices: jnp.ndarray
    tri_indices: jnp.ndarray
    mesh_closed: jnp.ndarray
    # Optional Robin coefficient intervals.
    tri_robin_bounds: jnp.ndarray
    node_robin_bounds: jnp.ndarray


def _int32_array(name, value):
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"tree {name} must contain integers")
    if raw.size and (np.any(raw < np.iinfo(np.int32).min)
                     or np.any(raw > np.iinfo(np.int32).max)):
        raise ValueError(f"tree {name} exceeds int32 indexing")
    return raw.astype(np.int32, copy=False)


def _validate_tree(tree, n_refs):
    """Return index arrays, depths and postorder; root must be 0."""
    arrays = {
        name: _int32_array(name, getattr(tree, name))
        for name in ("node_left", "node_right", "node_leaf_offset", "node_leaf_count")
    }
    n_nodes = arrays["node_left"].shape[0] if arrays["node_left"].ndim == 1 else -1
    if n_nodes < 1 or any(value.shape != (n_nodes,) for value in arrays.values()):
        raise ValueError("tree index and leaf arrays must have matching shape (N,)")
    root_array = _int32_array("root", getattr(tree, "root", 0))
    if root_array.size != 1 or int(root_array.reshape(())) != 0:
        raise ValueError("Warp BVH tree root must be node 0")
    if n_nodes >= _PACKED_PAYLOAD_LIMIT:
        raise ValueError("Warp BVH node indices exceed the 31-bit packed payload")

    left = arrays["node_left"]
    right = arrays["node_right"]
    offset = arrays["node_leaf_offset"]
    count = arrays["node_leaf_count"]
    refs_covered = np.zeros((n_refs,), dtype=bool)
    for node in range(n_nodes):
        if count[node] < 0:
            raise ValueError("tree leaf counts must be nonnegative")
        if count[node] > 0:
            begin = int(offset[node])
            end = begin + int(count[node])
            if begin < 0 or end > n_refs or end >= _PACKED_PAYLOAD_LIMIT:
                raise ValueError("tree leaf range is outside packed primitive references")
            if left[node] >= 0 or right[node] >= 0:
                raise ValueError("tree leaf nodes cannot have child indices")
            if np.any(refs_covered[begin:end]):
                raise ValueError("tree leaf primitive ranges overlap")
            refs_covered[begin:end] = True
        else:
            if offset[node] < 0:
                raise ValueError("tree leaf offsets must be nonnegative")
            if left[node] < 0 or right[node] < 0:
                raise ValueError("internal tree nodes must have two children")
            if left[node] == right[node]:
                raise ValueError("internal tree children must be distinct")
            if left[node] >= n_nodes or right[node] >= n_nodes:
                raise ValueError("tree child index is outside node arrays")
    if n_refs and not np.all(refs_covered):
        raise ValueError("tree leaf ranges do not cover packed primitive references")

    # Validate topology before accumulating areas.
    state = np.zeros((n_nodes,), dtype=np.uint8)
    depths = np.zeros((n_nodes,), dtype=np.int32)
    postorder = []
    stack = [(0, False, 0)]
    while stack:
        node, exiting, depth = stack.pop()
        if exiting:
            state[node] = 2
            postorder.append(node)
            continue
        if state[node] != 0:
            raise ValueError("tree topology contains a cycle or shared child")
        state[node] = 1
        depths[node] = depth
        stack.append((node, True, depth))
        if count[node] == 0:
            stack.append((int(right[node]), False, depth + 1))
            stack.append((int(left[node]), False, depth + 1))
    if np.any(state != 2):
        raise ValueError("tree contains nodes unreachable from root 0")
    return arrays, depths, postorder


def _pack_ray_nodes(node_lower, node_upper, left, right, offset, count):
    """float4 xyz=bounds; w holds uint32 bits, never numeric floats.
    Bit 31 marks leaves; low bits hold left/begin or right/end (exclusive)."""
    end = offset.astype(np.int64) + count.astype(np.int64)
    ray_lower = np.zeros((len(left), 4), dtype=np.float32)
    ray_upper = np.zeros_like(ray_lower)
    ray_lower[:, :3] = node_lower
    ray_upper[:, :3] = node_upper
    lower_bits = left.astype(np.uint32, copy=True)
    upper_bits = right.astype(np.uint32, copy=True)
    leaves = count > 0
    lower_bits[leaves] = offset[leaves].astype(np.uint32) | np.uint32(0x80000000)
    upper_bits[leaves] = end[leaves].astype(np.uint32) | np.uint32(0x80000000)
    ray_lower[:, 3] = lower_bits.view(np.float32)
    ray_upper[:, 3] = upper_bits.view(np.float32)
    return ray_lower, ray_upper


def _is_closed(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    if np.any(np.all(normals == 0, axis=1)):
        return False
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def make_warp_payload(tree, vertices_np, faces_np, snch=None, *, mesh_vertices=None):
    """Pack triangles in leaf order; preserve original face IDs."""
    vertices = np.asarray(vertices_np, dtype=np.float32)
    faces = np.asarray(faces_np, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("mesh vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("mesh faces must have shape (F, 3)")
    if faces.size and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("mesh faces contain an invalid vertex index")
    prim_ids = _int32_array("prim_ids", tree.prim_ids)
    if prim_ids.ndim != 1 or (prim_ids.size and (
            np.any(prim_ids < 0) or np.any(prim_ids >= len(faces)))):
        raise ValueError("tree primitive references contain an invalid face index")

    node_lower = np.asarray(tree.node_lower, dtype=np.float32)
    node_upper = np.asarray(tree.node_upper, dtype=np.float32)
    if node_lower.ndim != 2 or node_lower.shape[1:] != (3,):
        raise ValueError("tree lower bounds must have shape (N, 3)")
    if node_upper.shape != node_lower.shape:
        raise ValueError("tree upper bounds must match lower bound shape")
    if (not np.isfinite(node_lower).all() or not np.isfinite(node_upper).all()
            or np.any(node_lower > node_upper)):
        raise ValueError("tree bounds must be finite and ordered")
    n_nodes = node_lower.shape[0]
    tree_arrays, depths, postorder = _validate_tree(tree, prim_ids.size)
    if n_nodes != len(tree_arrays["node_left"]):
        raise ValueError("tree bounds and index arrays must have matching node counts")
    if prim_ids.size and not np.isfinite(vertices[faces[prim_ids]]).all():
        raise ValueError("packed triangle geometry requires finite referenced vertices")
    if prim_ids.size:
        tris = vertices[faces[prim_ids]]
        v0 = tris[:, 0]
        e1 = tris[:, 1] - v0
        e2 = tris[:, 2] - v0
        cross = np.cross(e1, e2)
        double_area = np.linalg.norm(cross, axis=1)
        area = 0.5 * double_area
        normal = cross / np.maximum(double_area[:, None], np.float32(1e-30))
        if not all(np.isfinite(a).all() for a in (v0, e1, e2, normal, area)):
            raise ValueError("packed triangle geometry overflows float32")
    else:
        v0 = e1 = e2 = normal = np.zeros((0, 3), dtype=np.float32)
        area = np.zeros((0,), dtype=np.float32)

    # Postorder supports arbitrary node numbering.
    left = tree_arrays["node_left"]
    right = tree_arrays["node_right"]
    offset = tree_arrays["node_leaf_offset"]
    count = tree_arrays["node_leaf_count"]
    node_area = np.zeros((n_nodes,), dtype=np.float32)
    for node in postorder:
        if count[node] > 0:
            a, b = int(offset[node]), int(offset[node] + count[node])
            node_area[node] = np.sum(area[a:b], dtype=np.float32)
        elif left[node] >= 0 and right[node] >= 0:
            node_area[node] = node_area[left[node]] + node_area[right[node]]
    if not np.isfinite(node_area).all():
        raise ValueError("subtree triangle area overflows float32")

    if snch is None:
        cone_axis = np.zeros((n_nodes, 3), dtype=np.float32)
        cone_sin_cos = np.tile(np.asarray([1., 0.], np.float32), (n_nodes, 1))
        cone_radius = np.zeros((n_nodes,), dtype=np.float32)
        silh_offset = silh_count = np.zeros((n_nodes,), dtype=np.int32)
        silh_refs = np.zeros((0,), dtype=np.int32)
        silh_v0 = silh_v1 = silh_n0 = silh_n1 = np.zeros((0, 3), dtype=np.float32)
    else:
        cone_axis = np.asarray(snch.node_cone_axis, dtype=np.float32)
        cone_half_angle = np.asarray(snch.node_cone_half_angle, dtype=np.float32)
        cone_sin_cos = np.stack([np.sin(cone_half_angle), np.cos(cone_half_angle)], axis=-1)
        cone_sin_cos[cone_half_angle >= 1.5707953] = [1., 0.]
        # Half the box diagonal encloses whole edges.
        cone_radius = .5 * np.linalg.norm(node_upper - node_lower, axis=1)
        silh_offset = np.asarray(snch.node_silhouette_offset, dtype=np.int32)
        silh_count = np.asarray(snch.node_silhouette_count, dtype=np.int32)
        silh_refs = np.asarray(snch.silhouette_refs, dtype=np.int32)
        # SNCH pads empty edge arrays with one safe row.
        silh_v0 = np.asarray(snch.silhouette_v0, dtype=np.float32)
        silh_v1 = np.asarray(snch.silhouette_v1, dtype=np.float32)
        silh_n0 = np.asarray(snch.silhouette_n0, dtype=np.float32)
        silh_n1 = np.asarray(snch.silhouette_n1, dtype=np.float32)

        has_silhouette = np.zeros((n_nodes,), dtype=bool)
        for node in postorder:
            if count[node] > 0:
                has_silhouette[node] = silh_count[node] > 0
            else:
                has_silhouette[node] = has_silhouette[left[node]] or has_silhouette[right[node]]
        # sin<0 marks empty subtrees; cos=0 disables cone pruning.
        cone_sin_cos[~has_silhouette] = [-1., 0.]

    ray_lower, ray_upper = _pack_ray_nodes(
        node_lower, node_upper, left, right, offset, count)

    return WarpBVHPayload(
        node_lower=jnp.asarray(node_lower),
        node_upper=jnp.asarray(node_upper),
        node_left=jnp.asarray(left),
        node_right=jnp.asarray(right),
        node_leaf_offset=jnp.asarray(offset),
        node_leaf_count=jnp.asarray(count),
        prim_ids=jnp.asarray(prim_ids),
        root=jnp.asarray(0, dtype=jnp.int32),
        node_cone_axis=jnp.asarray(cone_axis),
        node_cone_sin_cos=jnp.asarray(cone_sin_cos),
        node_cone_radius=jnp.asarray(cone_radius),
        node_silhouette_offset=jnp.asarray(silh_offset),
        node_silhouette_count=jnp.asarray(silh_count),
        silhouette_refs=jnp.asarray(silh_refs),
        silhouette_v0=jnp.asarray(silh_v0),
        silhouette_v1=jnp.asarray(silh_v1),
        silhouette_n0=jnp.asarray(silh_n0),
        silhouette_n1=jnp.asarray(silh_n1),
        tri_v0=jnp.asarray(v0),
        tri_e1=jnp.asarray(e1),
        tri_e2=jnp.asarray(e2),
        tri_normal=jnp.asarray(normal),
        tri_area=jnp.asarray(area),
        node_area=jnp.asarray(node_area),
        node_ray_lower=jnp.asarray(ray_lower),
        node_ray_upper=jnp.asarray(ray_upper),
        mesh_vertices=jnp.asarray(vertices if mesh_vertices is None else mesh_vertices,
                                  dtype=jnp.float32),
        tri_indices=jnp.asarray(faces[prim_ids]),
        mesh_closed=jnp.asarray([_is_closed(vertices, faces)], jnp.int32),
        tri_robin_bounds=jnp.zeros((0, 2), jnp.float32),
        node_robin_bounds=jnp.zeros((0, 2), jnp.float32),
    ), int(depths.max(initial=0))


def empty_warp_payload():
    empty3 = jnp.zeros((0, 3), dtype=jnp.float32)
    empty1 = jnp.zeros((0,), dtype=jnp.float32)
    emptyi = jnp.zeros((0,), dtype=jnp.int32)
    empty4 = jnp.zeros((0, 4), dtype=jnp.float32)
    return WarpBVHPayload(
        node_lower=empty3,
        node_upper=empty3,
        node_left=emptyi,
        node_right=emptyi,
        node_leaf_offset=emptyi,
        node_leaf_count=emptyi,
        prim_ids=emptyi,
        root=jnp.asarray(0, jnp.int32),
        node_cone_axis=empty3,
        node_cone_sin_cos=jnp.zeros((0, 2), dtype=jnp.float32),
        node_cone_radius=empty1,
        node_silhouette_offset=emptyi,
        node_silhouette_count=emptyi,
        silhouette_refs=emptyi,
        silhouette_v0=empty3,
        silhouette_v1=empty3,
        silhouette_n0=empty3,
        silhouette_n1=empty3,
        tri_v0=empty3,
        tri_e1=empty3,
        tri_e2=empty3,
        tri_normal=empty3,
        tri_area=empty1,
        node_area=empty1,
        node_ray_lower=empty4,
        node_ray_upper=empty4,
        mesh_vertices=empty3,
        tri_indices=jnp.zeros((0, 3), jnp.int32),
        mesh_closed=jnp.ones((1,), jnp.int32),
        tri_robin_bounds=jnp.zeros((0, 2), jnp.float32),
        node_robin_bounds=jnp.zeros((0, 2), jnp.float32),
    )


def with_robin_coefficients(payload, minimum, maximum=None):
    from ._warp_boundary_queries import prepare_robin_coefficients

    data = prepare_robin_coefficients(payload, minimum, maximum)
    return payload._replace(tri_robin_bounds=data.primitive_bounds, node_robin_bounds=data.node_bounds)
