"""Array-only JAX pytree for Warp BVHs."""
from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ._warp_boundary import BoundaryCoreData, BoundaryData, SilhouetteData
    from ._warp_tree import TreeData
    from .warp_queries.primitives.data import TrianglesArrays3
    from .warp_queries.queries.boundary import CoefficientData, SamplingData


_PACKED_PAYLOAD_LIMIT = 1 << 31


class WarpBVHPayload(NamedTuple):
    """Built mesh data using the generic query schemas.

    The legacy named array attributes remain available as views. Tuple fields
    follow the generic grouping, so flat positional construction and flat
    tuple serialization from earlier versions require migration.
    """

    tree: TreeData
    geometry: TrianglesArrays3
    boundary: BoundaryData[BoundaryCoreData, SilhouetteData]
    sampling: SamplingData
    coefficients: CoefficientData

    @property
    def root(self):
        return jnp.asarray(0, jnp.int32)

    @property
    def node_cone_axis(self):
        features = self.boundary.silhouette
        if features is None:
            return jnp.zeros_like(self.tree.node_lower)
        return features.node_cone_axis

    @property
    def node_cone_sin_cos(self):
        features = self.boundary.silhouette
        if features is None:
            return jnp.broadcast_to(jnp.asarray([1., 0.], jnp.float32), (len(self.tree.node_left), 2))
        return features.node_cone_sin_cos

    @property
    def node_cone_radius(self):
        features = self.boundary.silhouette
        if features is None:
            return jnp.zeros_like(self.sampling.node_weight)
        return features.node_cone_radius

    def _silhouette_view(self, name, *, index=False, node=False):
        features = self.boundary.silhouette
        if features is not None:
            return getattr(features, name)
        if node:
            return jnp.zeros_like(self.tree.node_left)
        if index:
            return self.tree.prim_ids[:0]
        return self.geometry.a[:0]

    @property
    def node_silhouette_offset(self):
        return self._silhouette_view("node_feature_offset", node=True)

    @property
    def node_silhouette_count(self):
        return self._silhouette_view("node_feature_count", node=True)

    @property
    def silhouette_refs(self):
        return self._silhouette_view("feature_refs", index=True)

    @property
    def silhouette_v0(self):
        return self._silhouette_view("feature_a")

    @property
    def silhouette_v1(self):
        return self._silhouette_view("feature_b")

    @property
    def silhouette_n0(self):
        return self._silhouette_view("feature_n0")

    @property
    def silhouette_n1(self):
        return self._silhouette_view("feature_n1")


def _payload_view(path):
    return property(attrgetter(path))


# Compatibility aliases for clients reading the original flat named arrays.
for _field in ("node_lower", "node_upper", "node_left", "node_right",
               "node_leaf_offset", "node_leaf_count", "prim_ids",
               "node_ray_lower", "node_ray_upper"):
    setattr(WarpBVHPayload, _field, _payload_view(f"tree.{_field}"))

for _field, _path in {
    "tri_v0": "geometry.a",
    "tri_e1": "geometry.e1",
    "tri_e2": "geometry.e2",
    "tri_b": "geometry.b",
    "tri_c": "geometry.c",
    "tri_frame": "geometry.frame",
    "mesh_vertices": "geometry.vertices",
    "tri_indices": "geometry.indices",
    "tri_normal": "boundary.core.primitive_normal",
    "tri_area": "boundary.core.primitive_measure",
    "mesh_closed": "boundary.core.closed",
    "node_area": "sampling.node_weight",
    "tri_robin_bounds": "coefficients.primitive_bounds",
    "node_robin_bounds": "coefficients.node_bounds",
}.items():
    setattr(WarpBVHPayload, _field, _payload_view(_path))


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
