"""Shared preparation for the named Warp triangle mesh backends."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.primitives import Primitives, TriMeshPrimitives

from ._warp_boundary import BoundaryData, SilhouetteData, _triangles_core
from ._warp_host_metadata import reduce_nodes
from ._warp_layout import WarpBVHPayload
from ._warp_tree import _build_tree
from .sah_jax import _build_payload
from .snch_jax import _build_payload_snch
from .warp_queries.primitives.builtin import Triangles
from .warp_queries.queries.boundary import CoefficientData, SamplingData


def _sampling_data(core, topology):
    if topology is None:
        return SamplingData(core.primitive_measure, core.primitive_measure)
    arrays, _, postorder = topology
    node_area = reduce_nodes(
        np.asarray(core.primitive_measure),
        arrays["node_left"], arrays["node_right"],
        arrays["node_leaf_offset"], arrays["node_leaf_count"], postorder,
        lambda values: np.sum(values, axis=0, dtype=np.float32),
    )
    if not np.isfinite(node_area).all():
        raise ValueError("subtree triangle area overflows float32")
    return SamplingData(core.primitive_measure, jnp.asarray(node_area))


def _snch_features(tree, raw, topology):
    if raw is None:
        empty = jnp.empty((0, 3), jnp.float32)
        return SilhouetteData(empty, empty, empty, empty, tree.prim_ids,
                              tree.node_leaf_offset, tree.node_leaf_count,
                              empty, jnp.empty((0, 2), jnp.float32), empty[:, 0])
    arrays, _, postorder = topology
    left, right, count = (arrays[name] for name in
                          ("node_left", "node_right", "node_leaf_count"))
    cone_half_angle = np.asarray(raw.node_cone_half_angle)
    angles = np.stack((np.sin(cone_half_angle), np.cos(cone_half_angle)), axis=-1)
    angles[cone_half_angle >= 1.5707953] = [1., 0.]
    node_feature_count = np.asarray(raw.node_silhouette_count)
    populated = np.zeros(len(left), dtype=bool)
    for node in postorder:
        if count[node] > 0:
            populated[node] = node_feature_count[node] > 0
        else:
            populated[node] = populated[left[node]] or populated[right[node]]
    # Expand legacy midpoint cones to enclose entire edges. Match the legacy
    # float32 calculation and mark empty subtrees with a negative sine.
    radii = .5 * np.linalg.norm(np.asarray(tree.node_upper) - np.asarray(tree.node_lower), axis=1)
    angles[~populated] = [-1., 0.]
    return SilhouetteData(raw.silhouette_v0, raw.silhouette_v1, raw.silhouette_n0,
                          raw.silhouette_n1, raw.silhouette_refs,
                          raw.node_silhouette_offset, raw.node_silhouette_count,
                          raw.node_cone_axis, jnp.asarray(angles), jnp.asarray(radii))


def build_warp_mesh(primitives: Primitives, config: Config, *, snch=False) -> WarpBVHPayload:
    name = "bvh.snch.warp" if snch else "bvh.sah.warp"
    if not isinstance(primitives, TriMeshPrimitives):
        raise TypeError(f"{name} requires TriMeshPrimitives")
    primitive = Triangles()
    vertices, faces = primitives.mesh.vertices, primitives.mesh.faces
    if faces.shape == (0, 3) and vertices.ndim == 2 and vertices.shape[1] == 3:
        # The legacy empty payload discards unreferenced vertices.
        vertices = vertices[:0]
    data = primitive.indexed(vertices, faces)
    core = _triangles_core(primitive, data, snch_compatibility=snch)
    if snch:
        build = partial(_build_payload_snch, faces_np=np.asarray(data.indices),
                        vertices_np=np.asarray(data.vertices), normals=core.primitive_normal)
    else:
        build = partial(_build_payload, dimension=primitive.dimension)
    tree, geometry, raw, topology = _build_tree(primitive, data, config, build)
    features = _snch_features(tree, raw, topology) if snch else None
    core = primitive.boundary_adapter.pack_core(core, data, tree, geometry)

    # Both names historically exposed subtree areas and rejected their
    # overflow, even though only SNCH binds the sampling query.
    sampling = _sampling_data(core, topology)
    empty_coefficients = jnp.empty((0, 2), jnp.float32)
    return WarpBVHPayload(tree, geometry, BoundaryData(core, features), sampling,
                          CoefficientData(empty_coefficients, empty_coefficients))
