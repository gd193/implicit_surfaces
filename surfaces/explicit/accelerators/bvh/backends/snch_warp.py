"""Warp SAH backend with SNCH data."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives

from ._warp_build import finite_primitive_bounds, validate_warp_mesh_and_config
from ._warp_layout import empty_warp_payload, make_warp_payload, with_robin_coefficients
from .sah_warp import BVHSahWarpBackend, _factory_data, _factory_boundary
from .snch_jax import _build_payload_snch


def _reject_nonmanifold(faces):
    edges = {}
    for face_id, face in enumerate(faces):
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append(face_id)
    bad = [edge for edge, incident in edges.items() if len(incident) > 2]
    if bad:
        raise ValueError(
            "bvh.snch.warp rejects non-manifold edges with more than two "
            f"incident faces; found {len(bad)}"
        )


@register_backend
class BVHSnchWarpBackend(BVHSahWarpBackend):

    name = "bvh.snch.warp"
    algorithm = "snch"
    payload_type = BVHSahWarpBackend.payload_type
    has_silhouette_query = True
    has_boundary_sampling = True
    has_robin_query = True

    def build(self, primitives: Primitives, config: Config):
        vertices, faces = validate_warp_mesh_and_config(
            primitives, config, "bvh.snch.warp")
        if faces.shape[0] == 0:
            return empty_warp_payload()
        used = vertices[faces]
        _reject_nonmanifold(faces)
        tri_cross = np.cross(used[:, 1] - used[:, 0], used[:, 2] - used[:, 0])
        if np.any(np.linalg.norm(tri_cross, axis=1) <= 1e-30):
            raise ValueError("bvh.snch.warp requires non-degenerate triangles")

        lo_np, hi_np = finite_primitive_bounds(primitives)
        tree = _build_payload_snch(
            lo_np, hi_np, int(config.leaf_size_threshold), faces, vertices)
        payload, depth = make_warp_payload(
            tree, vertices, faces, snch=tree, mesh_vertices=primitives.mesh.vertices)
        if depth > config.max_stack_depth:
            raise ValueError(
                f"built SNCH tree depth {depth} exceeds configured "
                f"max_stack_depth={config.max_stack_depth}"
            )

        return payload

    def closest_silhouette(self, payload, x, max_distance=jnp.inf):
        return self.closest_silhouette_with_status(
            payload, x, max_distance=max_distance)[:3]

    def closest_silhouette_with_status(self, payload, x,
                                       max_distance=jnp.inf):
        from ._warp_silhouette import make_silhouette

        primitive, tree, geometry = _factory_data(payload)
        return make_silhouette(primitive)(tree, geometry, _factory_boundary(payload), x, max_distance)

    def sample_boundary(self, payload, points, radii, randoms):
        from ._warp_boundary_queries import SamplingData, make_boundary_sample

        primitive, tree, geometry = _factory_data(payload)
        data = SamplingData(payload.tri_area, payload.node_area)
        return make_boundary_sample(primitive)(tree, geometry, data, points, radii, randoms)

    def star_radius(self, payload, points, cap):
        return self.star_radius_with_status(payload, points, cap)[0]

    def star_radius_with_status(self, payload, points, cap):
        from ._warp_silhouette import make_star_radius

        primitive, tree, geometry = _factory_data(payload)
        return make_star_radius(primitive)(tree, geometry, _factory_boundary(payload), points, cap)

    def with_robin_coefficients(self, payload, minimum, maximum=None):
        return with_robin_coefficients(payload, minimum, maximum)

    def robin_radius(self, payload, points, cap):
        return self.robin_radius_with_status(payload, points, cap)[0]

    def robin_radius_with_status(self, payload, points, cap):
        from ._warp_boundary_queries import CoefficientData, RobinData, make_robin_radius

        if payload.node_robin_bounds.shape != (payload.node_left.shape[0], 2):
            raise ValueError("configure Robin coefficients with with_robin_coefficients first")
        primitive, tree, geometry = _factory_data(payload)
        boundary = _factory_boundary(payload)
        data = RobinData(boundary.core, boundary.silhouette,
                         CoefficientData(payload.tri_robin_bounds, payload.node_robin_bounds))
        return make_robin_radius(primitive)(tree, geometry, data, points, cap)
