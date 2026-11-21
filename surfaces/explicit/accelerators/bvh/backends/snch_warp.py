"""Warp SAH backend with SNCH data."""
from __future__ import annotations

import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives

from ._warp_build import build_warp_mesh
from .sah_warp import BVHSahWarpBackend
from .warp_queries.primitives.builtin import Triangles


@register_backend
class BVHSnchWarpBackend(BVHSahWarpBackend):

    name = "bvh.snch.warp"
    algorithm = "snch"
    payload_type = BVHSahWarpBackend.payload_type
    has_silhouette_query = True
    has_boundary_sampling = True
    has_robin_query = True

    def build(self, primitives: Primitives, config: Config):
        return build_warp_mesh(primitives, config, snch=True)

    def closest_silhouette(self, payload, x, max_distance=jnp.inf):
        return self.closest_silhouette_with_status(
            payload, x, max_distance=max_distance)[:3]

    def closest_silhouette_with_status(self, payload, x,
                                       max_distance=jnp.inf):
        from .warp_queries.queries.silhouette import make_silhouette

        return make_silhouette(Triangles())(
            payload.tree, payload.geometry, payload.boundary, x, max_distance)

    def sample_boundary(self, payload, points, radii, randoms):
        from .warp_queries.queries.boundary import make_boundary_sample

        return make_boundary_sample(Triangles())(
            payload.tree, payload.geometry, payload.sampling, points, radii, randoms)

    def star_radius(self, payload, points, cap):
        return self.star_radius_with_status(payload, points, cap)[0]

    def star_radius_with_status(self, payload, points, cap):
        from .warp_queries.queries.silhouette import make_star_radius

        return make_star_radius(Triangles())(payload.tree, payload.geometry, payload.boundary, points, cap)

    def with_robin_coefficients(self, payload, minimum, maximum=None):
        from .warp_queries.queries.boundary import prepare_robin_coefficients

        data = prepare_robin_coefficients(payload.tree, minimum, maximum)
        return payload._replace(coefficients=data)

    def robin_radius(self, payload, points, cap):
        return self.robin_radius_with_status(payload, points, cap)[0]

    def robin_radius_with_status(self, payload, points, cap):
        from .warp_queries.queries.boundary import RobinData, make_robin_radius

        if payload.coefficients.node_bounds.shape != (payload.tree.node_left.shape[0], 2):
            raise ValueError("configure Robin coefficients with with_robin_coefficients first")
        data = RobinData(payload.boundary.core, payload.boundary.silhouette, payload.coefficients)
        return make_robin_radius(Triangles())(payload.tree, payload.geometry, data, points, cap)
