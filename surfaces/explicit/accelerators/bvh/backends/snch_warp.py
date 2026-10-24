"""Warp SAH backend with SNCH data."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives

from ._warp_build import finite_primitive_bounds, validate_warp_mesh_and_config
from ._warp_jax import query as _warp_query
from ._warp_layout import empty_warp_payload, make_warp_payload
from .sah_warp import BVHSahWarpBackend, _flatten_vec_queries, _restore
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
        payload, depth = make_warp_payload(tree, vertices, faces, snch=tree)
        if depth > config.max_stack_depth:
            raise ValueError(
                f"built SNCH tree depth {depth} exceeds configured "
                f"max_stack_depth={config.max_stack_depth}"
            )

        # Cone angle -1 marks empty subtrees.
        left = np.asarray(payload.node_left, np.int32)
        right = np.asarray(payload.node_right, np.int32)
        leaf_count = np.asarray(payload.node_leaf_count, np.int32)
        leaf_silh = np.asarray(payload.node_silhouette_count, np.int32)
        has_silh = np.zeros_like(leaf_count, dtype=bool)
        for i in range(len(left) - 1, -1, -1):
            if leaf_count[i] > 0:
                has_silh[i] = leaf_silh[i] > 0
            elif left[i] >= 0 and right[i] >= 0:
                has_silh[i] = has_silh[left[i]] or has_silh[right[i]]
        cone_angle = np.asarray(payload.node_cone_half_angle, np.float32).copy()
        cone_angle[~has_silh] = -1.0
        # Half the box diagonal encloses whole edges.
        bounds = np.asarray(payload.node_upper) - np.asarray(payload.node_lower)
        cone_radius = 0.5 * np.linalg.norm(np.maximum(bounds, 0.0), axis=1)
        payload = payload._replace(
            node_cone_half_angle=jnp.asarray(cone_angle),
            node_cone_radius=jnp.asarray(cone_radius, jnp.float32))
        return payload

    def closest_silhouette(self, payload, x, max_distance=jnp.inf):
        return self.closest_silhouette_with_status(
            payload, x, max_distance=max_distance)[:3]

    def closest_silhouette_with_status(self, payload, x,
                                       max_distance=jnp.inf):
        q, shape = _flatten_vec_queries(x)
        cap = jnp.broadcast_to(jnp.asarray(max_distance, jnp.float32), shape).reshape((-1,))
        cp, dist, edge, status = _warp_query("closest_silhouette_kernel", payload, q, cap)
        return (_restore(cp, shape),
                _restore(dist, shape),
                _restore(edge, shape),
                _restore(status, shape))

    def sample_boundary(self, payload, points, radii, randoms):
        """(point, normal, area_pdf, face, status); retain null trials (status 0)."""
        q, shape = _flatten_vec_queries(points)
        rad = jnp.broadcast_to(jnp.asarray(radii, jnp.float32), shape).reshape((-1,))
        rnd = jnp.asarray(randoms, jnp.float32)
        if rnd.shape[-1:] != (3,):
            raise ValueError("randoms must have shape (..., 3)")
        rnd = jnp.broadcast_to(rnd, shape + (3,)).reshape((-1, 3))
        point, normal, pdf, prim, status = _warp_query(
            "sample_boundary_kernel", payload, q, rad, rnd)
        return (_restore(point, shape), _restore(normal, shape),
                _restore(pdf, shape), _restore(prim, shape),
                _restore(status, shape))

    def star_radius(self, payload, points, cap):
        return self.star_radius_with_status(payload, points, cap)[0]

    def star_radius_with_status(self, payload, points, cap):
        _, distance, _, status = self.closest_silhouette_with_status(
            payload, points, max_distance=cap)
        cap = jnp.asarray(cap, jnp.float32)
        valid = (status >= 0) & (cap >= 0.0) & ~jnp.isnan(cap)
        radius = jnp.minimum(cap, distance)
        return (jnp.where(valid, radius, jnp.nan),
                jnp.where(valid, 1, -1).astype(jnp.int32))
