"""Layer-1 analytical fixtures.

Hand-computed scenarios with closed-form expected answers. Run before
oracle/brute-force tests as a smoke layer — if these fail, the
framework itself is broken.

Each test parametrizes over every registered backend of its flavor.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.primitives import TriMeshPrimitives

from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing,
                       DIST_RTOL, DIST_ATOL)


# =====================================================================
# Single-triangle fixtures with closed-form answers.
# =====================================================================


@pytest.fixture(scope="module")
def single_triangle() -> TriMeshPrimitives:
    """One triangle in z=0: vertices (0,0,0), (1,0,0), (0,1,0)."""
    v = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
    f = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    return TriMeshPrimitives(TriMesh(vertices=v, faces=f))


@parametrize_backends("bvh")
def test_closest_point_above_centroid_bvh(single_triangle, backend_name):
    """Query at (1/3, 1/3, 2): closest is (1/3, 1/3, 0), dist = 2."""
    skip_if_primitive_math_missing("closest_point_on_primitive",
                                   single_triangle)
    accel = build_accelerator("bvh", backend_name, single_triangle)
    x = jnp.asarray([1.0 / 3.0, 1.0 / 3.0, 2.0], dtype=jnp.float32)
    cp, dist, prim_idx = accel.closest_point(x)
    assert int(prim_idx) == 0
    np.testing.assert_allclose(np.asarray(cp),
                               np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
                               rtol=DIST_RTOL, atol=DIST_ATOL)
    np.testing.assert_allclose(float(dist), 2.0,
                               rtol=DIST_RTOL, atol=DIST_ATOL)


@parametrize_backends("bvh")
def test_first_hit_axis_ray_bvh(single_triangle, backend_name):
    """Ray from (0.25, 0.25, 5) heading in -z direction hits the
    triangle at t = 5.
    """
    skip_if_primitive_math_missing("ray_primitive_intersect",
                                   single_triangle)
    accel = build_accelerator("bvh", backend_name, single_triangle)
    ro = jnp.asarray([0.25, 0.25, 5.0], dtype=jnp.float32)
    rd = jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32)
    t_max = jnp.float32(100.0)
    t, prim_idx, hit = accel.first_hit(ro, rd, t_max)
    assert bool(hit)
    assert int(prim_idx) == 0
    np.testing.assert_allclose(float(t), 5.0, rtol=DIST_RTOL, atol=DIST_ATOL)


@parametrize_backends("bvh")
def test_aabb_overlap_known_bvh(single_triangle, backend_name):
    """AABB [(-0.1, -0.1, -0.1), (0.6, 0.6, 0.1)] clearly overlaps the
    triangle. AABB [(2, 2, 2), (3, 3, 3)] clearly does not.
    """
    skip_if_primitive_math_missing("primitive_aabb", single_triangle)
    accel = build_accelerator("bvh", backend_name, single_triangle)

    lo_in = jnp.asarray([-0.1, -0.1, -0.1], dtype=jnp.float32)
    hi_in = jnp.asarray([0.6, 0.6, 0.1], dtype=jnp.float32)
    assert bool(accel.aabb_overlap(lo_in, hi_in))

    lo_out = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float32)
    hi_out = jnp.asarray([3.0, 3.0, 3.0], dtype=jnp.float32)
    assert not bool(accel.aabb_overlap(lo_out, hi_out))
