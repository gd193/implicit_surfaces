"""Layer-5 cross-backend agreement.

For every pair of same-flavor backends (and only when ≥2 are
registered), assert their query outputs agree on identical inputs.
Catches bugs in shared infrastructure (registry, payload contract)
that the per-backend layer-2/3 tests can't isolate.

Skips automatically if fewer than 2 backends are registered for a
flavor — this whole module can be empty until the next backend lands.
"""
from __future__ import annotations

from itertools import combinations

import jax
import numpy as np
import pytest

from surfaces.explicit.accelerators import list_backends

from ._harness import (build_accelerator, skip_if_primitive_math_missing,
                       DIST_RTOL, DIST_ATOL, T_RTOL, T_ATOL)


def _pairs(flavor: str):
    backends = list_backends(flavor=flavor)
    return list(combinations(backends, 2))


_BVH_PAIRS = _pairs("bvh")


pytestmark = pytest.mark.skipif(not _BVH_PAIRS,
                                reason="<2 bvh backends registered")


@pytest.mark.parametrize("a,b", _BVH_PAIRS, ids=[f"{a}|{b}"
                                                  for a, b in _BVH_PAIRS])
def test_bvh_pair_closest_point_agrees(tri_fixture_sphere_4k, a, b):
    skip_if_primitive_math_missing("closest_point_on_primitive",
                                   tri_fixture_sphere_4k.primitives)

    accel_a = build_accelerator("bvh", a, tri_fixture_sphere_4k.primitives)
    accel_b = build_accelerator("bvh", b, tri_fixture_sphere_4k.primitives)
    qs = tri_fixture_sphere_4k.query_points

    _cp_a, d_a, _i_a = jax.vmap(lambda x: accel_a.closest_point(x))(qs)
    _cp_b, d_b, _i_b = jax.vmap(lambda x: accel_b.closest_point(x))(qs)

    np.testing.assert_allclose(np.asarray(d_a), np.asarray(d_b),
                               rtol=DIST_RTOL, atol=DIST_ATOL,
                               err_msg=f"distance disagrees: {a} vs {b}")


@pytest.mark.parametrize("a,b", _BVH_PAIRS, ids=[f"{a}|{b}"
                                                  for a, b in _BVH_PAIRS])
def test_bvh_pair_first_hit_agrees(tri_fixture_sphere_4k, a, b):
    skip_if_primitive_math_missing("ray_primitive_intersect",
                                   tri_fixture_sphere_4k.primitives)

    accel_a = build_accelerator("bvh", a, tri_fixture_sphere_4k.primitives)
    accel_b = build_accelerator("bvh", b, tri_fixture_sphere_4k.primitives)
    ros = tri_fixture_sphere_4k.query_ray_origins
    rds = tri_fixture_sphere_4k.query_ray_dirs
    mts = tri_fixture_sphere_4k.query_ray_max_t

    t_a, _i_a, h_a = jax.vmap(lambda o, d, t: accel_a.first_hit(o, d, t))(
        ros, rds, mts)
    t_b, _i_b, h_b = jax.vmap(lambda o, d, t: accel_b.first_hit(o, d, t))(
        ros, rds, mts)

    np.testing.assert_array_equal(np.asarray(h_a), np.asarray(h_b),
                                  err_msg=f"hit-mask disagrees: {a} vs {b}")
    mask = np.asarray(h_a) & np.asarray(h_b)
    if mask.any():
        np.testing.assert_allclose(
            np.asarray(t_a)[mask], np.asarray(t_b)[mask],
            rtol=T_RTOL, atol=T_ATOL,
            err_msg=f"t disagrees: {a} vs {b}")


@pytest.mark.parametrize("a,b", _BVH_PAIRS, ids=[f"{a}|{b}"
                                                  for a, b in _BVH_PAIRS])
def test_bvh_pair_aabb_overlap_agrees(tri_fixture_sphere_4k, a, b):
    skip_if_primitive_math_missing("primitive_aabb",
                                   tri_fixture_sphere_4k.primitives)

    accel_a = build_accelerator("bvh", a, tri_fixture_sphere_4k.primitives)
    accel_b = build_accelerator("bvh", b, tri_fixture_sphere_4k.primitives)
    los = tri_fixture_sphere_4k.query_aabb_lo
    his = tri_fixture_sphere_4k.query_aabb_hi

    o_a = jax.vmap(lambda l, h: accel_a.aabb_overlap(l, h))(los, his)
    o_b = jax.vmap(lambda l, h: accel_b.aabb_overlap(l, h))(los, his)

    np.testing.assert_array_equal(np.asarray(o_a), np.asarray(o_b),
                                  err_msg=f"aabb_overlap disagrees: {a} vs {b}")
