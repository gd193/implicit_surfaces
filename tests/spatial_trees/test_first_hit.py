"""Layer 2/3/4: first_hit (ray-primitive) query correctness."""
from __future__ import annotations

import jax
import numpy as np

from ._brute import brute_first_hit_batch
from ._oracle import oracle_first_hit
from ._invariants import assert_first_hit_consistent
from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing,
                       T_RTOL, T_ATOL)


def _backend_first_hit_batch(accel, ros, rds, max_t):
    return jax.vmap(lambda o, d, t: accel.first_hit(o, d, t))(ros, rds, max_t)


def _check_first_hit(flavor, backend_name, tri_fixture):
    skip_if_primitive_math_missing("ray_primitive_intersect",
                                   tri_fixture.primitives)

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    ros = tri_fixture.query_ray_origins
    rds = tri_fixture.query_ray_dirs
    mts = tri_fixture.query_ray_max_t

    t_t, idx_t, hit_t = _backend_first_hit_batch(accel, ros, rds, mts)

    # Layer 3 — brute.
    t_b, idx_b, hit_b = brute_first_hit_batch(tri_fixture.primitives,
                                              ros, rds, mts)
    np.testing.assert_array_equal(np.asarray(hit_t), np.asarray(hit_b),
                                  err_msg="hit-mask disagrees with brute")
    # Compare t only where both report a hit.
    mask = np.asarray(hit_t) & np.asarray(hit_b)
    if mask.any():
        np.testing.assert_allclose(
            np.asarray(t_t)[mask], np.asarray(t_b)[mask],
            rtol=T_RTOL, atol=T_ATOL,
            err_msg="t disagrees with brute on common hits")

    # Layer 2 — trimesh oracle.
    ros_np = np.asarray(ros)
    rds_np = np.asarray(rds)
    mts_np = np.asarray(mts)
    t_o, _idx_o, hit_o = oracle_first_hit(tri_fixture.tri_mesh,
                                          ros_np, rds_np, mts_np)
    np.testing.assert_array_equal(np.asarray(hit_t), hit_o,
                                  err_msg="hit-mask disagrees with trimesh")
    mask = np.asarray(hit_t) & hit_o
    if mask.any():
        np.testing.assert_allclose(
            np.asarray(t_t)[mask], t_o[mask],
            rtol=T_RTOL, atol=T_ATOL,
            err_msg="t disagrees with trimesh on common hits")

    # Layer 4 — self-consistency.
    for k in range(ros.shape[0]):
        assert_first_hit_consistent(
            tri_fixture.primitives, ros[k], rds[k], mts[k],
            t_t[k], idx_t[k], hit_t[k])


@parametrize_backends("bvh")
def test_bvh_first_hit_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_first_hit("bvh", backend_name, tri_fixture)


@parametrize_backends("kd")
def test_kd_first_hit_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_first_hit("kd", backend_name, tri_fixture)


@parametrize_backends("octree")
def test_octree_first_hit_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_first_hit("octree", backend_name, tri_fixture)
