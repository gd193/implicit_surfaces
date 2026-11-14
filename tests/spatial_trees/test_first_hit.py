"""First hit against external and exhaustive references."""
from __future__ import annotations

import jax
import numpy as np

from ._oracle import oracle_first_hit
from ._warp_helpers import _oracle_ray
from ._invariants import assert_first_hit_consistent
from ._harness import (parametrize_backends, build_accelerator,
                       T_RTOL, T_ATOL)


def _backend_first_hit_batch(accel, ros, rds, max_t):
    return jax.vmap(lambda o, d, t: accel.first_hit(o, d, t))(ros, rds, max_t)


@parametrize_backends()
def test_first_hit_matches_trimesh_and_numpy(tri_fixture, backend_name):
    flavor = backend_name.split(".")[0]

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    ros = tri_fixture.query_ray_origins
    rds = tri_fixture.query_ray_dirs
    mts = tri_fixture.query_ray_max_t

    t_t, idx_t, hit_t = _backend_first_hit_batch(accel, ros, rds, mts)

    mesh = tri_fixture.tri_mesh
    vertices = np.asarray(mesh.vertices, np.float64)
    t_b = np.asarray([_oracle_ray(vertices, mesh.faces, o, d, cap)[0]
                      for o, d, cap in zip(np.asarray(ros, np.float64),
                                           np.asarray(rds, np.float64), np.asarray(mts))])
    hit_b = np.isfinite(t_b)
    np.testing.assert_array_equal(np.asarray(hit_t), np.asarray(hit_b),
                                  err_msg="hit-mask disagrees with NumPy oracle")
    mask = np.asarray(hit_t) & np.asarray(hit_b)
    if mask.any():
        np.testing.assert_allclose(
            np.asarray(t_t)[mask], np.asarray(t_b)[mask],
            rtol=T_RTOL, atol=T_ATOL,
            err_msg="t disagrees with NumPy oracle on common hits")

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

    assert_first_hit_consistent(tri_fixture.primitives, ros, rds, mts, t_t, idx_t, hit_t)
