"""Closest point against external and exhaustive references."""
from __future__ import annotations

import jax
import numpy as np

from ._brute import brute_closest_point_batch
from ._oracle import oracle_closest_point
from ._invariants import assert_closest_point_consistent
from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing,
                       DIST_RTOL, DIST_ATOL)


def _backend_closest_point_batch(accel, queries):
    return jax.vmap(lambda x: accel.closest_point(x))(queries)


@parametrize_backends()
def test_closest_point_matches_oracle_and_brute(tri_fixture, backend_name):
    flavor = backend_name.split(".")[0]
    skip_if_primitive_math_missing("closest_point_on_primitive",
                                   tri_fixture.primitives)

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    qs = tri_fixture.query_points
    cp_t, d_t, idx_t = _backend_closest_point_batch(accel, qs)

    cp_b, d_b, idx_b = brute_closest_point_batch(tri_fixture.primitives, qs)
    np.testing.assert_allclose(np.asarray(d_t), np.asarray(d_b),
                               rtol=DIST_RTOL, atol=DIST_ATOL,
                               err_msg="distances disagree with brute")

    qs_np = np.asarray(qs)
    _cp_o, d_o, _face_o = oracle_closest_point(tri_fixture.tri_mesh, qs_np)
    np.testing.assert_allclose(np.asarray(d_t), d_o,
                               rtol=DIST_RTOL, atol=DIST_ATOL,
                               err_msg="distances disagree with trimesh oracle")

    assert_closest_point_consistent(tri_fixture.primitives, qs, cp_t, d_t, idx_t)
