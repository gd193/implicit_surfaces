"""Layer 2/3/4: closest_point query correctness.

For every registered backend × every primitive fixture:
  * Layer 2: distances match `trimesh.proximity.closest_point` (independent oracle).
  * Layer 3: distances match `_brute.brute_closest_point_batch` (project-internal reference).
  * Layer 4: per-result self-consistency (`||cp - x|| == d`, cp lies on prim_idx).

Skips per-flavor if no backends registered, per-test if the primitive
math callback used by query/oracle is not implemented yet.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._brute import brute_closest_point_batch
from ._oracle import oracle_closest_point
from ._invariants import assert_closest_point_consistent
from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing,
                       DIST_RTOL, DIST_ATOL)


def _backend_closest_point_batch(accel, queries):
    return jax.vmap(lambda x: accel.closest_point(x))(queries)


def _check_closest_point(flavor, backend_name, tri_fixture):
    skip_if_primitive_math_missing("closest_point_on_primitive",
                                   tri_fixture.primitives)

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    qs = tri_fixture.query_points
    cp_t, d_t, idx_t = _backend_closest_point_batch(accel, qs)

    # Layer 3 — brute via shared Primitives callbacks.
    cp_b, d_b, idx_b = brute_closest_point_batch(tri_fixture.primitives, qs)
    np.testing.assert_allclose(np.asarray(d_t), np.asarray(d_b),
                               rtol=DIST_RTOL, atol=DIST_ATOL,
                               err_msg="distances disagree with brute")

    # Layer 2 — independent oracle (trimesh).
    qs_np = np.asarray(qs)
    _cp_o, d_o, _face_o = oracle_closest_point(tri_fixture.tri_mesh, qs_np)
    np.testing.assert_allclose(np.asarray(d_t), d_o,
                               rtol=DIST_RTOL, atol=DIST_ATOL,
                               err_msg="distances disagree with trimesh oracle")

    # Layer 4 — self-consistency on every query.
    for k in range(qs.shape[0]):
        assert_closest_point_consistent(
            tri_fixture.primitives, qs[k], cp_t[k], d_t[k], idx_t[k])


@parametrize_backends("bvh")
def test_bvh_closest_point_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_closest_point("bvh", backend_name, tri_fixture)


@parametrize_backends("kd")
def test_kd_closest_point_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_closest_point("kd", backend_name, tri_fixture)


@parametrize_backends("octree")
def test_octree_closest_point_matches_oracle_and_brute(tri_fixture,
                                                       backend_name):
    _check_closest_point("octree", backend_name, tri_fixture)
