"""Layer 2/3/4: aabb_overlap query correctness.

Bool-exact equality (no tolerance) against both the brute-force
reference and the trimesh oracle.
"""
from __future__ import annotations

import jax
import numpy as np

from ._brute import brute_aabb_overlap_batch
from ._oracle import oracle_aabb_overlap
from ._invariants import assert_aabb_overlap_consistent
from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing)


def _backend_aabb_overlap_batch(accel, los, his):
    return jax.vmap(lambda lo, hi: accel.aabb_overlap(lo, hi))(los, his)


def _check_aabb_overlap(flavor, backend_name, tri_fixture):
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    out_t = _backend_aabb_overlap_batch(accel, los, his)

    # Layer 3 — brute.
    out_b = brute_aabb_overlap_batch(tri_fixture.primitives, los, his)
    np.testing.assert_array_equal(np.asarray(out_t), np.asarray(out_b),
                                  err_msg="aabb_overlap disagrees with brute")

    # Layer 2 — trimesh.
    out_o = oracle_aabb_overlap(tri_fixture.tri_mesh,
                                np.asarray(los), np.asarray(his))
    np.testing.assert_array_equal(np.asarray(out_t), out_o,
                                  err_msg="aabb_overlap disagrees with trimesh")

    # Layer 4 — independent O(N) sweep.
    for k in range(los.shape[0]):
        assert_aabb_overlap_consistent(
            tri_fixture.primitives, los[k], his[k], out_t[k])


@parametrize_backends("bvh")
def test_bvh_aabb_overlap_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_aabb_overlap("bvh", backend_name, tri_fixture)


@parametrize_backends("kd")
def test_kd_aabb_overlap_matches_oracle_and_brute(tri_fixture, backend_name):
    _check_aabb_overlap("kd", backend_name, tri_fixture)


@parametrize_backends("octree")
def test_octree_aabb_overlap_matches_oracle_and_brute(tri_fixture,
                                                      backend_name):
    _check_aabb_overlap("octree", backend_name, tri_fixture)
