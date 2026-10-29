"""Aabb overlap against external and exhaustive references."""
from __future__ import annotations

import jax
import numpy as np

from ._brute import brute_aabb_overlap_batch
from ._oracle import oracle_aabb_overlap
from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing)


def _backend_aabb_overlap_batch(accel, los, his):
    return jax.vmap(lambda lo, hi: accel.aabb_overlap(lo, hi))(los, his)


@parametrize_backends()
def test_aabb_overlap_matches_oracle_and_brute(tri_fixture, backend_name):
    flavor = backend_name.split(".")[0]
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    out_t = _backend_aabb_overlap_batch(accel, los, his)

    out_b = brute_aabb_overlap_batch(tri_fixture.primitives, los, his)
    np.testing.assert_array_equal(np.asarray(out_t), np.asarray(out_b),
                                  err_msg="aabb_overlap disagrees with brute")

    out_o = oracle_aabb_overlap(tri_fixture.tri_mesh,
                                np.asarray(los), np.asarray(his))
    np.testing.assert_array_equal(np.asarray(out_t), out_o,
                                  err_msg="aabb_overlap disagrees with trimesh")
