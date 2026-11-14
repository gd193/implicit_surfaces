"""Primitive AABB overlap against an independent NumPy reference."""
from __future__ import annotations

import jax
import numpy as np

from ._oracle import oracle_aabb_overlap
from ._harness import parametrize_backends, build_accelerator


def _backend_aabb_overlap_batch(accel, los, his):
    return jax.vmap(lambda lo, hi: accel.aabb_overlap(lo, hi))(los, his)


@parametrize_backends()
def test_aabb_overlap_matches_numpy_oracle(tri_fixture, backend_name):
    flavor = backend_name.split(".")[0]

    accel = build_accelerator(flavor, backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    out_t = _backend_aabb_overlap_batch(accel, los, his)

    out_o = oracle_aabb_overlap(tri_fixture.tri_mesh,
                                np.asarray(los), np.asarray(his))
    np.testing.assert_array_equal(np.asarray(out_t), out_o,
                                  err_msg="aabb_overlap disagrees with trimesh")
