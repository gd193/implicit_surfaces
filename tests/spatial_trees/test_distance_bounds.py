"""Layer 3/4: distance_bounds query correctness.

Capability-gated test: only runs against backends whose
`has_distance_bounds` flag is True. For backends that expose the
capability, validates that the returned `(d_lo, d_hi)` are sound
lower/upper bounds on the closest-primitive distance from the query
AABB.

Reference (Layer 3): brute-force min over all primitive AABBs of the
analytic box-box `(d_lo, d_hi)` formula. The backend must not exceed
the brute upper bound, must not undercut the brute lower bound, and
must always satisfy `d_lo <= d_hi`.

No trimesh oracle — distance-bounds is not in the trimesh API.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfaces.explicit.accelerators.registry import get_backend

from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing,
                       DIST_RTOL, DIST_ATOL)


# Slightly looser tolerance: distance_bounds is two reductions composed
# (per-axis sqrt + min), so accumulated float32 error is a touch worse
# than a single closest-point distance.
BOUND_RTOL = 5e-4
BOUND_ATOL = 5e-5


def _skip_if_capability_missing(backend_name: str) -> None:
    """Skip the test if the backend does not advertise has_distance_bounds."""
    cls = get_backend(backend_name)
    if not getattr(cls, "has_distance_bounds", False):
        pytest.skip(f"{backend_name} does not implement distance_bounds")


def _backend_distance_bounds_batch(accel, los, his):
    return jax.vmap(lambda lo, hi: accel.distance_bounds(lo, hi))(los, his)


def _box_box_bounds(a_lo: np.ndarray, a_hi: np.ndarray,
                    b_lo: np.ndarray, b_hi: np.ndarray):
    """Reference Euclidean (d_lo, d_hi) between two AABBs.

    Mirrors the analytic formula used by the backend (see
    `bvh/backends/sah_jax.py::_box_box_dist_bounds`):
      d_lo = sqrt(sum max(0, max(a_lo - b_hi, b_lo - a_hi))^2)
      d_hi = sqrt(sum max((a_hi - b_lo)^2, (b_hi - a_lo)^2))
    """
    sep = np.maximum(0.0, np.maximum(a_lo - b_hi, b_lo - a_hi))
    span_sq = np.maximum((a_hi - b_lo) ** 2, (b_hi - a_lo) ** 2)
    return float(np.sqrt(np.sum(sep ** 2))), float(np.sqrt(np.sum(span_sq)))


def _brute_distance_bounds(primitives, q_lo: np.ndarray, q_hi: np.ndarray,
                           p_los: np.ndarray, p_his: np.ndarray):
    """Brute Layer-3 reference: min over primitives of box-box bounds.

    Numpy-vectorised over primitives. Caller passes pre-fetched
    primitive AABB arrays so the per-query reduction is a single
    array op rather than N traced JAX calls.
    """
    sep = np.maximum(0.0, np.maximum(q_lo - p_his, p_los - q_hi))
    span_sq = np.maximum((q_hi - p_los) ** 2, (p_his - q_lo) ** 2)
    d_lo = np.sqrt(np.sum(sep * sep, axis=-1))
    d_hi = np.sqrt(np.sum(span_sq, axis=-1))
    return float(d_lo.min()), float(d_hi.min())


def _fetch_primitive_aabbs(primitives):
    """Pull all primitive AABBs as numpy arrays in one vmap call."""
    idx = jnp.arange(primitives.n_primitives, dtype=jnp.int32)
    p_lo, p_hi = jax.vmap(primitives.primitive_aabb)(idx)
    return np.asarray(p_lo), np.asarray(p_hi)


# =====================================================================
# bvh — capability-gated.
# =====================================================================


@parametrize_backends("bvh")
def test_bvh_distance_bounds_matches_brute(tri_fixture, backend_name):
    _skip_if_capability_missing(backend_name)
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator("bvh", backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    d_lo_t, d_hi_t = _backend_distance_bounds_batch(accel, los, his)
    d_lo_t = np.asarray(d_lo_t)
    d_hi_t = np.asarray(d_hi_t)

    los_np = np.asarray(los)
    his_np = np.asarray(his)
    p_los, p_his = _fetch_primitive_aabbs(tri_fixture.primitives)

    d_lo_b = np.empty(los.shape[0], dtype=np.float32)
    d_hi_b = np.empty(los.shape[0], dtype=np.float32)
    for k in range(los.shape[0]):
        lo, hi = _brute_distance_bounds(tri_fixture.primitives,
                                        los_np[k], his_np[k],
                                        p_los, p_his)
        d_lo_b[k] = lo
        d_hi_b[k] = hi

    np.testing.assert_allclose(d_lo_t, d_lo_b,
                               rtol=BOUND_RTOL, atol=BOUND_ATOL,
                               err_msg="distance_bounds d_lo disagrees with brute")
    np.testing.assert_allclose(d_hi_t, d_hi_b,
                               rtol=BOUND_RTOL, atol=BOUND_ATOL,
                               err_msg="distance_bounds d_hi disagrees with brute")


@parametrize_backends("bvh")
def test_bvh_distance_bounds_invariants(tri_fixture, backend_name):
    """Layer 4 self-consistency: d_lo <= d_hi, both finite, both >= 0."""
    _skip_if_capability_missing(backend_name)
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator("bvh", backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    d_lo, d_hi = _backend_distance_bounds_batch(accel, los, his)
    d_lo = np.asarray(d_lo)
    d_hi = np.asarray(d_hi)

    assert np.all(np.isfinite(d_lo)), "d_lo has non-finite entries"
    assert np.all(np.isfinite(d_hi)), "d_hi has non-finite entries"
    assert np.all(d_lo >= 0.0), "d_lo has negative entries"
    assert np.all(d_hi >= 0.0), "d_hi has negative entries"
    # Allow a single-ulp slack on the inequality (float32 rounding can
    # equalise within ~1 ulp at large magnitudes).
    assert np.all(d_lo <= d_hi + 1e-5), \
        f"d_lo > d_hi: max excess = {(d_lo - d_hi).max()}"


# =====================================================================
# kd / octree — defensively parametrize so the tests adapt when a
# distance-bounds-capable backend lands in either flavor.
# =====================================================================


@parametrize_backends("kd")
def test_kd_distance_bounds_matches_brute(tri_fixture, backend_name):
    _skip_if_capability_missing(backend_name)
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator("kd", backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi
    d_lo_t, d_hi_t = _backend_distance_bounds_batch(accel, los, his)

    los_np = np.asarray(los)
    his_np = np.asarray(his)
    p_los, p_his = _fetch_primitive_aabbs(tri_fixture.primitives)
    d_lo_b = np.empty(los.shape[0], dtype=np.float32)
    d_hi_b = np.empty(los.shape[0], dtype=np.float32)
    for k in range(los.shape[0]):
        lo, hi = _brute_distance_bounds(tri_fixture.primitives,
                                        los_np[k], his_np[k],
                                        p_los, p_his)
        d_lo_b[k] = lo
        d_hi_b[k] = hi

    np.testing.assert_allclose(np.asarray(d_lo_t), d_lo_b,
                               rtol=BOUND_RTOL, atol=BOUND_ATOL)
    np.testing.assert_allclose(np.asarray(d_hi_t), d_hi_b,
                               rtol=BOUND_RTOL, atol=BOUND_ATOL)


@parametrize_backends("octree")
def test_octree_distance_bounds_matches_brute(tri_fixture, backend_name):
    _skip_if_capability_missing(backend_name)
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator("octree", backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi
    d_lo_t, d_hi_t = _backend_distance_bounds_batch(accel, los, his)

    los_np = np.asarray(los)
    his_np = np.asarray(his)
    p_los, p_his = _fetch_primitive_aabbs(tri_fixture.primitives)
    d_lo_b = np.empty(los.shape[0], dtype=np.float32)
    d_hi_b = np.empty(los.shape[0], dtype=np.float32)
    for k in range(los.shape[0]):
        lo, hi = _brute_distance_bounds(tri_fixture.primitives,
                                        los_np[k], his_np[k],
                                        p_los, p_his)
        d_lo_b[k] = lo
        d_hi_b[k] = hi

    np.testing.assert_allclose(np.asarray(d_lo_t), d_lo_b,
                               rtol=BOUND_RTOL, atol=BOUND_ATOL)
    np.testing.assert_allclose(np.asarray(d_hi_t), d_hi_b,
                               rtol=BOUND_RTOL, atol=BOUND_ATOL)


@parametrize_backends("octree")
def test_octree_distance_bounds_invariants(tri_fixture, backend_name):
    """Layer 4 self-consistency: d_lo <= d_hi, both finite, both >= 0."""
    _skip_if_capability_missing(backend_name)
    skip_if_primitive_math_missing("primitive_aabb", tri_fixture.primitives)

    accel = build_accelerator("octree", backend_name, tri_fixture.primitives)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    d_lo, d_hi = _backend_distance_bounds_batch(accel, los, his)
    d_lo = np.asarray(d_lo)
    d_hi = np.asarray(d_hi)

    assert np.all(np.isfinite(d_lo)), "d_lo has non-finite entries"
    assert np.all(np.isfinite(d_hi)), "d_hi has non-finite entries"
    assert np.all(d_lo >= 0.0), "d_lo has negative entries"
    assert np.all(d_hi >= 0.0), "d_hi has negative entries"
    assert np.all(d_lo <= d_hi + 1e-5), \
        f"d_lo > d_hi: max excess = {(d_lo - d_hi).max()}"
