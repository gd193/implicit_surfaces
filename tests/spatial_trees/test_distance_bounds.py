"""Distance bounds against an independent box-distance reduction."""
import jax
import numpy as np

from ._harness import parametrize_backends, build_accelerator


@parametrize_backends(capability="has_distance_bounds")
def test_distance_bounds_matches_brute_and_invariants(tri_fixture, backend_name):
    accel = build_accelerator(backend_name.split('.')[0], backend_name, tri_fixture.primitives)
    los, his = tri_fixture.query_aabb_lo, tri_fixture.query_aabb_hi
    lower, upper = map(np.asarray, jax.vmap(accel.distance_bounds)(los, his))
    triangles = np.asarray(tri_fixture.tri_mesh.triangles)
    p_lo, p_hi = triangles.min(axis=1), triangles.max(axis=1)
    q_lo, q_hi = np.asarray(los)[:, None], np.asarray(his)[:, None]
    separation = np.maximum(0., np.maximum(q_lo-p_hi, p_lo-q_hi))
    span2 = np.maximum((q_hi-p_lo)**2, (p_hi-q_lo)**2)
    expected = [np.linalg.norm(separation, axis=-1).min(axis=1),
                np.sqrt(span2.sum(axis=-1)).min(axis=1)]
    np.testing.assert_allclose([lower, upper], expected, rtol=5e-4, atol=5e-5)
    assert np.isfinite([lower, upper]).all()
    assert np.all(lower >= 0) and np.all(lower <= upper+1e-5)
