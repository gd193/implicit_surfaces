"""Silhouette distances and result invariants against exhaustive edges."""
import jax
import numpy as np

from ._harness import parametrize_backends, build_accelerator
from ._warp_helpers import _oracle_silhouette


@parametrize_backends(capability="has_silhouette_query")
def test_closest_silhouette_matches_oracle_and_invariants(tri_fixture, backend_name):
    accel = build_accelerator(backend_name.split('.')[0], backend_name, tri_fixture.primitives)
    points = tri_fixture.query_points
    cp, distance, edge = map(np.asarray, jax.vmap(accel.closest_silhouette)(points))
    _, expected = _oracle_silhouette(tri_fixture.tri_mesh.vertices,
                                     tri_fixture.tri_mesh.faces, points)
    np.testing.assert_allclose(distance, expected, rtol=5e-4, atol=5e-5)
    found = np.isfinite(expected)
    assert found.any()
    assert np.all(edge[found] >= 0) and np.all(edge[~found] == -1)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(points)[found]-cp[found], axis=-1),
                               distance[found], rtol=5e-4, atol=5e-5)
