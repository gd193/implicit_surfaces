"""Batch result consistency, independent of tree traversal."""
import numpy as np

from ._warp_helpers import _closest_triangle


def _assert_on_triangles(primitives, points, indices):
    vertices = np.asarray(primitives.mesh.vertices, np.float64)
    triangles = vertices[np.asarray(primitives.mesh.faces)[np.asarray(indices)]]
    for point, triangle in zip(np.asarray(points, np.float64), triangles):
        np.testing.assert_allclose(_closest_triangle(point, triangle), point,
                                   rtol=0, atol=1e-4)


def assert_closest_point_consistent(primitives, x, cp, distance, index):
    assert np.all((np.asarray(index) >= 0) & (np.asarray(index) < primitives.n_primitives))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(cp)-np.asarray(x), axis=-1), distance, rtol=1e-4, atol=1e-5)
    _assert_on_triangles(primitives, cp, index)


def assert_first_hit_consistent(primitives, origin, direction, max_t, t, index, hit):
    hit = np.asarray(hit)
    if not hit.any():
        return
    times = np.asarray(t)[hit]
    assert np.all((times >= 0) & (times <= np.asarray(max_t)[hit]+1e-5))
    assert np.all((np.asarray(index)[hit] >= 0) & (np.asarray(index)[hit] < primitives.n_primitives))
    points = np.asarray(origin)[hit]+times[:, None]*np.asarray(direction)[hit]
    _assert_on_triangles(primitives, points, np.asarray(index)[hit])
