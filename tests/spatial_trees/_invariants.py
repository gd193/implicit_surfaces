"""Batch result consistency, independent of tree traversal."""
import jax
import numpy as np


def assert_closest_point_consistent(primitives, x, cp, distance, index):
    assert np.all((np.asarray(index) >= 0) & (np.asarray(index) < primitives.n_primitives))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(cp)-np.asarray(x), axis=-1), distance, rtol=1e-4, atol=1e-5)
    _, residual = jax.vmap(primitives.closest_point_on_primitive)(cp, index)
    assert np.all(np.asarray(residual) <= 1e-4)


def assert_first_hit_consistent(primitives, origin, direction, max_t, t, index, hit):
    hit = np.asarray(hit)
    if not hit.any():
        return
    times = np.asarray(t)[hit]
    assert np.all((times >= 0) & (times <= np.asarray(max_t)[hit]+1e-5))
    assert np.all((np.asarray(index)[hit] >= 0) & (np.asarray(index)[hit] < primitives.n_primitives))
    points = np.asarray(origin)[hit]+times[:, None]*np.asarray(direction)[hit]
    _, residual = jax.vmap(primitives.closest_point_on_primitive)(points, np.asarray(index)[hit])
    assert np.all(np.asarray(residual) <= 1e-4)
