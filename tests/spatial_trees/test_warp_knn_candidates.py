"""Candidate-storage boundaries and metric conversion failure semantics."""
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
wp = pytest.importorskip("warp")

from sengi.bvh.warp import Euclidean, Points, build_tree, make_knn  # noqa: E402
from sengi.protocol import Config  # noqa: E402


@wp.func
def signed_zero_score(query: wp.vec3, item: Any):
    return item.a, wp.where(item.a[0] < 0.0, -0.0, 0.0)


@wp.func
def zero_bound(query: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    return 0.0


@pytest.mark.parametrize("k", [16, 64, 65, 66, 67, 68, 128, 256])
@pytest.mark.parametrize("cache_bounds", [False, True])
@pytest.mark.parametrize("return_points", [False, True])
def test_signed_zero_ties_across_candidate_storage_sizes(k, cache_bounds, return_points):
    primitive = Points()
    rng = np.random.default_rng(248)
    points = rng.normal(size=(97, 3)).astype(np.float32)
    tree, data = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=1))
    query = make_knn(primitive, k, evaluate=signed_zero_score, lower_bound=zero_bound,
                     cache_bounds=cache_bounds, return_points=return_points)
    scores, neighbors, counts, status = map(np.asarray, jax.jit(query)(tree, data, jnp.zeros((2, 3)), jnp.array([0., jnp.inf])))
    count = min(k, len(points))
    if return_points:
        np.testing.assert_array_equal(neighbors[:, :count], np.broadcast_to(points[:count], (2, count, 3)))
        np.testing.assert_array_equal(neighbors[:, count:], 0.)
    else:
        np.testing.assert_array_equal(neighbors[:, :count], np.tile(np.arange(count), (2, 1)))
        np.testing.assert_array_equal(neighbors[:, count:], -1)
    np.testing.assert_array_equal(scores[:, :count], 0.)
    assert np.isposinf(scores[:, count:]).all()
    np.testing.assert_array_equal(counts, count)
    np.testing.assert_array_equal(status, 1)


@wp.func
def invalid_second_distance(score: float):
    return wp.where(score == 1.0, wp.nan, wp.sqrt(score))


@dataclass(frozen=True)
class InvalidConversion:
    def bind(self, primitive):
        return replace(Euclidean().bind(primitive), to_distance=invalid_second_distance)


@pytest.mark.parametrize("k", [3, 65])
def test_conversion_failure_clears_previously_written_results(k):
    primitive = Points()
    points = np.zeros((97, 3), np.float32)
    points[:, 0] = np.arange(len(points))
    tree, data = build_tree(primitive, primitive.data(points))
    query = make_knn(primitive, k, InvalidConversion())
    scores, ids, count, status = jax.jit(query)(tree, data, jnp.zeros(3))
    assert np.isposinf(scores).all()
    np.testing.assert_array_equal(ids, -1)
    assert int(count) == 0 and int(status) == -1
