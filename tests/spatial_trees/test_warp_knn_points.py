"""Optional closest-point output stays ordered, padded and fused."""
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from sengi.bvh.warp import (  # noqa: E402
    Euclidean, NearestCallbacks, Points, Segments, Triangles, build_tree,
    make_knn, make_nearest,
)
from sengi.protocol import Config  # noqa: E402
from tests.spatial_trees.test_warp_factories import (  # noqa: E402
    Balls, BallArrays, ball_evaluate, ball_lower_bound, weighted_callbacks,
)
from tests.spatial_trees.test_warp_knn import reference  # noqa: E402


def check_points(result, expected, coordinates):
    np.testing.assert_allclose(result[0], expected[0], rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(result[1], coordinates, rtol=5e-6, atol=2e-6)
    for actual, wanted in zip(result[2:], expected[2:], strict=True):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("k", [1, 4, 16, 17, 64, 65, 128])
def test_points_order_slots_caps_and_storage_paths(dimension, k):
    rng = np.random.default_rng(614)
    points = rng.integers(-8, 9, size=(97, dimension)).astype(np.float32)
    queries = rng.integers(-4, 5, size=(2, 3, dimension)).astype(np.float32)
    weights = (0.5, 2., 4.)[:dimension]
    evaluate, bound = weighted_callbacks(dimension, weights)
    primitive = Points(dimension)
    tree, data = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=1))
    assert not np.array_equal(tree.prim_ids, np.arange(len(points)))
    caps = np.array([[np.inf], [8.]], np.float32)
    scores = np.sum((queries[..., None, :] - points)**2 * weights, axis=-1)
    expected = reference(scores, k, caps)
    coordinates = np.where((expected[1] >= 0)[..., None], points[np.maximum(expected[1], 0)], 0.)
    call = make_knn(primitive, k, evaluate=evaluate, lower_bound=bound, return_points=True)
    assert call is make_knn(primitive, k, NearestCallbacks(evaluate, bound), return_points=True)
    ids_call = make_knn(primitive, k, evaluate=evaluate, lower_bound=bound)
    assert ids_call is make_knn(primitive, k, evaluate=evaluate, lower_bound=bound, return_points=False)
    assert call.kernel is not ids_call.kernel
    check_points(jax.jit(call)(tree, data, queries, caps), expected, coordinates)
    mapped = jax.jit(jax.vmap(jax.vmap(call, in_axes=(None, None, 0, 0)), in_axes=(None, None, 0, 0)))
    check_points(mapped(tree, data, queries, np.broadcast_to(caps, queries.shape[:-1])), expected, coordinates)
    hlo = jax.jit(call).lower(tree, data, queries, caps).as_text()
    assert hlo.count("stablehlo.custom_call") == 1
    assert "stablehlo.sort" not in hlo and "stablehlo.while" not in hlo
    if k == 1:
        nearest = make_nearest(primitive, evaluate=evaluate, lower_bound=bound)(tree, data, queries, caps)
        np.testing.assert_array_equal(call(tree, data, queries, caps)[1][..., 0, :], nearest[0])


@pytest.mark.parametrize("cached,packed,grouped", list(product((False, True), repeat=3)))
@pytest.mark.parametrize("k", [4, 65])
def test_extended_primitive_coordinates_and_independent_options(cached, packed, grouped, k):
    x = np.array([3., -2., 1., 0., 5.], np.float32)
    a = np.column_stack((x, np.zeros((5, 2), np.float32)))
    primitive = Segments()
    tree, data = build_tree(primitive, primitive.data(a, a + [0, 1, 0]), Config(leaf_size_threshold=1))
    queries = np.array([[.2, .25, .25], [-1., 1.5, .2]], np.float32)
    projected = np.broadcast_to(a, (2, 5, 3)).copy()
    projected[:, :, 1] = np.clip(queries[:, 1], 0., 1.)[:, None]
    expected = reference(np.linalg.norm(queries[:, None] - projected, axis=-1), k)
    coordinates = np.take_along_axis(projected, np.maximum(expected[1], 0)[..., None], axis=1)
    coordinates = np.where((expected[1] >= 0)[..., None], coordinates, 0.)
    call = make_knn(primitive, k, Euclidean(), return_points=True, cache_bounds=cached,
                    packed_nodes=packed, group_leaves=grouped)
    check_points(jax.jit(call)(tree, data, queries), expected, coordinates)


def test_triangle_coordinates_are_projections():
    primitive = Triangles()
    x = np.array([3., -2., 1., 0., 5.], np.float32)
    a = np.column_stack((x, np.zeros((5, 2), np.float32)))
    facets = np.stack((a, a + [0, 1, 0], a + [0, 0, 1]), axis=1)
    tree, data = build_tree(primitive, primitive.indexed(facets.reshape(-1, 3), np.arange(15).reshape(5, 3)),
                            Config(leaf_size_threshold=1))
    queries = np.array([[.2, .25, .25], [-1., .1, .2]], np.float32)
    expected = reference(np.abs(queries[:, None, 0] - x), 4)
    coordinates = np.broadcast_to(queries[:, None], (2, 4, 3)).copy()
    coordinates[:, :, 0] = x[expected[1]]
    check_points(jax.jit(make_knn(primitive, 4, Euclidean(), return_points=True))(tree, data, queries),
                 expected, coordinates)


def test_custom_ball_callback_and_predicate():
    primitive = Balls()
    centers = np.array([[4., 0, 0], [0., 0, 0], [-2., 0, 0], [1., 0, 0]], np.float32)
    radii = np.array([1., .5, .25, .5], np.float32)
    tree, data = build_tree(primitive, BallArrays(jnp.asarray(centers), jnp.asarray(radii)),
                            Config(leaf_size_threshold=1))

    @wp.func
    def accepts(q: wp.vec3, item: Any):
        return item.center[0] >= 0.0

    queries = np.array([[0., 0, 0], [3., 1., 0]], np.float32)
    delta = queries[:, None] - centers
    lengths = np.linalg.norm(delta, axis=-1)
    projected = centers + delta * np.minimum(1., radii / np.maximum(lengths, 1.e-10))[..., None]
    scores = np.maximum(lengths - radii, 0.)**2
    scores[:, 2] = np.inf
    expected = reference(scores, 5, 4.)
    coordinates = np.take_along_axis(projected, np.maximum(expected[1], 0)[..., None], axis=1)
    coordinates = np.where((expected[1] >= 0)[..., None], coordinates, 0.)
    call = make_knn(primitive, 5, evaluate=ball_evaluate, lower_bound=ball_lower_bound,
                    predicate=accepts, return_points=True)
    check_points(jax.jit(call)(tree, data, queries, 4.), expected, coordinates)


def test_empty_invalid_and_padding():
    primitive = Points(2)
    tree, data = build_tree(primitive, primitive.data([[2., 0.]]))
    call = make_knn(primitive, 3, Euclidean(), return_points=True)
    queries = np.array([[0., 0], [np.nan, 0], [0, 0], [0, 0]], np.float32)
    values, points, count, status = jax.jit(call)(tree, data, queries, np.array([np.inf, 1., 1., -1.]))
    np.testing.assert_array_equal(count, [1, 0, 0, 0])
    np.testing.assert_array_equal(status, [1, -1, 0, -1])
    np.testing.assert_array_equal(points[0], [[2., 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(points[1:], 0.)
    assert np.isposinf(np.asarray(values)[1:]).all()
    empty_tree, empty = build_tree(primitive, primitive.data(np.empty((0, 2))))
    result = call(empty_tree, empty, jnp.zeros(2))
    assert [v.shape for v in result] == [(3,), (3, 2), (), ()]
    np.testing.assert_array_equal(result[1], 0.)
    assert int(result[3]) == 0
    assert [v.shape for v in call(tree, data, jnp.empty((2, 0, 2)))] == [(2, 0, 3), (2, 0, 3, 2), (2, 0), (2, 0)]


@pytest.mark.parametrize("k", [4, 65])
def test_nonfinite_selected_coordinate_clears_row(k):
    @wp.func
    def evaluate(q: wp.vec3, item: Any):
        point = item.a
        if point[0] > 0.0:
            point[1] = wp.nan
        return point, wp.length_sq(q - item.a)

    @wp.func
    def bound(q: wp.vec3, lo: wp.vec3, hi: wp.vec3):
        return 0.0

    primitive = Points(3)
    tree, data = build_tree(primitive, primitive.data([[-1., 0, 0], [1., 0, 0]]))
    call = make_knn(primitive, k, evaluate=evaluate, lower_bound=bound, return_points=True)
    values, points, count, status = jax.jit(call)(tree, data, jnp.zeros(3))
    assert np.isposinf(values).all()
    np.testing.assert_array_equal(points, 0.)
    assert int(count) == 0 and int(status) == -1


@pytest.mark.parametrize("value", [1, "points", None])
def test_return_points_is_static_bool(value):
    with pytest.raises(TypeError, match="static Python bool"):
        make_knn(Points(3), 4, Euclidean(), return_points=value)
