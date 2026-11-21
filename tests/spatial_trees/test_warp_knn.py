"""Fused generic kNN contracts against independent NumPy references."""
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from sengi.bvh.warp import (  # noqa: E402
    Euclidean, NearestCallbacks, Points, Segments, Triangles, build_tree,
    make_knn, make_nearest, make_radius,
)
from sengi.protocol import Config  # noqa: E402
from tests.spatial_trees._warp_helpers import _closest_triangle  # noqa: E402
from tests.spatial_trees.test_warp_factories import (  # noqa: E402
    Balls, BallArrays, ball_evaluate, ball_lower_bound, weighted_callbacks,
)


def reference(scores, k, cap=np.inf):
    """Stable sorting preserves original IDs, independent of BVH packing."""
    scores = np.asarray(scores)
    shape = scores.shape[:-1]
    caps = np.broadcast_to(cap, shape)
    values = np.full(shape+(k,), np.inf, np.float32)
    ids = np.full(shape+(k,), -1, np.int32)
    count = np.zeros(shape, np.int32)
    for row in np.ndindex(shape):
        selected = np.argsort(scores[row], kind="stable")
        selected = selected[scores[row][selected] <= caps[row]][:k]
        count[row] = len(selected)
        values[row][:len(selected)] = scores[row][selected]
        ids[row][:len(selected)] = selected
    return values, ids, count, (count > 0).astype(np.int32)


def assert_result(result, expected):
    np.testing.assert_allclose(result[0], expected[0], rtol=5e-6, atol=2e-6)
    for actual, wanted in zip(result[1:], expected[1:], strict=True):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("cache_bounds", [False, True])
def test_weighted_points_numpy_jit_vmap_and_policy_identity(dimension, cache_bounds):
    rng = np.random.default_rng(392)
    points = rng.normal(size=(73, dimension)).astype(np.float32)
    queries = rng.normal(size=(2, 5, dimension)).astype(np.float32)
    weights = np.array([.01, 16., .5][:dimension], np.float32)
    evaluate, bound = weighted_callbacks(dimension, tuple(map(float, weights)))
    primitive = Points(dimension)
    tree, data = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=1))
    query = make_knn(primitive, 7, evaluate=evaluate, lower_bound=bound, cache_bounds=cache_bounds)
    policy = make_knn(primitive, 7, NearestCallbacks(evaluate, bound), cache_bounds=cache_bounds)
    assert query.kernel is policy.kernel
    assert query.kernel is make_knn(primitive, 7, evaluate=evaluate, lower_bound=bound, cache_bounds=cache_bounds).kernel
    other, _ = weighted_callbacks(dimension, tuple(map(float, weights[::-1])))
    assert query.kernel is not make_knn(primitive, 7, evaluate=other, lower_bound=bound, cache_bounds=cache_bounds).kernel
    scores = np.sum((queries[..., None, :]-points)**2*weights, axis=-1)
    caps = np.array([[.1], [10.]], np.float32)
    assert_result(jax.jit(query)(tree, data, queries, caps), reference(scores, 7, caps))
    mapped = jax.jit(jax.vmap(jax.vmap(query, in_axes=(None, None, 0)), in_axes=(None, None, 0)))
    assert_result(mapped(tree, data, queries), reference(scores, 7))
    hlo = jax.jit(query).lower(tree, data, queries).as_text()
    assert hlo.count("stablehlo.custom_call") == 1
    assert "stablehlo.sort" not in hlo and "stablehlo.while" not in hlo


@pytest.mark.parametrize("k", [1, 3, 9, 65])
def test_original_id_ties_caps_padding_and_nearest_equivalence(k):
    primitive = Points(2)
    points = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [3, 0]], np.float32)
    tree, data = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=1))
    query = make_knn(primitive, k, Euclidean())
    queries = np.zeros((3, 2), np.float32)
    caps = np.array([np.nextafter(np.float32(1), np.float32(0)), 1., np.inf], np.float32)
    expected = reference(np.broadcast_to(np.linalg.norm(points, axis=-1), (3, 5)), k, caps)
    got = jax.jit(query)(tree, data, queries, caps)
    assert_result(got, expected)
    if k == 1:
        nearest = make_nearest(primitive, Euclidean())(tree, data, queries, caps)
        np.testing.assert_array_equal(got[0][..., 0], nearest[1])
        np.testing.assert_array_equal(got[1][..., 0], nearest[2])
        np.testing.assert_array_equal(got[3], nearest[3])


@pytest.mark.parametrize("kind", ["segments", "triangles"])
def test_extended_primitives_numpy_reference(kind):
    # Parallel segments and right triangles permit direct independent projections.
    x = np.array([3., -2., 1., 0., 5.], np.float32)
    queries = np.array([[.2, .25, .25], [-1., .1, .2], [4., .3, .1]], np.float32)
    a = np.column_stack((x, np.zeros((5, 2), np.float32)))
    b = a + np.array([0, 1, 0], np.float32)
    if kind == "segments":
        primitive = Segments()
        raw = primitive.data(a, b)
        scores = np.sqrt((queries[:, None, 0]-x)**2 + queries[:, None, 2]**2)
    else:
        primitive = Triangles()
        c = a + np.array([0, 0, 1], np.float32)
        raw = primitive.indexed(np.stack((a, b, c), axis=1).reshape(-1, 3), np.arange(15).reshape(5, 3))
        # All projections lie strictly inside the right triangles.
        scores = np.abs(queries[:, None, 0]-x)
    tree, data = build_tree(primitive, raw, Config(leaf_size_threshold=1))
    assert_result(jax.jit(make_knn(primitive, 4, Euclidean()))(tree, data, queries), reference(scores, 4))


def test_custom_solid_balls_raw_scores_and_predicate():
    primitive = Balls()
    centers = np.array([[4., 0, 0], [0, 0, 0], [-2, 0, 0], [1, 0, 0]], np.float32)
    radii = np.array([1., .5, .25, .5], np.float32)
    queries = np.array([[0., 0, 0], [3., 0, 0]], np.float32)
    tree, data = build_tree(primitive, BallArrays(jnp.asarray(centers), jnp.asarray(radii)))

    @wp.func
    def accepts(q: wp.vec3, item: Any):
        return item.center[0] >= 0.0

    scores = np.maximum(np.linalg.norm(queries[:, None]-centers, axis=-1)-radii, 0)**2
    scores[:, 2] = np.inf
    query = make_knn(primitive, 5, evaluate=ball_evaluate, lower_bound=ball_lower_bound, predicate=accepts)
    assert_result(jax.jit(query)(tree, data, queries, 4.), reference(scores, 5, 4.))


def test_empty_invalid_queries_caps_and_shapes():
    primitive = Points(2)
    tree, data = build_tree(primitive, primitive.data([[0., 0.]]))
    query = make_knn(primitive, 3, Euclidean())
    queries = np.array([[0., 0], [np.nan, 0], [np.inf, 0], [0, 0], [0, 0]], np.float32)
    result = jax.jit(query)(tree, data, queries, np.array([np.inf, 1., 1., -1., np.nan]))
    np.testing.assert_array_equal(result[3], [1, -1, -1, -1, -1])
    np.testing.assert_array_equal(result[2], [1, 0, 0, 0, 0])
    assert np.isposinf(np.asarray(result[0])[1:]).all()
    np.testing.assert_array_equal(result[1][1:], -1)
    empty_tree, empty = build_tree(primitive, primitive.data(np.empty((0, 2))))
    assert_result(query(empty_tree, empty, jnp.zeros(2)), reference(np.empty((0,)), 3))
    empty_batch = query(tree, data, jnp.empty((2, 0, 2)))
    assert [v.shape for v in empty_batch] == [(2, 0, 3), (2, 0, 3), (2, 0), (2, 0)]
    with pytest.raises(ValueError, match="trailing shape"):
        query(tree, data, jnp.zeros(3))


@pytest.mark.parametrize("bad_score", [-1., np.inf, np.nan])
def test_bad_evaluated_score_clears_entire_row(bad_score):
    @wp.func(module="unique")
    def evaluate(q: wp.vec3, item: Any):
        return item.a, wp.static(float(bad_score))

    @wp.func
    def bound(q: wp.vec3, lo: wp.vec3, hi: wp.vec3):
        return 0.0

    primitive = Points()
    tree, data = build_tree(primitive, primitive.data([[0., 0, 0], [1., 0, 0]]))
    result = make_knn(primitive, 3, evaluate=evaluate, lower_bound=bound)(tree, data, jnp.zeros(3))
    assert int(result[3]) == -1 and int(result[2]) == 0
    assert np.isposinf(result[0]).all()
    np.testing.assert_array_equal(result[1], -1)


def test_stack_failure_clears_row():
    primitive = Points()
    points = np.random.default_rng(4).normal(size=(83, 3))
    tree, data = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=1))
    result = make_knn(primitive, 83, Euclidean(), stack_capacity=1)(tree, data, jnp.zeros(3))
    assert int(result[3]) == -1 and int(result[2]) == 0
    assert np.isposinf(result[0]).all()
    np.testing.assert_array_equal(result[1], -1)


@pytest.mark.parametrize("k", [True, False, 0, -1, 1.5, np.int32(2)])
def test_k_is_positive_static_python_int(k):
    with pytest.raises((TypeError, ValueError)):
        make_knn(Points(), k, Euclidean())


def test_explicit_metric_or_complete_callbacks_required():
    evaluate, bound = weighted_callbacks(2, (1., 4.))
    for kwargs in ({}, {"evaluate": evaluate}, {"lower_bound": bound},
                   {"metric": Euclidean(), "evaluate": evaluate, "lower_bound": bound},
                   {"metric": Euclidean(), "evaluate": evaluate},
                   {"metric": Euclidean(), "lower_bound": bound}):
        with pytest.raises((TypeError, ValueError)):
            make_knn(Points(2), 2, **kwargs)


def test_raw_score_k1_matches_nearest_and_closed_cap():
    primitive = Points(2)
    evaluate, bound = weighted_callbacks(2, (1., 4.))
    tree, data = build_tree(primitive, primitive.data([[0., 1.], [0., -1.], [5., 0.]]), Config(leaf_size_threshold=1))
    queries = jnp.zeros((3, 2))
    caps = jnp.array([2., 4., jnp.inf])
    result = make_knn(primitive, 1, evaluate=evaluate, lower_bound=bound)(tree, data, queries, caps)
    nearest = make_nearest(primitive, evaluate=evaluate, lower_bound=bound)(tree, data, queries, caps)
    np.testing.assert_array_equal(result[0][..., 0], nearest[1])
    np.testing.assert_array_equal(result[1][..., 0], nearest[2])
    np.testing.assert_array_equal(result[3], nearest[3])
    np.testing.assert_array_equal(result[2], [0, 1, 1])
    # The cap is a raw score: 2 excludes score 4, while 4 includes both ties.
    result = make_knn(primitive, 3, evaluate=evaluate, lower_bound=bound)(tree, data, jnp.zeros(2), 4.)
    assert_result(result, (np.array([4., 4., np.inf]), np.array([0, 1, -1]), 2, 1))


def test_euclidean_numerical_overflow_matches_nearest_validity():
    primitive = Points()
    tree, data = build_tree(primitive, primitive.data([[0., 0, 0]]))
    query = make_knn(primitive, 2, Euclidean())
    for cap, status in [(np.inf, -1), (.5, 0)]:
        result = query(tree, data, jnp.array([1.e30, 0, 0]), cap)
        assert int(result[3]) == status and int(result[2]) == 0
        assert np.isposinf(result[0]).all()
        np.testing.assert_array_equal(result[1], -1)


@pytest.mark.parametrize("return_points", [False, True])
def test_custom_scalar_query_policy(return_points):
    from dataclasses import dataclass, replace

    @dataclass(frozen=True)
    class AxisPolicy:
        def bind(self, primitive):
            base = Euclidean().bind(primitive)

            @wp.func
            def valid(query: float):
                return wp.isfinite(query)

            @wp.func
            def bound(query: float, lo: wp.vec3, hi: wp.vec3):
                return wp.static(base.node_bound)(wp.vec3(query, 0.0, 0.0), lo, hi)

            @wp.func
            def evaluate(query: float, item: Any):
                delta = item.a-wp.vec3(query, 0.0, 0.0)
                return item.a, wp.dot(delta, delta)

            @wp.func
            def cull(query: float, score: float):
                return score

            return replace(base, query_type=wp.float32, query_shape=(), valid=valid,
                           node_bound=bound, closest=evaluate, cull_limit=cull)

    primitive = Points()
    points = np.array([[4., 0, 0], [0, 0, 0], [-1, 0, 0]], np.float32)
    tree, data = build_tree(primitive, primitive.data(points))
    query = make_knn(primitive, 4, AxisPolicy(), return_points=return_points)
    queries = np.array([[0., 1., 3.], [-1., -2., 5.]], np.float32)
    scores = np.abs(queries[..., None]-points[:, 0])
    expected = reference(scores, 4)
    if return_points:
        coordinates = np.where((expected[1] >= 0)[..., None], points[np.maximum(expected[1], 0)], 0.)
        expected = (expected[0], coordinates, *expected[2:])
    assert_result(jax.jit(query)(tree, data, queries), expected)


@pytest.mark.parametrize("cache_bounds", [False, True])
def test_thin_triangle_capped_queries(cache_bounds):
    # The rounded vertices describe a thin, rotated triangle. Computing its
    # normal from two nearly parallel edges can admit a point outside the face.
    thin = np.array([
        [-2.09561824798584, -.8604152798652649, -.8177284002304077],
        [-.9771557450294495, -1.4173413515090942, .48430976271629333],
        [-1.4997109174728394, -1.1571329832077026, -.12399935722351074],
    ], np.float32)
    p = np.array([-1.710182785987854, -1.0626181364059448, -.3567012846469879],
                 np.float32)
    rival = np.broadcast_to(p + np.array([0., 0., .01045], np.float32), (3, 3))
    shift = np.array([8., -4., 16.], np.float32)
    contact_tri = np.array([[8., 0., 0.], [10., 1., 0.], [8., 1., 0.]], np.float32)
    # The first corner is opposite the longest edge. A plane equation
    # referenced to either endpoint can leave a residual at this corner.
    rotated_contact = np.array([
        [-.4503190517425537, -.28977879881858826, -1.2234320640563965],
        [1.5400711297988892, -.67762690782547, -.2773061990737915],
        [-1.630417823791504, .13754382729530334, -1.247592568397522],
    ], np.float32)
    triangles = np.stack([thin, rival, thin + shift, contact_tri, rotated_contact])
    near_contact = rotated_contact[0].copy()
    near_contact[2] = np.nextafter(near_contact[2], np.float32(np.inf))
    queries = np.array([
        p, p, p + shift, thin[0], rotated_contact[0], [8.5, .5, 0.], [9., .5, 0.],
        near_contact, p,
    ], np.float32)
    caps = np.array([np.inf, .01048, np.inf, 0., 0., 0., 0., 0., .0104], np.float32)
    points = np.array([
        [_closest_triangle(q.astype(np.float64), tri.astype(np.float64))
         for tri in triangles]
        for q in queries
    ])
    scores = np.linalg.norm(points - queries[:, None, :], axis=-1)
    k = len(triangles) + 1
    expected = reference(scores, k, caps)
    np.testing.assert_array_equal(expected[1][:, 0], [1, 1, 2, 0, 4, 3, 3, -1, -1])
    np.testing.assert_array_equal(expected[2], [5, 1, 5, 1, 1, 1, 1, 0, 0])

    primitive = Triangles()
    faces = np.arange(triangles.size // 3).reshape(-1, 3)
    raw = primitive.indexed(triangles.reshape(-1, 3), faces)
    tree, data = build_tree(primitive, raw, Config(leaf_size_threshold=1))
    options = dict(cache_bounds=cache_bounds)
    knn = make_knn(primitive, k, Euclidean(), **options)
    assert_result(jax.jit(knn)(tree, data, queries, caps), expected)
    nearest = make_nearest(primitive, Euclidean(), **options)
    cp, distance, ids, status = jax.jit(nearest)(tree, data, queries, caps)
    np.testing.assert_array_equal(ids, expected[1][:, 0])
    np.testing.assert_array_equal(status, expected[3])
    np.testing.assert_allclose(distance, expected[0][:, 0], rtol=5e-6, atol=2e-6)
    hits = ids >= 0
    nearest_points = points[np.flatnonzero(hits), ids[hits]]
    np.testing.assert_allclose(cp[hits], nearest_points, rtol=0, atol=2e-6)

    knn_points = make_knn(primitive, k, Euclidean(), return_points=True, **options)
    values, cp, counts, status = jax.jit(knn_points)(tree, data, queries, caps)
    order = expected[1]
    expected_points = np.where(
        (order >= 0)[..., None], points[np.arange(len(queries))[:, None], order], 0.
    )
    assert_result((values, counts, status), (expected[0], expected[2], expected[3]))
    np.testing.assert_allclose(cp, expected_points, rtol=0, atol=2e-6)


def test_triangle_adjacent_float32_caps():
    vertex = np.array([-.006331482, -9.0728, -9.025907], np.float32)
    point = np.array([-6.2419786, -8.897067, -4.5006127], np.float32)
    queries = np.broadcast_to(point, (2, 3))
    # Distances use the Euclidean policy's float32 score and cap convention.
    cap = np.float32(np.linalg.norm(point.astype(np.float64) - vertex))
    caps = np.array([np.nextafter(cap, np.float32(0)), cap], np.float32)

    primitive = Triangles()
    tree, data = build_tree(primitive, primitive.indexed([vertex], [[0, 0, 0]]))
    nearest = make_nearest(primitive, Euclidean())
    cp, distance, ids, status = jax.jit(nearest)(tree, data, queries, caps)
    np.testing.assert_array_equal(distance, [np.inf, cap])
    np.testing.assert_array_equal(ids, [-1, 0])
    np.testing.assert_array_equal(status, [0, 1])
    np.testing.assert_array_equal(cp[1], vertex)

    result = make_knn(primitive, 1, Euclidean())(tree, data, queries, caps)
    assert_result(result, reference([[cap], [cap]], 1, caps))
    points = make_knn(primitive, 1, Euclidean(), return_points=True)(tree, data, queries, caps)
    np.testing.assert_array_equal(points[1][:, 0], np.stack([np.zeros_like(vertex), vertex]))
    radius_count = make_radius(primitive, Euclidean(), mode="count")
    values, status = radius_count(tree, data, queries, caps)
    np.testing.assert_array_equal(values, [0, 1])
    np.testing.assert_array_equal(status, [0, 1])


def test_triangle_vertex_score_rounds_once_at_midpoint():
    # Face 2 has squared distance 1 + 2^-24, which rounds to 1. The tiny
    # positive z² terms put faces 0 and 1 above that midpoint, tied in float32.
    corners = np.array([[1., 2.**-12, z] for z in (2.**-50, 2.**-51, 0.)], np.float32)
    triangles = np.stack([corners, corners + [1, 0, 0], corners + [0, 1, 0]], axis=1)
    primitive, metric = Triangles(), Euclidean()
    mesh = primitive.indexed(triangles.reshape(-1, 3), np.arange(9).reshape(3, 3))
    tree, data = build_tree(primitive, mesh, Config(leaf_size_threshold=1))
    args = tree, data, np.zeros((2, 3), np.float32), np.array([1., np.inf], np.float32)
    point, distance, ids, status = map(np.asarray, make_nearest(primitive, metric)(*args))
    # The FP32 nearest kernel can round all three vertex scores to 1.
    assert np.isin(ids, [0, 1, 2]).all()
    np.testing.assert_array_equal(point, corners[ids])
    np.testing.assert_array_equal(distance, [1., 1.])
    np.testing.assert_array_equal(status, [1, 1])
    np.testing.assert_array_equal(make_knn(primitive, 3, metric)(*args)[1],
                                  [[2, -1, -1], [2, 0, 1]])
    np.testing.assert_array_equal(make_radius(primitive, metric, mode="count")(*args)[0], [1, 3])
