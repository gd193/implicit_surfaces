"""Static nearest/kNN composition keeps query identity and one fused call."""
from dataclasses import replace

import jax
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from sengi.bvh.warp import (  # noqa: E402
    Euclidean, Points, TriangleItem3, Triangles, build_tree, make_knn, make_nearest,
)
from sengi.bvh.backends.warp_queries.selection.candidates import (  # noqa: E402
    candidate_functions,
)


def test_candidate_storage_policy_is_cached_by_shape_and_output():
    ids = candidate_functions(65)
    points = candidate_functions(65, True)
    assert ids is candidate_functions(65)
    assert points is candidate_functions(65, True)
    assert ids.storage is not points.storage
    assert ids.slot is None and points.slot is not None
    assert ids.storage._fields_[0][0] == "entries"
    assert points.storage._fields_[0][0] == "slots"


def test_nearest_and_knn_output_policies_are_fused_and_cached():
    primitive = Points(2)
    tree, geometry = build_tree(primitive, primitive.data(np.array([[1., 0.], [-1., 0.]], np.float32)))
    nearest = make_nearest(primitive, Euclidean())
    ids = make_knn(primitive, 2, Euclidean())
    points = make_knn(primitive, 2, Euclidean(), return_points=True)
    assert nearest.kernel is make_nearest(primitive, Euclidean()).kernel
    assert ids.kernel is make_knn(primitive, 2, Euclidean()).kernel
    assert ids.kernel is not points.kernel
    assert nearest.describe()["components"] == {
        "node_access": "packed", "scheduling": "grouped",
        "deferred": "cached_bound", "selection": "single", "output": "coordinates",
    }
    assert ids.describe()["components"]["selection"] == "unrolled"
    assert ids.describe()["components"]["output"] == "ids"
    assert points.describe()["components"]["output"] == "coordinates"
    assert nearest.kernel is make_nearest(primitive, Euclidean()).kernel

    query = np.zeros((1, 2), np.float32)
    _, distance, item_id, status = jax.jit(nearest)(tree, geometry, query)
    scores, ids_result, count, knn_status = jax.jit(ids)(tree, geometry, query)
    point_scores, coordinates, point_count, point_status = jax.jit(points)(tree, geometry, query)
    np.testing.assert_array_equal(scores[0], [1., 1.])
    np.testing.assert_array_equal(ids_result[0], [0, 1])
    np.testing.assert_array_equal(coordinates[0], [[1., 0.], [-1., 0.]])
    np.testing.assert_array_equal(point_scores, scores)
    np.testing.assert_array_equal(distance, scores[:, 0])
    np.testing.assert_array_equal(item_id, ids_result[:, 0])
    np.testing.assert_array_equal([status[0], knn_status[0], point_status[0]], 1)
    np.testing.assert_array_equal([count[0], point_count[0]], 2)
    for composed in (nearest, ids, points):
        hlo = jax.jit(composed).lower(tree, geometry, query).as_text()
        assert hlo.count("stablehlo.custom_call") == 1


def test_triangle_fused_nearest_respects_replaced_geometry_and_metric():
    primitive = Triangles()
    metric = Euclidean()

    @wp.func
    def first_vertex(query: wp.vec3, item: TriangleItem3):
        return item.a

    @wp.func
    def weighted_vertex(query: wp.vec3, item: TriangleItem3):
        return item.a, 4.0 * wp.length_sq(query - item.a)

    class CustomMetric:
        def bind(self, primitive):
            return replace(metric.bind(primitive), closest=weighted_vertex, score_at=None, point_at=None)

    raw = primitive.indexed([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]])
    tree, data = build_tree(primitive, raw)
    query = np.array([.25, .25, 0.], np.float32)
    # The slot callbacks take precedence until a provider opts out of both.
    modified = replace(primitive, closest_euclidean=first_vertex)
    for provider, policy, point, distance in (
        (modified, metric, query, 0.),
        (replace(modified, score_euclidean_at=None, closest_euclidean_at=None),
         metric, [0., 0., 0.], np.linalg.norm(query)),
        (primitive, CustomMetric(), [0., 0., 0.], 2 * np.linalg.norm(query)),
    ):
        nearest = make_nearest(provider, policy)
        assert nearest.ops.bind_fused(nearest.ops) is None
        actual = nearest(tree, data, query)
        np.testing.assert_array_equal(actual[0], point)
        np.testing.assert_allclose(actual[1:], [distance, 0, 1])
