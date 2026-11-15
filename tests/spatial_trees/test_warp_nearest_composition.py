"""Static nearest/kNN composition keeps query identity and one fused call."""
import jax
import numpy as np
import pytest

pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    Euclidean, Points, build_tree, make_knn, make_nearest,
)
from surfaces.explicit.accelerators.bvh.backends.warp_queries.selection.candidates import (  # noqa: E402
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
