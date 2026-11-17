"""Selected-result policy device contract and native CPU compilation."""
import jax
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    DeviceTree, Euclidean, PointItem3, Points, build_tree,
    IdPoint, make_knn, make_nearest,
)
from tests.spatial_trees._warp_device import device_payload, run_kernel  # noqa: E402


def _tree_inputs(tree, geometry, geometry_type, device):
    return [device_payload(DeviceTree, tree, device),
            device_payload(geometry_type, geometry, device)]


@pytest.mark.parametrize("k", [16, 64, 65])
def test_id_point_selected_results(k):
    primitive = Points()
    points = np.zeros((70, 3), np.float32)
    points[:, 0] = np.arange(70)
    tree, geometry = build_tree(primitive, primitive.data(points))
    nearest = make_nearest(primitive, Euclidean(), result_policy=IdPoint())
    knn = make_knn(primitive, k, Euclidean(), result_policy=IdPoint())
    Result = nearest.result_policy.result_type
    assert knn.result_policy.result_type is Result
    with pytest.raises(TypeError, match="not callable"):
        nearest(tree, geometry, points[:1])
    with pytest.raises(TypeError, match="not callable"):
        knn(tree, geometry, points[:1])
    near_fn, knn_fn = nearest.device, knn.device
    Geometry = primitive.data_type

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: Geometry, queries: wp.array(dtype=wp.vec3),
               near_out: wp.array(dtype=Result), near_status: wp.array(dtype=wp.int32),
               values: wp.array2d(dtype=wp.float32), results: wp.array2d(dtype=Result),
               counts: wp.array(dtype=wp.int32), statuses: wp.array(dtype=wp.int32)):
        row = wp.tid()
        value, status = wp.static(near_fn)(queries[row], 2.0, tree, geometry)
        near_out[row] = value
        near_status[row] = status
        count, status = wp.static(knn_fn)(queries[row], 2.0, tree, geometry, values, results, row)
        counts[row] = count
        statuses[row] = status

    device = wp.device_from_jax(jax.devices()[0])
    q = np.array([[0., 0., 0.], [100., 0., 0.], [np.nan, 0., 0.]], np.float32)
    n = len(q)
    near, near_status, values, results, counts, statuses = run_kernel(
        kernel, n,
        [*_tree_inputs(tree, geometry, Geometry, device),
         wp.array(q, dtype=wp.vec3, device=device)],
        [(n, Result), (n, wp.int32), ((n, k), wp.float32),
         ((n, k), Result), (n, wp.int32), (n, wp.int32)],
        queries=(nearest, knn), device=device,
    )
    assert near['id'][0] == 0
    np.testing.assert_array_equal(near['point'][0], points[0])
    np.testing.assert_array_equal(near_status, [1, 0, -1])
    np.testing.assert_array_equal(counts, [3, 0, 0])
    np.testing.assert_array_equal(statuses, [1, 0, -1])
    np.testing.assert_array_equal(results['id'][0, :3], [0, 1, 2])
    np.testing.assert_array_equal(results['point'][0, :3], points[:3])
    assert np.isposinf(values[:, 3:]).all()
    assert np.isposinf(values[1:]).all()


@wp.struct
class CustomResult:
    original_id: int
    selected: wp.vec3
    attribute: float
    value: float


@wp.func
def custom_build(query: wp.vec3, item: PointItem3, original_id: int,
                 selected_point: wp.vec3, reported_value: float) -> CustomResult:
    result = CustomResult()
    result.original_id = original_id
    result.selected = selected_point
    result.attribute = item.a[0] * 2.0
    result.value = reported_value
    return result


@wp.func
def weighted(query: wp.vec3, item: PointItem3):
    delta = query - item.a
    return item.a, delta[0] * delta[0] + 4.0 * delta[1] * delta[1] + delta[2] * delta[2]


@wp.func
def weighted_bound(query: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    delta = wp.max(lower - query, wp.max(query - upper, wp.vec3(0.0)))
    return delta[0] * delta[0] + 4.0 * delta[1] * delta[1] + delta[2] * delta[2]


def test_custom_result_uses_original_id_item_and_reported_value():
    from surfaces.explicit.accelerators.bvh.backends.warp_queries.queries.results import ResultPolicy
    primitive = Points()
    points = np.array([[2., 0., 0.], [0., 1., 0.], [3., 0., 0.]], np.float32)
    tree, geometry = build_tree(primitive, primitive.data(points))
    policy = ResultPolicy(CustomResult, custom_build)
    nearest = make_nearest(primitive, evaluate=weighted, lower_bound=weighted_bound,
                           result_policy=policy)
    knn = make_knn(primitive, 3, evaluate=weighted, lower_bound=weighted_bound,
                   result_policy=policy)
    near_fn, knn_fn = nearest.device, knn.device
    Geometry = primitive.data_type

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: Geometry, queries: wp.array(dtype=wp.vec3),
               near: wp.array(dtype=CustomResult), near_status: wp.array(dtype=int),
               values: wp.array2d(dtype=float), results: wp.array2d(dtype=CustomResult),
               counts: wp.array(dtype=int), statuses: wp.array(dtype=int)):
        row = wp.tid()
        selected, status = wp.static(near_fn)(queries[row], 10.0, tree, geometry)
        near[row] = selected
        near_status[row] = status
        count, status = wp.static(knn_fn)(queries[row], 10.0, tree, geometry, values, results, row)
        counts[row] = count
        statuses[row] = status

    device = wp.device_from_jax(jax.devices()[0])
    near, near_status, values, results, counts, statuses = run_kernel(
        kernel, 1,
        [*_tree_inputs(tree, geometry, Geometry, device),
         wp.array(np.zeros((1, 3), np.float32), dtype=wp.vec3, device=device)],
        [(1, CustomResult), (1, int), ((1, 3), float),
         ((1, 3), CustomResult), (1, int), (1, int)],
        queries=(nearest, knn), device=device,
    )
    np.testing.assert_array_equal(near_status, [1])
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_array_equal(statuses, [1])
    np.testing.assert_array_equal(results['original_id'][0], [0, 1, 2])
    np.testing.assert_array_equal(results['selected'][0], points)
    np.testing.assert_array_equal(results['attribute'][0], [4., 0., 6.])
    np.testing.assert_array_equal(results['value'][0], [4., 4., 9.])
    np.testing.assert_array_equal(values[0], [4., 4., 9.])
    assert near['original_id'][0] == 0


def test_external_primitive_id_point_result():
    from tests.typing.warp_api.real_extension import (
        ExternalArrays, ExternalData, ExternalPoints, external_evaluate, external_bound,
    )
    primitive = ExternalPoints()
    positions = np.array([[4., 0., 0.], [1., 0., 0.]], np.float32)
    tree, geometry = build_tree(primitive, ExternalArrays(positions))
    query = make_nearest(primitive, evaluate=external_evaluate,
                         lower_bound=external_bound, result_policy=IdPoint())
    Result = query.result_policy.result_type
    device_fn = query.device

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: ExternalData,
               output: wp.array(dtype=Result), status: wp.array(dtype=int)):
        result, state = wp.static(device_fn)(wp.vec3(0.0), 100.0, tree, geometry)
        output[0] = result
        status[0] = state

    device = wp.device_from_jax(jax.devices()[0])
    output, status = run_kernel(
        kernel, 1, _tree_inputs(tree, geometry, ExternalData, device),
        [(1, Result), (1, int)], queries=(query,), device=device,
    )
    assert int(status[0]) == 1
    assert int(output['id'][0]) == 1
    np.testing.assert_array_equal(output['point'][0], positions[1])


@wp.func
def invalid_conversion(score: float):
    return wp.where(score == 1.0, wp.nan, wp.sqrt(score))


def test_custom_result_failure_clears_written_slots():
    from dataclasses import dataclass, replace

    @dataclass(frozen=True)
    class Conversion:
        def bind(self, primitive):
            return replace(Euclidean().bind(primitive), to_distance=invalid_conversion)

    primitive = Points()
    points = np.array([[0., 0., 0.], [1., 0., 0.]], np.float32)
    tree, geometry = build_tree(primitive, primitive.data(points))
    query = make_knn(primitive, 3, Conversion(), result_policy=IdPoint())
    Result = query.result_policy.result_type
    device_fn = query.device
    Geometry = primitive.data_type

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: Geometry,
               values: wp.array2d(dtype=float), results: wp.array2d(dtype=Result),
               counts: wp.array(dtype=int), statuses: wp.array(dtype=int)):
        count, status = wp.static(device_fn)(wp.vec3(0.0), 10.0, tree, geometry,
                                              values, results, 0)
        counts[0] = count
        statuses[0] = status

    device = wp.device_from_jax(jax.devices()[0])
    values, results, counts, statuses = run_kernel(
        kernel, 1, _tree_inputs(tree, geometry, Geometry, device),
        [((1, 3), float), ((1, 3), Result), (1, int), (1, int)],
        queries=(query,), device=device,
    )
    assert int(counts[0]) == 0
    assert int(statuses[0]) == -1
    assert np.isposinf(values).all()
    np.testing.assert_array_equal(results['id'], 0)
    np.testing.assert_array_equal(results['point'], 0.)


@pytest.mark.parametrize("k", [16, 64, 65])
def test_equal_score_id_order_in_custom_results(k):
    primitive = Points()
    points = np.zeros((70, 3), np.float32)
    tree, geometry = build_tree(primitive, primitive.data(points))
    query = make_knn(primitive, k, Euclidean(), result_policy=IdPoint())
    Result = query.result_policy.result_type
    device_fn = query.device
    Geometry = primitive.data_type

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: Geometry,
               values: wp.array2d(dtype=float), results: wp.array2d(dtype=Result)):
        wp.static(device_fn)(wp.vec3(0.0), 0.0, tree, geometry, values, results, 0)

    device = wp.device_from_jax(jax.devices()[0])
    values, results = run_kernel(
        kernel, 1, _tree_inputs(tree, geometry, Geometry, device),
        [((1, k), float), ((1, k), Result)], queries=(query,), device=device,
    )
    np.testing.assert_array_equal(results['id'][0], np.arange(k))
    np.testing.assert_array_equal(values, 0.)
