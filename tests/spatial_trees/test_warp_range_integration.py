"""Independent radius references and executable static reference typing."""
import itertools
import jax
import numpy as np
import pytest
import warp as wp
from sengi.bvh.warp import (
    DeviceTree, Euclidean, Points, build_tree, make_radius,
)
from sengi.protocol import Config
from tests.spatial_trees._warp_device import device_payload, run_kernel
from tests.typing.warp_api.range_valid import Context, SumState, reduction, traversal


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("cached,packed,grouped", list(itertools.product([False, True], repeat=3)))
def test_counts_against_brute_force_and_vmap(dimension, cached, packed, grouped):
    rng = np.random.default_rng(137)
    points = rng.uniform(-1, 1, (67, dimension)).astype(np.float32)
    queries = rng.uniform(-1.5, 1.5, (2, 17, dimension)).astype(np.float32)
    radii = np.linspace(0.0, 1.7, 34, dtype=np.float32).reshape(2, 17)
    primitive = Points(dimension)
    tree, geometry = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=3))
    query = make_radius(primitive, Euclidean(), mode="count", cache_bounds=cached,
                        packed_nodes=packed, group_leaves=grouped)
    distance_squared = np.sum((queries.astype(np.float64)[..., None, :] - points.astype(np.float64))**2, axis=-1)
    expected = np.sum(distance_squared <= radii.astype(np.float64)[..., None]**2, axis=-1)
    counts, statuses = jax.jit(jax.vmap(lambda q, r: query(tree, geometry, q, r)))(queries, radii)
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(statuses, expected > 0)
    lowered = jax.jit(lambda q, r: query(tree, geometry, q, r)).lower(queries[0], radii[0])
    assert str(lowered.compiler_ir()).count("stablehlo.custom_call") == 1


def test_typed_reference_callbacks_compile_and_mutate_caller_state():
    primitive = Points()
    tree, geometry = build_tree(primitive, primitive.data([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]))
    Geometry = primitive.data_type
    reduce_fn, walk_fn = reduction.device, traversal.device

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: Geometry, output: wp.array(dtype=wp.float32)):
        total, status = wp.static(reduce_fn)(wp.vec3(0.0), 1.0, tree, geometry)
        state = SumState()
        state.limit = wp.inf
        ctx = Context()
        wp.static(walk_fn)(tree, ctx, state)
        output[0] = total
        output[1] = state.total
        output[2] = float(status)

    device = wp.device_from_jax(jax.devices()[0])
    (output,) = run_kernel(
        kernel, 1, [device_payload(DeviceTree, tree, device),
                    device_payload(Geometry, geometry, device)],
        [((3,), wp.float32)], queries=(reduction, traversal), device=device,
    )
    np.testing.assert_array_equal(output, [1.0, 3.0, 1.0])
