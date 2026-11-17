"""Typed traversal and exact radius policy composition on CPU."""
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from surfaces.explicit.accelerators.bvh.warp import Points, PointItem3, build_tree
from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.bvh.backends._warp_tree import DeviceTree
from surfaces.explicit.accelerators.bvh.backends.warp_queries.metrics import Euclidean, NearestCallbacks
from surfaces.explicit.accelerators.bvh.backends.warp_queries.queries.range import (
    RadiusAccumulator, make_radius,
)
from surfaces.explicit.accelerators.bvh.backends.warp_queries.traversal.policy import (
    make_traversal, make_traversal_state,
)
from tests.spatial_trees._warp_device import device_payload, run_kernel


@wp.func
def score(q: wp.vec3, item: PointItem3):
    d = q - item.a
    return item.a, d[0] * d[0] + 4.0 * d[1] * d[1] + d[2] * d[2]


@wp.func
def bound(q: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    d = wp.max(lo - q, wp.max(q - hi, wp.vec3(0.0)))
    return d[0] * d[0] + 4.0 * d[1] * d[1] + d[2] * d[2]


@wp.func
def accept(q: wp.vec3, item: PointItem3):
    return item.a[0] >= 0.0


@pytest.mark.parametrize("cache_bounds", [True, False])
def test_radius_host_custom_metric_closed_caps_invalid_empty_and_jit(cache_bounds):
    primitive = Points()
    tree, data = build_tree(primitive, primitive.data([[-1., 0., 0.], [0., 1., 0.],
                                                       [0., -1., 0.], [5., 0., 0.]]))
    metric = NearestCallbacks(score, bound)
    queries = jnp.zeros((7, 3)).at[6, 0].set(jnp.nan)
    caps = jnp.array([0., 4., jnp.nextafter(jnp.float32(4.), jnp.float32(0.)),
                      25., jnp.inf, -1., 4.])
    any_query = make_radius(primitive, metric, predicate=accept,
                            cache_bounds=cache_bounds)
    count_query = make_radius(primitive, metric, predicate=accept, mode="count",
                              cache_bounds=cache_bounds)
    any_value, any_status = jax.jit(any_query)(tree, data, queries, caps)
    count, status = jax.jit(count_query)(tree, data, queries, caps)
    np.testing.assert_array_equal(any_value, [False, True, False, True, True, False, False])
    np.testing.assert_array_equal(count, [0, 2, 0, 3, 3, 0, 0])
    np.testing.assert_array_equal(status, [0, 1, 0, 1, 1, -1, -1])
    np.testing.assert_array_equal(any_status, status)
    empty_tree, empty = build_tree(primitive, primitive.data(np.empty((0, 3))))
    assert int(count_query(empty_tree, empty, jnp.zeros(3), 4.)[1]) == 0


def test_traversal_validation():
    State = make_traversal_state("PolicyValidationState", {"visits": wp.int32})
    assert {"limit", "failed", "stop", "visits"} <= set(State.vars)
    with pytest.raises(ValueError, match="reserved"):
        make_traversal_state("Invalid", {"limit": wp.int32})


AccumState = make_traversal_state("TestAccumState", {"count": wp.int32,
                                                    "total": wp.float32,
                                                    "matched": wp.bool})


@wp.struct
class AccumResult:
    count: wp.int32
    total: wp.float32


@wp.func
def accum_init(query: wp.vec3, cap: float):
    state = AccumState()
    state.limit = cap
    return state


@wp.func
def accum_update(query: wp.vec3, item: PointItem3, point: wp.vec3,
                 value: float, index: int, state: wp.ref[AccumState]):
    state.count += 1
    state.total += value
    if state.count == 2:
        state.stop = True


@wp.func
def accum_finish(state: AccumState):
    return AccumResult(state.count, state.total)


@wp.struct
class WalkContext:
    max_x: float


WalkState = make_traversal_state("TestWalkState", {"visits": wp.int32})


@wp.func
def walk_bound(ctx: WalkContext, tree: DeviceTree, node: int,
               lo: wp.vec3, hi: wp.vec3, state: wp.ref[WalkState]):
    return 0.0, lo[0] <= ctx.max_x, False


@wp.func
def walk_visit(ctx: WalkContext, tree: DeviceTree, node: int, begin: int,
               end: int, tag: bool, state: wp.ref[WalkState]):
    for slot in range(begin, end):
        state.visits += 1
        state.stop = True
        return


def test_direct_device_accumulation_and_public_walk():
    primitive = Points()
    tree, data = build_tree(primitive, primitive.data([[0., 1., 0.], [0., -1., 0.],
                                                       [5., 0., 0.]]))
    accumulator = RadiusAccumulator(AccumState, accum_init, accum_update, accum_finish)
    radius = make_radius(primitive, NearestCallbacks(score, bound),
                         mode="accumulate", accumulator=accumulator)
    traversal = make_traversal(WalkContext, WalkState, walk_bound, walk_visit)
    Geometry = primitive.data_type
    radius_fn, walk_fn = radius.device, traversal.device

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(tree: DeviceTree, geometry: Geometry, values: wp.array(dtype=wp.int32),
               totals: wp.array(dtype=wp.float32), statuses: wp.array(dtype=wp.int32),
               visits: wp.array(dtype=wp.int32)):
        i = wp.tid()
        cap = wp.where(i == 0, 4.0, wp.where(i == 1, 0.0, -1.0))
        result, status = wp.static(radius_fn)(wp.vec3(0.0), cap, tree, geometry)
        state = WalkState()
        state.limit = 0.0
        wp.static(walk_fn)(tree, WalkContext(10.0), state)
        values[i] = result.count
        totals[i] = result.total
        statuses[i] = status
        visits[i] = state.visits

    device = wp.device_from_jax(jax.devices()[0])
    counts, totals, statuses, visits = run_kernel(
        kernel, 3, [device_payload(DeviceTree, tree, device),
                    device_payload(Geometry, data, device)],
        [((3,), wp.int32), ((3,), wp.float32), ((3,), wp.int32), ((3,), wp.int32)],
        queries=(radius, traversal), device=device,
    )
    np.testing.assert_array_equal(counts, [2, 0, 0])
    np.testing.assert_array_equal(totals, [8., 0., 0.])
    np.testing.assert_array_equal(statuses, [1, 0, -1])
    np.testing.assert_array_equal(visits, [1, 1, 1])


@wp.struct
class ExternalData:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=wp.int32)


@wp.func
def external_load(data: Any, slot: int):
    return data.vertices[data.indices[slot]]


@wp.func
def external_bounds(item: wp.vec3):
    return item, item


@wp.func
def external_closest(query: wp.vec3, item: wp.vec3):
    return item


@dataclass(frozen=True)
class ExternalPoints:
    dimension: int = 3
    point_type: object = wp.vec3
    data_type: object = ExternalData
    load: object = external_load
    bounds_of_item: object = external_bounds
    closest_euclidean: object = external_closest

    def bounds(self, data):
        points = np.asarray(data.vertices)[np.asarray(data.indices)]
        return points, points

    def pack(self, data, ids):
        return data._replace(indices=data.indices[ids])


def test_external_primitive_and_stack_failure():
    Host = namedtuple("ExternalHost", "vertices indices")
    primitive = ExternalPoints()
    vertices = jnp.array([[0., 0., 0.], [3., 0., 0.], [8., 0., 0.]])
    data = Host(vertices, jnp.array([0, 1, 2], jnp.int32))
    tree, geometry = build_tree(primitive, data, Config(leaf_size_threshold=1))
    count = make_radius(primitive, Euclidean(), mode="count")
    values, statuses = jax.jit(count)(tree, geometry, jnp.zeros((2, 3)),
                                       jnp.array([3., 2.]))
    np.testing.assert_array_equal(values, [2, 1])
    np.testing.assert_array_equal(statuses, [1, 1])
    limited = make_radius(primitive, Euclidean(), mode="count", stack_capacity=1)
    failed_value, failed_status = limited(tree, geometry, jnp.zeros(3), jnp.inf)
    assert int(failed_value) == 0 and int(failed_status) == -1
