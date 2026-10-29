"""Boundary measure/PDF contracts and dimension-specific Robin bounds."""
from collections import namedtuple
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.protocol import Config  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_boundary import build_boundary  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_primitives import Points, Segments, Triangles  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_metrics import Angular  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_boundary_queries import (  # noqa: E402
    MeasureOps, RegionOps, RobinOps, _euclidean_robin, make_boundary_sample, make_robin_radius, prepare_boundary_sample,
    prepare_robin, prepare_robin_coefficients,
)


def _boundary(dimension, large=False):
    if dimension == 2:
        primitive = Segments(2)
        vertices = [[-10, 0], [10, 0]] if large else [[0, 0], [2, 0], [2, 1], [0, 1]]
        indices = [[0, 1]] if large else [[0, 1], [1, 2], [2, 3], [3, 0]]
    else:
        primitive = Triangles()
        vertices = [[-10, -10, 0], [10, -10, 0], [0, 10, 0]] if large else [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
        indices = [[0, 1, 2]]
    return primitive, *build_boundary(primitive, primitive.indexed(vertices, indices), Config(leaf_size_threshold=1))


@pytest.mark.parametrize("dimension", [2, 3])
def test_warp_boundary_sample_measure_batching_and_nulls(dimension):
    primitive, tree, geometry, boundary = _boundary(dimension)
    metadata = prepare_boundary_sample(tree, boundary)
    sampler = make_boundary_sample(primitive)
    randoms = jnp.asarray(np.column_stack(((np.arange(60)+.5)/60, np.full(60, .36), np.full(60, .25))), jnp.float32)
    points = jnp.zeros((60, dimension), jnp.float32)
    outputs = jax.jit(sampler)(tree, geometry, metadata, points, jnp.inf, randoms)
    point, normal, pdf, ids, status = map(np.asarray, outputs)
    np.testing.assert_array_equal(status, 1)
    np.testing.assert_allclose(pdf, 1./np.asarray(boundary.core.primitive_measure).sum(), rtol=3e-6)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1., atol=1e-6)
    if dimension == 2:
        np.testing.assert_array_equal(np.bincount(ids, minlength=4), [20, 10, 20, 10])
    else:
        np.testing.assert_allclose(point, np.tile([.9, .3, 0.], (60, 1)), atol=1e-6)
    nested = jax.jit(jax.vmap(sampler, in_axes=(None, None, None, 0, None, 0)))(
        tree, geometry, metadata, points.reshape(3, 20, dimension), jnp.inf, randoms.reshape(3, 20, 3))
    np.testing.assert_allclose(nested[0].reshape(60, dimension), outputs[0])
    rejected = sampler(tree, geometry, metadata, points, 0., randoms)
    np.testing.assert_array_equal(rejected[2], 0.)
    np.testing.assert_array_equal(rejected[3], -1)
    np.testing.assert_array_equal(rejected[4], 0)


@wp.func
def _nonuniform_sample(item: Any, u: wp.vec2):
    edge = item.b-item.a
    t = wp.sqrt(u[0])
    return item.a+t*edge, wp.vec2(0.0, -1.0), 2.0*t/wp.length(edge)


@wp.func
def _region_valid(query: float, cap: float):
    return wp.isfinite(query) and cap >= 0.0


@wp.func
def _region_overlaps(query: float, cap: float, lower: wp.vec3, upper: wp.vec3):
    return lower[0] <= query


@wp.func
def _region_contains(query: float, cap: float, point: wp.vec2):
    return point[0] <= query


def test_warp_boundary_custom_measure_and_exact_region_membership():
    primitive = Segments(2)
    tree, geometry, boundary = build_boundary(primitive, primitive.indexed([[0, 0], [1, 0], [2, 0], [4, 0]],
        [[0, 1], [2, 3]]), Config(leaf_size_threshold=1), silhouettes=False)
    measure = MeasureOps(_nonuniform_sample, lambda core: np.where(np.asarray(core.primitive_measure) == 1., 9., 1.))
    region = RegionOps(wp.float32, (), _region_valid, _region_overlaps, _region_contains)
    metadata = prepare_boundary_sample(tree, boundary, measure)
    sampler = make_boundary_sample(primitive, measure, region)
    point, _, pdf, ids, status = jax.jit(sampler)(tree, geometry, metadata, jnp.array([.25, .25, 4., 4.]), 0.,
        jnp.array([[.05, .04, .3], [.95, .64, .3], [.95, .64, .3], [.05, 0., .3]]))
    np.testing.assert_allclose(np.asarray(point)[[0, 2]], [[.2, 0], [3.6, 0]], atol=1e-6)
    np.testing.assert_allclose(pdf, [.4, 0, .08, 0], atol=1e-6)
    np.testing.assert_array_equal(ids, [0, -1, 1, -1])
    np.testing.assert_array_equal(status, [1, 0, 1, 0])


@pytest.mark.parametrize("dimension", [2, 3])
def test_warp_robin_dimension_correct_bounds_and_dynamic_coefficients(dimension):
    primitive, tree, geometry, boundary = _boundary(dimension, large=True)
    query = jax.jit(make_robin_radius(primitive, _euclidean_robin(primitive)))
    point = jnp.array([0., 1.] if dimension == 2 else [0., 0., 1.])
    expected = [np.sqrt(2.)*np.exp(.5), np.exp(.25)] if dimension == 2 else [1.5*np.sqrt(3.), 4./3.]
    for coefficient, radius in zip([1., 4., np.inf], [*expected, 1.], strict=True):
        metadata = prepare_robin(tree, boundary, coefficient)
        actual, status = query(tree, geometry, metadata, point, 10.)
        np.testing.assert_allclose(actual, radius, rtol=8e-6)
        assert int(status) == 1
    metadata = prepare_robin(tree, boundary, 0.)
    actual, status = query(tree, geometry, metadata, point, .5)
    np.testing.assert_array_equal([actual, status], [.5, 1.])
    actual, status = query(tree, geometry, metadata, jnp.full((2, dimension), jnp.nan), jnp.array([1., -1.]))
    assert np.isnan(actual).all()
    np.testing.assert_array_equal(status, -1)


def test_warp_boundary_policy_and_coefficient_validation():
    primitive, tree, geometry, boundary = _boundary(2)
    for bad in (-1., np.nan, 1e40, 1j):
        with pytest.raises((ValueError, TypeError)):
            prepare_robin_coefficients(tree, bad)
    with pytest.raises(ValueError, match="minimum exceeds"):
        prepare_robin_coefficients(tree, 1.00000004, 1.00000003)
    values = prepare_robin_coefficients(tree, .1).primitive_bounds
    assert np.all(np.asarray(values[:, 0], np.float64) <= .1)
    assert np.all(np.asarray(values[:, 1], np.float64) >= .1)
    with pytest.raises(TypeError, match="metric"):
        make_robin_radius(primitive, metric=Angular())
    for make in (make_robin_radius, make_boundary_sample):
        with pytest.raises(TypeError, match="explicit"):
            make(Points())
    with pytest.raises(ValueError, match="silhouette"):
        prepare_robin(tree, boundary._replace(silhouette=None), 1.)


@wp.struct
class _CustomRobinCore:
    limit: wp.array(dtype=wp.float32)


@wp.struct
class _CustomRobinFeatures:
    feature_refs: wp.array(dtype=wp.int32)
    node_feature_offset: wp.array(dtype=wp.int32)
    node_feature_count: wp.array(dtype=wp.int32)
    limit: wp.array(dtype=wp.float32)


def test_warp_custom_robin_policy_uses_conservative_closest_surface_ball():
    from surfaces.explicit.accelerators.bvh.backends._warp_metrics import Euclidean

    primitive, tree, geometry, boundary = _boundary(3, large=True)
    nearest = Euclidean().bind(primitive)

    @wp.func
    def node(query: wp.vec3, lower: wp.vec3, upper: wp.vec3, coefficients: wp.vec2):
        return wp.sqrt(wp.static(nearest.node_bound)(query, lower, upper))

    @wp.func
    def leaf(query: wp.vec3, item: Any, coefficient: float):
        point, score = wp.static(nearest.closest)(query, item)
        return wp.sqrt(score)

    @wp.func
    def feature(query: wp.vec3, a: wp.vec3, b: wp.vec3, n0: wp.vec3, n1: wp.vec3):
        return wp.inf

    policy = RobinOps(wp.vec3, (3,), nearest.valid, node, leaf, feature)
    result = jax.jit(make_robin_radius(primitive, policy))(
        tree, geometry, prepare_robin(tree, boundary, 1.), jnp.array([0., 0., 1.]), 10.)
    np.testing.assert_array_equal(result, [1., 1.])

    @wp.func
    def zero_node(query: wp.vec3, lower: wp.vec3, upper: wp.vec3, coefficients: wp.vec2):
        return 0.0

    @wp.func
    def feature_data(query: wp.vec3, tree: Any, geometry: Any, core: _CustomRobinCore,
                     features: _CustomRobinFeatures, coefficients: Any, node: int, index: int):
        value = features.limit[index]
        return wp.where(wp.isnan(value), value, wp.min(core.limit[0], value))

    policy = RobinOps(wp.vec3, (3,), nearest.valid, zero_node, leaf, None,
                      feature_bound_with_data=feature_data, core_type=_CustomRobinCore, feature_type=_CustomRobinFeatures)
    query = jax.jit(make_robin_radius(primitive, policy))
    Core = namedtuple("CustomRobinCore", _CustomRobinCore.vars)
    Features = namedtuple("CustomRobinFeatures", _CustomRobinFeatures.vars)
    core = Core(jnp.array([.75]))
    features = Features(jnp.array([0], jnp.int32), jnp.array([0], jnp.int32),
                        jnp.array([1], jnp.int32), jnp.array([.5]))
    metadata = prepare_robin(tree, boundary, 1.)._replace(core=core, silhouette=features)
    for limit in (.5, .25):
        metadata = metadata._replace(silhouette=features._replace(limit=jnp.array([limit])))
        np.testing.assert_array_equal(query(tree, geometry, metadata, jnp.array([0., 0., 1.]), 10.), [limit, 1.])
    for bad in (np.nan, -1.):
        for limits in ([bad, .5], [.5, bad]):
            invalid = Features(jnp.array([0, 1], jnp.int32), jnp.array([0], jnp.int32),
                               jnp.array([2], jnp.int32), jnp.array(limits))
            value, status = query(tree, geometry, metadata._replace(silhouette=invalid), jnp.array([0., 0., 1.]), 10.)
            assert np.isnan(value) and int(status) == -1
    with pytest.raises(TypeError, match="feature_bound_with_data"):
        make_robin_radius(primitive, RobinOps(wp.vec3, (3,), nearest.valid, node, leaf, feature,
                                            feature_type=_CustomRobinFeatures))
