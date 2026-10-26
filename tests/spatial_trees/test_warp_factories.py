"""Factory specialization against independent geometry and metric oracles."""
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    Angular, Euclidean, Points, Segments, Silhouette, SilhouetteEdges, Triangles,
    build_tree, make_nearest, make_ray_query,
)


@pytest.mark.parametrize("primitive", [Points(2), Points(3), Segments(2), Segments(3), Triangles()])
def test_warp_factory_nearest_oracle(primitive):
    rng = np.random.default_rng(73)
    dim = primitive.dimension
    vertices = rng.normal(size=(87, dim)).astype(np.float32)
    if primitive.name == "points":
        geometry = primitive.data(vertices)
    elif primitive.name == "segments":
        geometry = primitive.data(vertices, vertices+.5*rng.normal(size=vertices.shape))
    else:
        vertices = vertices[:81]
        geometry = primitive.data(vertices, np.arange(len(vertices)).reshape(-1, 3))
    tree, packed = build_tree(primitive, geometry)
    queries = rng.normal(size=(39, dim)).astype(np.float32)
    nearest = make_nearest(primitive, Euclidean())
    point, distance, index, status = jax.jit(nearest)(tree, packed, queries)
    np.testing.assert_array_equal(status, 1)
    for i, q in enumerate(queries.astype(np.float64)):
        if primitive.name == "points":
            candidates = np.asarray(geometry.a, np.float64)
        elif primitive.name == "segments":
            a, b = (np.asarray(v, np.float64) for v in geometry)
            edge = b-a
            t = np.clip(np.sum((q-a)*edge, axis=1)/np.sum(edge*edge, axis=1), 0, 1)
            candidates = a+t[:, None]*edge
        else:
            triangles = np.asarray(vertices, np.float64).reshape(-1, 3, 3)
            candidates = trimesh.triangles.closest_point(triangles, np.broadcast_to(q, (len(triangles), 3)))
        distances = np.linalg.norm(candidates-q, axis=1)
        expected = np.argmin(distances)
        assert int(index[i]) == expected
        np.testing.assert_allclose(distance[i], distances[expected], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(point[i], candidates[expected], rtol=3e-5, atol=3e-6)
    assert make_nearest(primitive, Euclidean()) is nearest


@dataclass(frozen=True)
class WeightedMetric:
    weights: tuple = (1., 25., .2)

    def bind(self, primitive):
        weights = wp.vec3(*self.weights)

        @wp.func
        def bound(q: wp.vec3, lo: wp.vec3, hi: wp.vec3):
            delta = wp.max(wp.max(lo-q, wp.vec3(0.0)), q-hi)
            return wp.dot(wp.cw_mul(delta, delta), wp.static(weights))

        @wp.func
        def closest(q: wp.vec3, item: Any):
            delta = q-item.a
            return item.a, wp.dot(wp.cw_mul(delta, delta), wp.static(weights))

        return replace(Euclidean().bind(primitive), node_bound=bound, closest=closest)


def test_warp_factory_custom_metric_and_static_isolation():
    primitive = Points()
    rng = np.random.default_rng(81)
    data = primitive.data(rng.normal(size=(101, 3)))
    tree, geometry = build_tree(primitive, data)
    queries = rng.normal(size=(47, 3)).astype(np.float32)
    variants = [(Euclidean(), (1, 1, 1)), (WeightedMetric(), (1, 25, .2)),
                (WeightedMetric((9, .1, 3)), (9, .1, 3))]
    for metric, weights in variants+variants[::-1]:
        got = jax.jit(make_nearest(primitive, metric))(tree, geometry, queries)
        d2 = np.sum((queries[:, None]-np.asarray(data.a)[None])**2*np.array(weights), axis=-1)
        np.testing.assert_array_equal(got[2], d2.argmin(axis=1))
        np.testing.assert_allclose(got[1], np.sqrt(d2.min(axis=1)), rtol=3e-6)


def test_warp_factory_batching_dynamic_data_and_cpu():
    primitive = Segments(2)
    tree, geometry = build_tree(primitive, primitive.data([[0., 0]], [[2., 0]]))
    nearest = make_nearest(primitive, Euclidean())
    q = jnp.array([[[1., 1], [3., 0]], [[-1., 0], [0., 2]]])
    batch = jax.jit(jax.vmap(jax.vmap(nearest, in_axes=(None, None, 0)), in_axes=(None, None, 0)))
    expected = batch(tree, geometry, q)
    np.testing.assert_allclose(expected[1], [[1, 1], [1, 2]])
    cpu = jax.devices("cpu")[0]
    on_cpu = batch(*(jax.device_put(v, cpu) for v in (tree, geometry, q)))
    for a, b in zip(expected, on_cpu, strict=True):
        np.testing.assert_array_equal(a, b)
    changed = geometry._replace(a=geometry.a+jnp.array([0., 1]), b=geometry.b+jnp.array([0., 1]))
    shifted_tree = tree._replace(node_lower=tree.node_lower+jnp.array([0., 1, 0]),
                                 node_upper=tree.node_upper+jnp.array([0., 1, 0]))
    np.testing.assert_allclose(batch(shifted_tree, changed, q)[1], [[0, np.sqrt(2)], [np.sqrt(2), 1]])
    assert nearest(tree, geometry, jnp.empty((2, 0, 2)))[0].shape == (2, 0, 2)


def test_warp_factory_caps_ties_empty_and_invalid():
    primitive = Points()
    tree, geometry = build_tree(primitive, primitive.data([[-1., 0, 0], [1., 0, 0]]))
    nearest = make_nearest(primitive, Euclidean())
    queries = jnp.zeros((7, 3)).at[4, 0].set(jnp.nan)
    caps = jnp.array([0., .5, 1., jnp.inf, 1., -1., jnp.nan])
    _, distance, ids, status = jax.jit(nearest)(tree, geometry, queries, caps)
    np.testing.assert_array_equal(status, [0, 0, 1, 1, -1, -1, -1])
    np.testing.assert_array_equal(ids, [-1, -1, 0, 0, -1, -1, -1])
    np.testing.assert_allclose(distance[2:4], 1.)
    empty_tree, empty = build_tree(primitive, primitive.data(np.empty((0, 3))))
    assert int(nearest(empty_tree, empty, jnp.zeros(3))[3]) == 0
    with pytest.raises(ValueError, match="trailing shape"):
        nearest(tree, geometry, jnp.zeros(2))
    with pytest.raises(TypeError, match="Angular"):
        make_nearest(Points(), Angular())


@wp.struct
class IndexedData:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=wp.int32)


@wp.func
def _indexed_load(data: Any, slot: int):
    return data.vertices[data.indices[slot]]


@wp.func
def _point_bounds(item: wp.vec3):
    return item, item


@wp.func
def _point_closest(query: wp.vec3, item: wp.vec3):
    return item


@dataclass(frozen=True)
class IndexedPoints:
    dimension: int = 3
    point_type: object = wp.vec3
    data_type: object = IndexedData
    load: object = _indexed_load
    bounds_of_item: object = _point_bounds
    closest_euclidean: object = _point_closest

    def bounds(self, data):
        points = np.asarray(data.vertices)[np.asarray(data.indices)]
        return points, points

    def pack(self, data, ids):
        return data._replace(indices=data.indices[ids])


def test_warp_factory_custom_indexed_primitive():
    Data = namedtuple("IndexedArrays", ["vertices", "indices"])
    primitive = IndexedPoints()
    data = Data(jnp.array([[0., 0, 0], [3., 0, 0], [1., 2, 0], [-1., 1, 0]]),
                jnp.array([3, 1, 0], jnp.int32))
    tree, geometry = build_tree(primitive, data)
    assert geometry.vertices is data.vertices
    point, distance, index, status = jax.jit(make_nearest(primitive, Euclidean()))(
        tree, geometry, jnp.array([[.1, 0, 0], [-2., 1, 0], [4., 0, 0]]))
    np.testing.assert_array_equal(index, [2, 0, 1])
    np.testing.assert_allclose(distance, [.1, 1, 1])
    np.testing.assert_array_equal(status, 1)
    np.testing.assert_array_equal(point, [[0, 0, 0], [-1, 1, 0], [3, 0, 0]])


def _angle(point, origin, direction):
    view = point-origin
    return np.arctan2(np.linalg.norm(np.cross(view, direction)), np.dot(view, direction))


def test_warp_factory_angular_segments_oracle():
    from scipy.optimize import minimize_scalar

    primitive = Segments()
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=(2, 29, 3)).astype(np.float32)
    tree, geometry = build_tree(primitive, primitive.data(a, b))
    origins, directions = rng.normal(size=(2, 12, 3)).astype(np.float32)
    queries = np.stack((origins, directions), axis=1)
    nearest = jax.jit(make_nearest(primitive, Angular()))
    got = nearest(tree, geometry, queries)
    cpu = jax.devices("cpu")[0]
    on_cpu = nearest(*(jax.device_put(v, cpu) for v in (tree, geometry, queries)))
    for actual, expected in zip(on_cpu, got, strict=True):
        np.testing.assert_allclose(actual, expected, atol=5e-6)
    for i, (o, d) in enumerate(zip(origins.astype(np.float64), directions.astype(np.float64))):
        scores = []
        for start, end in zip(a.astype(np.float64), b.astype(np.float64)):
            def objective(t):
                return _angle(start+t*(end-start), o, d)
            optimum = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
            scores.append(min(objective(0), objective(1), optimum.fun))
        assert int(got[2][i]) == np.argmin(scores)
        np.testing.assert_allclose(got[1][i], min(scores), atol=5e-6)
        np.testing.assert_allclose(_angle(np.asarray(got[0][i]), o, d), got[1][i], atol=5e-6)


@pytest.mark.parametrize("metric,queries", [
    (Euclidean(), jnp.zeros(3)),
    (Angular(), jnp.array([[0., 0, 0], [1., 0, 0]])),
])
def test_warp_factory_silhouette_eligibility(metric, queries):
    primitive = SilhouetteEdges()
    data = primitive.data([[1., -1, 0], [2., -1, 0]], [[1., 1, 0], [2., 1, 0]],
                          [[-1., 0, 0], [0., 1, 0]], [[-1., 0, 0], [0., -1, 0]])
    tree, geometry = build_tree(primitive, data)
    nearest = make_nearest(primitive, metric, predicate=Silhouette())
    point, distance, edge, status = jax.jit(nearest)(tree, geometry, queries)
    assert int(edge) == 1 and int(status) == 1
    np.testing.assert_allclose(point, [2, 0, 0], atol=1e-6)
    assert float(distance) == (2 if isinstance(metric, Euclidean) else 0)


def _square_segments():
    primitive = Segments(2)
    a = jnp.array([[-1., -1], [1., -1], [1., 1], [-1., 1]])
    return primitive, *build_tree(primitive, primitive.data(a, jnp.roll(a, -1, axis=0)))


def test_warp_factory_segment_rays_ownership_bounds_and_overflow():
    primitive, tree, data = _square_segments()
    origin, direction = jnp.zeros(2), jnp.array([1., 0])
    count = make_ray_query(primitive, "count")
    for d in (direction, -direction, jnp.array([1., 1]), jnp.array([-1., -1])):
        np.testing.assert_array_equal(count(tree, data, origin, d), [1, 1])
        np.testing.assert_array_equal(count(tree, data, origin, d, min_t=-jnp.inf), [2, 1])
    all_hits = make_ray_query(primitive, "all")
    times, faces, total, status = all_hits(tree, data, origin, direction, min_t=-jnp.inf, max_hits=1)
    assert int(total) == 2 and int(status) == -2
    assert np.isfinite(times).all() and np.all(np.asarray(faces) >= 0)
    times, _, total, status = all_hits(tree, data, origin, direction, min_t=-jnp.inf, max_hits=2)
    np.testing.assert_array_equal(np.sort(times), [-1, 1])
    assert int(total) == 2 and int(status) == 1
    np.testing.assert_array_equal(count(tree, data, origin, direction, 1., min_t=1.), [1, 1])
    first = make_ray_query(primitive, "first", exclude=True)
    t, face, hit, status = first(tree, data, origin, direction)
    assert float(t) == 1 and bool(hit) and int(status) == 1
    assert not bool(first(tree, data, origin, direction, exclude_face=face)[2])
    np.testing.assert_array_equal(count(tree, data, origin, direction, exclude_face=face), [0, 0])
    for d in (jnp.zeros(2), jnp.array([jnp.nan, 0])):
        result = all_hits(tree, data, origin, d, max_hits=2)
        np.testing.assert_array_equal(result[2:], -1)
        assert np.isposinf(result[0]).all() and np.all(np.asarray(result[1]) == -1)


def test_warp_factory_ray_batching_and_cpu():
    primitive, tree, geometry = _square_segments()
    all_hits = make_ray_query(primitive, "all")
    def query(t, g, o, d):
        return all_hits(t, g, o, d, max_hits=4)
    run = jax.jit(jax.vmap(jax.vmap(query, in_axes=(None, None, 0, 0)),
                            in_axes=(None, None, 0, 0)))
    origins = jnp.array([[[0., 0], [-2., 0]], [[2., 0], [0., .2]]])
    directions = jnp.broadcast_to(jnp.array([1., 0]), origins.shape)
    result = run(tree, geometry, origins, directions)
    np.testing.assert_array_equal(result[2], [[1, 2], [0, 1]])
    cpu = jax.devices("cpu")[0]
    expected = run(*(jax.device_put(v, cpu) for v in (tree, geometry, origins, directions)))
    for a, b in zip(result, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
def test_warp_factory_generic_triangle_ray_matches_specialization(mode):
    primitive = Triangles()
    mesh = trimesh.creation.icosphere(subdivisions=2)
    tree, geometry = build_tree(primitive, primitive.data(mesh.vertices, mesh.faces))
    rng = np.random.default_rng(92)
    origins = rng.normal(size=(83, 3)).astype(np.float32)
    directions = rng.normal(size=(83, 3)).astype(np.float32)
    options = {"max_hits": 4, "min_t": -jnp.inf} if mode == "all" else {}
    generic = make_ray_query(primitive, mode, stack_capacity=32)
    specialized = make_ray_query(primitive, mode)
    a = generic(tree, geometry, origins, directions, **options)
    b = specialized(tree, geometry, origins, directions, **options)
    if mode == "all":
        np.testing.assert_array_equal(np.sort(a[1], axis=-1), np.sort(b[1], axis=-1))
        np.testing.assert_allclose(np.sort(a[0], axis=-1), np.sort(b[0], axis=-1), atol=4e-6, rtol=4e-5)
        np.testing.assert_array_equal(a[2:], b[2:])
    else:
        for actual, expected in zip(a, b, strict=True):
            np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=4e-5)


def test_warp_factory_custom_ray_callback():
    primitive = Segments(2)
    original = primitive.intersect

    @wp.func
    def intersect(origin: wp.vec2, direction: wp.vec2, minimum: float, maximum: float,
                  data: Any, slot: int, half_open: bool, store_times: bool):
        if data.a[slot][0] == 2.0:
            return wp.vec2(0.0)
        return wp.static(original)(origin, direction, minimum, maximum, data, slot, half_open, store_times)

    custom = replace(primitive, intersect=intersect)
    geometry = custom.data([[1., -1], [2., -1], [3., -1]], [[1., 1], [2., 1], [3., 1]])
    tree, data = build_tree(custom, geometry)
    result = make_ray_query(custom, "all")(tree, data, jnp.zeros(2), jnp.array([1., 0]), max_hits=3)
    np.testing.assert_array_equal(result[2:], [2, 1])
    np.testing.assert_array_equal(np.sort(result[1]), [-1, 0, 2])
    np.testing.assert_array_equal(np.sort(result[0]), [1, 3, jnp.inf])


def test_warp_factory_numeric_and_stack_failures():
    primitive = Points()
    tree, data = build_tree(primitive, primitive.data([[0., 0, 0]]))
    nearest = make_nearest(primitive, Euclidean())
    assert int(nearest(tree, data, jnp.array([1.e30, 0, 0]))[3]) == -1
    assert int(nearest(tree, data, jnp.array([1.e30, 0, 0]), .5)[3]) == 0
    rng = np.random.default_rng(4)
    tree, data = build_tree(primitive, primitive.data(rng.normal(size=(83, 3))))
    limited = make_nearest(primitive, Euclidean(), stack_capacity=1)
    assert int(limited(tree, data, jnp.zeros(3))[3]) == -1
    for make, argument in [(make_nearest, Euclidean()), (make_ray_query, "count")]:
        for capacity in (0, 65):
            with pytest.raises(ValueError, match="stack"):
                make(primitive, argument, stack_capacity=capacity)


def test_warp_factory_outward_bounds_and_invalid_geometry():
    class RoundedBounds(IndexedPoints):
        def bounds(self, data):
            return np.full((1, 3), .1), np.full((1, 3), .7)

    Data = namedtuple("IndexedArrays", ["vertices", "indices"])
    data = Data(jnp.full((1, 3), .5), jnp.array([0], jnp.int32))
    tree, _ = build_tree(RoundedBounds(), data)
    assert np.all(np.asarray(tree.node_lower, np.float64) <= .1)
    assert np.all(np.asarray(tree.node_upper, np.float64) >= .7)
    for values in ([[jnp.inf, 0, 0]], [[jnp.nan, 0, 0]]):
        with pytest.raises(ValueError, match="finite"):
            Points().data(values)
    with pytest.raises(ValueError, match="index"):
        Triangles().data([[0., 0, 0]], [[0, 1, 2]])


@wp.struct
class MaskedData:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=wp.int32)
    node_mask: wp.array(dtype=wp.bool)


def test_warp_factory_metric_uses_dynamic_node_data():
    @dataclass(frozen=True)
    class MaskedPoints(IndexedPoints):
        data_type: object = MaskedData

    @dataclass(frozen=True)
    class MaskedMetric:
        def bind(self, primitive):
            base = Euclidean().bind(primitive)

            @wp.func
            def bound(q: wp.vec3, tree: Any, data: Any, node: int):
                if not data.node_mask[node]:
                    return wp.inf
                return wp.static(base.node_bound)(q, tree.node_lower[node], tree.node_upper[node])

            return replace(base, node_bound_with_data=bound)

    Data = namedtuple("MaskedArrays", ["vertices", "indices", "node_mask"])
    primitive = MaskedPoints()
    data = Data(jnp.zeros((1, 3)), jnp.array([0], jnp.int32), jnp.empty((0,), jnp.bool_))
    tree, geometry = build_tree(primitive, data)
    nearest = jax.jit(make_nearest(primitive, MaskedMetric()))
    geometry = geometry._replace(node_mask=jnp.ones((1,), jnp.bool_))
    assert int(nearest(tree, geometry, jnp.zeros(3))[3]) == 1
    geometry = geometry._replace(node_mask=jnp.zeros((1,), jnp.bool_))
    assert int(nearest(tree, geometry, jnp.zeros(3))[3]) == 0


def test_warp_factory_scalar_query_type():
    @dataclass(frozen=True)
    class AxisQuery:
        def bind(self, primitive):
            base = Euclidean().bind(primitive)

            @wp.func
            def valid(query: float):
                return wp.isfinite(query)

            @wp.func
            def bound(query: float, lo: wp.vec3, hi: wp.vec3):
                return wp.static(base.node_bound)(wp.vec3(query, 0.0, 0.0), lo, hi)

            @wp.func
            def closest(query: float, item: Any):
                delta = item.a-wp.vec3(query, 0.0, 0.0)
                return item.a, wp.dot(delta, delta)

            @wp.func
            def cull(query: float, score: float):
                return score

            return replace(base, query_type=wp.float32, query_shape=(), valid=valid,
                           node_bound=bound, closest=closest, cull_limit=cull)

    primitive = Points()
    tree, geometry = build_tree(primitive, primitive.data([[0., 0, 0], [2., 0, 0]]))
    nearest = jax.jit(make_nearest(primitive, AxisQuery()))
    np.testing.assert_array_equal(nearest(tree, geometry, jnp.array([[0., 1., 2.]]))[2], [[0, 0, 1]])
    np.testing.assert_array_equal(nearest(tree, geometry, jnp.float32(3))[1], 1)
