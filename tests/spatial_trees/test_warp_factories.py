"""Factory specialization against independent geometry and metric oracles."""
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.backends._warp_domain import Ray  # noqa: E402
from surfaces.explicit.accelerators.protocol import Config  # noqa: E402

from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    Angular, Euclidean, Points, Segments, Silhouette, SilhouetteEdges, Triangles,
    build_tree, make_nearest, make_ray_query,
)


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
    shifted_tree = tree._replace(
        node_ray_lower=tree.node_ray_lower.at[:, :3].add(jnp.array([0., 1, 0])),
        node_ray_upper=tree.node_ray_upper.at[:, :3].add(jnp.array([0., 1, 0])),
        node_lower=tree.node_lower+jnp.array([0., 1, 0]),
                                 node_upper=tree.node_upper+jnp.array([0., 1, 0]))
    np.testing.assert_allclose(batch(shifted_tree, changed, q)[1], [[0, np.sqrt(2)], [np.sqrt(2), 1]])
    assert nearest(tree, geometry, jnp.empty((2, 0, 2)))[0].shape == (2, 0, 2)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("cache_bounds", [False, True])
def test_warp_factory_caps_ties_empty_and_invalid(dimension, cache_bounds):
    primitive = Points(dimension)
    points = np.zeros((2, dimension), np.float32)
    points[:, 0] = [-1., 1.]
    tree, geometry = build_tree(primitive, primitive.data(points), Config(leaf_size_threshold=1))
    nearest = make_nearest(primitive, Euclidean(), cache_bounds=cache_bounds)
    queries = jnp.zeros((7, dimension)).at[4, 0].set(jnp.nan)
    caps = jnp.array([0., .5, 1., jnp.inf, 1., -1., jnp.nan])
    _, distance, ids, status = jax.jit(nearest)(tree, geometry, queries, caps)
    np.testing.assert_array_equal(status, [0, 0, 1, 1, -1, -1, -1])
    np.testing.assert_array_equal(ids, [-1, -1, 0, 0, -1, -1, -1])
    np.testing.assert_allclose(distance[2:4], 1.)
    empty_tree, empty = build_tree(primitive, primitive.data(np.empty((0, dimension))))
    assert int(nearest(empty_tree, empty, jnp.zeros(dimension))[3]) == 0
    with pytest.raises(ValueError, match="trailing shape"):
        nearest(tree, geometry, jnp.zeros(dimension-1))
    with pytest.raises(TypeError, match="Angular"):
        make_nearest(Triangles(), Angular())


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


def _assert_ray_hit(mode, result, expected):
    np.testing.assert_array_equal(result[-1], expected)
    np.testing.assert_array_equal(result[2] if mode in ("first", "all") else result[0], expected)


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
def test_segment_ray_interval_endpoints(mode):
    primitive = Segments(2)
    vertices = np.array([[-1., -1], [1., -1], [1., 1], [-1., 1]], np.float32)
    base = np.array([[1., 0], [1., 1e-30], [.75, 1.], [1., 1.]], np.float32)
    directions = np.concatenate([base*s for s in (2.**-30, 1., 2.**30, -1.)])
    times = 1/np.max(np.abs(directions), axis=1)
    previous, following = np.nextafter(times, -np.inf), np.nextafter(times, np.inf)
    maximum = np.column_stack((previous, times, following, np.full_like(times, np.inf))).ravel()
    minimum = np.column_stack((np.zeros((len(times), 3)), following)).astype(np.float32).ravel()
    directions = np.repeat(directions, 4, axis=0)
    expected = np.tile([False, True, True, False], len(times))
    query = jax.jit(make_ray_query(primitive, mode), static_argnames=("max_hits",))
    for reverse in (False, True):
        a, b = vertices, np.roll(vertices, -1, axis=0)
        tree, data = build_tree(primitive, primitive.data(b, a) if reverse else primitive.data(a, b),
                                Config(leaf_size_threshold=1))
        result = query(tree, data, np.zeros_like(directions), directions, maximum,
                       min_t=minimum, **({"max_hits": 2} if mode == "all" else {}))
        _assert_ray_hit(mode, result, expected)
        if mode in ("first", "all"):
            returned = np.asarray(result[0])[:, 0] if mode == "all" else np.asarray(result[0])
            np.testing.assert_array_equal(returned[expected], np.repeat(times, 4)[expected])


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
def test_ray_narrow_leaf_roundoff(mode, dimension):
    vertices = np.array([[0xbf2948b8, 0x3f4009f0], [0xbf296e69, 0x3f3fe8af],
                         [0xbf299415, 0x3f3fc767]], np.uint32).view(np.float32)
    origin, direction = np.array([[3193768802, 1019335998], [3204854055, 1062863525]],
                                 np.uint32).view(np.float32)
    primitive = Segments(2) if dimension == 2 else Triangles()
    a, b = np.vstack((vertices[:2], [10., 10.])), np.vstack((vertices[1:], [11., 11.]))
    if dimension == 2:
        raw = primitive.data(a, b)
    else:
        a, b = (np.pad(v, ((0, 0), (0, 1))) for v in (a, b))
        c = b+np.array([0., 0., 1.], np.float32)
        raw = primitive.indexed(np.stack((b, c, a), axis=1).reshape(-1, 3), np.arange(9).reshape(3, 3))
        origin, direction = np.append(origin, np.float32(.25)), np.append(direction, np.float32(.125))
    tree, data = build_tree(primitive, raw, Config(leaf_size_threshold=2))
    # The third primitive forces a child test whose unpadded slabs invert by one ULP.
    assert np.any(np.asarray(tree.node_leaf_count) == 2)
    result = jax.jit(make_ray_query(primitive, mode), static_argnames=("max_hits",))(
        tree, data, origin, direction, 10., **({"max_hits": 1} if mode == "all" else {}))
    _assert_ray_hit(mode, result, 1)
    if mode in ("first", "all"):
        assert np.asarray(result[1]).reshape(-1)[0] == 1
        expected = np.float32(0.8518937471553134)
        # First-hit triangles compute t in FP32, unlike the double segment path.
        np.testing.assert_allclose(np.asarray(result[0]).reshape(-1)[0], expected,
                                   atol=np.spacing(expected) if dimension == 3 and mode == "first" else 0., rtol=0.)


@pytest.mark.parametrize("scale", [2.**-120, 2.**-100, 1., 2.**100])
@pytest.mark.parametrize("translated", [False, True])
def test_segment_ray_filtered_endpoint_neighbours(scale, translated):
    primitive = Segments(2)
    shift = 2.**20*scale if translated else 0.
    slope = np.float32(2.**-20)
    tree, data = build_tree(primitive, primitive.data(
        [[shift+scale, -scale*slope]], [[shift+scale, scale*slope]]))
    directions = np.column_stack((np.ones(3, np.float32),
        [np.nextafter(slope, np.float32(0)), slope,
         np.nextafter(slope, np.float32(np.inf))]))
    # The CPU's existing DAZ conversion loses subnormal endpoint components.
    expected = ([False, False, False] if scale == 2.**-120
                and jax.devices()[0].platform == "cpu" else [True, True, False])
    for mode in ("first", "any"):
        result = jax.jit(make_ray_query(primitive, mode))(
            tree, data, np.tile(np.array([shift, 0], np.float32), (3, 1)), directions, np.float32(2*scale))
        _assert_ray_hit(mode, result, expected)
        if mode == "first":
            np.testing.assert_array_equal(np.asarray(result[0])[expected], np.float32(scale))


@pytest.mark.parametrize("bits,exponent,cpu_hits,gpu_hits", [
    ([0x31699a4a, 0xb5dcb961, 0x008d716b, 0x5b62ba12,
      0x969072b0, 0xf7860f0b, 0x049ffe95, 0x658ee884], 33,
     (False, False, False, False), (True, True, True, True)),
    ([0xef963dbe, 0x2338f59b, 0x259bfad0, 0x9cd1c63a,
      0x98d4b857, 0x18dc93ad, 0x5f1731bd, 0xc6d22ad4], 0,
     (True, True, False, False), (True, True, False, False)),
])
def test_segment_ray_unbalanced_endpoint_margin_uses_double_fallback(bits, exponent, cpu_hits, gpu_hits):
    a, b, origin, direction = np.array(bits, np.uint32).view(np.float32).reshape(4, 2)
    direction = np.ldexp(direction, exponent)
    primitive = Segments(2)
    tree, data = build_tree(primitive, primitive.data(a[None], b[None]))
    # Preserve measured original decisions where double cancellation differs
    # across CPU/CUDA, or root-box rejection differs from the closed predicate.
    expected = cpu_hits if jax.devices()[0].platform == "cpu" else gpu_hits
    for mode, hit in zip(("first", "any", "count", "all"), expected, strict=True):
        result = jax.jit(make_ray_query(primitive, mode), static_argnames=("max_hits",))(
            tree, data, origin, direction, 10., **({"max_hits": 1} if mode == "all" else {}))
        _assert_ray_hit(mode, result, hit)


@pytest.mark.parametrize("mode", ["first", "all"])
def test_segment_ray_axis_edge_factor_rounding(mode):
    primitive = Segments(2)
    query = jax.jit(make_ray_query(primitive, mode), static_argnames=("max_hits",))
    cases = [
        ((0x3f9e402e, 0x32dea7f8, 0x3f57f6f4), 0., 0x3fbb964a),
        ((0x3f9e402e, 0x32dea7f8, 0x3f57f6f4), 2.**-25, 0x3fbb964b),
        ((0x3f800000, 0x00000000, 0x3f15ca56), 0., 0x3fdac251),
        ((0x3f800000, 0xb3800000, 0x3f15ca56), 0., 0x3fdac252),
    ]
    # Rounding either a nonpower-of-two edge or an inexact offset to FP32
    # before cancellation/division changes these results by one FP32 ULP.
    for bits, extra, expected in cases:
        coordinate, offset, speed = np.array(bits, np.uint32).view(np.float32)
        a = np.array([[coordinate, -extra], [-extra, coordinate]], np.float32)
        b = np.array([[coordinate, 1.], [1., coordinate]], np.float32)
        origins = np.array([[offset, .5], [.5, offset]], np.float32)
        directions = np.array([[speed, 0.], [0., speed]], np.float32)
        tree, data = build_tree(primitive, primitive.data(a, b))
        result = query(tree, data, origins, directions, **({"max_hits": 1} if mode == "all" else {}))
        _assert_ray_hit(mode, result, 1)
        np.testing.assert_array_equal(np.asarray(result[0]).view(np.uint32), expected)


def test_warp_factory_custom_ray_callback():
    primitive = Segments(2)
    original = primitive.intersect

    @wp.func
    def intersect(ray: Ray, minimum: float, maximum: float,
                  data: Any, slot: int, half_open: bool, store_times: bool):
        if data.a[slot][0] == 2.0:
            return wp.vec2(0.0)
        return wp.static(original)(ray, minimum, maximum, data, slot, half_open, store_times)

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
    @dataclass(frozen=True)
    class InvalidCandidate:
        def bind(self, primitive):
            @wp.func
            def closest(query: wp.vec3, item: Any):
                score = wp.length_sq(query-item.a)
                return item.a, wp.where(item.a[0] == 0.0 and query[0] == 0.0, wp.nan, score)
            return replace(Euclidean().bind(primitive), closest=closest)

    tree, data = build_tree(primitive, primitive.data([[0., 0, 0], [1., 0, 0]]))
    result = make_nearest(primitive, InvalidCandidate())(tree, data, jnp.array([[0., 0, 0], [2., 0, 0]]))
    np.testing.assert_array_equal(result[1], [jnp.inf, 1.])
    np.testing.assert_array_equal(result[2], [-1, 1])
    np.testing.assert_array_equal(result[3], [-1, 1])
    for make, argument in [(make_nearest, Euclidean()), (make_ray_query, "count")]:
        for capacity in (0, 65):
            with pytest.raises(ValueError, match="stack"):
                make(primitive, argument, stack_capacity=capacity)


def test_warp_factory_negative_infinite_bound_disables_pruning():
    @wp.func
    def bound(query: wp.vec3, lo: wp.vec3, hi: wp.vec3):
        return -wp.inf

    @dataclass(frozen=True)
    class Unpruned:
        def bind(self, primitive):
            return replace(Euclidean().bind(primitive), node_bound=bound)

    primitive = Points()
    tree, data = build_tree(primitive, primitive.data(np.random.default_rng(4).normal(size=(83, 3))))
    queries = jnp.array([[0., 0, 0], [8., 0, 0]])
    normal, unpruned = (jax.jit(make_nearest(primitive, metric)) for metric in (Euclidean(), Unpruned()))
    for cap in (jnp.inf, .5):
        for actual, expected in zip(unpruned(tree, data, queries, cap), normal(tree, data, queries, cap), strict=True):
            np.testing.assert_array_equal(actual, expected)


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


@pytest.mark.parametrize("primitive", [Points(2), Points(3), Segments(2), Segments(3)], ids=lambda p: f"{p.name}{p.dimension}")
def test_warp_tree_splits_collinear_geometry(primitive):
    for scale in (1., 1e-8):
        a = np.zeros((48, primitive.dimension), np.float32)
        a[:, -1] = np.arange(len(a))*scale
        b = a.copy()
        b[:, -1] += .25*scale
        raw = primitive.data(a) if primitive.name == "points" else primitive.data(a, b)
        tree, _ = build_tree(primitive, raw, Config(leaf_size_threshold=4))
        assert np.max(tree.node_leaf_count) <= 4
        np.testing.assert_array_equal(np.sort(tree.prim_ids), np.arange(len(a)))


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
