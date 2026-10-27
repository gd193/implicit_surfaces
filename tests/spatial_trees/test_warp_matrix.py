"""Every supported builtin combination against independent geometry oracles."""
from functools import partial
from dataclasses import replace

import jax
import numpy as np
import pytest
import trimesh
from scipy.optimize import minimize_scalar

wp = pytest.importorskip("warp")

from scripts.bench.spatial_trees._warp_cases import (  # noqa: E402
    CASES, PRIMITIVES, METRICS, RAYS, BOUNDARIES, QueryCase, arguments, query_factory, scene,
)
from scripts.bench.spatial_trees.warp_query_matrix import kernel_inputs  # noqa: E402
from scripts.bench.spatial_trees._warp_timing import _make_case  # noqa: E402
from ._robin_oracle import brute_robin_radius  # noqa: E402


def _angle(point, origin, direction):
    view = point-origin
    if not np.any(view):
        return np.inf
    cross = np.cross(np.pad(view, (0, 3-len(view))), np.pad(direction, (0, 3-len(direction))))
    return np.arctan2(np.linalg.norm(cross), view @ direction)


def _segments(raw):
    a = np.asarray(raw.a, np.float64)
    return a, np.asarray(raw.b, np.float64) if hasattr(raw, "b") else a


def _segment_point(a, b, origin, direction=None):
    edge = b-a
    if direction is None:
        length2 = np.sum(edge*edge, axis=-1)
        t = np.clip(np.sum((origin-a)*edge, axis=-1)/np.where(length2 > 0, length2, 1.), 0, 1)
        return a+np.expand_dims(t, -1)*edge
    if np.array_equal(a, b):
        return a
    def objective(t):
        return _angle(a+t*edge, origin, direction)
    optimum = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    return a+min((0., 1., optimum.x), key=objective)*edge


def _candidates(case, raw, query):
    angular = case.metric == "angular"
    origin, direction = query if angular else (query, None)
    if case.primitive == "triangles":
        triangles = np.asarray(raw.vertices, np.float64)[np.asarray(raw.indices)]
        points = trimesh.triangles.closest_point(triangles, np.broadcast_to(origin, (len(triangles), 3)))
    else:
        a, b = _segments(raw)
        points = (np.asarray([_segment_point(start, end, origin, direction) for start, end in zip(a, b, strict=True)])
                  if angular else _segment_point(a, b, origin))
    distances = (np.array([_angle(p, origin, direction) for p in points]) if angular
                 else np.linalg.norm(points-origin, axis=1))
    return points, distances


def _assert_member(case, raw, point, index):
    _, distances = _candidates(QueryCase(case.primitive, "nearest", "euclidean"), raw, point.astype(np.float64))
    assert 0 <= index < len(distances) and distances[index] < 8e-6


def _feature_distances(boundary, query, metric):
    origin, direction = query if metric == "angular" else (query, None)
    origin = np.pad(origin, (0, 3-len(origin)))
    if direction is not None:
        direction = np.pad(direction, (0, 3-len(direction)))
    meta = boundary.silhouette
    distances = np.full(len(meta.feature_a), np.inf)
    for i, (a, b, n0, n1) in enumerate(zip(*(np.asarray(x, np.float64) for x in
                              (meta.feature_a, meta.feature_b, meta.feature_n0, meta.feature_n1)), strict=True)):
        if ((origin-a) @ n0)*((origin-a) @ n1) > 0:
            continue
        point = _segment_point(a, b, origin, direction)
        distances[i] = np.linalg.norm(point-origin) if direction is None else _angle(point, origin, direction)
    return distances


def _robin2(raw, boundary, point, cap):
    result = min(cap, _feature_distances(boundary, point, "euclidean").min(initial=np.inf))
    for a, b in zip(*_segments(raw), strict=True):
        edge = b-a
        height = abs((point-a) @ np.array([edge[1], -edge[0]]))/np.linalg.norm(edge)
        def radius(t):
            distance2 = np.sum((a+t*edge-point)**2)
            if distance2 == 0:
                return 0.
            return np.sqrt(distance2)*np.exp(min(height/distance2, 700.))
        optimum = minimize_scalar(radius, bounds=(0, 1), method="bounded")
        result = min(result, radius(0), radius(1), optimum.fun)
    return result


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_builtin_query_matrix(case):
    geometry = scene(case.primitive)
    args, kwargs = arguments(case, geometry, 19)
    if case.query == "overlap":
        args = (*args[:2], args[2].at[-2].set(np.nan).at[-1].set(2.), args[3].at[-1].set(1.))
    got = tuple(map(np.asarray, jax.jit(partial(query_factory(case), **kwargs))(*args)))
    call, inputs, shapes = kernel_inputs(case, geometry, 19)
    device = wp.device_from_jax(jax.devices()[0])
    timed = _make_case(case.query, "custom", call.kernel, inputs, 19, device, call.block_dim, shapes)
    wp.launch(call.kernel, 19, inputs=timed.inputs, outputs=timed.outputs, device=device, block_dim=call.block_dim)
    direct = tuple(v.numpy() for v in timed.outputs)
    if case.query == "star":
        direct = (np.minimum(direct[1], .75), np.where(direct[3] >= 0, 1, -1))
    for a, b in zip(direct, got, strict=True):
        if case.query == "overlap":
            a, b = a[:-2], b[:-2]
        if np.issubdtype(a.dtype, np.floating):
            np.testing.assert_allclose(a, b, atol=5e-6, rtol=3e-5)
        else:
            np.testing.assert_array_equal(a, b)
    raw, primitive = geometry.raw, PRIMITIVES[case.primitive]
    if case.query == "nearest":
        for i, query in enumerate(np.asarray(args[2], np.float64)):
            candidates, distances = _candidates(case, raw, query)
            expected = distances.min()
            found = expected <= float(args[3][i])
            assert got[3][i] == int(found)
            if found:
                index = got[2][i]
                assert 0 <= index < len(candidates)
                np.testing.assert_allclose(got[1][i], expected, atol=5e-6, rtol=3e-5)
                # Angular minima can tie across primitives or along a whole segment.
                a, b = (candidates[index], candidates[index]) if case.primitive == "triangles" else tuple(v[index] for v in _segments(raw))
                np.testing.assert_allclose(got[0][i], _segment_point(a, b, got[0][i]), atol=8e-6, rtol=3e-5)
                score = _angle(got[0][i], *query) if case.metric == "angular" else np.linalg.norm(got[0][i]-query)
                np.testing.assert_allclose(score, expected, atol=5e-6, rtol=3e-5)
            else:
                assert np.isposinf(got[1][i]) and got[2][i] == -1
    elif case.query == "overlap":
        lo, hi = primitive.bounds(raw)
        lower, upper = map(np.asarray, args[2:])
        hits = np.any(np.all((lo[None] <= upper[:, None]) & (hi[None] >= lower[:, None]), axis=-1), axis=1)
        np.testing.assert_array_equal(got[0], hits)
        valid = np.isfinite(lower).all(axis=1) & np.all(lower <= upper, axis=1)
        np.testing.assert_array_equal(got[1], np.where(valid, hits.astype(int), -1))
    elif case.query in RAYS:
        origin, direction = map(np.asarray, args[2:])
        a, b = (-1-origin)/direction, (1-origin)/direction
        enter, leave = np.min(np.stack((a, b)), axis=0).max(axis=1), np.max(np.stack((a, b)), axis=0).min(axis=1)
        times = np.column_stack((enter, leave))
        lower = np.broadcast_to(np.asarray(kwargs.get("min_t", 0.)), (len(origin),))
        valid = (enter <= leave)[:, None] & (times >= lower[:, None]) & (times <= 10.)
        times = np.sort(np.where(valid, times, np.inf), axis=1)
        count = valid.sum(axis=1)
        np.testing.assert_array_equal(got[-1], (count > 0).astype(int))
        if case.query == "first":
            np.testing.assert_allclose(got[0], np.minimum(times[:, 0], 10.), atol=5e-6)
            np.testing.assert_array_equal(got[2], count > 0)
            np.testing.assert_array_equal(got[1] >= 0, count > 0)
        elif case.query == "all":
            np.testing.assert_array_equal(got[2], count)
            np.testing.assert_allclose(np.sort(got[0], axis=1), np.pad(times, ((0, 0), (0, 2)), constant_values=np.inf), atol=5e-6)
            np.testing.assert_array_equal((got[1] >= 0).sum(axis=1), count)
        else:
            np.testing.assert_array_equal(got[0], count > 0 if case.query == "any" else count)
        if case.query in ("first", "all"):
            for o, d, ts, ids in zip(origin, direction, got[0], got[1], strict=True):
                for t, index in zip(np.atleast_1d(ts), np.atleast_1d(ids), strict=True):
                    if index >= 0:
                        _assert_member(case, raw, o+t*d, index)
                    else:
                        assert index == -1
    elif case.query == "classify":
        points = np.asarray(args[3])
        expected = np.where(np.max(np.abs(points), axis=1) < 1, -1., 1.)
        np.testing.assert_array_equal(got, [expected, np.ones(len(points))])
    elif case.query in ("silhouette", "star"):
        queries = np.asarray(args[3], np.float64)
        distances = np.array([_feature_distances(geometry.boundary, q, case.metric) for q in queries])
        expected = distances.min(axis=1, initial=np.inf)
        caps = np.asarray(args[4])
        if case.query == "star":
            np.testing.assert_allclose(got[0], np.minimum(expected, caps), atol=5e-6, rtol=3e-5)
            np.testing.assert_array_equal(got[1], 1)
        else:
            np.testing.assert_allclose(got[1], np.where(expected <= caps, expected, np.inf), atol=5e-6, rtol=3e-5)
            np.testing.assert_array_equal(got[3], (expected <= caps).astype(int))
            meta = geometry.boundary.silhouette
            a, b = (np.asarray(v)[:, :primitive.dimension] for v in (meta.feature_a, meta.feature_b))
            for i, index in enumerate(got[2]):
                if got[3][i] == 1:
                    assert 0 <= index < len(a)
                    np.testing.assert_allclose(distances[i, index], expected[i], atol=5e-6, rtol=3e-5)
                    np.testing.assert_allclose(got[0][i], _segment_point(a[index], b[index], got[0][i]), atol=8e-6)
                    score = _angle(got[0][i], *queries[i]) if case.metric == "angular" else np.linalg.norm(got[0][i]-queries[i])
                    np.testing.assert_allclose(score, expected[i], atol=5e-6, rtol=3e-5)
                else:
                    assert index == -1
    elif case.query == "robin":
        points, caps = map(np.asarray, args[3:])
        expected = [(_robin2(raw, geometry.boundary, q.astype(np.float64), float(cap)) if primitive.dimension == 2
                     else brute_robin_radius(np.asarray(raw.vertices), np.asarray(raw.indices), q.astype(np.float64), 1., float(cap)))
                    for q, cap in zip(points, caps, strict=True)]
        np.testing.assert_allclose(got[0], expected, atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(got[1], 1)
    else:
        np.testing.assert_array_equal(got[4], 1)
        np.testing.assert_allclose(got[2], 1/np.asarray(geometry.boundary.core.primitive_measure).sum(), rtol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(got[1], axis=1), 1., atol=3e-6)
        for point, index in zip(got[0], got[3], strict=True):
            _assert_member(case, raw, point, index)


def test_unsupported_builtin_combinations_fail_explicitly():
    assert len({c.id for c in CASES}) == len(CASES)
    for name, primitive in PRIMITIVES.items():
        missing = ([QueryCase(name, "nearest", "angular")] if primitive.segment_endpoints is None else [])
        if primitive.intersect is None:
            missing += [QueryCase(name, q) for q in RAYS]
        if name not in BOUNDARIES:
            missing += [QueryCase(name, q, "euclidean") for q in ("silhouette", "star", "robin", "classify", "sample")]
        for case in missing:
            with pytest.raises(TypeError):
                query_factory(case)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_equivalent_static_policies_reuse_kernel(case):
    from surfaces.explicit.accelerators.bvh.warp import RayOps

    from surfaces.explicit.accelerators.bvh.warp import (
        make_nearest, make_ray_query, make_star_radius, make_silhouette, make_classify, make_boundary_sample, make_robin_radius, make_overlap,
    )
    from surfaces.explicit.accelerators.bvh.backends._warp_boundary import resolve_boundary_adapter
    from surfaces.explicit.accelerators.bvh.backends._warp_boundary_queries import _uniform_measure, _euclidean_robin

    class DelegatingMetric:
        def bind(self, primitive):
            return metric.bind(primitive)

    mode, primitive, metric = case.query, PRIMITIVES[case.primitive], METRICS.get(case.metric)
    default = query_factory(case)
    if mode == "nearest":
        explicit = make_nearest(primitive, DelegatingMetric())
    elif mode == "silhouette":
        explicit = make_silhouette(primitive, DelegatingMetric())
    elif mode == "star":
        default = default.silhouette
        explicit = make_star_radius(primitive, adapter=resolve_boundary_adapter(primitive)).silhouette
    elif mode == "classify":
        explicit = make_classify(primitive, resolve_boundary_adapter(primitive).classification(primitive))
    elif mode == "sample":
        explicit = make_boundary_sample(primitive, _uniform_measure(primitive))
    elif mode == "robin":
        explicit = make_robin_radius(primitive, _euclidean_robin(primitive))
    elif mode == "overlap":
        explicit = make_overlap(replace(primitive, name="same_math"))
    else:
        explicit = make_ray_query(primitive, mode, ops=RayOps(intersect=primitive.intersect))
    assert default.kernel is explicit.kernel
