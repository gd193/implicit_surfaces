"""Boundary events, broad-phase overlap and multi-root custom primitives."""
from collections import namedtuple
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.backends._warp_domain import Ray  # noqa: E402

from surfaces.explicit.accelerators.bvh.backends._warp_boundary import build_boundary, resolve_boundary_adapter  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_domain import ClassificationOps, RayOps  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_inside import make_classify  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_overlap import make_overlap  # noqa: E402
from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    Points, Segments, Triangles, build_tree, make_ray_query,
)


@wp.struct
class SphereData:
    centers: wp.array(dtype=wp.vec3)
    radii: wp.array(dtype=wp.float32)


@wp.func
def _sphere_load(data: SphereData, slot: int):
    p = data.centers[slot]
    return wp.vec4(p[0], p[1], p[2], data.radii[slot])


@wp.func
def _sphere_bounds(item: wp.vec4):
    center = wp.vec3(item[0], item[1], item[2])
    return center-wp.vec3(item[3]), center+wp.vec3(item[3])


@dataclass(frozen=True)
class Spheres:
    dimension: int = 3
    point_type: object = wp.vec3
    data_type: object = SphereData
    load: object = _sphere_load
    bounds_of_item: object = _sphere_bounds

    def bounds(self, data):
        return np.asarray(data.centers)-np.asarray(data.radii)[:, None], np.asarray(data.centers)+np.asarray(data.radii)[:, None]

    def pack(self, data, ids):
        return type(data)(*(v[ids] for v in data))


@wp.func
def _sphere_next(ray: Ray, minimum: float, maximum: float,
                 data: SphereData, slot: int, state: int, half_open: bool, store_times: bool):
    origin, direction = ray.origin, ray.direction
    view = origin-data.centers[slot]
    a, b = wp.dot(direction, direction), wp.dot(view, direction)
    discriminant = b*b-a*(wp.dot(view, view)-data.radii[slot]*data.radii[slot])
    cursor = state+0
    if discriminant < 0.0:
        return 2, 0.0, 0
    while cursor < 2:
        sign = wp.where(cursor == 0, -1.0, 1.0)
        t = (-b+sign*wp.sqrt(discriminant))/a
        cursor += 1
        if discriminant == 0.0:
            cursor = 2
        if minimum <= t and t <= maximum:
            return cursor, t, 1
    return 2, 0.0, 0


@wp.struct
class Cursor:
    index: wp.int32


@wp.func
def _sphere_next_struct(ray: Ray, minimum: float, maximum: float,
                        data: SphereData, slot: int, state: Cursor, half_open: bool, store_times: bool):
    index, t, flag = _sphere_next(ray, minimum, maximum, data, slot, state.index, half_open, store_times)
    next_state = Cursor()
    next_state.index = index
    return next_state, t, flag


@wp.struct
class SphereDomain:
    closed: wp.array(dtype=wp.int32)


@wp.func
def _sphere_valid(meta: SphereDomain):
    return meta.closed[0] != 0


@wp.func
def _sphere_event(point: wp.vec3, data: SphereData, slot: int, original: int, meta: SphereDomain):
    view = point-data.centers[slot]
    value = wp.dot(view, view)-data.radii[slot]*data.radii[slot]
    if value == 0.0:
        return 2
    return wp.int32(value < 0.0)


def _sphere_tree():
    Data = namedtuple("SphereArrays", ["centers", "radii"])
    primitive = Spheres()
    return primitive, *build_tree(primitive, Data(jnp.zeros((1, 3)), jnp.ones(1)))


def test_warp_custom_overlap_factory():
    primitive, tree, geometry = _sphere_tree()
    lower = jnp.array([[0., 0, 0], [10., 10, 10], [jnp.nan, 0, 0], [2., 2, 2]])
    upper = jnp.array([[0., 0, 0], [11., 11, 11], [1., 1, 1], [1., 1, 1]])
    hit, status = jax.jit(make_overlap(primitive))(tree, geometry, lower, upper)
    np.testing.assert_array_equal(hit, [True, False, False, False])
    np.testing.assert_array_equal(status, [1, 0, -1, -1])


@pytest.mark.parametrize("dimension", [2, 3])
def test_warp_classification_factory_boundary_events(dimension):
    if dimension == 2:
        primitive = Segments(2)
        vertices = np.array([[-1., -1], [1., -1], [1., 1], [-1., 1]], np.float32)
        indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], np.int32)
        data = primitive.indexed(vertices, indices)
        points = [[0, 0], [-2, 1], [0, 1], [1, 1], [np.nextafter(np.float32(1), np.float32(np.inf)), 0]]
    else:
        import trimesh
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        primitive = Triangles()
        data = primitive.data(mesh.vertices, mesh.faces)
        points = [[0, 0, 0], [-2, 1, 1], [0, 1, 0], [1, 1, 1], [np.nextafter(np.float32(1), np.float32(np.inf)), 0, 0]]
    with pytest.raises(TypeError, match="classification"):
        make_classify(primitive, adapter=replace(resolve_boundary_adapter(primitive), classification=None))
    with pytest.raises(TypeError, match="BoundaryAdapter"):
        make_classify(Points(dimension))
    tree, geometry, boundary = build_boundary(primitive, data, silhouettes=False)
    points = jnp.asarray(points, jnp.float32)
    query = jax.jit(make_classify(primitive, stack_capacity=32))
    sign, status = query(tree, geometry, boundary, points)
    np.testing.assert_array_equal(sign, [-1, 1, 0, 0, 1])
    np.testing.assert_array_equal(status, 1)
    invalid = boundary._replace(core=boundary.core._replace(closed=jnp.array([0], jnp.int32)))
    sign, status = query(tree, geometry, invalid, points)
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, -1)


def test_warp_custom_multi_root_events_and_tangent_classification():
    primitive, tree, geometry = _sphere_tree()
    ops = RayOps(state_type=wp.int32, next_hit=_sphere_next)
    origins = jnp.array([[-2., 0, 0], [0., 0, 0], [-2., 1, 0]])
    directions = jnp.broadcast_to(jnp.array([1., 0, 0]), origins.shape)
    structured = RayOps(state_type=Cursor, next_hit=_sphere_next_struct)
    np.testing.assert_array_equal(make_ray_query(primitive, "count", ops=structured)(tree, geometry, origins, directions)[0], [2, 1, 1])
    count = make_ray_query(primitive, "count", ops=ops)
    np.testing.assert_array_equal(count(tree, geometry, origins, directions)[0], [2, 1, 1])
    np.testing.assert_array_equal(count(tree, geometry, origins, directions, min_t=-jnp.inf)[0], [2, 2, 1])
    all_hits = make_ray_query(primitive, "all", ops=ops)
    times, ids, total, status = all_hits(tree, geometry, origins[:1], directions[:1], max_hits=1)
    np.testing.assert_array_equal([total, status], [[2], [-2]])
    assert np.isfinite(times).all() and np.all(np.asarray(ids) == 0)
    times, _, total, status = all_hits(tree, geometry, origins[:1], directions[:1], max_hits=2)
    np.testing.assert_array_equal(times, [[1., 3.]])
    np.testing.assert_array_equal([total, status], [[2], [1]])
    np.testing.assert_array_equal(count(tree, geometry, origins[:1], directions[:1], 1., min_t=1.)[0], 1)
    np.testing.assert_array_equal(count(tree, geometry, origins, directions, exclude_face=0)[0], 0)
    first = make_ray_query(primitive, "first", ops=ops)(tree, geometry, origins, directions)
    np.testing.assert_array_equal(first[0], [1., 1., 2.])
    np.testing.assert_array_equal(make_ray_query(primitive, "any", ops=ops)(tree, geometry, origins, directions)[0], True)
    Meta = namedtuple("SphereDomainData", ["closed"])
    policy = ClassificationOps(SphereDomain, _sphere_valid, _sphere_event)
    sign, status = make_classify(primitive, policy)(tree, geometry, Meta(jnp.array([1], jnp.int32)), origins)
    np.testing.assert_array_equal(sign, [1., -1., 1.])
    np.testing.assert_array_equal(status, 1)
