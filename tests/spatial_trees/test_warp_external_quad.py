"""An indexed convex quad extends nearest through the public primitive API."""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from sengi.bvh.warp import (  # noqa: E402
    DeviceTree, Euclidean, build_tree, make_nearest, make_ray_query,
)
from sengi.protocol import Config  # noqa: E402
from ._warp_device import device_payload, run_kernel  # noqa: E402


@wp.struct
class QuadData:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=wp.vec4i)


Quad = wp.types.matrix(shape=(4, 3), dtype=wp.float32)


@wp.func
def load_quad(data: QuadData, slot: int):
    i = data.indices[slot]
    return wp.matrix_from_rows(data.vertices[i[0]], data.vertices[i[1]],
                               data.vertices[i[2]], data.vertices[i[3]])


@wp.func
def quad_bounds(q: Quad):
    return (wp.min(wp.min(q[0], q[1]), wp.min(q[2], q[3])),
            wp.max(wp.max(q[0], q[1]), wp.max(q[2], q[3])))


@wp.func
def closest_on_quad(query: wp.vec3, q: Quad):
    normal = wp.normalize(wp.cross(q[1] - q[0], q[3] - q[0]))
    inside = True
    best = q[0]
    for i in range(4):
        edge = q[(i + 1) % 4] - q[i]
        delta = query - q[i]
        inside = inside and wp.dot(wp.cross(edge, delta), normal) >= 0.
        candidate = q[i] + edge * wp.clamp(wp.dot(delta, edge) / wp.length_sq(edge), 0., 1.)
        if wp.length_sq(query - candidate) < wp.length_sq(query - best):
            best = candidate
    return query - wp.dot(query - q[0], normal) * normal if inside else best


class QuadArrays(NamedTuple):
    vertices: jax.Array
    indices: jax.Array


@dataclass(frozen=True)
class PlanarQuads:
    dimension = 3
    point_type = wp.vec3
    data_type = QuadData
    item_type = Quad
    load = load_quad
    bounds_of_item = quad_bounds
    closest_euclidean = closest_on_quad

    def bounds(self, data):
        corners = np.asarray(data.vertices)[np.asarray(data.indices)]
        return corners.min(axis=1), corners.max(axis=1)

    def pack(self, data, ids):
        return QuadArrays(data.vertices, data.indices[ids])


def test_external_indexed_quads_nearest():
    primitive = PlanarQuads()
    face = np.array([[0, 0, 0], [3, 0, 0], [2, 2, 0], [0, 2, 0]], np.float32)
    vertices = np.vstack((face + [10, 0, 0], face))
    raw = QuadArrays(jnp.asarray(vertices, jnp.float32), jnp.arange(8, dtype=jnp.int32).reshape(2, 4))
    tree, data = build_tree(primitive, raw, Config(leaf_size_threshold=1), builder="lbvh")
    np.testing.assert_array_equal(tree.prim_ids, [1, 0])
    with pytest.raises(TypeError, match="intersection"):
        make_ray_query(primitive)

    # Interior at its cap, slanted edge, corner and the same interior just outside its cap.
    queries = np.array([[[11, 1, 2], [3, 2, 0]], [[-1, -1, 1], [11, 1, 2]]], np.float32)
    caps = np.array([[2, 1], [2, np.nextafter(np.float32(2), np.float32(0))]], np.float32)
    nearest = make_nearest(primitive, Euclidean())

    def scalar(q, cap):
        return nearest(tree, data, q, cap)

    point, distance, index, status = jax.jit(jax.vmap(jax.vmap(scalar)))(queries, caps)
    np.testing.assert_allclose(point, [[[11, 1, 0], [2.2, 1.6, 0]], [[0, 0, 0], [0, 0, 0]]])
    np.testing.assert_allclose(distance, [[2, np.sqrt(.8)], [np.sqrt(3), np.inf]])
    np.testing.assert_array_equal(index, [[0, 1], [1, -1]])
    np.testing.assert_array_equal(status, [[1, 1], [1, 0]])

    closest = nearest.device

    @wp.kernel(module="unique", enable_backward=False)
    def caller(tree: DeviceTree, geometry: QuadData, ids: wp.array(dtype=int)):
        point, distance, index, status = wp.static(closest)(wp.vec3(11., 1., 2.), 2., tree, geometry)
        ids[0] = index

    device = wp.device_from_jax(jax.devices()[0])
    args = [device_payload(DeviceTree, tree, device), device_payload(QuadData, data, device)]
    ids, = run_kernel(caller, 1, args, [(1, int)], queries=(nearest,), device=device)
    np.testing.assert_array_equal(ids, [0])
