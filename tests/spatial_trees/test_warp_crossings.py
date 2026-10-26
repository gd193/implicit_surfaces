"""Ray/line crossings against analytic cases and a float64 oracle."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from surfaces.explicit.accelerators import BVHAccelerator, Config
from ._warp_helpers import WARP_BVHS, _mesh, _build_mesh as _build

pytestmark = pytest.mark.skipif(not WARP_BVHS, reason="Warp is unavailable")


def _oracle(vertices, faces, origin, direction, low, high, excluded=-1):
    triangles = np.asarray(vertices, np.float64)[faces]
    a, b, c = triangles.transpose(1, 0, 2)
    e1, e2 = b-a, c-a
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    valid = det != 0
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    s = origin-a
    u = np.einsum("ij,ij->i", s, h)*inv
    q = np.cross(s, e1)
    v = (q @ direction)*inv
    t = np.einsum("ij,ij->i", e2, q)*inv
    # Random oracle rays avoid edges/vertices; analytic tests cover ownership.
    valid &= (u >= 0) & (v >= 0) & (u+v <= 1) & (t >= low) & (t <= high)
    valid &= np.arange(len(faces)) != excluded
    ids = np.flatnonzero(valid)
    return t[ids], ids


@pytest.mark.parametrize("backend", WARP_BVHS)
@pytest.mark.parametrize("line", [False, True])
@pytest.mark.parametrize("shape", ["sphere", "torus"])
def test_warp_crossings_matches_oracle(backend, line, shape):
    mesh = (trimesh.creation.icosphere(subdivisions=2) if shape == "sphere" else
            trimesh.creation.torus(major_radius=.7, minor_radius=.3,
                                   major_sections=24, minor_sections=12))
    a = _build(mesh, backend)
    rng = np.random.default_rng(231)
    origins = rng.uniform(-1.2, 1.2, (89, 3)).astype(np.float32)
    directions = rng.normal(size=origins.shape).astype(np.float32)
    low = np.full(len(origins), -np.inf if line else 0, np.float32)
    high = np.full(len(origins), np.inf, np.float32)
    low[:10] = -.3 if line else .1
    high[:10] = .5
    exclude = rng.integers(-1, len(mesh.faces), len(origins), dtype=np.int32)
    count, status = jax.jit(a.count_hits_with_status)(
        origins, directions, high, min_t=low, exclude_face=exclude)
    times, ids, totals, stored_status = jax.jit(lambda o, d, lo, hi, ex:
        a.all_hits_with_status(o, d, hi, min_t=lo, max_hits=8, exclude_face=ex))(
            origins, directions, low, high, exclude)
    np.testing.assert_array_equal(count, totals)
    np.testing.assert_array_equal(status, stored_status)
    for i in range(len(origins)):
        t, face = _oracle(np.asarray(mesh.vertices, np.float32), mesh.faces,
                           origins[i], directions[i], low[i], high[i], exclude[i])
        n = len(t)
        assert int(count[i]) == n
        assert int(status[i]) == int(n > 0)
        order = np.argsort(np.asarray(ids[i, :n]))
        np.testing.assert_array_equal(np.asarray(ids[i, :n])[order], face)
        np.testing.assert_allclose(np.asarray(times[i, :n])[order], t, atol=3e-6, rtol=3e-5)
        np.testing.assert_array_equal(ids[i, n:], -1)
        assert np.isposinf(times[i, n:]).all()
    assert a.has_all_hits


@pytest.mark.parametrize("backend", WARP_BVHS)
@pytest.mark.parametrize("leaf", [1, 4, 32])
def test_warp_crossings_shared_diagonals_winding_and_signed_bounds(backend, leaf):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh.faces[::2] = mesh.faces[::2, ::-1]
    a = _build(mesh, backend, leaf)
    origins = jnp.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], jnp.float32)
    for axis in range(3):
        o = jnp.roll(origins, axis, axis=1)
        d = jnp.broadcast_to(jnp.eye(3)[axis], o.shape)
        np.testing.assert_array_equal(a.count_hits(o, d), [2, 1, 0])
        np.testing.assert_array_equal(a.count_hits(o, d, min_t=-jnp.inf), [2, 2, 2])
        np.testing.assert_array_equal(a.count_hits(o, -d), [0, 1, 2])
    times, ids, n, status = a.all_hits_with_status(origins[0], jnp.array([1., 0, 0]),
                                                 3., min_t=1., max_hits=2)
    np.testing.assert_array_equal(np.sort(times), [1, 3])
    assert int(n) == 2 and int(status) == 1
    assert int(a.count_hits(origins[0], jnp.array([1., 0, 0]), 1., min_t=1.)) == 1
    assert int(a.count_hits(origins[0], jnp.array([1., 0, 0]),
                           exclude_face=ids[np.argmin(times)])) == 1
    origin_face = ids[np.argmin(times)]
    assert int(a.count_hits(jnp.array([-1., 0, 0]), jnp.array([1., 0, 0]),
                           exclude_face=origin_face)) == 1


def test_warp_crossings_overflow_reports_full_count():
    meshes = [trimesh.creation.box(extents=[s, s, s]) for s in (2, 4, 6, 8, 10)]
    a = _build(trimesh.util.concatenate(meshes))
    o, d = jnp.zeros(3), jnp.array([1., 0, 0])
    times, faces, count, status = a.all_hits_with_status(o, d, min_t=-jnp.inf, max_hits=3)
    assert int(count) == 10 and int(status) == -2
    assert np.isfinite(times).all() and np.all(np.asarray(faces) >= 0)
    assert int(a.count_hits(o, d, min_t=-jnp.inf)) == 10
    times, _, count, status = a.all_hits_with_status(o, d, min_t=-jnp.inf, max_hits=10)
    assert int(count) == 10 and int(status) == 1
    np.testing.assert_array_equal(np.sort(times), [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])


@pytest.mark.parametrize("backend", WARP_BVHS)
def test_warp_crossings_batching_cpu_and_dynamic_geometry(backend):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]), backend)
    b = _build(trimesh.creation.box(extents=[4, 4, 4]), backend)
    o = jnp.array([[[0, 0, 0], [3, 0, 0]], [[1.5, 0, 0], [-3, 0, 0]]], jnp.float32)
    d = jnp.broadcast_to(jnp.array([1., 0, 0]), o.shape)
    count = jax.jit(jax.vmap(jax.vmap(a.backend.count_hits, in_axes=(None, 0, 0)),
                            in_axes=(None, 0, 0)))
    np.testing.assert_array_equal(count(a.payload, o, d), [[1, 0], [0, 2]])
    np.testing.assert_array_equal(count(b.payload, o, d), [[1, 0], [1, 2]])
    all_hits = jax.jit(jax.vmap(jax.vmap(lambda payload, x, v:
        a.backend.all_hits_with_status(payload, x, v, max_hits=3), in_axes=(None, 0, 0)),
        in_axes=(None, 0, 0)))
    result = all_hits(a.payload, o, d)
    assert result[0].shape == (2, 2, 3)
    np.testing.assert_array_equal(result[2], [[1, 0], [0, 2]])
    cpu = jax.devices("cpu")[0]
    payload, oc, dc = (jax.device_put(x, cpu) for x in (a.payload, o, d))
    cpu_result = all_hits(payload, oc, dc)
    assert cpu_result[0].device.platform == "cpu"
    for expected, actual in zip(result, cpu_result, strict=True):
        np.testing.assert_allclose(actual, expected)
    assert a.count_hits(jnp.empty((2, 0, 3)), jnp.empty((2, 0, 3))).shape == (2, 0)
    assert a.all_hits(jnp.empty((2, 0, 3)), jnp.empty((2, 0, 3)), max_hits=4)[0].shape == (2, 0, 4)


def test_warp_crossings_invalid_inputs_and_empty_geometry():
    a = _build(trimesh.creation.box())
    points = jnp.array([[jnp.nan, 0, 0], [jnp.inf, 0, 0], [0, 0, 0]], jnp.float32)
    directions = jnp.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], jnp.float32)
    n, status = a.count_hits_with_status(points, directions)
    np.testing.assert_array_equal(n, -1)
    np.testing.assert_array_equal(status, -1)
    o, d = jnp.zeros(3), jnp.array([1., 0, 0])
    for lo, hi, ex in [(1, 0, -1), (jnp.nan, 1, -1), (0, jnp.nan, -1),
                       (jnp.inf, jnp.inf, -1), (-jnp.inf, -jnp.inf, -1),
                       (0, 1, -2), (0, 1, 12), (0, 1, jnp.uint32(2**32-1))]:
        ts, ids, n, status = a.all_hits_with_status(o, d, hi, min_t=lo, exclude_face=ex, max_hits=2)
        assert int(n) == -1 and int(status) == -1
        assert np.isposinf(ts).all()
        np.testing.assert_array_equal(ids, -1)
    for capacity in (0, -1, 2**31):
        with pytest.raises(ValueError, match="capacity"):
            a.all_hits(o, d, max_hits=capacity)
    with pytest.raises(TypeError):
        a.all_hits(o, d, max_hits=1.5)
    with pytest.raises(TypeError, match="integer"):
        a.count_hits(o, d, exclude_face=1.5)
    with pytest.raises(ValueError, match="batch shapes"):
        a.count_hits(jnp.zeros((2, 3)), d)
    empty = BVHAccelerator.build(_mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32))[0],
                                 "bvh.sah.warp")
    times, ids, n, status = empty.all_hits_with_status(o, d, max_hits=2)
    assert int(n) == 0 and int(status) == 0
    assert np.isposinf(times).all()
    np.testing.assert_array_equal(ids, -1)


@pytest.mark.parametrize("scale,shift", [(1.e-3, 0), (1., 10000), (1.e3, -10000)])
def test_warp_crossings_scale_translation_and_coplanar_faces(scale, shift):
    mesh = trimesh.creation.box(extents=[2*scale]*3)
    mesh.apply_translation([shift]*3)
    a = _build(mesh)
    o, d = jnp.full(3, shift, jnp.float32), jnp.array([scale, 0, 0], jnp.float32)
    times, _, n, status = a.all_hits_with_status(o, d, min_t=-jnp.inf, max_hits=2)
    np.testing.assert_allclose(np.sort(times), [-1, 1], rtol=3e-5, atol=3e-6)
    assert int(n) == 2 and int(status) == 1
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    a = BVHAccelerator.build(_mesh(vertices, [[0, 1, 2]])[0], "bvh.sah.warp")
    assert int(a.count_hits(jnp.array([.2, .2, 0]), jnp.array([1., 0, 0]), min_t=-jnp.inf)) == 0


def test_warp_crossings_degenerate_faces_and_many_hits():
    n = 129
    vertices = np.array([[[i, -1, -1], [i, 1, -1], [i, 0, 1]] for i in range(1, n+1)], np.float32)
    faces = np.arange(3*n, dtype=np.int32).reshape(-1, 3)
    vertices = vertices.reshape(-1, 3)
    a = BVHAccelerator.build(_mesh(vertices, faces)[0], "bvh.sah.warp",
                             Config(leaf_size_threshold=4))
    o, d = jnp.zeros(3), jnp.array([1., 0, 0])
    assert int(a.count_hits(o, d)) == n
    times, _, count, status = a.all_hits_with_status(o, d, max_hits=n)
    assert int(count) == n and int(status) == 1
    np.testing.assert_array_equal(np.sort(times), np.arange(1, n+1))
    a = BVHAccelerator.build(_mesh([[0,0,0],[1,0,0],[2,0,0]], [[0,1,2]])[0], "bvh.sah.warp")
    assert int(a.count_hits(o, d, min_t=-jnp.inf)) == 0


@pytest.mark.parametrize("magnitude", [1.e-12, 1.e12])
def test_warp_crossings_direction_scaling(magnitude):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]))
    o, d = jnp.zeros(3), jnp.array([magnitude, 0, 0], jnp.float32)
    ts, _, n, status = a.all_hits_with_status(o, d, min_t=-jnp.inf, max_hits=2)
    assert int(n) == 2 and int(status) == 1
    np.testing.assert_allclose(np.sort(ts), [-1/magnitude, 1/magnitude], rtol=3e-5)


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.uint32])
def test_warp_crossings_integer_face_ids_broadcast(dtype):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]))
    o, d = jnp.array([-2., 0, 0]), jnp.array([1., 0, 0])
    times, faces, _ = a.all_hits(o, d, max_hits=2)
    excluded = jnp.full((2, 1), faces[jnp.argmin(times)], dtype=dtype)
    origins = jnp.broadcast_to(o, (2, 3, 3))
    directions = jnp.broadcast_to(d, origins.shape)
    np.testing.assert_array_equal(a.count_hits(origins, directions, exclude_face=excluded), 1)


def test_warp_crossings_exact_endpoint_nonaligned_rays():
    a = BVHAccelerator.build(_mesh([[0,-1,-1],[0,1,-1],[0,0,1]], [[0,1,2]])[0],
                             "bvh.sah.warp")
    directions = np.random.default_rng(2).uniform(.1,10,(1000,3)).astype(np.float32)
    directions[::2] *= -1
    for device in (jax.devices()[0], jax.devices("cpu")[0]):
        payload, d = (jax.device_put(v, device) for v in (a.payload, directions))
        for endpoint in (-1., 1.):
            o = -endpoint*d
            result = jax.jit(lambda tree, x, v: a.backend.all_hits_with_status(
                tree,x,v,endpoint,min_t=endpoint,max_hits=1))(payload,o,d)
            np.testing.assert_array_equal(result[0], endpoint)
            np.testing.assert_array_equal(result[2:], 1)
