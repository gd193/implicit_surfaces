"""Classification against solid angles and analytic closed meshes."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from surfaces.explicit.accelerators import BVHAccelerator
from surfaces.explicit.trimesh import TriMeshSurface
from ._warp_helpers import WARP_BVHS, _mesh, _build_mesh as _build

pytestmark = pytest.mark.skipif(not WARP_BVHS, reason="Warp is unavailable")


def _winding(vertices, faces, points):
    triangles = np.asarray(vertices, np.float64)[faces]
    result = []
    for p in points:
        a, b, c = (triangles - p).transpose(1, 0, 2)
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (la*lb*lc + np.einsum("ij,ij->i", a, b)*lc
                       + np.einsum("ij,ij->i", b, c)*la
                       + np.einsum("ij,ij->i", c, a)*lb)
        result.append(-1 if abs(np.sum(2*np.arctan2(numerator, denominator))) > 2*np.pi else 1)
    return result


@pytest.mark.parametrize("backend", WARP_BVHS)
@pytest.mark.parametrize("shape", ["sphere", "torus"])
def test_warp_inside_matches_solid_angles(backend, shape):
    mesh = (trimesh.creation.icosphere(subdivisions=2) if shape == "sphere" else
            trimesh.creation.torus(major_radius=.7, minor_radius=.3,
                                   major_sections=24, minor_sections=12))
    a = _build(mesh, backend)
    points = np.random.default_rng(407).uniform(-1.2, 1.2, (181, 3)).astype(np.float32)
    points = np.concatenate([points, [[0, 0, 0], [.7, 0, 0]]]).astype(np.float32)
    sign, status = jax.jit(a.backend.classify_point_with_status)(a.payload, points)
    np.testing.assert_array_equal(sign, _winding(np.asarray(mesh.vertices, np.float32),
                                               mesh.faces, points))
    np.testing.assert_array_equal(status, 1)
    assert a.has_point_classification
    assert a.payload.mesh_vertices is a.primitives.mesh.vertices


@pytest.mark.parametrize("backend", WARP_BVHS)
@pytest.mark.parametrize("leaf", [1, 4, 32])
def test_warp_inside_shared_edges_vertices_and_boundary(backend, leaf):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    # All these +X rays cross triangulation diagonals, edges or vertices.
    points = np.array([[0, 0, 0], [-2, 0, 0], [2, 0, 0],
                       [-2, 1, 1], [-2, -1, -1], [-2, 1, 0],
                       [0, 1, 0], [0, 0, 1], [1, .5, .25],
                       [1, 1, 0], [-1, -1, -1]], np.float32)
    expected = [-1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # Independent face reversal must preserve parity and edge ownership.
    rng = np.random.default_rng(88)
    for reverse in (False, True):
        if reverse:
            mask = rng.random(len(mesh.faces)) < .5
            mesh.faces[mask] = mesh.faces[mask, ::-1]
            rng.shuffle(mesh.faces)
        a = _build(mesh, backend, leaf)
        signs, statuses = jax.jit(a.classify_point_with_status)(points)
        np.testing.assert_array_equal(signs, expected)
        np.testing.assert_array_equal(statuses, 1)


@pytest.mark.parametrize("scale,shift", [(1e-3, 0), (1, 10000), (1e3, -10000)])
def test_warp_inside_scales_and_nextafter_contacts(scale, shift):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh.vertices = mesh.vertices * scale + shift
    a = _build(mesh)
    lo, hi = np.asarray(mesh.bounds, np.float32)
    points = np.tile(np.float32(shift), (7, 3))
    points[1] = hi
    points[2, 0] = np.nextafter(hi[0], np.float32(np.inf))
    points[3, 0] = np.nextafter(hi[0], np.float32(-np.inf))
    points[4, 1] = np.nextafter(lo[1], np.float32(-np.inf))
    points[5, 1] = np.nextafter(lo[1], np.float32(np.inf))
    points[6, 2] = hi[2]
    np.testing.assert_array_equal(a.classify_point(points), [-1, 0, 1, -1, 1, -1, 0])


def test_warp_inside_disconnected_and_nested_shells():
    outer = trimesh.creation.box(extents=[4, 4, 4])
    inner = trimesh.creation.box(extents=[2, 2, 2])
    other = trimesh.creation.box(extents=[2, 2, 2])
    other.apply_translation([5, 0, 0])
    a = _build(trimesh.util.concatenate([outer, inner, other]))
    np.testing.assert_array_equal(a.classify_point(jnp.array(
        [[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [5, 0, 0], [7, 0, 0]])),
        [1, -1, 1, -1, 1])


@pytest.mark.parametrize("damage", ["open", "nonmanifold", "degenerate"])
def test_warp_inside_invalid_meshes_fail(damage):
    mesh = trimesh.creation.box()
    if damage == "open":
        mesh.update_faces(np.arange(len(mesh.faces)-1))
    elif damage == "nonmanifold":
        mesh.faces = np.concatenate([mesh.faces, mesh.faces[:1]])
    else:
        mesh.vertices[mesh.faces[0, 1]] = mesh.vertices[mesh.faces[0, 0]]
    a = _build(mesh)
    sign, status = a.classify_point_with_status(jnp.array([[0, 0, 0], [10, 10, 10]]))
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, -1)


def test_warp_inside_invalid_points_empty_mesh_and_batch():
    a = _build(trimesh.creation.box())
    points = jnp.array([[jnp.nan, 0, 0], [0, jnp.inf, 0], [0, 0, -jnp.inf]])
    sign, status = a.classify_point_with_status(points)
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, -1)
    empty = BVHAccelerator.build(_mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32))[0],
                                 "bvh.sah.warp")
    sign, status = empty.classify_point_with_status(jnp.zeros((3, 3)))
    np.testing.assert_array_equal(sign, 1)
    np.testing.assert_array_equal(status, 0)
    assert a.classify_point(jnp.empty((2, 0, 3))).shape == (2, 0)
    with pytest.raises(ValueError, match="query shape"):
        a.classify_point(jnp.zeros(2))


@pytest.mark.parametrize("backend", WARP_BVHS)
def test_warp_inside_cpu_nested_vmap_and_dynamic_geometry(backend):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]), backend)
    b = _build(trimesh.creation.box(extents=[4, 4, 4]), backend)
    points = jnp.array([[[0, 0, 0], [1.5, 0, 0]], [[0, 1, 0], [3, 3, 3]]], jnp.float32)
    query = jax.jit(jax.vmap(jax.vmap(
        a.backend.classify_point, in_axes=(None, 0)), in_axes=(None, 0)))
    np.testing.assert_array_equal(query(a.payload, points), [[-1, 1], [0, 1]])
    np.testing.assert_array_equal(query(b.payload, points), [[-1, -1], [-1, 1]])
    cpu = jax.devices("cpu")[0]
    payload = jax.device_put(a.payload, cpu)
    output = jax.jit(a.backend.classify_point)(payload, jax.device_put(points, cpu))
    assert output.device.platform == "cpu"
    np.testing.assert_array_equal(output, [[-1, 1], [0, 1]])


def test_warp_inside_trimesh_surface_signed_value():
    a = _build(trimesh.creation.box(extents=[2, 2, 2]))
    surface = TriMeshSurface(a.primitives.mesh, a)
    points = jnp.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], jnp.float32)
    np.testing.assert_allclose(jax.jit(surface.signed_value)(None, points), [-1, 1, 0])
    value, radius, _ = surface.empty_sphere(None, points, 4.)
    np.testing.assert_allclose(value, [-1, 1, 0])
    np.testing.assert_allclose(radius, [1, 1, 0])
    for center, expected in (([0, 0, 0], -1), ([2, 0, 0], 1), ([1, 0, 0], 0)):
        center = jnp.array(center, jnp.float32)
        assert int(surface.classify_box(None, center-.1, center+.1)) == expected


def test_warp_inside_vertex_rays_and_rotated_contacts():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    a = _build(mesh)
    vertices = np.asarray(mesh.vertices, np.float32)
    points = vertices - np.array([3., 0., 0.], np.float32)
    np.testing.assert_array_equal(a.classify_point(points), 1)
    np.testing.assert_array_equal(a.classify_point(vertices), 0)
    # Dyadic coordinates give exactly representable slanted face contacts.
    vertices = np.array([[0, 0, 0], [2, 0, 1], [0, 2, 1], [0, 0, 2]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    a = BVHAccelerator.build(_mesh(vertices, faces)[0], "bvh.sah.warp")
    points = np.array([[.5, .5, .5], [1, 1, 1], [.25, .25, 1], [1, 1, .5]], np.float32)
    np.testing.assert_array_equal(a.classify_point(points), [0, 0, -1, 1])


@pytest.mark.parametrize("aspect", [1.e-4, 1., 1.e4])
def test_warp_inside_thin_rotated_mesh_near_faces(aspect):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh.vertices *= [1., aspect, .7]
    rotation = trimesh.transformations.rotation_matrix(.83, [.2, .7, .4])
    mesh.apply_transform(rotation)
    a = _build(mesh)
    vertices = np.asarray(mesh.vertices, np.float32)
    triangles = vertices[mesh.faces].astype(np.float64)
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    delta = 1.e-5 * min(1., aspect)
    points = np.concatenate([centers + delta*normals, centers - delta*normals]).astype(np.float32)
    np.testing.assert_array_equal(a.classify_point(points), _winding(vertices, mesh.faces, points))
