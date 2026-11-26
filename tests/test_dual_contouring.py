"""Adaptive dual-contouring geometry and topology checks."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfaces.implicit.lipschitz import lipschitzImplicit
from surfaces.meshes import TriMesh
from meshExtractor import make_dual_contouring


def _mesh_checks(vertices, faces, closed=False):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    assert np.isfinite(vertices).all() and faces.dtype == np.int32
    assert vertices.ndim == faces.ndim == 2 and vertices.shape[1] == faces.shape[1] == 3
    if len(faces):
        assert faces.min() >= 0 and faces.max() < len(vertices)
        assert np.all(np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                                              vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1) > 1e-10)
        if closed:
            edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
            assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    return vertices, faces


@pytest.mark.parametrize("shape", ["sphere", "box"])
def test_public_closed_geometry_dynamic_parameters(shape):
    def field(params, point):
        q = point - params["center"]
        return jnp.linalg.norm(q) - params["radius"] if shape == "sphere" else jnp.max(jnp.abs(q) - params["radius"])
    surface = lipschitzImplicit(1., fn=field, level_set=0., dim=3)
    extract = make_dual_contouring(surface, ((-1., -1.2, -1.4), (1.1, 1.3, 1.2)),
                                   tolerance=.27, max_depth=5)
    retained = []
    for radius in (.41, .57):
        params = {"center": jnp.array([.023, -.031, .017]), "radius": jnp.float32(radius)}
        mesh = extract(params).mesh
        assert isinstance(mesh, TriMesh)
        vertices, faces = mesh.vertices, mesh.elements
        assert isinstance(vertices, jax.Array)
        retained.extend((array, np.asarray(array).copy()) for array in (vertices, faces))
        vertices, faces = _mesh_checks(vertices, faces, closed=True)
        assert len(faces) > 50
        normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
        assert np.all(np.sum(normals * (vertices[faces].mean(1)-np.asarray(params["center"])), axis=1) > 0.)
        residual = np.asarray(jax.vmap(field, (None, 0))(params, jnp.array(vertices)))
        assert np.max(np.abs(residual)) < (.04 if shape == "sphere" else 2e-5)
        if shape == "box":
            corners = np.array(list(itertools.product((-radius, radius), repeat=3))) + np.asarray(params["center"])
            assert np.max(np.min(np.linalg.norm(corners[:, None]-vertices[None], axis=2), axis=1)) < 3e-5
    mesh = extract({"center": jnp.zeros(3), "radius": jnp.float32(5.)}).mesh
    assert mesh.vertices.shape == mesh.elements.shape == (0, 3)
    for array, expected in retained:
        np.testing.assert_array_equal(array, expected)


@pytest.mark.parametrize("axis,normal_kind", [
    (0, "analytic"), (1, "analytic"), (2, "analytic"),
    (1, "zero"), (1, "nonfinite"),
])
def test_plane_winding_explicit_gradient_and_fallback(axis, normal_kind):
    def field(params, point):
        return params[0] * (point[axis] - params[1])
    def gradient(params, point):
        if normal_kind == "zero":
            return jnp.zeros(3)
        if normal_kind == "nonfinite":
            return jnp.full(3, jnp.nan)
        return params[0] * jnp.eye(3)[axis]
    class Plane:
        signed_value = staticmethod(field)
        spatial_derivative = staticmethod(gradient)

    extract = make_dual_contouring(Plane(), ((-1., -2., -3.), (1., 2., 3.)),
                                   tolerance=2., max_depth=2)
    for sign in (-1., 1.):
        mesh = extract(jnp.array([sign, .071])).mesh
        vertices, faces = _mesh_checks(mesh.vertices, mesh.elements)
        assert len(faces)
        np.testing.assert_allclose(vertices[:, axis], .071, atol=2e-6)
        normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
        assert np.all(sign * normals[:, axis] > 0)


def test_grid_corner_ties_share_indices():
    surface = lipschitzImplicit(1., fn=lambda radius, p: jnp.linalg.norm(p) - radius,
                                level_set=0., dim=3)
    extract = make_dual_contouring(surface, ((-1.,) * 3, (1.,) * 3),
                                   tolerance=.22, max_depth=5)
    mesh = extract(jnp.float32(.75)).mesh
    assert len(mesh.elements)
    _mesh_checks(mesh.vertices, mesh.elements, closed=True)


def test_two_contours_in_one_cell_require_subdivision():
    class CornerPatches:
        def signed_value(self, params, p):
            return jnp.minimum(jnp.linalg.norm(p), jnp.linalg.norm(p - 1)) - .3

    bounds = ((0.,) * 3, (1.,) * 3)
    coarse = make_dual_contouring(CornerPatches(), bounds, tolerance=2., max_depth=0)(None)
    assert len(coarse.mesh.elements) == 0
    np.testing.assert_array_equal(coarse.unresolved_boxes, [bounds])
    refined = make_dual_contouring(CornerPatches(), bounds, tolerance=2., max_depth=1)(None)
    vertices, faces = _mesh_checks(refined.mesh.vertices, refined.mesh.elements)
    assert len(faces)
    assert np.all(np.ptp(vertices[faces], axis=1) < .5)
