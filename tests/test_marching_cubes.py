"""Geometry and certificate contracts through the public adaptive extractor."""
import jax.numpy as jnp
import numpy as np
import pytest

from surfaces.meshes import TriMesh
from meshExtractor import make_dual_contouring, make_marching_cubes
from surfaces.implicit.lipschitz import lipschitzImplicit

BOUNDS = ((-1., -1., -1.), (1., 1., 1.))


def mesh_edges(faces):
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    return np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)


@pytest.mark.parametrize("factory", [make_marching_cubes, make_dual_contouring])
def test_certified_root_skips_point_evaluation(factory):
    class Empty:
        def classify_box(self, params, lower, upper):
            return -1

        def signed_value(self, params, point):
            raise AssertionError("certified box reached point sampler")

    mesh, unresolved = factory(Empty(), BOUNDS, tolerance=.01)(None)
    assert mesh.vertices.shape == mesh.elements.shape == (0, 3)
    assert unresolved.shape == (0, 2, 3)


@pytest.mark.parametrize("factory", [make_marching_cubes, make_dual_contouring])
def test_hidden_component_and_point_only_depth_limit(factory):
    class PointOnly:
        def signed_value(self, params, p):
            return jnp.linalg.norm(p) - .1

    mesh, unresolved = factory(PointOnly(), BOUNDS, tolerance=4., max_depth=0)(None)
    assert not len(mesh.elements)
    np.testing.assert_array_equal(unresolved, [BOUNDS])


def test_level_set_and_dynamic_parameters_closed_sphere():
    surface = lipschitzImplicit(1., fn=lambda radius, p: jnp.linalg.norm(p) - radius,
                                level_set=.1, dim=3)
    extract = make_marching_cubes(surface, BOUNDS, tolerance=.25, max_depth=4)
    for radius in (.42, .61):
        mesh = extract(jnp.float32(radius)).mesh
        assert isinstance(mesh, TriMesh)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.elements)
        assert len(faces) > 0
        assert np.max(np.abs(np.linalg.norm(vertices, axis=1) - radius - .1)) < .025
        _, counts = mesh_edges(faces)
        assert np.all(counts == 2)
        a, b, c = vertices[faces].transpose(1, 0, 2)
        assert np.all(np.sum(np.cross(b - a, c - a) * a, axis=1) > 0)


class BentGraph:
    """Exact monotone range. The left half is planar and the right half curves."""
    def height(self, x):
        return .16 + .58 * jnp.maximum(x, 0) ** 2

    def signed_value(self, params, p):
        return p[2] - self.height(p[0])

    def classify_box(self, params, lo, hi):
        return self.signs(lo, hi)

    def classify_subcubes(self, params, parent_lo, parent_hi, lo, hi):
        return self.signs(lo, hi)

    def signs(self, lo, hi):
        return jnp.where(lo[..., 2] > self.height(hi[..., 0]), 1,
                         jnp.where(hi[..., 2] < self.height(lo[..., 0]), -1, 0))


@pytest.mark.parametrize("factory", [make_marching_cubes, make_dual_contouring])
def test_mixed_depth_stitches_shared_edges(factory):
    surface = BentGraph()
    mesh, unresolved = factory(surface, BOUNDS, tolerance=.14, max_depth=5)(None)
    vertices, faces = map(np.asarray, (mesh.vertices, mesh.elements))
    assert not len(unresolved)
    edges, counts = mesh_edges(faces)
    assert max(counts) == 2
    boundary = vertices[edges[counts == 1], :2]
    assert np.all(np.any((boundary[:, 0] == boundary[:, 1]) &
                         (np.abs(boundary[:, 0]) == 1), axis=1))
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    assert lengths.max() >= (1 if factory is make_marching_cubes else .5) and lengths.min() < .1
    # The triangles cover the whole graph once in projection, including the seam.
    a, b, c = vertices[faces].transpose(1, 0, 2)
    areas = np.cross(b - a, c - a)[:, 2] * .5
    assert np.all(areas > 0)
    np.testing.assert_allclose(areas.sum(), 4., atol=1e-5)
    assert np.max(np.abs(vertices[:, 2] - surface.height(vertices[:, 0]))) < .14


@pytest.mark.parametrize("factory", [make_marching_cubes, make_dual_contouring])
def test_autobound_plane_accepted_at_root(factory):
    pytest.importorskip("jnp_bounds")
    from surfaces.implicit.jnp_bounded import jnpboundedImplicit
    surface = jnpboundedImplicit(autobounds_max_degree=2, use_warp=False,
                                 fn=lambda p, x: x[2] + p, level_set=.25, dim=3)
    mesh, unresolved = factory(surface, BOUNDS, tolerance=.01, max_depth=0)(jnp.float32(.125))
    if factory is make_marching_cubes:
        assert mesh.elements.shape == (2, 3) and mesh.vertices.shape == (4, 3)
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.elements)]
    a, b, c = triangles.transpose(1, 0, 2)
    area = np.cross(b - a, c - a)[:, 2] * .5
    assert np.all(area > 0)
    np.testing.assert_allclose(area.sum(), 4.)
    assert not len(unresolved)
    np.testing.assert_allclose(np.asarray(mesh.vertices)[:, 2], .125)


@pytest.mark.parametrize("factory", [make_marching_cubes, make_dual_contouring])
def test_tangent_corner_refines_to_depth_limit(factory):
    class Tangency:
        def signed_value(self, params, p):
            return p.sum() - 3.

        def classify_box(self, params, lower, upper):
            return jnp.where(upper.sum() < 3., -1, 0)

    bounds = ((-1., -.5, 0.), (1., 1., 1.))
    mesh, unresolved = factory(Tangency(), bounds, tolerance=4., max_depth=3)(None)
    assert not len(mesh.elements) and not len(mesh.vertices)
    np.testing.assert_array_equal(unresolved, [[[.75, .8125, .875], [1., 1., 1.]]])
