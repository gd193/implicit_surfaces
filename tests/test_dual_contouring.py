"""Independent constrained least-squares and public dual-contouring checks."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from surfaces.implicit.lipschitz import lipschitzImplicit
from surfaces.meshes import TriMesh
from meshExtractor import CapacityOverflow, DC_COMPILER_OPTIONS, make_dual_contouring, make_dual_contouring_resident
from meshExtractor._dual_qef import solve_qef


@wp.kernel
def _solve(matrices: wp.array(dtype=wp.mat33), rhs: wp.array(dtype=wp.vec3),
           centers: wp.array(dtype=wp.vec3), lower: wp.array(dtype=wp.vec3),
           upper: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    output[i] = solve_qef(matrices[i], rhs[i], centers[i], lower[i], upper[i])


def _oracle(normals, offsets, center, lower, upper):
    candidates = []
    for states in itertools.product(range(3), repeat=3):
        free = np.array(states) == 0
        point = center.copy()
        point[np.array(states) == 1] = lower[np.array(states) == 1]
        point[np.array(states) == 2] = upper[np.array(states) == 2]
        point[free] += np.linalg.lstsq(normals[:, free], offsets - normals @ point, rcond=1e-3)[0]
        if np.all(point >= lower - 1e-10) and np.all(point <= upper + 1e-10):
            candidates.append((np.linalg.norm(normals @ point - offsets)**2,
                               np.linalg.norm(point-center)**2, point))
    error = min(item[0] for item in candidates)
    return min((item for item in candidates if item[0] <= error + 1e-12), key=lambda item: item[1])[2]


def test_bounded_qef_against_independent_svd():
    rng = np.random.default_rng(18)
    cases = [(np.zeros((1, 3)), np.zeros(1))]
    for rank in (1, 2, 3):
        for _ in range(5):
            normals = rng.normal(size=(8, rank)) @ np.linalg.qr(rng.normal(size=(3, 3)))[0][:rank]
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            cases.append((normals, normals @ rng.uniform(-.7, 1.7, 3)))
    cases.extend([(np.array([[1., 0., 0.], [1., 1e-4, 0.]]), np.array([.3, .30002])),
                  (np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]]), np.array([2.5, .2, .9]))])
    count = len(cases)
    lower = np.zeros((count, 3), np.float32)
    upper = np.tile([1., .7, 1.4], (count, 1)).astype(np.float32)
    centers = .37 * upper
    matrices = np.array([n.T @ n for n, _ in cases], np.float32)
    rhs = np.array([n.T @ d for n, d in cases], np.float32)
    output = wp.empty(count, dtype=wp.vec3, device="cpu")
    inputs = [wp.array(a, dtype=t, device="cpu") for a, t in
              [(matrices, wp.mat33), (rhs, wp.vec3), (centers, wp.vec3), (lower, wp.vec3), (upper, wp.vec3)]]
    wp.launch(_solve, count, inputs=inputs, outputs=[output], device="cpu")
    result = output.numpy()
    for i, (normals, offsets) in enumerate(cases):
        reference = _oracle(normals, offsets, centers[i].astype(float), lower[i], upper[i])
        # Compare residuals because null-space positions need not be unique.
        actual_error = np.linalg.norm(normals @ result[i] - offsets)**2
        expected_error = np.linalg.norm(normals @ reference - offsets)**2
        assert actual_error <= expected_error + 2e-5
        assert np.all(result[i] >= lower[i]) and np.all(result[i] <= upper[i])
    np.testing.assert_allclose(result[0], centers[0], atol=1e-6)


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


@pytest.mark.parametrize("explicit_gradient", [False, True])
def test_root_refinement_reduces_smooth_surface_residual(explicit_gradient):
    def field(radius, point):
        return jnp.exp(point[0]) - radius
    gradient = (lambda radius, point: jnp.array([jnp.exp(point[0]), 0., 0.])) if explicit_gradient else None
    errors = []
    for steps in (0, 5):
        extract = make_dual_contouring_resident(field, ((-1.,)*3, (1.,)*3), (9, 11, 13),
                                      capacity=(1287, 1536, 2048), root_steps=steps, gradient=gradient, level=.125)
        mesh = jax.jit(extract, compiler_options=DC_COMPILER_OPTIONS)(jnp.float32(1.13)).compact()
        vertices, faces = _mesh_checks(mesh.vertices, mesh.elements)
        errors.append(np.mean(np.abs(np.exp(vertices[:, 0])-1.255)))
    assert errors[1] < errors[0]


@pytest.mark.parametrize("sparse,steps,analytic,batch", [
    (False, 0, False, 64), (False, 5, True, 127),
    (True, 0, True, 127), (True, 5, False, 64),
])
def test_resident_composition_geometry_and_retention(sparse, steps, analytic, batch):
    def field(params, p):
        return jnp.linalg.norm(p-params["center"])-params["radius"]
    def gradient(params, p):
        q = p-params["center"]
        return q/jnp.maximum(jnp.linalg.norm(q), jnp.float32(1e-20))
    bounds, shape = ((-1., -1.2, -1.4), (1.1, 1.3, 1.2)), (7, 9, 11)
    capacity = (693, 2880, 4158)
    options = dict(capacity=capacity, sparse_capacity=(4096, 5544) if sparse else None,
                   lipschitz_bound=1. if sparse else None, root_steps=steps,
                   batch_size=batch, gradient=gradient if analytic else None)
    resident = make_dual_contouring_resident(field, bounds, shape, **options)
    nested = jax.jit(lambda params: resident(params), compiler_options=DC_COMPILER_OPTIONS)
    retained = []
    for radius in (.41, .57, 5.):
        params = {"center": jnp.array([.023, -.031, .017]), "radius": jnp.float32(radius)}
        mesh = nested(params)
        jax.block_until_ready(mesh)
        counts = np.asarray(mesh.counts)
        assert mesh.vertices.dtype == jnp.float32 and mesh.faces.dtype == jnp.int32
        assert counts.dtype == np.int32 and counts[3] == 0
        assert mesh.vertices.shape == (capacity[0], 3) and mesh.faces.shape == (capacity[2], 3)
        vertices, faces = _mesh_checks(np.asarray(mesh.vertices)[:counts[0]],
                                      np.asarray(mesh.faces)[:counts[2]], closed=True)
        expected = mesh.compact()
        np.testing.assert_array_equal(faces, expected.elements)
        np.testing.assert_allclose(vertices, expected.vertices, atol=1e-6, rtol=0)
        if radius == 5.:
            assert tuple(counts[:3]) == (0, 0, 0)
        else:
            assert len(faces) > 20 and counts[1]*2 == len(faces)
            normal = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                              vertices[faces[:, 2]]-vertices[faces[:, 0]])
            assert np.all(np.sum(normal*(vertices[faces].mean(1)-np.asarray(params["center"])), axis=1) > 0)
            residual = np.asarray(jax.vmap(field, (None, 0))(params, jnp.asarray(vertices)))
            assert np.max(np.abs(residual)) < .08
        retained.extend((array, np.asarray(array).copy().view(np.uint8)) for array in mesh)
    for array, snapshot in retained:
        np.testing.assert_array_equal(np.asarray(array).view(np.uint8), snapshot)


@pytest.mark.parametrize("limited", ["C", "E", "T", "F", "K"])
def test_resident_overflow_rejects_exact_output(limited):
    def field(radius, p):
        return jnp.linalg.norm(p-jnp.array([.023, -.031, .017]))-radius
    sparse = limited in ("F", "K")
    bounds, shape = ((-1., -1.2, -1.4), (1.1, 1.3, 1.2)), (7, 9, 11)
    capacity, sparse_capacity = [693, 2880, 4158], [4096, 5544]
    batch = 127 if limited in ("E", "F") else 64
    options = dict(capacity=tuple(capacity), sparse_capacity=tuple(sparse_capacity) if sparse else None,
                   lipschitz_bound=1. if sparse else None, batch_size=batch)
    params = jnp.float32(.57)
    full = make_dual_contouring_resident(field, bounds, shape, **options)
    required = np.asarray(jax.device_get(jax.jit(full, compiler_options=DC_COMPILER_OPTIONS)(params).counts))
    slot, bit = {"C": (0, 1), "E": (1, 2), "T": (2, 4), "F": (4, 8), "K": (5, 16)}[limited]
    assert required[3] == 0 and required[slot] > 1
    if sparse:
        sparse_capacity[slot-4] = int(required[slot])-1
        options["sparse_capacity"] = tuple(sparse_capacity)
    else:
        capacity[slot] = int(required[slot])-1
        options["capacity"] = tuple(capacity)
    resident = make_dual_contouring_resident(field, bounds, shape, **options)
    mesh = jax.jit(resident, compiler_options=DC_COMPILER_OPTIONS)(params)
    counts = np.asarray(jax.device_get(mesh.counts))
    assert counts[3] & bit and counts[slot] >= required[slot]
    if limited in ("C", "E", "F", "K"):
        assert counts[2] == -1
    assert mesh.vertices.shape == (capacity[0], 3) and mesh.faces.shape == (capacity[2], 3)
    with pytest.raises(CapacityOverflow) as error:
        mesh.compact()
    assert error.value.counts == tuple(map(int, counts))


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
