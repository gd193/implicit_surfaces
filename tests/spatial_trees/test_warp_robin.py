"""Robin radius contracts and exhaustive geometry checks."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from surfaces.explicit.accelerators import BVHAccelerator, list_backends
from surfaces.explicit.accelerators.protocol import Config

from ._robin_oracle import brute_robin_radius
from ._warp_helpers import _mesh


pytestmark = pytest.mark.skipif("bvh.snch.warp" not in list_backends("bvh", runtime="warp"),
                                reason="Warp SNCH backend is unavailable")

def _tree(vertices, faces, backend="bvh.snch.warp", leaf_size=1):
    primitives, _, _ = _mesh(vertices, faces)
    return BVHAccelerator.build(primitives, backend, Config(leaf_size_threshold=leaf_size))

def test_robin_mu_zero_is_existing_silhouette_radius():
    # A single open triangle has boundary silhouette edges at every view.
    vertices = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    faces = [[0, 1, 2]]
    tree = _tree(vertices, faces)
    robin = tree.with_robin_coefficients(0.)
    points = jnp.asarray([[.2, .3, 1.], [.6, .4, 2.]], dtype=jnp.float32)
    cap = jnp.asarray([10., .5], dtype=jnp.float32)
    expected = jax.vmap(tree.star_radius)(points, cap)
    actual, status = jax.jit(jax.vmap(robin.robin_radius_with_status))(
        points, cap)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(status, [1, 1])


def test_robin_infinite_coefficient_is_capped_closest_surface_distance():
    vertices = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    tree = _tree(vertices, [[0, 1, 2]])
    robin = tree.with_robin_coefficients(np.inf)
    points = jnp.asarray([[.2, .3, 1.], [3., 3., 2.], [.2, .3, 0.]],
                         dtype=jnp.float32)
    caps = jnp.asarray([10., 10., 3.], dtype=jnp.float32)
    actual = jax.jit(jax.vmap(robin.robin_radius))(points, caps)
    _, outside_distance, _ = tree.closest_point(points[1])
    expected = np.minimum(np.asarray(caps),
                          np.asarray([1., float(outside_distance), 0.]))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_robin_finite_coefficient_matches_triangle_bound_and_monotonicity():
    # The center has no visible silhouette; Robin still restricts its radius.
    vertices = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    tree = _tree(vertices, faces)
    point = jnp.asarray([0., 0., 0.], dtype=jnp.float32)
    cap = jnp.asarray(10., dtype=jnp.float32)

    coefficients = [.5, 1., 2., 4., 8.]
    radii = []
    for mu in coefficients:
        robin = tree.with_robin_coefficients(mu)
        got = float(robin.robin_radius(point, cap))
        radii.append(got)
        expected = brute_robin_radius(vertices, faces, np.zeros(3), mu, 10.)
        np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)
    assert np.all(np.diff(radii) <= 2e-5)


def test_robin_interval_uses_upper_bound_and_updates_share_geometry():
    tree = _tree([[-10, -10, 0], [10, -10, 0], [0, 10, 0]],
                 [[0, 1, 2]])
    point = jnp.asarray([0., 0., 1.], dtype=jnp.float32)
    cap = jnp.asarray(10., dtype=jnp.float32)
    lower_zero = tree.with_robin_coefficients(0., maximum=2.)
    lower_positive = tree.with_robin_coefficients(1., maximum=2.)
    np.testing.assert_allclose(lower_zero.robin_radius(point, cap),
                               lower_positive.robin_radius(point, cap),
                               atol=2e-5, rtol=2e-5)
    updated = lower_positive.with_robin_coefficients(1.5, maximum=3.)
    assert updated is not lower_positive
    assert updated.primitives is lower_positive.primitives is tree.primitives
    assert updated.payload is not lower_positive.payload
    for name in ("node_lower", "node_left", "tri_v0", "tri_normal"):
        assert getattr(updated.payload, name) is getattr(tree.payload, name)
    assert tree.payload.tri_robin_bounds.shape == (0, 2)
    np.testing.assert_array_equal(lower_positive.payload.tri_robin_bounds, [[1., 2.]])


def test_robin_nested_vmap_over_shared_geometry():
    tree = _tree([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    robin = tree.with_robin_coefficients(np.inf)
    points = jnp.asarray([[[.2, .3, 1.], [.3, .2, 2.]],
                          [[.5, .5, 1.], [.1, .1, 3.]]], dtype=jnp.float32)
    caps = jnp.asarray([[2., 1.], [4., 2.]], dtype=jnp.float32)
    query = jax.jit(jax.vmap(jax.vmap(robin.robin_radius)))
    actual = query(points, caps)
    np.testing.assert_allclose(actual, np.minimum(caps, points[..., 2]),
                               atol=2e-5, rtol=2e-5)


def test_robin_face_coefficients_follow_original_face_order():
    # Face order opposes BVH order; the distant face must not change the bound.
    vertices = [[-3, -1, 0], [-1, -1, 0], [-2, 1, 0],
                [1, -1, 0], [3, -1, 0], [2, 1, 0]]
    faces = [[3, 4, 5], [0, 1, 2]]
    tree = _tree(vertices, faces, leaf_size=1)
    np.testing.assert_array_equal(tree.payload.prim_ids, [1, 0])
    p = jnp.asarray([-2., 0., 1.], dtype=jnp.float32)
    cap = jnp.asarray(10., dtype=jnp.float32)
    a = tree.with_robin_coefficients([8., .5]).robin_radius(p, cap)
    b = tree.with_robin_coefficients([.1, .5]).robin_radius(p, cap)
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)


def test_robin_invalid_query_status_and_host_validation():
    tree = _tree([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    robin = tree.with_robin_coefficients(.5)
    points = jnp.asarray([[jnp.nan, 0, 0], [0, jnp.inf, 0], [0, 0, 1],
                          [0, 0, 1]], dtype=jnp.float32)
    caps = jnp.asarray([1., 1., jnp.nan, -1.], dtype=jnp.float32)
    radius, status = jax.jit(jax.vmap(robin.robin_radius_with_status))(
        points, caps)
    np.testing.assert_array_equal(status, [-1, -1, -1, -1])
    assert np.isnan(np.asarray(radius)).all()

    for bad in (-1., np.nan, [-1.], [0., 1.], [np.inf, np.nan], 1e40, 1j):
        with pytest.raises((ValueError, TypeError)):
            tree.with_robin_coefficients(bad)
    with pytest.raises((ValueError, TypeError)):
        tree.with_robin_coefficients(.5, maximum=-.1)


def test_robin_empty_geometry_zero_cap_and_dynamic_cap():
    tree = _tree(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    robin = tree.with_robin_coefficients(1.)
    points = jnp.asarray([[0., 0., 0.], [1., 2., 3.]], dtype=jnp.float32)
    query = jax.jit(jax.vmap(robin.robin_radius_with_status))
    r0, s0 = query(points, jnp.asarray([0., 2.]))
    np.testing.assert_array_equal(r0, [0., 2.])
    np.testing.assert_array_equal(s0, [1, 1])
    r1, s1 = query(points, jnp.asarray([1., 0.]))
    np.testing.assert_array_equal(r1, [1., 0.])
    np.testing.assert_array_equal(s1, [1, 1])


@pytest.mark.parametrize("shape", ["sphere", "torus"])
@pytest.mark.parametrize("coefficient", [None, 1., 10.], ids=["varying", "mu1", "mu10"])
def test_robin_tree_pruning_matches_exhaustive_float64_oracle(shape, coefficient):
    mesh = (trimesh.creation.icosphere(subdivisions=1) if shape == "sphere" else
            trimesh.creation.torus(major_radius=.7, minor_radius=.3,
                                   major_sections=12, minor_sections=8))
    vertices, faces = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
    rng = np.random.default_rng(1053)
    coefficients = np.power(10., rng.uniform(-2., 2., len(faces))).astype(np.float32)
    if coefficient is not None:
        coefficients.fill(coefficient)
    points = rng.uniform(-1.4, 1.4, (36, 3)).astype(np.float32)
    caps = np.resize(np.asarray([.125, 1., 5., 100.], np.float32), len(points))
    tree = _tree(vertices, faces, leaf_size=2)
    robin = tree.with_robin_coefficients(coefficients * .1, coefficients)
    radius, status = jax.jit(robin.robin_radius_with_status)(points, caps)
    expected = [brute_robin_radius(vertices, faces, p, coefficients, cap)
                for p, cap in zip(points, caps, strict=True)]
    np.testing.assert_array_equal(status, np.ones(len(points), np.int32))
    np.testing.assert_allclose(radius, expected, atol=3e-5, rtol=1e-4)


def test_robin_cpu_execution_and_dynamic_coefficients():
    tree = _tree([[-10, -10, 0], [10, -10, 0], [0, 10, 0]],
                 [[0, 1, 2]])
    cpu = jax.devices("cpu")[0]
    tree.payload = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu), tree.payload)
    point = jax.device_put(np.asarray([0., 0., 1.], np.float32), cpu)
    cap = jax.device_put(np.float32(10), cpu)
    query = jax.jit(tree.backend.robin_radius_with_status)
    outputs = []
    for coefficient in (1., 4.):
        payload = tree.with_robin_coefficients(coefficient).payload
        assert all(x.device.platform == "cpu" for x in jax.tree_util.tree_leaves(payload))
        radius, status = query(payload, point, cap)
        assert radius.device.platform == status.device.platform == "cpu"
        assert int(status) == 1
        outputs.append(float(radius))
    assert outputs[1] < outputs[0]


def test_robin_configuration_and_interval_rounding():
    tree = _tree([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    with pytest.raises(ValueError, match="configure Robin"):
        tree.robin_radius(jnp.zeros(3), 1.)
    with pytest.raises(ValueError, match="minimum exceeds"):
        tree.with_robin_coefficients(1.00000004, 1.00000003)
    value = .1
    configured = tree.with_robin_coefficients(value)
    lower, upper = np.asarray(configured.payload.tri_robin_bounds)[0].astype(np.float64)
    assert lower <= value <= upper
    sah = _tree([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], "bvh.sah.warp")
    with pytest.raises(NotImplementedError, match="Robin"):
        sah.with_robin_coefficients(1.)


@pytest.mark.parametrize("scale", [1e-3, 1., 1e3])
def test_robin_radius_scales_with_geometry_and_inverse_coefficient(scale):
    mesh = trimesh.creation.box(extents=[2., 2., 2.])
    tree = _tree(np.asarray(mesh.vertices) * scale, mesh.faces)
    radius, status = tree.with_robin_coefficients(1. / scale).robin_radius_with_status(
        jnp.zeros(3), 10. * scale)
    assert int(status) == 1
    np.testing.assert_allclose(radius, 1.5 * np.sqrt(3.) * scale, rtol=2e-5)


def test_robin_coplanar_queries_and_empty_batch():
    tree = _tree([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    robin = tree.with_robin_coefficients(.1)
    points = jnp.asarray([[.2, .3, 0.], [3., 3., 0.]], dtype=jnp.float32)
    radius, status = jax.jit(robin.robin_radius_with_status)(points, 10.)
    np.testing.assert_array_equal(status, [1, 1])
    np.testing.assert_allclose(radius, [0., np.sqrt(8.)], atol=2e-5, rtol=2e-5)
    empty_radius, empty_status = jax.jit(robin.robin_radius_with_status)(
        jnp.zeros((0, 3), jnp.float32), jnp.zeros(0, jnp.float32))
    assert empty_radius.shape == empty_status.shape == (0,)


def test_robin_infinite_cap_preserves_unrestricted_radius():
    mesh = trimesh.creation.box(extents=[2., 2., 2.])
    tree = _tree(mesh.vertices, mesh.faces)
    for coefficient in (0., .01):
        radius, status = tree.with_robin_coefficients(coefficient).robin_radius_with_status(
            jnp.zeros(3), jnp.inf)
        assert int(status) == 1 and np.isposinf(float(radius))


def test_robin_radius_respects_triangle_distance_interval():
    mesh = trimesh.creation.icosphere(subdivisions=5)
    tree = _tree(mesh.vertices, mesh.faces, leaf_size=4)
    robin = tree.with_robin_coefficients(1.)
    points = jnp.asarray([[-.21108818, .40054315, -.81500834],
                          [.77663988, -.34156108, -.27615789],
                          [.74352419, -.45358282, .13813464],
                          [-.83153093, -.51310819, .03087507],
                          [.23223017, -.28848827, -.75982475]], dtype=jnp.float32)
    # Exhaustive float64 bounds on the same float32 mesh.
    expected = [1.695200758, 2., 2., .905456576, 2.]
    radius, status = jax.jit(robin.robin_radius_with_status)(points, 2.)
    np.testing.assert_array_equal(status, [1, 1, 1, 1, 1])
    np.testing.assert_allclose(radius, expected, atol=3e-5, rtol=3e-5)
