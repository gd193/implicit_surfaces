"""Pressure CUDA kernels against independent direct sums and interpolation."""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('warp')
requires_cuda = pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA kernels')


def _gather_reference(points, velocities, records, regularization):
    tiny = np.finfo(points.dtype).tiny
    records, velocities = records.astype(np.float64), velocities.astype(np.float64)
    delta = records[None, :, :3]-points[:, None]
    r2 = np.sum(delta*delta, axis=-1)
    weights = records[None, :, 3]+np.sum(records[None, :, 4:]*velocities[:, None], axis=-1)
    terms = delta*(weights/np.maximum(4*np.pi*np.sqrt(r2)*np.maximum(r2, regularization), tiny))[..., None]
    return terms.sum(axis=1), np.abs(terms).sum(axis=1)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('queries,walk_count', [(259, 1), (63, 4681)])
@pytest.mark.parametrize('layout,regularization', [('dense', 0.), ('split', 1e-5), ('walks', 1e-5)])
@requires_cuda
def test_boundary_gather(dtype, queries, walk_count, layout, regularization):
    from examples.fluids.pressure_prototype.warp_gather import boundary_gather, boundary_gather_split

    rng = np.random.default_rng(81)
    points, velocities = (rng.normal(size=(queries, 3)).astype(dtype) for _ in range(2))
    roots = rng.normal(size=(walk_count+2, 7)).astype(dtype)
    scalar = rng.normal(size=(walk_count*6, 4)).astype(dtype)
    roots[0, :3] = points[0]  # The coincident term is exactly zero.
    if layout == 'walks':
        walks = np.zeros((walk_count, 7, 7), dtype)
        walks[:, 0] = roots[:walk_count]
        walks[:, 1:, :4] = scalar.reshape(walk_count, 6, 4)
        records = np.concatenate((walks.reshape(-1, 7), roots[walk_count:]))
    else:
        records = np.concatenate((roots, np.pad(scalar, ((0, 0), (0, 3)))))
    expected, scale = _gather_reference(points, velocities, records, regularization)
    with jax.enable_x64(dtype == np.float64):
        x, u, r, s = map(jnp.asarray, (points, velocities, roots, scalar))
        if layout == 'dense':
            result = boundary_gather(x, u, jnp.asarray(records), regularization)
        else:
            result = boundary_gather_split(x, u, r, s, regularization,
                                            continuations_per_root=6 if layout == 'walks' else 0)
        error = np.abs(np.asarray(result)-expected)
        tolerance = 8e-6 if dtype == np.float32 else 8e-14
        assert np.all(error <= tolerance*np.maximum(scale, 1.))
        if layout == 'dense' and walk_count == 1:
            assert boundary_gather(x[:0], u[:0], r, regularization).shape == (0, 3)
            np.testing.assert_array_equal(boundary_gather(x, u, r[:0], regularization), 0.)
            expected, _ = _gather_reference(points, velocities, roots[:1], regularization)
            np.testing.assert_allclose(boundary_gather(x, u, r[:1], regularization),
                                       expected, rtol=tolerance, atol=tolerance)


def _interpolate(field, points, lower=(-1.,)*3, spacing=(.25,)*3):
    index = (points.astype(np.float64)-lower)/spacing
    low = np.floor(index).astype(int)
    fraction = index-low
    result = np.zeros(points.shape, np.float64)
    for corner in product((0, 1), repeat=3):
        cell = np.clip(low+corner, 0, np.asarray(field.shape[:3])-1)
        weight = np.prod(np.where(corner, fraction, 1-fraction), axis=-1)
        result += weight[..., None]*field[cell[..., 0], cell[..., 1], cell[..., 2]]
    return result


@pytest.mark.parametrize('dtype,kind,pairs', [
    (np.float32, 'box', 3), (np.float32, 'torus', 65),
    (np.float64, 'box', 33), (np.float64, 'torus', 512),
])
@requires_cuda
def test_grid_volume(dtype, kind, pairs):
    from examples.fluids.grid import GridVelocity
    from examples.fluids.pressure_prototype.warp_direct import grid_volume, grid_volume_prepared
    from examples.fluids.pressure_prototype.warp_volume_geometry import VolumeGeometry

    rng = np.random.default_rng(12)
    field = rng.normal(size=(9, 7, 5, 3)).astype(dtype)
    lower, spacing = (-.5, -.75, -1.), (.125, .25, .5)
    points = rng.uniform(-.9, .9, (17, 3)).astype(dtype)
    directions = rng.normal(size=(17, pairs, 3))
    directions = (directions/np.linalg.norm(directions, axis=-1, keepdims=True)).astype(dtype)
    radial = rng.uniform(.02, 1., (17, pairs)).astype(dtype)
    radii = rng.uniform(.3, 1.2, 17).astype(dtype)
    distances = radii[:, None]*radial
    delta = distances[..., None]*directions
    positions = np.stack((points[:, None]+delta, points[:, None]-delta), axis=2)
    valid = np.all((positions > -1.) & (positions < 1.), axis=-1)
    geometry = VolumeGeometry(kind, (-1.,)*3, (1.,)*3, major=.4, minor=.15)
    if kind == 'torus':
        radial_distance = np.sqrt(positions[..., 0]**2+positions[..., 2]**2+dtype(1e-12))-dtype(.4)
        valid &= np.sqrt(radial_distance**2+positions[..., 1]**2+dtype(1e-12)) > dtype(.15)
    change = (_interpolate(field, positions, lower, spacing)
              -_interpolate(field, points, lower, spacing)[:, None, None])
    d = directions.astype(np.float64)[:, :, None]
    terms = (radii[:, None]/distances)[..., None, None]*(3*d*np.sum(d*change, axis=-1, keepdims=True)-change)
    expected = np.where(valid[..., None], terms, 0.).sum(axis=(1, 2))/(2*pairs)
    with jax.enable_x64(dtype == np.float64):
        x, directions, radial, radii, distances, valid = map(jnp.asarray,
            (points, directions, radial, radii, distances, valid))
        params = {'velocity': jnp.asarray(field)}
        velocity = GridVelocity(lower, spacing)
        unprepared = grid_volume(velocity, params, x, directions, distances, radii, valid)
        prepared = grid_volume_prepared(geometry, velocity, params, x, directions, radial, radii)
        tolerance = 2e-5 if dtype == np.float32 else 2e-13
        for result in (unprepared, prepared):
            np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('dtype,count', [(np.float32, 17), (np.float64, 257)])
@requires_cuda
def test_grid_pseudo(dtype, count):
    from examples.fluids.grid import GridVelocity
    from examples.fluids.pressure_prototype.warp_direct import grid_pseudo

    rng = np.random.default_rng(28)
    field = rng.normal(size=(9, 9, 9, 3)).astype(dtype)
    points = rng.uniform(-.8, .8, (9, 3)).astype(dtype)
    positions = rng.uniform(-1., 1., (9, count, 3)).astype(dtype)
    axes = rng.integers(0, 3, (9, count))
    normals = np.eye(3, dtype=dtype)[axes]*rng.choice([-1., 1.], (9, count, 1)).astype(dtype)
    positions = np.where(normals != 0, normals, positions)
    area = np.full((9, count, 1), 24., dtype)
    samples = np.concatenate((positions, normals, area), axis=-1)
    delta = positions.astype(np.float64)-points[:, None]
    r2 = np.sum(delta*delta, axis=-1)
    du = _interpolate(field, positions)-_interpolate(field, points)[:, None]
    weights = area[..., 0]*np.sum(normals*du, axis=-1)
    expected = np.mean(delta*(weights/(4*np.pi*np.sqrt(r2)*np.maximum(r2, 1e-5)))[..., None], axis=1)
    with jax.enable_x64(dtype == np.float64):
        result = grid_pseudo(GridVelocity((-1.,)*3, (.25,)*3), {'velocity': jnp.asarray(field)},
                             jnp.asarray(points), jnp.asarray(samples), 1e-5)
        tolerance = 3e-6 if dtype == np.float32 else 3e-13
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('platform', ['cpu', pytest.param('gpu', marks=requires_cuda)])
def test_projector_dispatch(platform):
    """Tracing the unused CUDA branch must preserve the CPU fallback."""
    from examples.fluids.grid import GridVelocity
    from examples.fluids.pressure_prototype.boundary_sampling import SphereBoundary
    from examples.fluids.pressure_prototype.pressure import ClosedPressureProjector

    with jax.default_device(jax.devices(platform)[0]):
        solver = ClosedPressureProjector(SphereBoundary(), GridVelocity((-1.,)*3, (1.,)*3),
            volume_samples=4, boundary_volume_samples=4, regularization=.001)
        points = jnp.zeros((1, 3), jnp.float32)
        params = {'velocity': jnp.ones((3, 3, 3, 3), jnp.float32)}
        records = jnp.zeros((5, 7), jnp.float32).at[:, 0].set(1.).at[:, 3].set(1.)
        for _ in range(2):
            for result in (solver._gather(params, points, records),
                           solver._gather_walks(params, points, records, n_walks=1)):
                np.testing.assert_allclose(result, [[5/(4*np.pi), 0., 0.]], rtol=1e-6)
            volume = solver._direct(params, points, jax.random.split(jax.random.key(5), 1))
            np.testing.assert_allclose(volume, 0., atol=1e-5)
