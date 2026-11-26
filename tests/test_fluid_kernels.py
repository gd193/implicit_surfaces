"""Fluid CUDA kernels against independent direct sums and interpolation."""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('warp')
requires_cuda = pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA kernels')


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


def _philox(query, sample, block, key):
    """Independent integer Random123 Philox4x32-10 reference."""
    mask = 2**32-1
    index = 2*sample+block
    c = [query & mask, query >> 32, index & mask, index >> 32]
    k = list(map(int, key))
    for _ in range(10):
        a, b = c[0]*0xD2511F53, c[2]*0xCD9E8D57
        c = [((b >> 32)^c[1]^k[0]) & mask, b & mask,
             ((a >> 32)^c[3]^k[1]) & mask, a & mask]
        k = [(k[0]+0x9E3779B9) & mask, (k[1]+0xBB67AE85) & mask]
    return c


def _uniforms(query, sample, key, dtype):
    a = _philox(query, sample, 0, key)
    if dtype == np.float32:
        return np.asarray([v >> 8 for v in a[:3]], dtype)*dtype(2.**-24)
    b = _philox(query, sample, 1, key)
    return np.asarray([(a[0] << 21) | (a[1] >> 11),
                       (a[2] << 21) | (a[3] >> 11),
                       (b[0] << 21) | (b[1] >> 11)], dtype)*2.**-53


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@requires_cuda
def test_philox(dtype):
    import warp as wp
    from examples.fluids.warp_local import philox, uniform32, uniform64

    scalar = wp.float32 if dtype == np.float32 else wp.float64

    @wp.kernel(enable_backward=False, module='unique')
    def sample(counters: wp.array2d(dtype=wp.uint64), keys: wp.array2d(dtype=wp.uint32),
               hashes: wp.array2d(dtype=wp.uint32), values: wp.array2d(dtype=scalar)):
        i = wp.tid()
        a = philox(counters[i, 0], counters[i, 1], counters[i, 2], keys[i, 0], keys[i, 1])
        for j in range(4):
            hashes[i, j] = a[j]
        if wp.static(scalar == wp.float32):
            for j in range(3):
                values[i, j] = uniform32(a[j])
        else:
            b = philox(counters[i, 0], counters[i, 1], wp.uint64(1), keys[i, 0], keys[i, 1])
            values[i, 0] = uniform64(a[0], a[1])
            values[i, 1] = uniform64(a[2], a[3])
            values[i, 2] = uniform64(b[0], b[1])

    with jax.enable_x64(True):
        n = 65536
        counters = np.zeros((n, 3), np.uint64)
        counters[:, 1] = np.arange(n)
        counters[-3:] = [(2**40+3, 2**39+5, 0), (2**63+7, 2**61+9, 1), (1, 2, 1)]
        keys = np.zeros((n, 2), np.uint32)
        keys[-3:] = [(17, 2**32-1), (2**31, 9), (42, 123)]
        hashes, values = wp.jax_kernel(sample, num_outputs=2)(
            jnp.asarray(counters), jnp.asarray(keys),
            output_dims={'hashes': (n, 4), 'values': (n, 3)})
        hashes, values = np.asarray(hashes), np.asarray(values)
        np.testing.assert_array_equal(hashes[0], [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
        for i in [0, 1, 31, 32, n-3, n-2, n-1]:
            q, t, block = map(int, counters[i])
            np.testing.assert_array_equal(hashes[i], _philox(q, t, block, keys[i]))
        for i in [0, 1, 31, n-3]:
            np.testing.assert_array_equal(values[i], _uniforms(*map(int, counters[i, :2]), keys[i], dtype))
        u = values[:-3].astype(np.float64)
        assert np.all((u >= 0.) & (u < 1.))
        np.testing.assert_allclose(u.mean(0), .5, atol=.006, rtol=0.)
        np.testing.assert_allclose(u.var(0), 1/12, atol=.002, rtol=0.)
        np.testing.assert_allclose(np.corrcoef(u.T), np.eye(3), atol=.02, rtol=0.)
        z, phi = 2*u[:, 1]-1, 2*np.pi*u[:, 2]
        rr = np.sqrt(1-z*z)
        d = np.stack((rr*np.cos(phi), rr*np.sin(phi), z), axis=1)
        np.testing.assert_allclose(d.mean(0), 0., atol=.01, rtol=0.)
        np.testing.assert_allclose(d.T@d/len(d), np.eye(3)/3, atol=.008, rtol=0.)
        if dtype == np.float64:
            assert np.count_nonzero(np.modf(u[:, 0]*2**24)[0]) > .99*len(u)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('pairs,draws', [(1, 1), (3, 45), (64, 16)])
@requires_cuda
def test_grid_local_gradient(dtype, pairs, draws):
    from examples.fluids.warp_local import grid_local_gradient

    rng = np.random.default_rng(41)
    shape = (1, 2, 3) if draws == 1 else (7, 5, 9)
    field = rng.normal(size=(*shape, 3)).astype(dtype)
    points = rng.uniform(-2., 2., (17, 3)).astype(dtype)
    radii = rng.uniform(0., .5, len(points)).astype(dtype)
    radii[0] = 0.
    lower, spacing = (-.5, -.75, -1.), (.25, .5, .125)
    with jax.enable_x64(dtype == np.float64):
        key = (jax.random.PRNGKey(7) if draws == 1 else
               jax.random.key(7, impl='rbg' if draws == 45 else 'threefry2x32'))

        seed = np.asarray(jax.random.bits(jax.random.fold_in(key, 0x4c4f434c),
                                          shape=(2,), dtype=jnp.uint32))
        u = np.asarray([[_uniforms(q, i, seed, dtype) for i in range(pairs*draws)]
                        for q in range(len(points))])
        t, z, phi = dtype(1.)-u[..., 0], dtype(2.)*u[..., 1]-dtype(1.), dtype(2*np.pi)*u[..., 2]
        t = (np.arange(pairs*draws, dtype=dtype)+t)/dtype(pairs*draws)
        rr = np.sqrt(np.maximum(dtype(0.), dtype(1.)-z*z))
        d = np.stack((rr*np.cos(phi), rr*np.sin(phi), z), axis=-1)
        positions = points[:, None]+(radii[:, None]*t)[..., None]*d
        values = _interpolate(field, positions, lower, spacing)
        expected = np.einsum('qsi,qsj,qs->qij', values, d,
            radii[:, None]*(1-t*t*t))/(pairs*draws)
        f, x, r = map(jnp.asarray, (field, points, radii))
        options = dict(lower=lower, spacing=spacing, pairs=pairs, draws=draws)
        result = grid_local_gradient(f, x, r, key, **options)
        tolerance = 2e-6 if dtype == np.float32 else 2e-14
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(result[0], 0.)
        assert grid_local_gradient(f, x[:0], r[:0], key, **options).shape == (0, 3, 3)

        if pairs == 64:
            # Interior affine field: E[J] = R² A/10, including a nonzero offset.
            a = np.array([[1., 2., -1.], [-.5, .75, 1.5], [2., -.25, .5]])
            grid = np.stack(np.meshgrid(*[lower[i]+spacing[i]*np.arange(shape[i])
                                         for i in range(3)], indexing='ij'), -1)
            affine = jnp.asarray(grid@a.T+[.25, -.125, .0625], dtype)
            center = np.asarray(lower)+np.asarray(spacing)*(np.asarray(shape)-1)/2
            x = jnp.broadcast_to(jnp.asarray(center, dtype), (4096, 3))
            r = jnp.full((len(x),), .2, dtype)
            values = np.asarray(grid_local_gradient(affine, x, r, key, **options), np.float64)
            error = np.abs(values.mean(0)-(.2**2/10)*a)
            sem = values.std(0, ddof=1)/np.sqrt(len(values))
            assert np.all(error <= 6*sem+tolerance)
        if dtype == np.float32 and draws == 1:
            with pytest.raises(ValueError, match='int32 indexing'):
                jax.eval_shape(lambda: grid_local_gradient(f, x, r, key,
                    **dict(options, pairs=2**31)))
            with pytest.raises(ValueError, match='int32 indexing'):
                jax.eval_shape(lambda xx, rr: grid_local_gradient(f, xx, rr, key, **options),
                    jax.ShapeDtypeStruct((2**26, 3), x.dtype),
                    jax.ShapeDtypeStruct((2**26,), r.dtype))


@pytest.mark.parametrize('platform,supported', [
    ('cpu', True), pytest.param('gpu', True, marks=requires_cuda),
    pytest.param('gpu', False, marks=requires_cuda),
])
def test_local_gradient_dispatch(platform, supported):
    from examples.fluids.dirichlet import make_dirichlet_gradient_solve
    signature = np.arange(1, 10, dtype=np.float32).reshape(3, 3)/8

    def source(params, x):
        return jnp.zeros_like(x)

    def cuda(params, points, radii, key, *, pairs):
        if not supported:
            return NotImplemented
        return jnp.broadcast_to(jnp.asarray(signature), (len(points), 3, 3))

    with jax.default_device(jax.devices(platform)[0]):
        params = {'omega': jnp.arange(81, dtype=jnp.float32).reshape(3, 3, 3, 3)/81}
        points = jnp.array([[.1, .2, .3], [-.2, .1, .1]], jnp.float32)
        results = []
        for callback in (None, cuda):
            solve = make_dirichlet_gradient_solve(
                lambda p, x, tr: (jnp.linalg.norm(x)-1., 1.-jnp.linalg.norm(x), tr),
                lambda x: jnp.zeros(3, x.dtype), source, dim=3, source_takes_params=True,
                max_steps=2, gradient_source_samples=3, local_gradient_cuda=callback)
            results.append(solve(params, points, jax.random.PRNGKey(4), 1., 2,
                return_details=True, max_active_states=4, steps_between_refill=2,
                n_compactions=0, jit_buck_size=2))
        expected = (np.asarray(results[1].successful_samples)[:, None, None]*signature
                    if platform == 'gpu' and supported else np.zeros((len(points), 3, 3)))
        assert np.all(np.asarray(results[1].successful_samples) > 0)
        np.testing.assert_array_equal(results[0].contribution_sum, 0.)
        np.testing.assert_allclose(results[1].contribution_sum, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(results[1].termination_counts, results[0].termination_counts)
        np.testing.assert_array_equal(results[1].successful_samples, results[0].successful_samples)
