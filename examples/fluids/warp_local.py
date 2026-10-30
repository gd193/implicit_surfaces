"""Fused sampling and evaluation of the first-ball source Jacobian."""
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import warp as wp

_BLOCK_SIZE = 512
_WARP_SIZE = 32
_NATIVE_DIR = Path(__file__).with_name('native')
_BUILD_OPTIONS = wp.ModuleBuildOptions(
    extra_cpu_include_dirs=[_NATIVE_DIR], extra_cuda_include_dirs=[_NATIVE_DIR],
    extra_cpu_preamble='#include "sampling.h"', extra_cuda_preamble='#include "sampling.h"',
    extra_build_dependencies=[_NATIVE_DIR/'sampling.h'],
)


# Philox4x32-10: Salmon et al., SC11, doi:10.1145/2063384.2063405.
@wp.func
def philox(query: wp.uint64, sample: wp.uint64, block: wp.uint64,
           k0: wp.uint32, k1: wp.uint32) -> wp.vec4ui:
    index = sample*wp.uint64(2)+block
    c0 = wp.uint32(query)
    c1 = wp.uint32(query >> wp.uint64(32))
    c2 = wp.uint32(index)
    c3 = wp.uint32(index >> wp.uint64(32))
    for _ in range(10):
        a = wp.uint64(c0)*wp.uint64(0xD2511F53)
        b = wp.uint64(c2)*wp.uint64(0xCD9E8D57)
        n0 = wp.uint32(b >> wp.uint64(32)) ^ c1 ^ k0
        n2 = wp.uint32(a >> wp.uint64(32)) ^ c3 ^ k1
        c0 = n0
        c1 = wp.uint32(b)
        c2 = n2
        c3 = wp.uint32(a)
        k0 += wp.uint32(0x9E3779B9)
        k1 += wp.uint32(0xBB67AE85)
    return wp.vec4ui(c0, c1, c2, c3)


@wp.func
def uniform32(a: wp.uint32) -> wp.float32:
    return wp.float32(a >> wp.uint32(8))*wp.float32(wp.static(2.**-24))


@wp.func
def uniform64(a: wp.uint32, b: wp.uint32) -> wp.float64:
    bits = (wp.uint64(a) << wp.uint64(21)) | (wp.uint64(b) >> wp.uint64(11))
    return wp.float64(bits)*wp.float64(wp.static(2.**-53))


@lru_cache(maxsize=16)
def _kernel(dtype, lower_values, spacing_values, pairs, draws, grid_shape):
    if (len(lower_values) != 3 or len(spacing_values) != 3
            or not np.isfinite([lower_values, spacing_values]).all() or min(spacing_values) <= 0):
        raise ValueError('expected finite grid origin and positive spacing in three dimensions')
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    vector = wp.vec3 if scalar == wp.float32 else wp.vec3d
    matrix = wp.mat33 if scalar == wp.float32 else wp.mat33d
    two_pi = wp.constant(scalar(2.*np.pi))
    lower = wp.constant(vector(*lower_values))
    spacing = wp.constant(vector(*spacing_values))

    nx, ny, nz = grid_shape
    stride_x = wp.constant(wp.int64(ny*nz))
    stride_y = wp.constant(wp.int64(nz))

    @wp.func_native("return wp::warp_shuffle_down(value, offset, WP_TILE_LANE_MASK_ALL);")
    def shuffle_down(value: matrix, offset: int) -> matrix: ...

    @wp.func_native("return fluid::sphere_direction(rr, phi, z);")
    def direction(rr: scalar, phi: scalar, z: scalar) -> vector: ...

    @wp.func
    def mix(a: vector, b: vector, t: scalar) -> vector:
        return (scalar(1.)-t)*a+t*b

    # JAX supplies contiguous grids; retain wide offsets for large volumes.
    @wp.func_native("return field.data[index];")
    def load(field: wp.array3d(dtype=vector), index: wp.int64) -> vector: ...

    @wp.func
    def sample(field: wp.array3d(dtype=vector), index: vector) -> vector:
        base = wp.vec3i(int(wp.floor(index[0])), int(wp.floor(index[1])),
                       int(wp.floor(index[2])))
        f = index-vector(scalar(base[0]), scalar(base[1]), scalar(base[2]))
        i, ii = wp.clamp(base[0], 0, nx-1), wp.clamp(base[0]+1, 0, nx-1)
        j, jj = wp.clamp(base[1], 0, ny-1), wp.clamp(base[1]+1, 0, ny-1)
        k, kk = wp.clamp(base[2], 0, nz-1), wp.clamp(base[2]+1, 0, nz-1)
        i0, i1 = wp.int64(i)*stride_x, wp.int64(ii)*stride_x
        j0, j1 = wp.int64(j)*stride_y, wp.int64(jj)*stride_y
        k0, k1 = wp.int64(k), wp.int64(kk)
        a = mix(load(field, i0+j0+k0), load(field, i0+j0+k1), f[2])
        b = mix(load(field, i0+j1+k0), load(field, i0+j1+k1), f[2])
        c = mix(load(field, i1+j0+k0), load(field, i1+j0+k1), f[2])
        d = mix(load(field, i1+j1+k0), load(field, i1+j1+k1), f[2])
        return mix(mix(a, b, f[1]), mix(c, d, f[1]), f[0])

    @wp.kernel(enable_backward=False, module='unique', module_options={'fuse_fp': True, 'extra_build_options': _BUILD_OPTIONS})
    def integral(
        field: wp.array3d(dtype=vector),
        points: wp.array2d(dtype=scalar),
        radii: wp.array(dtype=scalar),
        key: wp.array(dtype=wp.uint32),
        output: wp.array3d(dtype=scalar),
    ):
        tid = wp.tid()
        query, lane = tid//_WARP_SIZE, tid % _WARP_SIZE
        x = vector(points[query, 0], points[query, 1], points[query, 2])
        radius = radii[query]
        index0 = wp.cw_div(x-lower, spacing)
        scale = wp.cw_div(vector(radius), spacing)
        total = matrix(scalar(0.))
        for draw in range(lane, pairs*draws, _WARP_SIZE):
            words = philox(wp.uint64(query), wp.uint64(draw), wp.uint64(0), key[0], key[1])
            t = scalar(0.)
            z = scalar(0.)
            phi = scalar(0.)
            if wp.static(scalar == wp.float32):
                t = scalar(1.)-uniform32(words[0])
                z = scalar(2.)*uniform32(words[1])-scalar(1.)
                phi = two_pi*uniform32(words[2])
            else:
                more = philox(wp.uint64(query), wp.uint64(draw), wp.uint64(1), key[0], key[1])
                t = scalar(1.)-uniform64(words[0], words[1])
                z = scalar(2.)*uniform64(words[2], words[3])-scalar(1.)
                phi = two_pi*uniform64(more[0], more[1])
            # One jittered radius per stratum, ordered for locality.
            t = (scalar(draw)+t)/scalar(pairs*draws)
            rr = wp.sqrt(wp.max(scalar(0.), scalar(1.)-z*z))
            d = direction(rr, phi, z)
            value = sample(field, index0+wp.cw_mul(scale, t*d))
            total += wp.outer(value*(radius*(scalar(1.)-t*t*t)), d)
        # Q*32 threads keep every query warp complete, including the final block.
        for i in range(wp.static(5)):
            total += shuffle_down(total, 16 >> i)
        if lane == 0:
            for a in range(3):
                for b in range(3):
                    output[query, a, b] = total[a, b]/scalar(pairs*draws)

    return wp.jax_kernel(integral, block_dim=_BLOCK_SIZE,
                         module_preload_mode=wp.JaxModulePreloadMode.NONE)


@partial(jax.jit, static_argnames=('lower', 'spacing', 'pairs', 'draws'))
def grid_local_gradient(field, points, radii, key, *, lower, spacing, pairs, draws):
    """Average R(1-t³) ω(x+Rtd) dᵀ with radially stratified Philox samples."""
    if (field.ndim != 4 or field.shape[-1] != 3 or min(field.shape[:3]) < 1
            or points.ndim != 2 or points.shape[1] != 3 or radii.shape != points.shape[:1]):
        raise ValueError('expected a vector grid, (Q,3) points and (Q,) radii')
    if points.dtype not in (jnp.float32, jnp.float64) or field.dtype != points.dtype or radii.dtype != points.dtype:
        raise TypeError('field, points and radii must share float32 or float64 dtype')
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (pairs, draws)):
        raise ValueError('positive integer pair and draw counts required')
    if pairs*draws > 2**31-_WARP_SIZE or len(points)*_WARP_SIZE >= 2**31:
        raise ValueError('sample or query count exceeds Warp int32 indexing')
    call = _kernel(points.dtype, lower, spacing, pairs, draws, field.shape[:3])
    if not len(points):
        return jnp.zeros((0, 3, 3), points.dtype)
    # Separate this stream from the continuation walks, for any JAX key type.
    seed = jr.bits(jr.fold_in(key, 0x4c4f434c), shape=(2,), dtype=jnp.uint32)
    return call(field, points, radii, seed, launch_dims=len(points)*_WARP_SIZE,
                output_dims={'output': (len(points), 3, 3)})[0]
