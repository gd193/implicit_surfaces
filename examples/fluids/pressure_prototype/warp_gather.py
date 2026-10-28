"""Tiled CUDA evaluation of cached pressure-boundary records.

Each thread owns a target and its xyz accumulator. Source tiles are shared
across the block. Records are padded once for aligned shared-memory loads;
no target-by-source array is allocated.
"""
import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

# jax_kernel launches 256 threads per block. Round the launch up so every
# lane reaches the cooperative tile loads, including the last target block.
_BLOCK_SIZE = 256
_SOURCE_TILE = 128
# Measured generic f32 choice: 38 registers, no spills. Larger source tiles lost to
# register/shared-memory pressure. Keep f64's register budget unconstrained.
_MIN_BLOCKS = 5
_GRID_STRIDE = False


@lru_cache(maxsize=32)
def _green_factor(dtype, regularization):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    info = np.finfo(dtype)
    tiny = float(info.tiny)
    reg = float(dtype.type(regularization))
    inverse_reg = wp.constant(scalar(min(1.0 / reg, float(info.max)) if reg else info.max))
    four_pi_value = float(dtype.type(4.0 * math.pi))
    four_pi = wp.constant(scalar(four_pi_value))
    inverse_four_pi = wp.constant(scalar(1.0 / four_pi_value))
    # Keep every fast-path intermediate normal and the original denominator
    # finite. Rare extremes use the original expression, including its floor.
    lower = wp.constant(scalar(4 * math.exp((2 / 3) * (math.log(tiny) - math.log(4 * math.pi)))))
    upper_log = (2 / 3) * (math.log(float(info.max)) - math.log(4 * math.pi))
    if reg:
        upper_log = min(upper_log, 2 * (math.log(float(info.max)) - math.log(4 * math.pi) - math.log(reg)))
    upper = wp.constant(scalar(math.exp(upper_log) / 4))
    regularization_value = wp.constant(scalar(reg))

    # The guard below guarantees positive normal inputs. A direct f32 RSQ
    # avoids rsqrtf's redundant subnormal scaling; extremes keep the fallback.
    @wp.func_native('''
        float result;
        asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
        return result;
        ''' if scalar == wp.float32 else "return rsqrt(x);")
    def inverse_sqrt(x: scalar) -> scalar:
        ...

    @wp.func
    def factor(r2: scalar) -> scalar:
        inverse = scalar(0.0)
        if r2 >= lower and r2 <= upper:
            inverse_r = inverse_sqrt(r2)
            inverse = (inverse_four_pi * inverse_r) * wp.min(
                inverse_r * inverse_r, inverse_reg)
        else:
            denominator = wp.max(four_pi * wp.sqrt(r2)
                * wp.max(r2, regularization_value), scalar(tiny))
            inverse = scalar(1.0) / denominator
        return inverse

    return factor


@lru_cache(maxsize=32)
def _gather_kernel(dtype, regularization, split=False, continuations_per_root=0):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    vector = wp.vec3 if scalar == wp.float32 else wp.vec3d
    factor = _green_factor(dtype, regularization)
    roots_per_tile = max(1, _SOURCE_TILE // max(1, continuations_per_root))
    scalars_per_tile = roots_per_tile * max(1, continuations_per_root)

    @wp.func
    def green(r: vector) -> vector:
        # Preserve the pressure gather's multiplication order and zero handling.
        return r * factor(wp.dot(r, r))

    @wp.kernel(enable_backward=False, module="unique",
               launch_bounds=(_BLOCK_SIZE, _MIN_BLOCKS if scalar == wp.float32 else 1),
               grid_stride=_GRID_STRIDE)
    def boundary_gather(
        points: wp.array2d(dtype=scalar),
        velocities: wp.array2d(dtype=scalar),
        samples: wp.array2d(dtype=scalar),
        scalar_samples: wp.array2d(dtype=scalar),
        output: wp.array2d(dtype=scalar),
    ):
        i = wp.tid()
        x = vector(scalar(0.0))
        u = vector(scalar(0.0))
        if i < points.shape[0]:
            x = vector(points[i, 0], points[i, 1], points[i, 2])
            u = vector(velocities[i, 0], velocities[i, 1], velocities[i, 2])
        total = vector(scalar(0.0))
        affine_start = int(0)  # noqa: UP018, RUF046 - Warp requires a mutable typed local.
        if wp.static(continuations_per_root > 0):
            affine_start = scalar_samples.shape[0] // continuations_per_root
            # Match the original root, hop1, ..., hopK source order exactly.
            # Only storage is split: each scalar continuation has four fields.
            for tile in range((affine_start + roots_per_tile - 1) // roots_per_tile):
                root_start = tile * roots_per_tile
                roots = wp.tile_load(samples, shape=(roots_per_tile, 8),
                                     offset=(root_start, 0), storage="shared")
                continuations = wp.tile_load(scalar_samples,
                    shape=(scalars_per_tile, 4),
                    offset=(root_start * continuations_per_root, 0), storage="shared")
                for j in range(wp.min(roots_per_tile, affine_start - root_start)):
                    r = vector(roots[j, 0], roots[j, 1], roots[j, 2]) - x
                    weight = roots[j, 3] + wp.dot(
                        vector(roots[j, 4], roots[j, 5], roots[j, 6]), u)
                    total += green(r) * weight
                    for hop in range(wp.static(continuations_per_root)):
                        k = j * continuations_per_root + hop
                        r = vector(continuations[k, 0], continuations[k, 1], continuations[k, 2]) - x
                        total += green(r) * continuations[k, 3]
        for tile in range((samples.shape[0] - affine_start + _SOURCE_TILE - 1) // _SOURCE_TILE):
            start = affine_start + tile * _SOURCE_TILE
            source = wp.tile_load(samples, shape=(_SOURCE_TILE, 8),
                                  offset=(start, 0), storage="shared")
            for j in range(wp.min(_SOURCE_TILE, samples.shape[0] - start)):
                r = vector(source[j, 0], source[j, 1], source[j, 2]) - x
                weight = source[j, 3] + wp.dot(
                    vector(source[j, 4], source[j, 5], source[j, 6]), u)
                total += green(r) * weight
        if wp.static(split and continuations_per_root == 0):
            # Continuation records have no velocity coefficient. Their four
            # fields fit a smaller shared tile, and this phase has no b loads
            # or dot product. Both phases accumulate in this one launch.
            for tile in range((scalar_samples.shape[0] + _SOURCE_TILE - 1) // _SOURCE_TILE):
                start = tile * _SOURCE_TILE
                source_scalar = wp.tile_load(scalar_samples, shape=(_SOURCE_TILE, 4),
                                             offset=(start, 0), storage="shared")
                for j in range(wp.min(_SOURCE_TILE, scalar_samples.shape[0] - start)):
                    r = vector(source_scalar[j, 0], source_scalar[j, 1], source_scalar[j, 2]) - x
                    total += green(r) * source_scalar[j, 3]
        if i < points.shape[0]:
            output[i, 0] = total[0]
            output[i, 1] = total[1]
            output[i, 2] = total[2]

    return wp.jax_kernel(boundary_gather,
                         module_preload_mode=wp.JaxModulePreloadMode.ALL_DEVICES)


def boundary_gather(points: jax.Array, velocities: jax.Array,
                    samples: jax.Array, regularization: float) -> jax.Array:
    """Return the unnormalized cached sum, with the same kernel as _green.

    Inputs are (Q,3), (Q,3), and (M,7), all float32 or all float64. Walk-count
    normalization remains the projector's responsibility. CUDA/JIT only.
    """
    if points.ndim != 2 or points.shape[1] != 3 or velocities.shape != points.shape:
        raise ValueError('points and velocities must have shape (Q, 3)')
    if samples.ndim != 2 or samples.shape[1] != 7:
        raise ValueError('cached boundary records must have shape (M, 7)')
    if points.dtype not in (jnp.float32, jnp.float64) or not (
            points.dtype == velocities.dtype == samples.dtype):
        raise TypeError('gather inputs must share float32 or float64 dtype')
    if not len(points) or not len(samples):
        return jnp.zeros_like(points)
    # Eight scalars per row align shared loads. This O(M) pack preserves the
    # public seven-field format and replaces repeated per-interaction loads.
    samples = jnp.pad(samples, ((0, 0), (0, 1)))
    launch_size = ((len(points) + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    return _gather_kernel(points.dtype, regularization)(
        points, velocities, samples, jnp.empty((0, 4), points.dtype),
        launch_dims=launch_size, output_dims={'output': points.shape})[0]


def boundary_gather_split(points: jax.Array, velocities: jax.Array,
                          affine_samples: jax.Array, scalar_samples: jax.Array,
                          regularization: float, *,
                          continuations_per_root: int = 0) -> jax.Array:
    """Gather affine (M,7) and scalar (N,4) records in one CUDA launch.

    Scalar records contain (y.x, y.y, y.z, a) and represent b == 0 exactly.
    The caller must establish that invariant; arbitrary seven-field records
    belong in ``boundary_gather``. By default, affine sources are summed first,
    followed by scalar sources, so f32 sums may differ from an interleaved order.
    With continuations_per_root=K>0, scalar records are K consecutive hops
    per root. The first N/K affine records are those roots; remaining affine
    records form a trailing pseudo-source phase. This preserves the original
    root/hops/pseudo summation order. Both arrays share points' float32/float64
    dtype. No normalization occurs.
    """
    if points.ndim != 2 or points.shape[1] != 3 or velocities.shape != points.shape:
        raise ValueError('points and velocities must have shape (Q, 3)')
    if affine_samples.ndim != 2 or affine_samples.shape[1] != 7:
        raise ValueError('affine boundary records must have shape (M, 7)')
    if scalar_samples.ndim != 2 or scalar_samples.shape[1] != 4:
        raise ValueError('scalar boundary records must have shape (N, 4)')
    if points.dtype not in (jnp.float32, jnp.float64) or not (
            points.dtype == velocities.dtype == affine_samples.dtype == scalar_samples.dtype):
        raise TypeError('gather inputs must share float32 or float64 dtype')
    if (not isinstance(continuations_per_root, int) or continuations_per_root < 0
            or continuations_per_root > _SOURCE_TILE):
        raise ValueError('continuations_per_root must be an integer in [0, 128]')
    if continuations_per_root and (len(scalar_samples) % continuations_per_root
            or len(scalar_samples) // continuations_per_root > len(affine_samples)):
        raise ValueError('scalar records must contain K hops for each leading affine root')
    if not len(points) or not (len(affine_samples) or len(scalar_samples)):
        return jnp.zeros_like(points)
    affine_samples = jnp.pad(affine_samples, ((0, 0), (0, 1)))
    launch_size = ((len(points) + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
    return _gather_kernel(points.dtype, regularization, True, continuations_per_root)(
        points, velocities, affine_samples, scalar_samples,
        launch_dims=launch_size, output_dims={'output': points.shape})[0]
