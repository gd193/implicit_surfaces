"""Tiled CUDA evaluation of cached pressure-boundary records.

Float32 shares source tiles within target blocks. Float64 divides each block
across target and striped source lanes, reduced in place. Records are padded
once for aligned shared loads; no target-by-source array is allocated.
"""
import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

# Round the launch so every lane reaches the cooperative tile loads,
# including the last target block.
_BLOCK_SIZE = 128
_SOURCE_TILE = 128
# Keep f64's register budget unconstrained.
_MIN_BLOCKS = 5


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
def _gather_kernel(dtype, regularization, split=False, continuations_per_root=0,
                   partitions=1):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    vector = wp.vec3 if scalar == wp.float32 else wp.vec3d
    factor = _green_factor(dtype, regularization)
    cooperative = scalar == wp.float64
    block_size = _BLOCK_SIZE
    source_tile = 1 if cooperative else _SOURCE_TILE
    targets_per_block = block_size // partitions if cooperative else block_size
    roots_per_tile = max(1, source_tile // max(1, continuations_per_root))
    scalars_per_tile = roots_per_tile * max(1, continuations_per_root)

    @wp.func
    def green(r: vector) -> vector:
        # Preserve the pressure gather's multiplication order and zero handling.
        return r * factor(wp.dot(r, r))

    @wp.kernel(enable_backward=False, module="unique",
               launch_bounds=(block_size, _MIN_BLOCKS if scalar == wp.float32 else 1),
               grid_stride=False)
    def boundary_gather(
        points: wp.array2d(dtype=scalar),
        velocities: wp.array2d(dtype=scalar),
        samples: wp.array2d(dtype=scalar),
        scalar_samples: wp.array2d(dtype=scalar),
        output: wp.array2d(dtype=scalar),
    ):
        i = wp.tid()
        part = int(0)
        if wp.static(cooperative):
            lane = i % block_size
            part = lane // targets_per_block
            i = (i // block_size) * targets_per_block + lane % targets_per_block
        elif wp.static(partitions > 1):
            rounded_queries = ((points.shape[0] + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE
            part = i // rounded_queries
            i = i % rounded_queries
        x = vector(scalar(0.0))
        u = vector(scalar(0.0))
        if i < points.shape[0]:
            x = vector(points[i, 0], points[i, 1], points[i, 2])
            u = vector(velocities[i, 0], velocities[i, 1], velocities[i, 2])
        total = vector(scalar(0.0))
        affine_start = int(0)  # noqa: UP018, RUF046 - Warp requires a mutable typed local.
        if wp.static(continuations_per_root > 0):
            affine_start = scalar_samples.shape[0] // continuations_per_root
            # Preserve root/hop order within each source partition.
            # Only storage is split: each scalar continuation has four fields.
            ntiles = (affine_start + roots_per_tile - 1) // roots_per_tile
            perpart = (ntiles + partitions - 1) // partitions
            for tile in range(part if wp.static(cooperative) else part * perpart,
                              ntiles if wp.static(cooperative) else wp.min((part + 1) * perpart, ntiles),
                              partitions if wp.static(cooperative) else 1):
                root_start = tile * roots_per_tile
                if wp.static(not cooperative):
                    roots = wp.tile_load(samples, shape=(roots_per_tile, 8),
                                         offset=(root_start, 0), storage="shared")
                    continuations = wp.tile_load(scalar_samples,
                        shape=(scalars_per_tile, 4),
                        offset=(root_start * continuations_per_root, 0), storage="shared")
                for j in range(wp.min(roots_per_tile, affine_start - root_start)):
                    if wp.static(cooperative):
                        r = vector(samples[root_start + j, 0], samples[root_start + j, 1], samples[root_start + j, 2]) - x
                        weight = samples[root_start + j, 3] + wp.dot(
                            vector(samples[root_start + j, 4], samples[root_start + j, 5], samples[root_start + j, 6]), u)
                    else:
                        r = vector(roots[j, 0], roots[j, 1], roots[j, 2]) - x
                        weight = roots[j, 3] + wp.dot(
                            vector(roots[j, 4], roots[j, 5], roots[j, 6]), u)
                    total += green(r) * weight
                    for hop in range(wp.static(continuations_per_root)):
                        k = j * continuations_per_root + hop
                        if wp.static(cooperative):
                            r = vector(scalar_samples[root_start * continuations_per_root + k, 0], scalar_samples[root_start * continuations_per_root + k, 1], scalar_samples[root_start * continuations_per_root + k, 2]) - x
                            weight = scalar_samples[root_start * continuations_per_root + k, 3]
                        else:
                            r = vector(continuations[k, 0], continuations[k, 1], continuations[k, 2]) - x
                            weight = continuations[k, 3]
                        total += green(r) * weight
        ntiles = (samples.shape[0] - affine_start + source_tile - 1) // source_tile
        perpart = (ntiles + partitions - 1) // partitions
        for tile in range(part if wp.static(cooperative) else part * perpart,
                              ntiles if wp.static(cooperative) else wp.min((part + 1) * perpart, ntiles),
                              partitions if wp.static(cooperative) else 1):
            start = affine_start + tile * source_tile
            if wp.static(not cooperative):
                source = wp.tile_load(samples, shape=(source_tile, 8),
                                      offset=(start, 0), storage="shared")
            for j in range(wp.min(source_tile, samples.shape[0] - start)):
                if wp.static(cooperative):
                    r = vector(samples[start + j, 0], samples[start + j, 1], samples[start + j, 2]) - x
                    weight = samples[start + j, 3] + wp.dot(
                        vector(samples[start + j, 4], samples[start + j, 5], samples[start + j, 6]), u)
                else:
                    r = vector(source[j, 0], source[j, 1], source[j, 2]) - x
                    weight = source[j, 3] + wp.dot(
                        vector(source[j, 4], source[j, 5], source[j, 6]), u)
                total += green(r) * weight
        if wp.static(split and continuations_per_root == 0):
            # Continuation records have no velocity coefficient. Their four
            # fields fit a smaller shared tile, and this phase has no b loads
            # or dot product. Both phases accumulate in the same partial sum.
            ntiles = (scalar_samples.shape[0] + source_tile - 1) // source_tile
            perpart = (ntiles + partitions - 1) // partitions
            for tile in range(part if wp.static(cooperative) else part * perpart,
                              ntiles if wp.static(cooperative) else wp.min((part + 1) * perpart, ntiles),
                              partitions if wp.static(cooperative) else 1):
                start = tile * source_tile
                if wp.static(not cooperative):
                    source_scalar = wp.tile_load(scalar_samples, shape=(source_tile, 4),
                                                 offset=(start, 0), storage="shared")
                for j in range(wp.min(source_tile, scalar_samples.shape[0] - start)):
                    if wp.static(cooperative):
                        r = vector(scalar_samples[start + j, 0], scalar_samples[start + j, 1], scalar_samples[start + j, 2]) - x
                        weight = scalar_samples[start + j, 3]
                    else:
                        r = vector(source_scalar[j, 0], source_scalar[j, 1], source_scalar[j, 2]) - x
                        weight = source_scalar[j, 3]
                    total += green(r) * weight
        if wp.static(cooperative and partitions == block_size):
            total = wp.tile_sum(wp.tile(total, preserve_type=True))[0]
        elif wp.static(cooperative and partitions > 1):
            rx = wp.tile_sum(wp.tile_reshape(wp.tile(total[0]), shape=(partitions, targets_per_block)), axis=0)
            ry = wp.tile_sum(wp.tile_reshape(wp.tile(total[1]), shape=(partitions, targets_per_block)), axis=0)
            rz = wp.tile_sum(wp.tile_reshape(wp.tile(total[2]), shape=(partitions, targets_per_block)), axis=0)
            total = vector(rx[lane % targets_per_block], ry[lane % targets_per_block], rz[lane % targets_per_block])
        if i < points.shape[0] and (not wp.static(cooperative) or part == 0):
            output[(0 if wp.static(cooperative) else part) * points.shape[0] + i, 0] = total[0]
            output[(0 if wp.static(cooperative) else part) * points.shape[0] + i, 1] = total[1]
            output[(0 if wp.static(cooperative) else part) * points.shape[0] + i, 2] = total[2]

    # CPU tracing visits the CUDA branch too; load on the execution device.
    return wp.jax_kernel(boundary_gather,
                         block_dim=block_size,
                         module_preload_mode=wp.JaxModulePreloadMode.NONE)


def _source_lanes(affine, scalar, continuations_per_root):
    roots = scalar // continuations_per_root if continuations_per_root else 0
    # Root continuations are serial within each independent source lane.
    units = max(roots, affine - roots) if continuations_per_root else max(affine, scalar)
    return min(_BLOCK_SIZE, 1 << max(0, units - 1).bit_length())


def _partition_count(queries, sources):
    # Bound scratch to 32 xyz vectors per query and avoid tiny source chunks.
    blocks = (queries + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    return min(32, max(1, (512 + blocks - 1) // blocks), max(1, sources // 2048))


@lru_cache(maxsize=16)
def _reduce_kernel(dtype, partitions):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64

    @wp.kernel(enable_backward=False, module="unique")
    def reduce(partial: wp.array2d(dtype=scalar), output: wp.array2d(dtype=scalar)):
        i = wp.tid()
        for axis in range(3):
            total = scalar(0.0)
            for part in range(partitions):
                total += partial[part * output.shape[0] + i, axis]
            output[i, axis] = total

    return wp.jax_kernel(reduce, module_preload_mode=wp.JaxModulePreloadMode.NONE)


def _launch_gather(points, velocities, samples, scalar_samples, regularization,
                   *, split=False, continuations_per_root=0):
    # Eight scalars per affine row align shared loads while preserving the
    # public seven-field format. Packing is O(M), with no target/source matrix.
    samples = jnp.pad(samples, ((0, 0), (0, 1)))
    cooperative = points.dtype == np.dtype("float64")
    block = _BLOCK_SIZE
    if cooperative:
        partitions = _source_lanes(len(samples), len(scalar_samples), continuations_per_root)
    else:
        partitions = _partition_count(len(points), len(samples) + len(scalar_samples))
    target_lanes = block // partitions if cooperative else block
    launch_size = ((len(points) + target_lanes - 1) // target_lanes) * target_lanes
    partial = _gather_kernel(points.dtype, regularization, split,
                             continuations_per_root, partitions)(
        points, velocities, samples, scalar_samples,
        launch_dims=launch_size * partitions,
        output_dims={'output': (len(points) if cooperative else len(points) * partitions, 3)})[0]
    if cooperative or partitions == 1:
        return partial
    return _reduce_kernel(points.dtype, partitions)(
        partial, launch_dims=len(points), output_dims={'output': points.shape})[0]


def boundary_gather(points: jax.Array, velocities: jax.Array,
                    samples: jax.Array, regularization: float) -> jax.Array:
    """Return the unnormalized cached sum, with the same kernel as _green.

    Float32 uses shared source tiles and bounded ordered partial sums. Float64
    uses up to 128 source lanes per target, reduced within the same block.
    These preserve the mathematical sum but change floating-point grouping.

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
    return _launch_gather(points, velocities, samples,
                          jnp.empty((0, 4), points.dtype), regularization)


def boundary_gather_split(points: jax.Array, velocities: jax.Array,
                          affine_samples: jax.Array, scalar_samples: jax.Array,
                          regularization: float, *,
                          continuations_per_root: int = 0) -> jax.Array:
    """Gather affine (M,7) and scalar (N,4) records on CUDA.

    Scalar rows are (position, a) with b=0; callers must ensure this invariant.
    For K>0, each leading affine root owns K consecutive scalar hops;
    remaining affine rows are pseudo sources. For K=0, phases are independent.
    Inputs share float32/float64 dtype. The sum is unnormalized; parallel
    reduction changes floating-point grouping.
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
    return _launch_gather(points, velocities, affine_samples, scalar_samples,
                          regularization, split=True,
                          continuations_per_root=continuations_per_root)
