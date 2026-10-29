"""Fused grid interpolation, pressure integrands and cooperative reductions.

JAX supplies the original random draws in bounded query batches. Optional
analytic volume capabilities fuse sample transforms and geometry masks too.
Warp keeps interpolation indices and per-sample contributions local;
only one vector per query leaves each reduction kernel.
"""
import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from examples.fluids.pressure_prototype.boundary_integrator import (
    ExteriorWoBPressureIntegrator,
    volume_sample_pairs,
)
from examples.fluids.pressure_prototype.warp_volume_geometry import volume_geometry, warp_volume_contains

_BLOCK_SIZE = 256
_QUERY_TILE = 256


def _block_size(samples):
    # A warp suffices for small sample sets; larger sets keep parallel loads.
    return min(_BLOCK_SIZE, max(32, 1 << (samples-1).bit_length()))


@lru_cache(maxsize=16)
def _grid_sampler(dtype, lower_values, spacing_values):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    vector = wp.vec3 if scalar == wp.float32 else wp.vec3d
    lower = wp.constant(vector(*lower_values))
    inverse_spacing = wp.constant(vector(*(dtype.type(1.)/dtype.type(v) for v in spacing_values)))

    @wp.func
    def sample_vector(field: wp.array4d(dtype=scalar), y: vector) -> vector:
        index = wp.cw_mul(y-lower, inverse_spacing)
        lo = wp.vec3i(int(wp.floor(index[0])), int(wp.floor(index[1])),
                     int(wp.floor(index[2])))
        fraction = index-vector(scalar(lo[0]), scalar(lo[1]), scalar(lo[2]))
        total = vector(scalar(0.))
        for a in range(2):
            i = wp.clamp(lo[0]+a, 0, field.shape[0]-1)
            wa = fraction[0]
            if a == 0:
                wa = scalar(1.)-wa
            for b in range(2):
                j = wp.clamp(lo[1]+b, 0, field.shape[1]-1)
                wb = fraction[1]
                if b == 0:
                    wb = scalar(1.)-wb
                for c in range(2):
                    k = wp.clamp(lo[2]+c, 0, field.shape[2]-1)
                    wc = fraction[2]
                    if c == 0:
                        wc = scalar(1.)-wc
                    value = vector(field[i, j, k, 0], field[i, j, k, 1],
                                   field[i, j, k, 2])
                    total += value*((wa*wb)*wc)
        return total

    return sample_vector


@lru_cache(maxsize=16)
def _kernels(dtype, lower_values, spacing_values, regularization, geometry=None,
             block_size=_BLOCK_SIZE):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    vector = wp.vec3 if scalar == wp.float32 else wp.vec3d
    lower = wp.constant(vector(*lower_values))
    spacing = wp.constant(vector(*spacing_values))
    inverse_spacing = wp.constant(vector(*(dtype.type(1.)/dtype.type(v) for v in spacing_values)))
    four_pi = wp.constant(scalar(4*math.pi))
    reg = wp.constant(scalar(regularization))
    tiny = wp.constant(scalar(np.finfo(dtype).tiny))
    sample_vector = _grid_sampler(dtype, lower_values, spacing_values)

    def face_sampler(axis):
        # Compile the component and tangential axes as constants. Dynamic
        # vector indexing otherwise turns these four loads into local-array
        # address calculations in every sample interaction.
        a, b = (i for i in range(3) if i != axis)

        @wp.func
        def sample_face(field: wp.array4d(dtype=scalar), y: vector) -> scalar:
            dims = wp.vec3i(field.shape[0], field.shape[1], field.shape[2])
            index = wp.cw_mul(y-lower, inverse_spacing)
            ia, ib = int(wp.floor(index[a])), int(wp.floor(index[b]))
            fa, fb = index[a]-scalar(ia), index[b]-scalar(ib)
            point = wp.vec3i(0)
            if y[axis] > lower[axis]:
                point[axis] = dims[axis]-1
            total = scalar(0.)
            for da in range(2):
                point[a] = wp.clamp(ia+da, 0, dims[a]-1)
                wa = fa
                if da == 0:
                    wa = scalar(1.)-fa
                for db in range(2):
                    point[b] = wp.clamp(ib+db, 0, dims[b]-1)
                    wb = fb
                    if db == 0:
                        wb = scalar(1.)-fb
                    total += (wa*wb)*field[point[0], point[1], point[2], axis]
            return total

        return sample_face

    sample_x, sample_y, sample_z = (face_sampler(axis) for axis in range(3))

    @wp.func
    def volume_contribution(field: wp.array4d(dtype=scalar), y: vector, u: vector,
                            d: vector, factor: scalar) -> vector:
        delta = sample_vector(field, y)-u
        return factor*(scalar(3.)*d*wp.dot(d, delta)-delta)

    # The reference rounds each sample position before interpolating. Contracting
    # its multiply/add changes near-query samples and R/r amplifies that error.
    @wp.kernel(enable_backward=False, module='unique', module_options={'fuse_fp': False})
    def volume_kernel(
        field: wp.array4d(dtype=scalar),
        points: wp.array2d(dtype=scalar),
        velocities: wp.array2d(dtype=scalar),
        directions: wp.array3d(dtype=scalar),
        distances: wp.array2d(dtype=scalar),
        radii: wp.array(dtype=scalar),
        valid: wp.array3d(dtype=wp.bool),
        output: wp.array2d(dtype=scalar),
    ):
        tid = wp.tid()
        query, lane = tid//block_size, tid % block_size
        x = vector(points[query, 0], points[query, 1], points[query, 2])
        u = vector(velocities[query, 0], velocities[query, 1], velocities[query, 2])
        total = vector(scalar(0.))
        for k in range(lane, distances.shape[1], block_size):
            d = vector(directions[query, k, 0], directions[query, k, 1],
                       directions[query, k, 2])
            r = distances[query, k]
            factor = radii[query]/r
            if valid[query, k, 0]:
                total += volume_contribution(field, x+r*d, u, d, factor)
            if valid[query, k, 1]:
                total += volume_contribution(field, x-r*d, u, d, factor)
        reduced = wp.tile_sum(wp.tile(total, preserve_type=True))[0]
        if lane == 0:
            result = reduced/scalar(2*distances.shape[1])
            output[query, 0] = result[0]
            output[query, 1] = result[1]
            output[query, 2] = result[2]

    @wp.kernel(enable_backward=False, module='unique')
    def pseudo_kernel(
        field: wp.array4d(dtype=scalar),
        points: wp.array2d(dtype=scalar),
        velocities: wp.array2d(dtype=scalar),
        samples: wp.array3d(dtype=scalar),
        output: wp.array2d(dtype=scalar),
    ):
        tid = wp.tid()
        query, lane = tid//block_size, tid % block_size
        x = vector(points[query, 0], points[query, 1], points[query, 2])
        u = vector(velocities[query, 0], velocities[query, 1], velocities[query, 2])
        upper = lower+wp.cw_mul(spacing, vector(scalar(field.shape[0]-1),
            scalar(field.shape[1]-1), scalar(field.shape[2]-1)))
        total = vector(scalar(0.))
        for k in range(lane, samples.shape[1], block_size):
            y = vector(samples[query, k, 0], samples[query, k, 1], samples[query, k, 2])
            normal = vector(samples[query, k, 3], samples[query, k, 4], samples[query, k, 5])
            flux = scalar(0.)
            if (wp.abs(normal[0]) == scalar(1.) and normal[1] == scalar(0.) and normal[2] == scalar(0.)
                    and (y[0] <= lower[0] or y[0] >= upper[0])):
                flux = normal[0]*(sample_x(field, y)-u[0])
            elif (wp.abs(normal[1]) == scalar(1.) and normal[0] == scalar(0.) and normal[2] == scalar(0.)
                    and (y[1] <= lower[1] or y[1] >= upper[1])):
                flux = normal[1]*(sample_y(field, y)-u[1])
            elif (wp.abs(normal[2]) == scalar(1.) and normal[0] == scalar(0.) and normal[1] == scalar(0.)
                    and (y[2] <= lower[2] or y[2] >= upper[2])):
                flux = normal[2]*(sample_z(field, y)-u[2])
            else:
                flux = wp.dot(normal, sample_vector(field, y)-u)
            r = y-x
            r2 = wp.dot(r, r)
            denominator = wp.max(four_pi*wp.sqrt(r2)*wp.max(r2, reg), tiny)
            total += (r/denominator)*(samples[query, k, 6]*flux)
        reduced = wp.tile_sum(wp.tile(total, preserve_type=True))[0]
        if lane == 0:
            result = reduced/scalar(samples.shape[1])
            output[query, 0] = result[0]
            output[query, 1] = result[1]
            output[query, 2] = result[2]

    kernels = [volume_kernel, pseudo_kernel]
    if geometry is not None:
        contains = warp_volume_contains(dtype, geometry)

        @wp.kernel(enable_backward=False, module='unique', module_options={'fuse_fp': False})
        def prepared_volume_kernel(
            field: wp.array4d(dtype=scalar),
            points: wp.array2d(dtype=scalar),
            velocities: wp.array2d(dtype=scalar),
            directions: wp.array3d(dtype=scalar),
            radial_uniforms: wp.array2d(dtype=scalar),
            radii: wp.array(dtype=scalar),
            output: wp.array2d(dtype=scalar),
        ):
            tid = wp.tid()
            query, lane = tid//block_size, tid % block_size
            x = vector(points[query, 0], points[query, 1], points[query, 2])
            u = vector(velocities[query, 0], velocities[query, 1], velocities[query, 2])
            radius = radii[query]
            total = vector(scalar(0.))
            for k in range(lane, radial_uniforms.shape[1], block_size):
                d = vector(directions[query, k, 0], directions[query, k, 1],
                           directions[query, k, 2])
                r = radius*radial_uniforms[query, k]
                plus, minus = x+r*d, x-r*d
                factor = radius/r
                if contains(plus):
                    total += volume_contribution(field, plus, u, d, factor)
                if contains(minus):
                    total += volume_contribution(field, minus, u, d, factor)
            reduced = wp.tile_sum(wp.tile(total, preserve_type=True))[0]
            if lane == 0:
                result = reduced/scalar(2*radial_uniforms.shape[1])
                output[query, 0] = result[0]
                output[query, 1] = result[1]
                output[query, 2] = result[2]

        kernels.append(prepared_volume_kernel)
    return tuple(wp.jax_kernel(kernel, block_dim=block_size,
        module_preload_mode=wp.JaxModulePreloadMode.NONE)
        for kernel in kernels)


def grid_volume(velocity, params, points, directions, distances, radii, valid):
    """Evaluate exactly the supplied radial pairs and masks on CUDA."""
    field = params['velocity']
    u = jax.vmap(velocity, (None, 0))(params, points)
    block_size = _block_size(distances.shape[1])
    kernel = _kernels(points.dtype, velocity.lower, velocity.spacing, 0.,
                      block_size=block_size)[0]
    return kernel(field, points, u, directions, distances, radii, valid,
                  launch_dims=len(points)*block_size,
                  output_dims={'output': points.shape})[0]


def grid_pseudo(velocity, params, points, samples, regularization):
    """Evaluate supplied (position, normal, inverse-area-PDF) records on CUDA."""
    field = params['velocity']
    u = jax.vmap(velocity, (None, 0))(params, points)
    block_size = _block_size(samples.shape[1])
    kernel = _kernels(points.dtype, velocity.lower, velocity.spacing, regularization,
                      block_size=block_size)[1]
    return kernel(field, points, u, samples, launch_dims=len(points)*block_size,
                  output_dims={'output': points.shape})[0]


def grid_volume_prepared(geometry, velocity, params, points, directions,
                         radial_uniforms, radii):
    """Fuse radial distances, rounded sample positions and analytic masks.

    Directions, radii and u(x) retain the reference JAX arithmetic: moving their
    transforms into Warp changes near-query rounding amplified by R/r.
    Radial uniforms retain the original keys, shapes and epsilon floor.
    """
    u = jax.vmap(velocity, (None, 0))(params, points)
    block_size = _block_size(radial_uniforms.shape[1])
    kernel = _kernels(points.dtype, velocity.lower, velocity.spacing, 0., geometry,
                      block_size)[2]
    return kernel(params['velocity'], points, u, directions, radial_uniforms, radii,
                  launch_dims=len(points)*block_size,
                  output_dims={'output': points.shape})[0]


def grid_source(integrator, velocity, params, points, keys, query_batch_size,
                *, include_pseudo=True):
    """Bound sample storage while preserving the original per-target keys."""
    return _grid_source(integrator, velocity, params, points, keys, query_batch_size,
                        include_pseudo=include_pseudo, prepare_volume=True)


def _grid_source(integrator, velocity, params, points, keys, query_batch_size,
                 *, include_pseudo, prepare_volume):
    if points.dtype not in (jnp.float32, jnp.float64) or params['velocity'].dtype != points.dtype:
        raise TypeError('fused grid velocity and points must share float32 or float64 dtype')
    if not len(points):
        return jnp.zeros_like(points)
    exterior = isinstance(integrator, ExteriorWoBPressureIntegrator)
    capability = volume_geometry(integrator.geometry) if prepare_volume else None

    def evaluate(batch):
        x, key = batch
        if exterior:
            split = jax.vmap(jax.random.split)(key)
            volume_keys, pseudo_keys = split[:, 0], split[:, 1]
        else:
            volume_keys = key

        def draw_volume(point, k):
            d, r, radius = volume_sample_pairs(integrator.geometry, point, k,
                                                integrator.volume_samples)
            contains = jax.vmap(integrator.geometry.contains)
            valid = jnp.stack((contains(point+r*d), contains(point-r*d)), axis=-1)
            return d, r[:, 0], radius, valid

        if capability is None:
            d, r, radii, valid = jax.vmap(draw_volume)(x, volume_keys)
            volume = grid_volume(velocity, params, x, d, r, radii, valid)
        else:
            def draw_uniforms(point, k):
                from utils.sampling_primitives import sample_uniform_2sphere
                kd, kr = jax.random.split(k)
                pairs = integrator.volume_samples//2
                return (sample_uniform_2sphere(kd, x.dtype, shape=(pairs,)),
                        jax.random.uniform(kr, (pairs, 1), dtype=x.dtype,
                                           minval=jnp.finfo(x.dtype).eps)[:, 0],
                        integrator.geometry.volume_radius(point))
            directions, radial, radii = jax.vmap(draw_uniforms)(x, volume_keys)
            volume = grid_volume_prepared(capability, velocity, params, x,
                                           directions, radial, radii)
        if not exterior or not include_pseudo:
            return volume

        def draw_pseudo(k):
            samples = jax.random.split(k, integrator.pseudo_samples)
            y, normal, area = jax.vmap(integrator.geometry.sample_pseudo_boundary,
                                      (0, None))(samples, x.dtype)
            return jnp.concatenate((y, normal, area[:, None]), axis=-1)

        records = jax.vmap(draw_pseudo)(pseudo_keys)
        return volume+grid_pseudo(velocity, params, x, records, integrator.regularization)

    batch_size = min(query_batch_size, _QUERY_TILE, len(points))
    # Map complete blocks and the unpadded tail separately: padding would also
    # create extra random targets and unnecessary work at arbitrary geometries.
    count = len(points)//batch_size*batch_size
    full = jax.lax.map(evaluate, (points[:count].reshape(-1, batch_size, 3),
                                 keys[:count].reshape((-1, batch_size)+keys.shape[1:])))
    result = full.reshape(-1, 3)
    if count != len(points):
        result = jnp.concatenate((result, evaluate((points[count:], keys[count:]))))
    return result
