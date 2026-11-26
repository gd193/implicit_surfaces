"""Shared bounded JAX field evaluation and extraction-grid validation."""
from functools import lru_cache
from math import prod
from operator import index

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp


def _map_batches(fn, points, batch_size):
    count = points.shape[0]
    if count <= batch_size:
        return fn(points)
    padded = jnp.pad(points, ((0, -count % batch_size), (0, 0)))
    mapped = jax.lax.map(fn, padded.reshape(-1, batch_size, 3))
    return mapped.reshape((-1,) + mapped.shape[2:])[:count]


class _FieldEvaluator:
    def __init__(self, field, batch_size, *, output_shape=()):
        self.batch_size = batch_size
        self.output_shape = output_shape

        def scalar(params, point):
            value = jnp.asarray(field(params, point))
            if value.shape != output_shape or value.dtype != jnp.float32:
                expected = "a scalar float32" if output_shape == () else "a float32 vector of shape (3,)"
                raise TypeError(f"field/gradient must return {expected}")
            return value

        self.evaluate = jax.jit(jax.vmap(scalar, in_axes=(None, 0)))
        self.values = jax.jit(self._values)

    def _values(self, params, points):
        return _map_batches(lambda p: self.evaluate(params, p), points, self.batch_size)

    def __call__(self, params, points):
        return wp.from_jax(self.values(params, wp.to_jax(points)),
                           dtype=wp.vec3 if self.output_shape == (3,) else None)

    def dense(self, params, shape, lower, upper):
        batch = self._dense_batch(tuple(shape), tuple(lower), tuple(upper))
        out = wp.empty(shape, dtype=wp.float32)
        flat = out.flatten()
        copied = wp.Event() if out.device.is_cuda else None
        previous = None
        for start in range(0, flat.size, self.batch_size):
            values = wp.from_jax(batch(jnp.int32(start), params))
            if copied is not None and previous is not None:
                wp.synchronize_event(copied)
            wp.copy(flat, values, dest_offset=start, count=min(self.batch_size, flat.size-start))
            if copied is not None:
                wp.record_event(copied)
            previous = values  # Retain the JAX source until its asynchronous copy completes.
        if copied is not None:
            wp.synchronize_event(copied)
        return out

    @lru_cache(maxsize=4)
    def _dense_batch(self, shape, lower, upper):
        nx, ny, nz = shape

        @jax.jit
        def batch(start, params):
            i = jnp.minimum(start + jnp.arange(self.batch_size), nx*ny*nz - 1)
            p = jnp.stack((i // (ny*nz), (i // nz) % ny, i % nz), axis=-1)
            lo, hi = jnp.asarray(lower, jnp.float32), jnp.asarray(upper, jnp.float32)
            delta = (hi-lo) / jnp.asarray((nx-1, ny-1, nz-1), jnp.float32)
            return self.evaluate(params, lo + p.astype(jnp.float32)*delta)

        return batch


def _validate_grid(bounds, resolution, level, lipschitz_bound, batch_size):
    cells = (index(resolution),)*3 if np.isscalar(resolution) else tuple(map(index, resolution))
    lower, upper = (np.asarray(b, np.float32) for b in bounds)
    batch_size = index(batch_size)
    if len(cells) != 3 or min(cells) < 1 or batch_size < 1:
        raise ValueError("resolution and batch_size must be positive")
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
        raise ValueError("bounds must be finite increasing 3D corners")
    if not np.isfinite(level) or (lipschitz_bound is not None and
                                (not np.isfinite(lipschitz_bound) or lipschitz_bound <= 0)):
        raise ValueError("level must be finite and Lipschitz bound positive and finite")
    nodes = tuple(n+1 for n in cells)
    if lipschitz_bound is None and max(3*prod(nodes), 15*prod(cells)) > np.iinfo(np.int32).max:
        raise ValueError("dense grid exceeds int32 mesh indexing")
    return cells, nodes, lower, upper, batch_size


def _parameter_device(params):
    devices = {v.device for v in jax.tree_util.tree_leaves(params) if isinstance(v, jax.Array)}
    if len(devices) > 1:
        raise ValueError("parameters must share one device")
    device = next(iter(devices), jax.devices()[0])
    return wp.device_from_jax(device)
