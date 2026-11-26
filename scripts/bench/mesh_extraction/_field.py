"""Batched scalar field sampling for mesh extraction benchmarks."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import warp as wp


def _map_batches(fn, points, batch_size):
    count = points.shape[0]
    if count <= batch_size:
        return fn(points)
    padded = jnp.pad(points, ((0, -count % batch_size), (0, 0)))
    mapped = jax.lax.map(fn, padded.reshape(-1, batch_size, 3))
    return mapped.reshape((-1,) + mapped.shape[2:])[:count]


class _FieldEvaluator:
    def __init__(self, field, batch_size):
        self.batch_size = batch_size

        def scalar(params, point):
            value = jnp.asarray(field(params, point))
            if value.shape != () or value.dtype != jnp.float32:
                raise TypeError("field must return a scalar float32")
            return value

        self.evaluate = jax.jit(jax.vmap(scalar, in_axes=(None, 0)))
        self.values = jax.jit(self._values)

    def _values(self, params, points):
        return _map_batches(lambda p: self.evaluate(params, p), points, self.batch_size)

    def __call__(self, params, points):
        return wp.from_jax(self.values(params, wp.to_jax(points)))

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
