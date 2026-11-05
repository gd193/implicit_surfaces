"""Shared grid sampling and edge refinement for dual contouring."""
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp


def _grid_points(indices, nodes, origin, spacing):
    # Preserve grid rounding before field/refinement arithmetic (native/dual_grid.h).
    ny, nz = nodes[1:]
    coords = jnp.stack((indices // (ny * nz), (indices // nz) % ny, indices % nz), axis=-1)
    scaled = jax.lax.optimization_barrier(coords.astype(jnp.float32) * jnp.asarray(spacing, jnp.float32))
    return jax.lax.optimization_barrier(jnp.asarray(origin, jnp.float32) + scaled)


def _make_dense_sampler(evaluate, nodes, origin, spacing):
    count = prod(nodes)
    batch_size = min(count, evaluate.batch_size)
    @jax.jit
    def sample(params):
        def batch(i, values):
            # Overlap the last batch to avoid a padded full-grid scalar copy.
            start = jnp.minimum(i * batch_size, count - batch_size)
            points = _grid_points(start + jnp.arange(batch_size), nodes, origin, spacing)
            return jax.lax.dynamic_update_slice(values, evaluate.evaluate(params, points), (start,))
        return jax.lax.fori_loop(0, (count + batch_size - 1) // batch_size,
                                 batch, jnp.empty(count, dtype=jnp.float32))
    return sample


def _refine_edges(evaluate, params, lo, hi, flo, fhi, level, steps):
    def step(_, state):
        lo, hi, flo, fhi = state
        t = jnp.clip(flo / (flo - fhi), .1, .9)
        middle = lo + t[:, None] * (hi - lo)
        fm = evaluate(params, middle) - level
        same = (fm < 0) == (flo < 0)
        return (jnp.where(same[:, None], middle, lo), jnp.where(same[:, None], hi, middle),
                jnp.where(same, fm, flo), jnp.where(same, fhi, fm))
    lo, hi, flo, fhi = jax.lax.fori_loop(0, steps, step, (lo, hi, flo, fhi))
    return lo + jnp.clip(flo / (flo - fhi), 0., 1.)[:, None] * (hi - lo)


def _make_cell_test(evaluate, cells_shape, lower, upper, level, bound):
    depth = (max(cells_shape)-1).bit_length()
    root_width = wp.vec3(*((np.asarray(upper)-lower)/np.asarray(cells_shape))) * float(1 << depth)

    def test(params, cells, d):
        width = root_width * (1.0 / float(1 << d))
        band = jnp.float32(bound * .5 * wp.length(width))
        centers = jnp.asarray(lower, jnp.float32) + jnp.asarray(width)*(cells.astype(jnp.float32)+.5)
        values = evaluate.evaluate(params, jnp.clip(centers, jnp.asarray(lower), jnp.asarray(upper)))
        keep = jnp.abs(values-level) <= band
        if d == depth:
            keep &= jnp.all(cells < jnp.asarray(cells_shape), axis=1)
        return keep.astype(jnp.int32)

    return test
