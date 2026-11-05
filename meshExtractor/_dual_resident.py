"""Fixed-capacity dual contouring; payload prefixes are valid only with status zero."""

from functools import partial
from math import prod
from operator import index
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import warp as wp
from meshExtractor._dual_common import _make_dense_sampler, _grid_points, _refine_edges
from meshExtractor._field import _FieldEvaluator, _validate_grid
from meshExtractor import _dual_resident_warp as kernels
from meshExtractor._dual_scan import scan


class ResidentMesh(NamedTuple):
    vertices: jax.Array
    faces: jax.Array
    counts: jax.Array  # C, E, T, overflow bits, frontier, raw corners, unique corners


def make_resident(field, bounds, resolution, *, capacity, batch_size, gradient=None, level=0.0, root_steps=0,
                  lipschitz_bound=None, sparse_capacity=None):
    if wp.__version__.split("+")[0] != "1.18.0":
        raise RuntimeError("dual contouring requires Warp 1.18.0")
    if lipschitz_bound is None and sparse_capacity is not None:
        raise ValueError("sparse_capacity requires a Lipschitz bound")
    cells, nodes, lower, upper, batch_size = _validate_grid(bounds, resolution, level, lipschitz_bound, batch_size)
    C, E, T = map(index, capacity)
    root_steps = index(root_steps)
    if min(C, E, T) < 1 or max(12 * C, 4 * E, 3 * T) > np.iinfo(np.int32).max:
        raise ValueError("capacities must be positive and int32-indexable")
    if root_steps < 0:
        raise ValueError("root_steps must be nonnegative")
    sparse = lipschitz_bound is not None
    if sparse:
        if sparse_capacity is None:
            raise ValueError("sparse_capacity=(F,K) is required")
        F, K = map(index, sparse_capacity)
        if min(F, K) < 1 or max(8 * F, 3 * K) > np.iinfo(np.int32).max:
            raise ValueError("F and K must be positive and int32-indexable")
    lower = tuple(lower)
    spacing = tuple(wp.vec3(*((upper - lower) / np.asarray(cells))))
    evaluate = _FieldEvaluator(field, batch_size)
    normals = _FieldEvaluator(jax.grad(field, 1) if gradient is None else gradient, batch_size, output_shape=(3,))
    if sparse:
        from meshExtractor._dual_sparse import make_sparse_inputs

        sample = make_sparse_inputs(
            evaluate, cells, lower, tuple(upper), level, lipschitz_bound, sparse_capacity=(F, K)
        )
    else:
        sample = _make_dense_sampler(evaluate, nodes, lower, spacing)
    wrap = partial(wp.jax_kernel, module_preload_mode=wp.JaxModulePreloadMode.NONE)
    classify = wrap(kernels.classify, num_outputs=2)
    compact = wrap(kernels.compact, num_outputs=2)
    incidence = wrap(kernels.incidence, num_outputs=2, in_out_argnames=["ids"])
    emit_edges = wrap(kernels.edges, num_outputs=3)
    if sparse:
        classify_sparse = wrap(kernels.classify_sparse, num_outputs=2, in_out_argnames=["edge_flags"])
        compact_sparse = wrap(kernels.compact_sparse, num_outputs=2)
    place = wrap(kernels.place)
    faces = wrap(kernels.faces)
    n = prod(cells)
    m = prod(nodes)
    B = min(batch_size, E)

    def hermite(params, positions, endpoints, values, counts, corner_points):
        live = jnp.where(counts[3] == 0, counts[1], jnp.int32(0))

        def batch(i, state):
            refined, gradient_values = state
            start = i * B
            ids = start + jnp.arange(B)
            valid = ids < live
            safe_ids = jnp.minimum(ids, live - 1)
            points = jnp.where(valid[:, None], positions[safe_ids], jnp.asarray(lower))
            if root_steps:
                pairs = endpoints[safe_ids]
                if sparse:
                    lo, hi = corner_points[pairs[:, 0]], corner_points[pairs[:, 1]]
                else:
                    lo = _grid_points(pairs[:, 0], nodes, lower, spacing)
                    hi = _grid_points(pairs[:, 1], nodes, lower, spacing)
                lo = jnp.where(valid[:, None], lo, jnp.asarray(lower))
                hi = jnp.where(valid[:, None], hi, jnp.asarray(lower))
                flo = jnp.where(valid, values[pairs[:, 0]] - level, -1.0)
                fhi = jnp.where(valid, values[pairs[:, 1]] - level, 1.0)
                points = _refine_edges(evaluate.evaluate, params, lo, hi, flo, fhi, level, root_steps)
            points = jnp.where(valid[:, None], points, jnp.asarray(lower))
            g = normals.evaluate(params, jax.lax.optimization_barrier(points))
            if root_steps:
                refined = refined.at[ids].set(
                    jnp.where(valid[:, None], points, 0.0), mode="drop", indices_are_sorted=True, unique_indices=True
                )
            return (
                refined,
                gradient_values.at[ids].set(
                    jnp.where(valid[:, None], g, 0.0), mode="drop", indices_are_sorted=True, unique_indices=True
                ),
            )

        return jax.lax.fori_loop(0, (live + B - 1) // B, batch, (positions, jnp.empty((E, 3), jnp.float32)))

    def extract(params):
        if sparse:
            candidates, corner_map, points, values, upstream = sample(params)
            ef, cf = classify_sparse(
                corner_map,
                values,
                upstream,
                level,
                jnp.zeros(3 * K, jnp.int32),
                launch_dims=F,
                output_dims={"cell_flags": F},
            )
            es, cs = scan(ef), scan(cf)
            nc = jnp.where(upstream[3] == 0, cs[-1], jnp.int32(-1))
            ne = jnp.where(upstream[3] == 0, es[-1], jnp.int32(-1))
            overflow = upstream[3] + (nc > C).astype(jnp.int32) + 2 * (ne > E).astype(jnp.int32)
            counts = upstream.at[0].set(nc).at[1].set(ne).at[3].set(overflow)
            active, cc = compact_sparse(
                candidates, corner_map, cs, counts, launch_dims=F, output_dims={"cells": C, "cc": (C, 8)}
            )
            ny, nz = 0, 0
        else:
            values = sample(params)
            points = jnp.empty((0, 3), jnp.float32)
            ef, cf = classify(
                values, *cells, level, launch_dims=max(3 * m, n), output_dims={"edge_flags": 3 * m, "cell_flags": n}
            )
            es, cs = scan(ef), scan(cf)
            overflow = (cs[-1] > C).astype(jnp.int32) + 2 * (es[-1] > E).astype(jnp.int32)
            counts = jnp.stack((cs[-1], es[-1], jnp.int32(-1), overflow, jnp.int32(0), jnp.int32(0), jnp.int32(0)))
            active, cc = compact(cs, counts, cells[1], cells[2], launch_dims=n, output_dims={"cells": C, "cc": (C, 8)})
            ny, nz = nodes[1], nodes[2]
        ids, ce = incidence(
            cc,
            values,
            level,
            es,
            counts,
            jnp.full((E, 4), -1, jnp.int32),
            launch_dims=C,
            output_dims={"cell_edges": (C, 12)},
        )
        positions, endpoints, flags = emit_edges(
            cc,
            values,
            points,
            level,
            ids,
            ce,
            counts,
            ny,
            nz,
            *lower,
            *spacing,
            launch_dims=E,
            output_dims={"positions": E, "endpoints": (E, 2), "flags": E},
        )
        fs = scan(flags)
        required = jnp.where(overflow == 0, 2 * fs[-1], jnp.int32(-1))
        counts = counts.at[2].set(required).at[3].set(overflow + 4 * (required > T).astype(jnp.int32))
        positions, g = hermite(params, positions, endpoints, values, counts, points)
        vertices = place(active, ce, positions, g, counts, *lower, *spacing, launch_dims=C, output_dims=C)[0]
        result = faces(ids, endpoints, values, level, fs, vertices, counts, launch_dims=E, output_dims=3 * T)[0]
        return ResidentMesh(vertices, result.reshape(T, 3), counts)

    return extract
