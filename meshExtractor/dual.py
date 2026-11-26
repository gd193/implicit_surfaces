"""Dual contouring with bounded JAX field and Hermite evaluation."""
from functools import partial
from operator import index
import importlib

import jax
import numpy as np
import warp as wp
from warp._src import utils

from surfaces.meshes import TriMesh
from meshExtractor._field import _FieldEvaluator, _validate_grid, _map_batches, _parameter_device
from meshExtractor._dual_common import _grid_points, _make_dense_sampler, _refine_edges, _make_cell_test
from meshExtractor._dual_resident import make_dual_contouring_resident
from meshExtractor import _dual_warp as topology

sparse = importlib.import_module('warp._src.geometry.sparse_marching_cubes')

DC_COMPILER_OPTIONS = {
    'xla_gpu_enable_command_buffer':
        'CONDITIONAL,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,DYNAMIC_SLICE_FUSION,FUSION,WHILE',
    'xla_gpu_graph_min_graph_size': 1,
}


def _make_pruner(evaluate, cells_shape, lower, upper, level, bound):
    test = _make_cell_test(evaluate, cells_shape, lower, upper, level, bound)
    def prune(params, cells, d):
        return _map_batches(lambda c: test(params, c, d), cells, evaluate.batch_size)
    return jax.jit(prune, static_argnums=2)


def _sample_sparse(evaluate, prune, params, cells_shape, origin, spacing):
    device = wp.get_device()
    depth = (max(cells_shape)-1).bit_length()
    resolution = 1 << depth
    cells = wp.array([[0, 0, 0]], dtype=wp.vec3i, device=device)
    for d in range(depth+1):
        owner = prune(params, wp.to_jax(cells), d)
        scan = wp.empty(cells.size, dtype=int, device=device)
        utils.array_scan(wp.from_jax(owner), scan, inclusive=True)
        count = int(scan[-1:].numpy()[0])
        if not count:
            return None
        out = wp.empty(count if d == depth else 8*count, dtype=wp.vec3i, device=device)
        wp.launch(topology.compact_children, cells.size, [cells, scan, out, d == depth], device=device)
        cells = out
    del owner, scan  # Release pruning temporaries before corner deduplication.
    corners, points, _, _ = sparse._dedupe_corners(cells, cells.size, (0, 0, 0),
                                                   resolution + 1, origin, spacing, device)
    values = evaluate(params, points)
    return cells, corners, points, values, origin, spacing


def _make_refiner(evaluate, normals, level, root_steps, dense_grid=None):
    @jax.jit
    def refine(params, point_array, value_array, endpoints):
        if dense_grid is None:
            lo, hi = point_array[endpoints[:, 0]], point_array[endpoints[:, 1]]
        else:
            nodes, origin, spacing = dense_grid
            lo = _grid_points(endpoints[:, 0], nodes, origin, spacing)
            hi = _grid_points(endpoints[:, 1], nodes, origin, spacing)
        flo, fhi = value_array[endpoints[:, 0]] - level, value_array[endpoints[:, 1]] - level
        positions = _refine_edges(evaluate.values, params, lo, hi, flo, fhi, level, root_steps)
        return positions, normals.values(params, positions)
    return refine


def _extract_samples(params, samples, *, normals, level, refiner):
    if samples is None:
        return wp.empty(0, dtype=wp.vec3), wp.empty(0, dtype=wp.int32)
    cells, corners, points, values, origin, spacing = samples
    workspace = topology.prepare(cells, corners, points, values, origin, spacing, level)
    if refiner is not None and workspace.positions.size:
        positions_jax, gradients_jax = refiner(params, wp.to_jax(points) if points is not None else None, wp.to_jax(values),
                                             wp.to_jax(workspace.edge_endpoints))
        workspace.positions = wp.from_jax(positions_jax, dtype=wp.vec3)
        gradient_values = wp.from_jax(gradients_jax, dtype=wp.vec3)
    else:
        gradient_values = normals(params, workspace.positions)
    result = topology.finish(workspace, gradient_values)
    if wp.get_device().is_cuda:
        wp.synchronize_stream()  # Retain all shared JAX owners until Warp consumes them.
    return result


def make_dual_contouring(field, bounds, resolution=128, *, gradient=None, level=0.,
                         lipschitz_bound=None, batch_size=65536, root_steps=0,
                         capacity=None, sparse_capacity=None):
    """Return ``extract(params) -> TriMesh``.

    A conservative Lipschitz bound enables sparse sampling. Explicit capacities
    use the resident core with one count readback. Field/gradient contracts,
    refinement and topology limits are documented in docs/mesh_extraction.md.
    """
    if capacity is not None:
        run = jax.jit(make_dual_contouring_resident(
            field, bounds, resolution, gradient=gradient, level=level,
            lipschitz_bound=lipschitz_bound, batch_size=batch_size, root_steps=root_steps,
            capacity=capacity, sparse_capacity=sparse_capacity), compiler_options=DC_COMPILER_OPTIONS)

        def exact(params):
            return run(params).compact()
        return exact
    if sparse_capacity is not None:
        raise ValueError('sparse_capacity requires capacity')
    sample, triangulate = _make_dual_stages(
        field, bounds, resolution, gradient=gradient, level=level, lipschitz_bound=lipschitz_bound,
        batch_size=batch_size, root_steps=root_steps)

    def extract(params):
        with wp.ScopedDevice(_parameter_device(params)):
            vertices, indices = triangulate(params, sample(params))
            return TriMesh(wp.to_jax(vertices), wp.to_jax(indices).reshape(-1, 3))
    return extract


def _make_dual_stages(field, bounds, resolution, *, gradient=None, level, lipschitz_bound, batch_size, root_steps):
    """Prepare sampling and triangulation, which can be timed independently."""
    if wp.__version__.split('+')[0] != '1.18.0':
        raise RuntimeError('dual contouring requires Warp 1.18.0')
    cells, nodes, lower, upper, batch_size = _validate_grid(bounds, resolution, level, lipschitz_bound, batch_size)
    root_steps = index(root_steps)
    if root_steps < 0:
        raise ValueError('root_steps must be nonnegative')
    lower, upper = tuple(lower), tuple(upper)
    evaluate = _FieldEvaluator(field, batch_size)
    normals = _FieldEvaluator(jax.grad(field, argnums=1) if gradient is None else gradient, batch_size, output_shape=(3,))

    origin = wp.vec3(*lower)
    spacing = wp.vec3(*((np.asarray(upper) - lower) / np.asarray(cells)))
    if lipschitz_bound is None:
        dense_grid = (nodes, lower, tuple(spacing))
        dense_sampler = _make_dense_sampler(evaluate, *dense_grid)

        def sample(params):
            return cells, None, None, wp.from_jax(dense_sampler(params)), origin, spacing
    else:
        dense_grid = None
        prune = _make_pruner(evaluate, cells, lower, upper, level, lipschitz_bound)

        def sample(params):
            return _sample_sparse(evaluate, prune, params, cells, origin, spacing)

    refiner = _make_refiner(evaluate, normals, level, root_steps, dense_grid) if root_steps else None

    return sample, partial(_extract_samples, normals=normals, level=level, refiner=refiner)
