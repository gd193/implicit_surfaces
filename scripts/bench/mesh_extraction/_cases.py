"""Shared JAX fields and adapters for mesh benchmarks."""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
import warp.geometry as geometry

from surfaces.meshes import TriMesh
from meshExtractor import (
    AdaptiveMeshResult, DC_COMPILER_OPTIONS, DualContouringBuffers,
    make_dual_contouring, make_dual_contouring_resident, make_marching_cubes,
)
from surfaces.implicit.lipschitz import lipschitzImplicit
from meshExtractor._field import _FieldEvaluator
from scripts.bench.mesh_extraction._warp_mc import _extract_dense, _extract_sparse

FIELDS = ("sphere", "torus", "neural", "gyroid")
FIXED_MC_BACKENDS = ("warp-dense", "warp-sparse", "compact-dense")
ADAPTIVE_BACKENDS = {"mc-adaptive": make_marching_cubes, "dual-adaptive": make_dual_contouring}
BACKENDS = (*ADAPTIVE_BACKENDS, *FIXED_MC_BACKENDS, "dual-dense", "dual-sparse")
LOWER, UPPER = (-1., -1., -1.), (1., 1., 1.)


def field(name, params, points):
    p = points - jnp.array([.031, -.023, .017], dtype=jnp.float32)
    sphere = jnp.linalg.norm(p, axis=-1) - .613
    if name == "sphere":
        return sphere
    if name == "torus":
        q = jnp.stack((jnp.linalg.norm(p[..., :2], axis=-1) - .53, p[..., 2]), axis=-1)
        return jnp.linalg.norm(q, axis=-1) - .173
    if name == "gyroid":
        q = p * (3 * jnp.pi)
        return jnp.sum(jnp.sin(q) * jnp.cos(jnp.roll(q, -1, axis=-1)), axis=-1)
    x = p
    for weight, bias in params[:-1]:
        x = jnp.tanh(x @ weight + bias)
    weight, bias = params[-1]
    return sphere + .08 * (x @ weight + bias)[..., 0]


@dataclass
class Fixture:
    name: str
    params: tuple
    lipschitz: float
    level: float = 0.

    def __call__(self, params, points):
        return field(self.name, params, points)


def fixture(name, level=0.):
    if name not in FIELDS:
        raise ValueError(name)
    rng = np.random.default_rng(817)
    params = []
    if name == "neural":
        for rows, cols in ((3, 32), (32, 32), (32, 1)):
            w = rng.normal(size=(rows, cols))
            w *= .8 / np.linalg.norm(w, 2)
            params.append((jnp.asarray(w, dtype=jnp.float32),
                           jnp.asarray(rng.normal(0, .1, cols), dtype=jnp.float32)))
    bound = 6 * np.pi * np.sqrt(3) if name == "gyroid" else 1.05 if name == "neural" else 1.
    return Fixture(name, tuple(params), float(bound), level)


def make_runner(backend, scene, cells, phase, batch_size, root_steps=0, *,
                capacity=None, sparse_capacity=None, resident=False):
    """Prepare resident inputs and return the measured operation."""
    details = {"input": "implicit_function", "field_evaluation_in_timed_region": True}
    if backend in ADAPTIVE_BACKENDS:
        if phase != "pipeline":
            raise ValueError(f"{backend} supports pipeline timing only")
        depth = (cells - 1).bit_length()
        tolerance = np.nextafter(2. * np.sqrt(3.) / cells, np.inf)
        surface = lipschitzImplicit(scene.lipschitz, fn=scene, level_set=scene.level, dim=3)
        extract = ADAPTIVE_BACKENDS[backend](surface, (LOWER, UPPER), tolerance=tolerance, max_depth=depth)
        details.update(max_depth=depth, tolerance=tolerance,
                       gradient_evaluation_in_timed_region=backend == "dual-adaptive")
        return partial(extract, scene.params), details
    if backend in ("dual-dense", "dual-sparse"):
        if capacity is None or phase != "pipeline":
            raise ValueError("fixed-capacity dual contouring needs a pipeline run and capacity")
        bound = scene.lipschitz if backend == "dual-sparse" else None
        extract = make_dual_contouring_resident(
            scene, (LOWER, UPPER), cells, lipschitz_bound=bound, level=scene.level,
            batch_size=batch_size, root_steps=root_steps, capacity=capacity, sparse_capacity=sparse_capacity)
        compiled = jax.jit(extract, compiler_options=DC_COMPILER_OPTIONS)

        def run():
            buffers = compiled(scene.params)
            return buffers if resident else buffers.compact()

        details.update(gradient_evaluation_in_timed_region=True, root_steps=root_steps,
                       capacity=capacity, sparse_capacity=sparse_capacity, resident=resident)
        return run, details

    evaluate = _FieldEvaluator(scene, batch_size)
    if backend in ("warp-dense", "compact-dense"):
        implementation = _extract_dense if backend == "compact-dense" else geometry.IsoSurfaceMarchingCubes.extract

        def extract(values):
            return implementation(values, threshold=scene.level, lower=LOWER, upper=UPPER)

        if phase == "extraction":
            values = evaluate.dense(scene.params, (cells+1,)*3, LOWER, UPPER)
            run = partial(extract, values)
            details.update(input="dense_values", field_evaluation_in_timed_region=False)
        else:
            def run():
                return extract(evaluate.dense(scene.params, (cells+1,)*3, LOWER, UPPER))
    elif backend == "warp-sparse":
        if phase == "extraction":
            if cells & (cells - 1):
                raise ValueError("sparse extraction-only requires a power-of-two cell resolution")
            occupied, width = geometry.lipschitz_octree(
                partial(evaluate, scene.params), LOWER, 2., cells.bit_length() - 1,
                threshold=scene.level, lipschitz_bound=scene.lipschitz)
            offsets = jnp.asarray(geometry.IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS)
            points = -1. + (wp.to_jax(occupied)[:, None, :] + offsets) * width
            values = wp.from_jax(evaluate.values(scene.params, points.reshape(-1, 3)).reshape(-1, 8))
            run = partial(geometry.sparse_marching_cubes_from_cells, occupied, values,
                          origin=LOWER, cell_width=width, threshold=scene.level)
            details.update(input="sparse_cells_and_corner_values", cells=occupied.shape[0],
                           field_evaluation_in_timed_region=False)
        else:
            run = partial(_extract_sparse, evaluate, scene.params, (cells+1,)*3,
                          LOWER, UPPER, scene.level, scene.lipschitz)
    else:
        raise ValueError(backend)
    return run, details


def host_mesh(mesh):
    if isinstance(mesh, AdaptiveMeshResult):
        mesh = mesh.mesh
    if isinstance(mesh, DualContouringBuffers):
        mesh = mesh.compact()
    if isinstance(mesh, TriMesh):
        return np.asarray(mesh.vertices), np.asarray(mesh.elements)
    return tuple((x.numpy() if isinstance(x, wp.array) else np.asarray(x)).reshape(-1, 3) for x in mesh)


def validate_mesh(vertices, faces, scene, cells, *, tolerance=None):
    if not np.isfinite(vertices).all() or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
        raise AssertionError("invalid mesh coordinates or indices")
    if vertices.size and (vertices.min() < -1.00001 or vertices.max() > 1.00001):
        raise AssertionError("mesh outside domain")
    values = np.asarray(_FieldEvaluator(scene, 65536).values(scene.params, jnp.asarray(vertices))) - scene.level
    residual = float(np.max(np.abs(values), initial=0.))
    distance_limit = 2. / cells if tolerance is None else tolerance
    if residual > scene.lipschitz * distance_limit + 1e-5:
        raise AssertionError(f"field residual too large: {residual}")
    return {"vertices": len(vertices), "triangles": len(faces),
            "output_bytes": vertices.nbytes + faces.nbytes, "max_field_residual": residual}
