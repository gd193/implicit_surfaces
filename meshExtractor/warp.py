"""JAX scalar fields to indexed meshes; uses pinned Warp 1.18 kernels."""
from functools import lru_cache
from math import prod
from operator import index

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
import warp.geometry as geometry
from warp._src import utils
from warp._src.geometry import marching_cubes as mc
from warp._src.geometry.iso_surface import resolve_domain_bounds, validate_field


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
        count = points.shape[0]
        if count <= self.batch_size:
            return self.evaluate(params, points)
        padded = jnp.pad(points, ((0, -count % self.batch_size), (0, 0)))
        chunks = padded.reshape(-1, self.batch_size, 3)
        return jax.lax.map(lambda p: self.evaluate(params, p), chunks).reshape(-1)[:count]

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


def _extract_dense(field, threshold=0., *, lower=None, upper=None):
    # Private Warp APIs: revalidate the alias proof when upgrading the version pin.
    if wp.__version__.split("+")[0] != "1.18.0":
        raise RuntimeError("compact marching cubes requires Warp 1.18.0")
    if field.requires_grad:
        return mc.IsoSurfaceMarchingCubes.extract(field, threshold, lower=lower, upper=upper)
    validate_field(field)
    lower, delta = resolve_domain_bounds(field.shape, lower, upper)
    device, nodes = field.device, tuple(field.shape)
    if max(3*prod(nodes), 15*prod(n-1 for n in nodes)) > np.iinfo(np.int32).max:
        raise ValueError("dense grid exceeds int32 mesh indexing")
    edge_scan = wp.zeros(prod(nodes)*3, dtype=wp.int32, device=device)
    wp.launch(mc.extract_vertices_kernel, dim=nodes+(3,),
              inputs=[field, threshold, lower, delta, None, True],
              outputs=[edge_scan, None, None, None], device=device)
    utils.array_scan(edge_scan, edge_scan, inclusive=True)
    vertices = wp.empty(int(edge_scan[-1:].numpy()[0]), dtype=wp.vec3, device=device)
    boundary = wp.empty(vertices.shape[0], dtype=wp.bool, device=device)
    # Each emission thread reads its own scan slot, then replaces that same slot.
    edge_map = edge_scan.reshape(nodes+(3,))
    wp.launch(mc.extract_vertices_kernel, dim=nodes+(3,),
              inputs=[field, threshold, lower, delta, edge_scan, False],
              outputs=[None, vertices, boundary, edge_map], device=device)
    cells = tuple(n-1 for n in nodes)
    face_scan = wp.empty(prod(cells), dtype=wp.int32, device=device)
    tables = [mc._get_mc_case_to_tri_range_table(device),
              mc._get_mc_tri_local_inds_table(device), mc._get_mc_edge_offset_table(device)]
    wp.launch(mc.extract_faces_kernel, dim=cells,
              inputs=[field, threshold, edge_map, None, *tables, True],
              outputs=[face_scan, None], device=device)
    utils.array_scan(face_scan, face_scan, inclusive=True)
    indices = wp.empty(3*int(face_scan[-1:].numpy()[0]), dtype=wp.int32, device=device)
    wp.launch(mc.extract_faces_kernel, dim=cells,
              inputs=[field, threshold, edge_map, face_scan, *tables, False],
              outputs=[None, indices], device=device)
    return vertices, indices


def _extract_sparse(evaluate, params, nodes, lower, upper, level, bound):
    last_values = None

    def sample(points):
        nonlocal last_values
        # Each octree level synchronizes its count before the next evaluation.
        last_values = evaluate(params, points)
        return last_values

    mesh = geometry.sparse_marching_cubes(sample, *nodes, lower=lower, upper=upper,
                                        threshold=level, lipschitz_bound=bound)
    if wp.get_device().is_cuda:
        wp.synchronize_stream()  # Final face emission still reads the JAX corner values.
    return mesh


def make_marching_cubes(field, bounds, resolution=128, *, level=0., lipschitz_bound=None, batch_size=65536):
    """Return ``extract(params) -> (JAX vertices, faces)`` for a scalar FP32 field.

    Resolution counts cells. A supplied conservative Lipschitz bound enables
    sparse sampling; otherwise sample densely. Parameters remain dynamic.
    Call outside JAX transformations: mesh sizes depend on field values.
    """
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
    lower, upper = tuple(lower), tuple(upper)
    # Warp pads sparse root grids; projection extends the same Lipschitz bound.
    padded = lipschitz_bound is not None and (len(set(cells)) != 1 or cells[0] & (cells[0]-1))
    evaluate = _FieldEvaluator(
        (lambda params, point: field(params, jnp.clip(point, jnp.asarray(lower), jnp.asarray(upper))))
        if padded else field, batch_size)

    def extract(params):
        devices = {v.device for v in jax.tree_util.tree_leaves(params) if isinstance(v, jax.Array)}
        if len(devices) > 1:
            raise ValueError("parameters must share one device")
        device = next(iter(devices), jax.devices()[0])
        with wp.ScopedDevice(wp.device_from_jax(device)):
            if lipschitz_bound is None:
                values = evaluate.dense(params, nodes, lower, upper)
                vertices, faces = _extract_dense(values, level, lower=lower, upper=upper)
            else:
                vertices, faces = _extract_sparse(evaluate, params, nodes, lower, upper, level, lipschitz_bound)
            return wp.to_jax(vertices), wp.to_jax(faces).reshape(-1, 3)

    return extract
