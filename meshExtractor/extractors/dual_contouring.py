"""Dual contouring extractor (generic path).

Per crossing cell:
  * find edge crossings via linear interpolation (`_dual_topology.edge_crossings`),
  * query `surface.spatial_derivative` at each crossing for the normal,
  * solve a per-cell QEF over the (p_i, n_i) Hermite data (`_qef.solve_qef_batch`),
  * connect with the standard dual-cell-to-quad walk (`_dual_topology.emit_dual_quads`).

Connectivity is identical to SurfaceNets — the only difference is vertex placement.
"""
from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from meshExtractor._dual_topology import (
    CORNER_OFFSETS,
    cell_sign_change_mask,
    edge_crossings,
    emit_dual_quads,
)
from meshExtractor._qef import solve_qef_batch
from meshExtractor.base import Leaves, MeshExtractor


class DualContouring(MeshExtractor):
    accepts_leaf_uniformity = ("uniform",)
    required_surface_methods = ("__call__", "spatial_derivative")

    def __init__(self, regularization: float = 1e-2):
        self.regularization = float(regularization)

    def extract(self, surface, params: Any, leaves: Leaves) -> Tuple[jnp.ndarray, jnp.ndarray]:
        valid_mask = np.asarray(leaves.valid)
        lo = np.asarray(leaves.lower)[valid_mask]
        hi = np.asarray(leaves.upper)[valid_mask]
        n_cells = lo.shape[0]
        if n_cells == 0:
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=jnp.int32)

        cell_size = (hi[0] - lo[0]).astype(np.float32)
        level_set = float(surface.level_set)

        def eval_shifted(x):
            return surface(params, x).squeeze() - level_set

        def grad_at(x):
            return surface.spatial_derivative(params, x)

        corners = lo[:, None, :] + CORNER_OFFSETS[None, :, :] * cell_size[None, None, :]
        flat_corners = jnp.asarray(corners.reshape(-1, 3))
        flat_vals = np.asarray(jax.vmap(eval_shifted)(flat_corners))
        vals = flat_vals.reshape(n_cells, 8)

        active = cell_sign_change_mask(vals)
        if not active.any():
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=jnp.int32)

        crossings, has_cross = edge_crossings(corners, vals)
        active_idx = np.nonzero(active)[0]
        a_pos = crossings[active_idx]
        a_has = has_cross[active_idx]
        a_lo = lo[active_idx]
        a_hi = hi[active_idx]
        a_centers = 0.5 * (a_lo + a_hi)

        flat_pts = jnp.asarray(a_pos.reshape(-1, 3))
        flat_grads = np.asarray(jax.vmap(grad_at)(flat_pts))
        grads = flat_grads.reshape(-1, 12, 3)
        norm = np.linalg.norm(grads, axis=-1, keepdims=True)
        normals = np.where(norm > 1e-12, grads / np.maximum(norm, 1e-12), 0.0)

        v_active = solve_qef_batch(
            positions=a_pos,
            normals=normals,
            mask=a_has,
            cell_centers=a_centers,
            cell_lower=a_lo,
            cell_upper=a_hi,
            regularization=self.regularization,
        )

        verts = np.zeros((n_cells, 3), dtype=np.float32)
        verts[active_idx] = v_active

        tris = emit_dual_quads(lo, vals, cell_size, active)
        if tris.shape[0] == 0:
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=jnp.int32)

        used = np.unique(tris)
        remap = -np.ones(n_cells, dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        return jnp.asarray(verts[used]), jnp.asarray(remap[tris])
