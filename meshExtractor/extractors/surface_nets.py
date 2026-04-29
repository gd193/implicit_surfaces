"""Naive SurfaceNets extractor.

Per crossing cell, place one vertex at the centroid of edge crossings (fallback:
cell midpoint). Connectivity is the standard dual-cell-to-quad walk shared with
DualContouring; this extractor differs from DC only in vertex placement.
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
from meshExtractor.base import Leaves, MeshExtractor


class SurfaceNets(MeshExtractor):
    accepts_leaf_uniformity = ("uniform",)
    required_surface_methods = ("__call__",)

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

        # Per-cell corner positions and values (natural bit-ordering corners).
        corners = lo[:, None, :] + CORNER_OFFSETS[None, :, :] * cell_size[None, None, :]
        flat_corners = jnp.asarray(corners.reshape(-1, 3))
        flat_vals = np.asarray(jax.vmap(eval_shifted)(flat_corners))
        vals = flat_vals.reshape(n_cells, 8)

        active = cell_sign_change_mask(vals)
        if not active.any():
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=jnp.int32)

        # Vertex placement: centroid of edge crossings (cell midpoint as fallback).
        crossings, has_cross = edge_crossings(corners, vals)  # (N, 12, 3), (N, 12)
        n_per_cell = has_cross.sum(axis=1)  # (N,)
        sums = (crossings * has_cross[..., None]).sum(axis=1)  # (N, 3)
        safe_n = np.where(n_per_cell > 0, n_per_cell, 1)[..., None]
        verts = np.where(n_per_cell[..., None] > 0,
                         sums / safe_n,
                         0.5 * (lo + hi))

        # Connectivity (shared helper).
        tris = emit_dual_quads(lo, vals, cell_size, active)
        if tris.shape[0] == 0:
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=jnp.int32)

        # Compact vertices to those referenced by triangles.
        used = np.unique(tris)
        remap = -np.ones(n_cells, dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        return jnp.asarray(verts[used]), jnp.asarray(remap[tris])
