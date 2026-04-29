"""Marching cubes extractor — NIQ-style.

Per-cell MC kernels ported from `neural-implicit-queries/src/extract_cell.py`. The outer
batched driver follows the same template as `meshExtractor.trees._bucketed_build`:

  * A single JIT'd kernel `_extract_batch` runs MC over the entire padded leaf bucket
    in one trace, vmapped per leaf.
  * `surface` and `subcell_depth` are static so the JIT cache keys by (surface identity,
    subcell_depth, bucket_size). Same surface + same depth = always a cache hit.
  * The padded `Leaves.valid` mask flows through to a per-triangle valid mask
    (`tri_valid &= leaf_valid[:, None]`); padding cells produce no triangles.
  * `vmap` everywhere — no `lax.map`.

The only Python/numpy work is final triangle compaction, which is unavoidable because the
output mesh size is data-dependent.
"""
from __future__ import annotations

import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from meshExtractor._mc_tables import (
    EDGE_VERTS,
    MC_TRI_TABLE,
    VERT_LOGICAL_COORDS,
)
from meshExtractor.base import Leaves, MeshExtractor


def _extract_triangles_from_cell(cell_lower, cell_upper, vert_vals):
    """MC for a single cube. Returns (5, 3, 3) tri positions and (5,) valid mask."""
    vert_pos = jnp.where(VERT_LOGICAL_COORDS, cell_upper[None, :], cell_lower[None, :])

    def check_edge(inds):
        ia, ib = inds[0], inds[1]
        va, vb = vert_vals[ia], vert_vals[ib]
        pa, pb = vert_pos[ia, :], vert_pos[ib, :]
        t = -va / (vb - va)
        t = jnp.nan_to_num(t)
        t = jnp.clip(t, a_min=0.0, a_max=1.0)
        cross = (1.0 - t) * pa + t * pb
        return cross

    edge_cross_loc = jax.vmap(check_edge)(EDGE_VERTS)

    case_bits = jnp.power(2, jnp.arange(8))
    case_id = jnp.sum((vert_vals < 0) * case_bits)
    case_tris = jnp.reshape(MC_TRI_TABLE[case_id, :15], (-1, 3))

    def get_tri_pos(tri):
        is_valid = tri[0] != -1
        i0 = jnp.clip(tri[0], a_min=0)
        i1 = jnp.clip(tri[1], a_min=0)
        i2 = jnp.clip(tri[2], a_min=0)
        return jnp.stack(
            (edge_cross_loc[i0, :], edge_cross_loc[i1, :], edge_cross_loc[i2, :]),
            axis=0,
        ), is_valid

    return jax.vmap(get_tri_pos)(case_tris)


def _extract_triangles_from_subcells(eval_fn, subcell_depth, cell_lower, cell_upper):
    """Dice cell into 2^subcell_depth uniform subcells per axis and run MC on each."""
    side_n_sub = 2 ** subcell_depth
    side_n_pts = side_n_sub + 1

    s0 = jnp.linspace(cell_lower[0], cell_upper[0], num=side_n_pts)
    s1 = jnp.linspace(cell_lower[1], cell_upper[1], num=side_n_pts)
    s2 = jnp.linspace(cell_lower[2], cell_upper[2], num=side_n_pts)
    g0, g1, g2 = jnp.meshgrid(s0, s1, s2, indexing="ij")
    grid_coords = jnp.stack((g0, g1, g2), axis=-1)

    flat_coords = jnp.reshape(grid_coords, (-1, 3))
    flat_vals = jax.vmap(eval_fn)(flat_coords)
    grid_vals = jnp.reshape(flat_vals, (side_n_pts, side_n_pts, side_n_pts))

    side_inds = jnp.arange(side_n_sub)
    gi0, gi1, gi2 = jnp.meshgrid(side_inds, side_inds, side_inds, indexing="ij")
    subcell_inds = jnp.stack((gi0, gi1, gi2), axis=-1).reshape(-1, 3)

    delta = (cell_upper - cell_lower) / side_n_sub
    sub_lower = cell_lower[None, :] + subcell_inds * delta[None, :]
    sub_upper = sub_lower + delta[None, :]

    vert_inds = subcell_inds[:, None, :] + VERT_LOGICAL_COORDS[None, :, :]
    sub_vert_vals = grid_vals[
        vert_inds[:, :, 0], vert_inds[:, :, 1], vert_inds[:, :, 2]
    ]

    tri_pos, tri_valid = jax.vmap(_extract_triangles_from_cell)(
        sub_lower, sub_upper, sub_vert_vals
    )
    return tri_pos.reshape(-1, 3, 3), tri_valid.reshape(-1)


@functools.partial(jax.jit, static_argnames=("surface", "subcell_depth"))
def _extract_batch(surface, params, level_set, subcell_depth: int,
                   leaf_valid, leaf_lower, leaf_upper):
    """Run MC over an entire padded leaf bucket. One JIT trace per (surface, subcell_depth, bucket_size)."""
    def eval_shifted(x):
        return surface(params, x).squeeze() - level_set

    if subcell_depth == 0:
        def per_leaf(lo, hi):
            vert_pos = jnp.where(VERT_LOGICAL_COORDS, hi[None, :], lo[None, :])
            vert_vals = jax.vmap(eval_shifted)(vert_pos)
            return _extract_triangles_from_cell(lo, hi, vert_vals)
    else:
        def per_leaf(lo, hi):
            return _extract_triangles_from_subcells(eval_shifted, subcell_depth, lo, hi)

    tri_pos, tri_valid = jax.vmap(per_leaf)(leaf_lower, leaf_upper)
    # tri_pos: (B, T_per_leaf, 3, 3), tri_valid: (B, T_per_leaf)
    tri_valid = tri_valid & leaf_valid[:, None]
    return tri_pos, tri_valid


class MarchingCubes(MeshExtractor):
    accepts_leaf_uniformity = ("uniform",)
    required_surface_methods = ("__call__",)

    def __init__(self, subcell_depth: int = 0):
        if subcell_depth < 0:
            raise ValueError("subcell_depth must be non-negative")
        self.subcell_depth = int(subcell_depth)

    def extract(self, surface, params: Any, leaves: Leaves) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if leaves.lower.shape[0] == 0:
            return jnp.zeros((0, 3)), jnp.zeros((0, 3), dtype=jnp.int32)

        level_set = jnp.asarray(surface.level_set)
        tri_pos, tri_valid = _extract_batch(
            surface, params, level_set, self.subcell_depth,
            leaves.valid, leaves.lower, leaves.upper,
        )

        # Final compaction. Variable output size → numpy after JIT graph completes.
        tri_pos_np = np.asarray(tri_pos.reshape(-1, 3, 3))
        tri_valid_np = np.asarray(tri_valid.reshape(-1))
        valid_tris = tri_pos_np[tri_valid_np]
        n = valid_tris.shape[0]
        verts = valid_tris.reshape(-1, 3)
        tris = np.arange(3 * n, dtype=np.int32).reshape(n, 3)
        return jnp.asarray(verts), jnp.asarray(tris)
