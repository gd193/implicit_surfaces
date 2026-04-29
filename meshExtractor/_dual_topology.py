"""Cube topology + dual-cell-to-quad connectivity, shared by dual extractors.

A "dual" extractor (SurfaceNets, DualContouring) emits one vertex per crossing cell
and connects them with quads, where each quad is the set of 4 cells sharing a
crossing lattice edge in the cube grid. This module provides:

  - cube corner/edge constants in the natural bit-ordering convention
    (corner k has logical coords (k & 1, (k>>1) & 1, (k>>2) & 1)). NOT the Paul
    Bourke MC convention — those tables live in `_mc_tables.py`.
  - `emit_dual_quads`: walks a uniform leaf set, emits triangulated quads per
    crossing lattice edge whose 4 surrounding cells are all crossing cells.

Both extractors then differ only in vertex placement.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# 8 cube corners in natural bit-ordering. corner_k = (k & 1, (k>>1) & 1, (k>>2) & 1).
CORNER_OFFSETS = np.array(
    [[k & 1, (k >> 1) & 1, (k >> 2) & 1] for k in range(8)],
    dtype=np.float32,
)

# 12 cube edges as (corner_a, corner_b) pairs (with a < b, hamming distance 1).
EDGE_PAIRS = np.array(
    [(a, b) for a in range(8) for b in range(a + 1, 8)
     if int(np.sum(np.abs(CORNER_OFFSETS[a] - CORNER_OFFSETS[b]))) == 1],
    dtype=np.int32,
)


def _quad_cell_offsets(axis: int) -> np.ndarray:
    """Cell-grid offsets for the 4 cells sharing a +axis-directed lattice edge.

    The edge runs from the lower corner of cell (i, j, k) along +axis. The 4
    cells whose union of cells contains that edge differ from (i, j, k) by 0
    or -1 in the two axes orthogonal to `axis`. The quad winding is consistent
    across axes so callers can flip orientation uniformly with `[::-1]` based
    on the sign at the lower endpoint.
    """
    other = [a for a in range(3) if a != axis]
    o = np.zeros((4, 3), dtype=np.int32)
    for q, (da, db) in enumerate([(0, 0), (0, -1), (-1, -1), (-1, 0)]):
        o[q, other[0]] = da
        o[q, other[1]] = db
    return o


QUAD_OFFSETS = {a: _quad_cell_offsets(a) for a in range(3)}


def emit_dual_quads(cell_lower: np.ndarray,
                    cell_corner_vals: np.ndarray,
                    cell_size: np.ndarray,
                    active: np.ndarray) -> np.ndarray:
    """Walk active cells and emit triangulated quads per crossing lattice edge.

    Args:
        cell_lower: (N, 3) lower corner of each cell. All cells must share the
            same `cell_size` (i.e. the leaf set must be uniform).
        cell_corner_vals: (N, 8) function values at the 8 corners of each cell,
            with corner ordering matching CORNER_OFFSETS (natural bit-ordering).
            Values are assumed shifted so that the inside test is `< 0`.
        cell_size: (3,) edge length of each cell along each axis.
        active: (N,) bool — which cells are crossing cells. Each active cell
            contributes one vertex; its index in the output mesh is its row
            number in `cell_lower` (the caller is expected to compact non-active
            rows out *after* receiving these triangles, or use `active` to mask).

    Returns:
        tris: (T, 3) int32 triangle indices into the per-cell vertex array
            (i.e., values are row indices of `cell_lower`).
    """
    n = cell_lower.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.int32)

    cell_size = np.asarray(cell_size, dtype=np.float64)
    anchor = cell_lower.min(axis=0)
    gi = np.round((cell_lower - anchor) / cell_size).astype(np.int64)
    cell_index = {tuple(g): i for i, g in enumerate(gi)}

    tris: list[tuple[int, int, int]] = []
    for ci in range(n):
        if not active[ci]:
            continue
        cv = cell_corner_vals[ci]
        v_low = cv[0]
        for axis in range(3):
            v_high = cv[1 << axis]
            if (v_low < 0) == (v_high < 0):
                continue
            offsets = QUAD_OFFSETS[axis]
            quad = []
            ok = True
            for off in offsets:
                nb = tuple((gi[ci] + off).tolist())
                idx = cell_index.get(nb, -1)
                if idx < 0 or not active[idx]:
                    ok = False
                    break
                quad.append(idx)
            if not ok:
                continue
            if v_low > 0:
                quad = quad[::-1]
            a, b, c, d = quad
            tris.append((a, b, c))
            tris.append((a, c, d))

    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(tris, dtype=np.int32)


def cell_sign_change_mask(cell_corner_vals: np.ndarray) -> np.ndarray:
    """(N,) bool: True iff some corner is < 0 and some >= 0."""
    return (cell_corner_vals.min(axis=1) < 0) & (cell_corner_vals.max(axis=1) >= 0)


def edge_crossings(cell_corner_pos: np.ndarray,
                   cell_corner_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell list of edge crossing points.

    Args:
        cell_corner_pos: (N, 8, 3) per-cell corner positions.
        cell_corner_vals: (N, 8) corresponding values (shifted so inside is < 0).

    Returns:
        crossings: (N, 12, 3) — crossing point per edge (zeros where no crossing).
        has_crossing: (N, 12) bool — which edges have a crossing.
    """
    va = cell_corner_vals[:, EDGE_PAIRS[:, 0]]  # (N, 12)
    vb = cell_corner_vals[:, EDGE_PAIRS[:, 1]]
    pa = cell_corner_pos[:, EDGE_PAIRS[:, 0], :]  # (N, 12, 3)
    pb = cell_corner_pos[:, EDGE_PAIRS[:, 1], :]

    has_crossing = (va < 0) != (vb < 0)
    denom = vb - va
    safe_denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)
    t = -va / safe_denom
    t = np.clip(t, 0.0, 1.0)
    cross = (1.0 - t)[..., None] * pa + t[..., None] * pb
    cross = np.where(has_crossing[..., None], cross, 0.0)
    return cross, has_crossing
