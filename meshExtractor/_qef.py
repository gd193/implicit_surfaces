"""Batched QEF solver for dual-contouring vertex placement.

For each cell with crossing-edge Hermite data {(p_i, n_i)}, place the cell vertex
at the point minimising

    Σ_i ((v - p_i) · n_i)²  +  λ ‖v - c‖²

where c is the cell centre and λ > 0 is a Tikhonov regulariser. The closed form
is `(Σ n_i n_iᵀ + λ I) v = Σ (p_i · n_i) n_i + λ c`. Solved per cell as a 3×3
linear system, then clamped to the cell AABB so the QEF can't push v outside the
cell (standard DC practice).
"""
from __future__ import annotations

import numpy as np


def solve_qef_batch(positions: np.ndarray,
                    normals: np.ndarray,
                    mask: np.ndarray,
                    cell_centers: np.ndarray,
                    cell_lower: np.ndarray,
                    cell_upper: np.ndarray,
                    regularization: float = 1e-2) -> np.ndarray:
    """Solve a QEF per cell, return (N, 3) cell-vertex positions clamped to each cell.

    Args:
        positions:    (N, K, 3) Hermite intersection points (zero-padded).
        normals:      (N, K, 3) unit-length normals at those points (zero-padded).
        mask:         (N, K)    bool — which Hermite samples are real.
        cell_centers: (N, 3)
        cell_lower:   (N, 3)
        cell_upper:   (N, 3)
        regularization: λ.
    """
    n_cells = positions.shape[0]
    if n_cells == 0:
        return np.zeros((0, 3), dtype=np.float32)

    m = mask.astype(np.float32)[..., None]                         # (N, K, 1)
    n_outer = normals[..., :, None] * normals[..., None, :]        # (N, K, 3, 3)
    A = (m[..., None] * n_outer).sum(axis=1)                       # (N, 3, 3)
    A = A + regularization * np.eye(3, dtype=A.dtype)[None, :, :]

    n_dot_p = (normals * positions).sum(axis=-1, keepdims=True)    # (N, K, 1)
    b = (m * n_dot_p * normals).sum(axis=1)                        # (N, 3)
    b = b + regularization * cell_centers

    # numpy<2 wants b shape (..., m, k); expand to (N, 3, 1) and squeeze.
    v = np.linalg.solve(A, b[..., None]).squeeze(-1)
    v = np.clip(v, cell_lower, cell_upper)
    return v.astype(np.float32)
