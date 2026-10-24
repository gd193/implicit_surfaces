"""Layer-2 reference: external-library oracle (`trimesh`, `scipy`).

Independent implementation of each query, used to catch bugs that fool
both the project's `Primitives` callbacks and `_brute.py` (they share
the same callbacks; if the callback is wrong, both fail equally).
trimesh and scipy are independent codebases — disagreement points at
our code with high confidence.

All oracle functions take a `trimesh.Trimesh` and NumPy queries; they
return NumPy results. Tests convert to/from jnp at boundaries.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh


# =====================================================================
# closest_point.
# =====================================================================


def oracle_closest_point(tm: trimesh.Trimesh,
                         queries: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(cp, dist, prim_idx)` from `trimesh.proximity.closest_point`.
    `prim_idx` is the face index in `tm.faces`.
    """
    cp, dist, face_idx = trimesh.proximity.closest_point(tm, queries)
    return (np.asarray(cp, dtype=np.float32),
            np.asarray(dist, dtype=np.float32),
            np.asarray(face_idx, dtype=np.int32))


# =====================================================================
# first_hit.
# =====================================================================


def oracle_first_hit(tm: trimesh.Trimesh,
                     ray_origins: np.ndarray,
                     ray_directions: np.ndarray,
                     max_t: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(t, face_idx, hit_flag)` per ray.

    Uses trimesh's RayMeshIntersector, picks the smallest-t hit per
    ray, then masks any t > max_t as miss. trimesh returns hits in
    arbitrary order so we sort host-side.
    """
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(tm)
    locations, ray_idx, face_idx = intersector.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions,
        multiple_hits=True)

    n = ray_origins.shape[0]
    out_t = np.full((n,), np.inf, dtype=np.float32)
    out_face = np.full((n,), -1, dtype=np.int32)
    out_hit = np.zeros((n,), dtype=bool)

    if locations.shape[0] > 0:
        # t = (location - origin) projected on direction (direction not
        # required unit). Direction unit length per fixture; safe to dot.
        rel = locations - ray_origins[ray_idx]
        ts = np.einsum("ij,ij->i", rel, ray_directions[ray_idx])
        for k in range(locations.shape[0]):
            i = int(ray_idx[k])
            t_k = float(ts[k])
            if t_k < 0.0 or t_k > float(max_t[i]):
                continue
            if t_k < float(out_t[i]):
                out_t[i] = t_k
                out_face[i] = int(face_idx[k])
                out_hit[i] = True

    return out_t, out_face, out_hit


# =====================================================================
# aabb_overlap.
# =====================================================================


def oracle_aabb_overlap(tm: trimesh.Trimesh,
                        q_los: np.ndarray, q_his: np.ndarray) -> np.ndarray:
    """Per-query bool: does any face's AABB overlap `[q_lo, q_hi]`?

    Uses trimesh's per-face vertex extraction, then a simple SoA AABB
    check — independent from `TriMeshPrimitives` AABB code.
    """
    v = np.asarray(tm.vertices, dtype=np.float32)
    f = np.asarray(tm.faces, dtype=np.int32)
    tri_v = v[f]                                     # (F, 3, 3)
    f_lo = tri_v.min(axis=1)                         # (F, 3)
    f_hi = tri_v.max(axis=1)                         # (F, 3)

    # (M, F) overlap matrix via broadcasting.
    overlaps = ((f_lo[None, :, :] <= q_his[:, None, :]).all(axis=-1) &
                (q_los[:, None, :] <= f_hi[None, :, :]).all(axis=-1))
    return overlaps.any(axis=1)
