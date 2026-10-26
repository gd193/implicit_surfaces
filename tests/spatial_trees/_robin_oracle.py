"""Float64 oracle using numerical minimization in distance space."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ._warp_helpers import _oracle_silhouette, _closest_triangle


def brute_robin_radius(vertices, faces, point, maximum_coefficients, cap):
    """Exhaustive capped radius for scalar or per-face maximum coefficients."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    point = np.asarray(point, dtype=np.float64)
    cap = float(cap)
    if cap < 0 or not np.isfinite(cap):
        raise ValueError("cap must be finite and nonnegative")
    coeff = np.asarray(maximum_coefficients, dtype=np.float64)
    if coeff.ndim == 0:
        coeff = np.full(len(faces), float(coeff))
    if coeff.shape != (len(faces),):
        raise ValueError("expected a scalar coefficient or one coefficient per face")

    best = cap
    _, silhouette_distance = _oracle_silhouette(vertices, faces, point)
    if np.isfinite(silhouette_distance):
        best = min(best, float(silhouette_distance))

    for face, mu in zip(faces, coeff, strict=True):
        tri = vertices[face]
        near_point = _closest_triangle(point, tri)
        dnear = float(np.linalg.norm(near_point - point))
        if dnear >= best:
            continue
        if mu == 0:
            continue
        if np.isinf(mu) or dnear == 0:
            best = dnear
            continue

        edge0, edge1 = tri[1] - tri[0], tri[2] - tri[0]
        cross = np.cross(edge0, edge1)
        cross_norm = float(np.linalg.norm(cross))
        h = abs(float((point - tri[0]) @ cross)) / cross_norm if cross_norm > 0 else 0.
        if h == 0:
            best = min(best, dnear)
            continue
        dfar = float(np.max(np.linalg.norm(tri - point, axis=1)))
        threshold = np.sqrt(h / mu)
        # At or below the threshold, this triangle has no finite Robin bound.
        if dfar <= threshold:
            continue

        lo = max(dnear, np.nextafter(threshold, np.inf))
        hi = dfar
        if lo > hi:
            continue

        def radius(r):
            denominator = 1.0 - h / (mu * r * r)
            return r / denominator if denominator > 0 else np.inf

        candidates = [radius(lo), radius(hi)]
        if hi > lo:
            result = minimize_scalar(radius, bounds=(lo, hi), method="bounded",
                                     options={"xatol": 1e-13})
            if result.success:
                candidates.append(float(result.fun))
        primitive_bound = min(candidates)
        best = min(best, primitive_bound)

    return min(cap, best)
