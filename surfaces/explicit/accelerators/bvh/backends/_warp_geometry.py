"""Warp geometry helpers."""
from __future__ import annotations

import warp as wp


@wp.func
def point_box_distance_sq(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.float32:
    dx = wp.max(wp.max(lo[0] - p[0], 0.0), p[0] - hi[0])
    dy = wp.max(wp.max(lo[1] - p[1], 0.0), p[1] - hi[1])
    dz = wp.max(wp.max(lo[2] - p[2], 0.0), p[2] - hi[2])
    return dx * dx + dy * dy + dz * dz


@wp.func
def _closest_segment(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> wp.vec3:
    ab = b - a
    denom = wp.dot(ab, ab)
    t = wp.float32(0.0)
    if denom > 0.0:
        t = wp.clamp(wp.dot(p - a, ab) / denom, wp.float32(0.0), wp.float32(1.0))
    return a + t * ab


@wp.func
def closest_triangle(p: wp.vec3, a: wp.vec3, e1: wp.vec3, e2: wp.vec3) -> wp.vec3:
    """Degenerate triangles fall back to segments."""
    b = a + e1
    c = a + e2
    n = wp.cross(e1, e2)
    n2 = wp.dot(n, n)

    # Cross-product barycentrics stay stable on thin triangles.
    if n2 > 0.0:
        projected = p - (wp.dot(p - a, n) / n2) * n
        v = projected - a
        u = wp.dot(wp.cross(v, e2), n) / n2
        w = wp.dot(wp.cross(e1, v), n) / n2
        if u >= 0.0 and w >= 0.0 and u + w <= 1.0:
            return projected

    best = _closest_segment(p, a, b)
    d2 = wp.dot(p - best, p - best)
    q = _closest_segment(p, b, c)
    qd2 = wp.dot(p - q, p - q)
    if qd2 < d2:
        best = q
        d2 = qd2
    q = _closest_segment(p, c, a)
    qd2 = wp.dot(p - q, p - q)
    if qd2 < d2:
        best = q
        d2 = qd2

    return best


@wp.func
def boxes_overlap(a_lo: wp.vec3, a_hi: wp.vec3, b_lo: wp.vec3, b_hi: wp.vec3) -> bool:
    return (a_lo[0] <= b_hi[0] and b_lo[0] <= a_hi[0]
            and a_lo[1] <= b_hi[1] and b_lo[1] <= a_hi[1]
            and a_lo[2] <= b_hi[2] and b_lo[2] <= a_hi[2])


@wp.func
def finite3(v: wp.vec3) -> bool:
    return (v[0] == v[0] and v[1] == v[1] and v[2] == v[2]
            and wp.abs(v[0]) < wp.float32(3.4028234e38)
            and wp.abs(v[1]) < wp.float32(3.4028234e38)
            and wp.abs(v[2]) < wp.float32(3.4028234e38))
