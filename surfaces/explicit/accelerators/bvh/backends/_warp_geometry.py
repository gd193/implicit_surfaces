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
