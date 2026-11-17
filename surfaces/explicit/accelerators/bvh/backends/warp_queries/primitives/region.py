"""Exact, closed-box item intersection tests for builtin primitives."""

import warp as wp

from .data import PointsData2, PointsData3, SegmentsData2, SegmentsData3, TrianglesData3
from ..io.arrays import load_array


@wp.func
def _inside(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> bool:
    return (lo[0] <= p[0] and p[0] <= hi[0] and
            lo[1] <= p[1] and p[1] <= hi[1] and
            lo[2] <= p[2] and p[2] <= hi[2])


@wp.func
def point2_intersects(data: PointsData2, slot: int, lo: wp.vec3, hi: wp.vec3) -> bool:
    a = load_array(data.a, slot)
    return _inside(wp.vec3(a[0], a[1], 0.0), lo, hi)


@wp.func
def point3_intersects(data: PointsData3, slot: int, lo: wp.vec3, hi: wp.vec3) -> bool:
    return _inside(load_array(data.a, slot), lo, hi)


@wp.func
def _segment(a: wp.vec3, b: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> bool:
    low = wp.float64(0.0)
    high = wp.float64(1.0)
    for axis in range(3):
        d = wp.float64(b[axis]) - wp.float64(a[axis])
        if d == 0.0:
            if a[axis] < lo[axis] or a[axis] > hi[axis]:
                return False
        else:
            t0 = (wp.float64(lo[axis]) - wp.float64(a[axis])) / d
            t1 = (wp.float64(hi[axis]) - wp.float64(a[axis])) / d
            low = wp.max(low, wp.min(t0, t1))
            high = wp.min(high, wp.max(t0, t1))
            if low > high:
                return False
    return True


@wp.func
def segment2_intersects(data: SegmentsData2, slot: int, lo: wp.vec3, hi: wp.vec3) -> bool:
    a = load_array(data.a, slot)
    b = load_array(data.b, slot)
    return _segment(wp.vec3(a[0], a[1], 0.0), wp.vec3(b[0], b[1], 0.0), lo, hi)


@wp.func
def segment3_intersects(data: SegmentsData3, slot: int, lo: wp.vec3, hi: wp.vec3) -> bool:
    return _segment(load_array(data.a, slot), load_array(data.b, slot), lo, hi)


@wp.func
def _separated(axis: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
               center: wp.vec3d, half: wp.vec3d) -> bool:
    p0 = wp.dot(axis, a - center)
    p1 = wp.dot(axis, b - center)
    p2 = wp.dot(axis, c - center)
    radius = wp.dot(wp.abs(axis), half)
    return wp.min(p0, wp.min(p1, p2)) > radius or wp.max(p0, wp.max(p1, p2)) < -radius


@wp.func
def _edge_separated(edge: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
                    center: wp.vec3d, half: wp.vec3d) -> bool:
    return (_separated(wp.cross(edge, wp.vec3d(1.0, 0.0, 0.0)), a, b, c, center, half) or
            _separated(wp.cross(edge, wp.vec3d(0.0, 1.0, 0.0)), a, b, c, center, half) or
            _separated(wp.cross(edge, wp.vec3d(0.0, 0.0, 1.0)), a, b, c, center, half))


@wp.func
def triangle3_intersects(data: TrianglesData3, slot: int, lo: wp.vec3, hi: wp.vec3) -> bool:
    # Packed indices retain the input vertices; float32 edges may not reconstruct
    # them when a triangle spans very different coordinate magnitudes.
    indices = load_array(data.indices, slot)
    a = wp.vec3d(load_array(data.vertices, indices[0]))
    b = wp.vec3d(load_array(data.vertices, indices[1]))
    c = wp.vec3d(load_array(data.vertices, indices[2]))
    # The box axes test the three coordinate ranges, including flat boxes.
    for axis in range(3):
        if wp.min(a[axis], wp.min(b[axis], c[axis])) > hi[axis] or wp.max(a[axis], wp.max(b[axis], c[axis])) < lo[axis]:
            return False
    center = wp.vec3d(lo) * wp.float64(0.5) + wp.vec3d(hi) * wp.float64(0.5)
    half = wp.vec3d(hi) * wp.float64(0.5) - wp.vec3d(lo) * wp.float64(0.5)
    if _edge_separated(b - a, a, b, c, center, half):
        return False
    if _edge_separated(c - b, a, b, c, center, half):
        return False
    if _edge_separated(a - c, a, b, c, center, half):
        return False
    return not _separated(wp.cross(b - a, c - a), a, b, c, center, half)
