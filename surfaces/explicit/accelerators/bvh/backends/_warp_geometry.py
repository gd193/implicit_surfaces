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


# Native snippets share the query point p.
_NATIVE_FINITE3 = r"""
    const auto finite3 = [](const wp::vec3& v) {
        const float limit = 3.4028234e38f;
        return v[0] == v[0] && v[1] == v[1] && v[2] == v[2]
            && wp::abs(v[0]) < limit && wp::abs(v[1]) < limit
            && wp::abs(v[2]) < limit;
    };
"""

_NATIVE_POINT_GEOMETRY = "const wp::vec3 p = query;\n" + _NATIVE_FINITE3 + r"""
    const auto box_distance_sq = [&](const wp::vec3& lower,
                                     const wp::vec3& upper) {
        const float dx = wp::max(wp::max(lower[0] - p[0], 0.0f), p[0] - upper[0]);
        const float dy = wp::max(wp::max(lower[1] - p[1], 0.0f), p[1] - upper[1]);
        const float dz = wp::max(wp::max(lower[2] - p[2], 0.0f), p[2] - upper[2]);
        return dx * dx + dy * dy + dz * dz;
    };
"""

_NATIVE_SEGMENT_CLOSEST = r"""
    const auto closest_segment = [&](const wp::vec3& a, const wp::vec3& b) {
        const wp::vec3 ab = b - a;
        const float denom = wp::dot(ab, ab);
        float t = 0.0f;
        if (denom > 0.0f)
            t = wp::clamp(wp::dot(p - a, ab) / denom, 0.0f, 1.0f);
        return a + t * ab;
    };
"""

_NATIVE_TRIANGLE_CLOSEST = _NATIVE_SEGMENT_CLOSEST + r"""
    const auto closest_triangle = [&](const wp::vec3& a, const wp::vec3& e1,
                                      const wp::vec3& e2) {
        const wp::vec3 b = a + e1;
        const wp::vec3 c = a + e2;
        const wp::vec3 n = wp::cross(e1, e2);
        const float n2 = wp::dot(n, n);
        const wp::vec3 e3 = e2-e1;
        const float edge_scale = wp::max(wp::dot(e3, e3),
            wp::max(wp::dot(e1, e1), wp::dot(e2, e2)));
        // Keep thin and degenerate triangles on the robust path.
        if (n2 > 1.e-2f * edge_scale * edge_scale) {
            const wp::vec2 uv = wp::closest_point_to_triangle(a, b, c, p);
            return a + uv[1]*e1 + (1.0f-uv[0]-uv[1])*e2;
        }
        if (n2 > 0.0f) {
            const wp::vec3 projected = p - (wp::dot(p - a, n) / n2) * n;
            const wp::vec3 v = projected - a;
            const float u = wp::dot(wp::cross(v, e2), n) / n2;
            const float w = wp::dot(wp::cross(e1, v), n) / n2;
            if (u >= 0.0f && w >= 0.0f && u + w <= 1.0f)
                return projected;
        }
        wp::vec3 best = closest_segment(a, b);
        float best_d2 = wp::dot(p - best, p - best);
        wp::vec3 candidate = closest_segment(b, c);
        float candidate_d2 = wp::dot(p - candidate, p - candidate);
        if (candidate_d2 < best_d2) {
            best = candidate;
            best_d2 = candidate_d2;
        }
        candidate = closest_segment(c, a);
        candidate_d2 = wp::dot(p - candidate, p - candidate);
        if (candidate_d2 < best_d2) best = candidate;
        return best;
    };
"""
