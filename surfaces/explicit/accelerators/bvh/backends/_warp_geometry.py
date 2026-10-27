"""Warp geometry helpers."""
from __future__ import annotations

import warp as wp


@wp.func
def boxes_overlap(a_lo: wp.vec3, a_hi: wp.vec3, b_lo: wp.vec3, b_hi: wp.vec3) -> bool:
    return (a_lo[0] <= b_hi[0] and b_lo[0] <= a_hi[0]
            and a_lo[1] <= b_hi[1] and b_lo[1] <= a_hi[1]
            and a_lo[2] <= b_hi[2] and b_lo[2] <= a_hi[2])


# Native snippets share the query point p.
_NATIVE_POINT_GEOMETRY = "const wp::vec3 p = query;\n" + r"""
    const auto box_distance_sq = [&](const wp::vec3& lower,
                                     const wp::vec3& upper) {
        const float dx = wp::max(wp::max(lower[0] - p[0], 0.0f), p[0] - upper[0]);
        const float dy = wp::max(wp::max(lower[1] - p[1], 0.0f), p[1] - upper[1]);
        const float dz = wp::max(wp::max(lower[2] - p[2], 0.0f), p[2] - upper[2]);
        return dx * dx + dy * dy + dz * dz;
    };
"""

_NATIVE_SEGMENT_CLOSEST = r"""
    const auto closest_segment = [&](const auto& a, const auto& b) {
        const auto ab = b - a;
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


_NATIVE_RAY_BOX_TEST = r"""
    const auto box_interval = [&](const wp::vec3& lower, const wp::vec3& upper,
                           float tmin, float tmax, float& enter) {
        enter = tmin;
        float leave = tmax;
        const auto update = [&](float t0, float t1) {
            float near_t = wp::min(t0,t1), far_t = wp::max(t0,t1);
            enter = wp::max(enter,near_t);
            leave = wp::min(leave,far_t);
        };
        if (fast_aabb) {
            for (int axis = 0; axis < dimension; ++axis) {
                const float t0 = (lower[axis] - o[axis]) * inv_dir[axis];
                const float t1 = (upper[axis] - o[axis]) * inv_dir[axis];
                update(t0,t1);
            }
        } else {
            for (int axis = 0; axis < dimension; ++axis) {
                const float component = d[axis];
                const float origin_component = o[axis];
                if (component == 0.0f) {
                    if (origin_component < lower[axis]
                        || origin_component > upper[axis]) return false;
                } else {
                    const float t0 = (lower[axis] - origin_component) / component;
                    const float t1 = (upper[axis] - origin_component) / component;
                    update(t0,t1);
                }
            }
        }
        // Expand the reduced interval; narrow slabs can invert through rounding.
        const float padding = 9.5367431640625e-7f
            * wp::max(wp::abs(enter),wp::abs(leave)) + 1.e-37f;
        if (wp::isfinite(padding)) {
            enter -= padding;
            return enter <= leave+padding;
        }
        enter = tmin;
        return true;
    };
"""


_NATIVE_EDGE_PREDICATES = r"""
    // Separate rounding preserves antisymmetry on shared projected edges.
    const auto cross2 = [](double ax, double ay, double bx, double by) {
#if defined(__CUDA_ARCH__)
        return __dadd_rn(__dmul_rn(ax, by), -__dmul_rn(ay, bx));
#else
        volatile double a = ax * by;
        volatile double b = ay * bx;
        return a - b;
#endif
    };
    const auto owns_edge = [](double ax, double ay, double bx, double by, double orientation) {
        const double dx = orientation*(bx-ax), dy = orientation*(by-ay);
        return dy > 0 || (dy == 0 && dx < 0);
    };
"""


_NATIVE_CONE_VISIBILITY = r"""
    const auto cone_may_have_silhouette = [&](int node) {
        const wp::vec2 angles = node_cone_sin_cos.data[node];
        const float sin_alpha = angles[0];
        const float cos_alpha = angles[1];
        if (sin_alpha < 0.0f) return false;
        if (cos_alpha <= 0.0f) return true;
        const wp::vec3 lower = node_lower.data[node];
        const wp::vec3 upper = node_upper.data[node];
        const wp::vec3 view = 0.5f * (lower + upper) - p;
        const wp::vec3 extent = 0.5f * (upper - lower);
        const wp::vec3 axis = node_cone_axis.data[node];
        const float projection = wp::dot(axis, view);
        // Box support along the cone axis bounds every edge endpoint.
        const float support = wp::abs(axis[0]) * extent[0]
                            + wp::abs(axis[1]) * extent[1]
                            + wp::abs(axis[2]) * extent[2];
        const float gap = wp::abs(projection) - support;
        if (gap <= 0.0f) return true;
        const wp::vec3 perpendicular = view - projection * axis;
        const float radius = node_cone_radius.data[node];
        const float limit = sin_alpha * (wp::sqrt(wp::dot(perpendicular, perpendicular)) + radius);
        const float slack = 2.0e-6f * (wp::sqrt(wp::dot(view, view)) + radius);
        return gap * cos_alpha <= limit + slack;
    };

"""


_NATIVE_TRIANGLE_RAY = r"""
        const auto triangle_t = [&](const wp::vec3& a, const wp::vec3& e1,
                                    const wp::vec3& e2, float tmin, float tmax,
                                    float& result) {
            const wp::vec3 pvec = wp::cross(d, e2);
            const float det = wp::dot(e1, pvec);
            const float scale = wp::sqrt(wp::dot(e1, e1) * wp::dot(e2, e2)
                                         * wp::dot(d, d));
            const float det_eps = 1.0e-7f * scale;
            result = 0.0f;
            if (wp::abs(det) > det_eps) {
                const float inv_det = 1.0f / det;
                const wp::vec3 tvec = o - a;
                const float u = wp::dot(tvec, pvec) * inv_det;
                const wp::vec3 qvec = wp::cross(tvec, e1);
                const float v = wp::dot(d, qvec) * inv_det;
                const float t = wp::dot(e2, qvec) * inv_det;
                result = t;
                const float slack = 1.0e-6f;
                return u >= -slack && v >= -slack && u + v <= 1.0f + slack
                    && t >= tmin && t <= tmax;
            }
            return false;
        };

"""
