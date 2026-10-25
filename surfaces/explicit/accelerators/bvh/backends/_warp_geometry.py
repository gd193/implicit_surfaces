"""Warp geometry helpers."""
from __future__ import annotations

import warp as wp

from ._warp_specs import STACK_CAPACITY


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


# Packed metadata adapted from NVIDIA Warp's bvh.h helpers.
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. Apache-2.0.
_NATIVE_PACKED_TRAVERSAL = f"constexpr int stack_capacity = {STACK_CAPACITY};\n" + r"""
    const auto load_vec4_nc = [](const wp::vec4* ptr, int index) {
        wp::vec4 result;
#if defined(__CUDA_ARCH__)
        const float4 value = __ldg(reinterpret_cast<const float4*>(ptr + index));
        result = wp::vec4(value.x, value.y, value.z, value.w);
#else
        result = ptr[index];
#endif
        return result;
    };
    const auto pack_node = [](const wp::vec4& lower, const wp::vec4& upper) {
        uint32_t lower_bits;
        uint32_t upper_bits;
        memcpy(&lower_bits, &lower.c[3], sizeof(lower_bits));
        memcpy(&upper_bits, &upper.c[3], sizeof(upper_bits));
        const uint64_t leaf = uint64_t(lower_bits >> 31);
        const uint64_t lower_payload = uint64_t(lower_bits & 0x7fffffffu);
        const uint64_t upper_payload = uint64_t(upper_bits & 0x7fffffffu);
        return (leaf << 62) | (upper_payload << 31) | lower_payload;
    };
    const auto lower_payload = [](uint64_t meta) {
        return static_cast<int>(meta & 0x7fffffffull);
    };
    const auto upper_payload = [](uint64_t meta) {
        return static_cast<int>((meta >> 31) & 0x7fffffffull);
    };
    const auto is_leaf = [](uint64_t meta) { return (meta >> 62) != 0; };

    // Callbacks inline; visit returns 0 to continue, nonzero to stop.
    const auto traverse = [&](auto&& bounds, auto&& visit,
                               bool near_first = false, bool test_root = true) {
        if (!node_count) return 0;
        const wp::vec4* lower = node_ray_lower.data;
        const wp::vec4* upper = node_ray_upper.data;
        const wp::vec4 root_lo = load_vec4_nc(lower, 0), root_hi = load_vec4_nc(upper, 0);
        float enter;
        if (test_root && !bounds(root_lo, root_hi, enter)) return 0;
        uint64_t current = pack_node(root_lo, root_hi);
        uint64_t stack[stack_capacity];
        int size = 0;
        while (true) {
            if (is_leaf(current)) {
                for (int k = lower_payload(current); k < upper_payload(current); ++k) {
                    const int result = visit(k);
                    if (result) return result;
                }
            } else {
                const int left = lower_payload(current), right = upper_payload(current);
                const wp::vec4 ll = load_vec4_nc(lower, left), lu = load_vec4_nc(upper, left);
                const wp::vec4 rl = load_vec4_nc(lower, right), ru = load_vec4_nc(upper, right);
                const uint64_t left_node = pack_node(ll, lu), right_node = pack_node(rl, ru);
                float tl, tr;
                const bool l = bounds(ll, lu, tl), r = bounds(rl, ru, tr);
                if (l && r) {
                    if (size == stack_capacity) return -1;
                    const bool swap = near_first && tr < tl;
                    stack[size++] = swap ? left_node : right_node;
                    current = swap ? right_node : left_node;
                    continue;
                }
                if (l) { current = left_node; continue; }
                if (r) { current = right_node; continue; }
            }
            if (!size) return 0;
            current = stack[--size];
        }
    };
"""


_NATIVE_RAY_BOX = r"""
    const bool fast_aabb = wp::abs(d[0]) >= 1.0e-20f
        && wp::abs(d[1]) >= 1.0e-20f && wp::abs(d[2]) >= 1.0e-20f;
    wp::vec3 inv_dir(0.0f);
    if (fast_aabb) {
        inv_dir = wp::vec3(1.0f / d[0], 1.0f / d[1], 1.0f / d[2]);
    }

    const auto box_interval = [&](const wp::vec3& lower, const wp::vec3& upper,
                           float tmin, float tmax, float& enter, bool conservative = false) {
        enter = tmin;
        float leave = tmax;
        const auto update = [&](float t0, float t1) {
            float near_t = wp::min(t0,t1), far_t = wp::max(t0,t1);
            if (conservative) {
                const float padding = 9.5367431640625e-7f
                    * wp::max(wp::abs(t0),wp::abs(t1)) + 1.e-37f;
                if (wp::isfinite(padding)) { near_t -= padding; far_t += padding; }
                else { near_t = -INFINITY; far_t = INFINITY; }
            }
            enter = wp::max(enter,near_t);
            leave = wp::min(leave,far_t);
        };
        if (fast_aabb) {
            float t0 = (lower[0] - o[0]) * inv_dir[0];
            float t1 = (upper[0] - o[0]) * inv_dir[0];
            update(t0,t1);
            t0 = (lower[1] - o[1]) * inv_dir[1];
            t1 = (upper[1] - o[1]) * inv_dir[1];
            update(t0,t1);
            t0 = (lower[2] - o[2]) * inv_dir[2];
            t1 = (upper[2] - o[2]) * inv_dir[2];
            update(t0,t1);
        } else {
            for (int axis = 0; axis < 3; ++axis) {
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
        return enter <= leave;
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
