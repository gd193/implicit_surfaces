"""Robin star-radius bounds on triangle SNCH trees."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import _NATIVE_POINT_GEOMETRY, _NATIVE_TRIANGLE_CLOSEST
from ._warp_specs import STACK_CAPACITY


_NATIVE_ROBIN = f"constexpr int stack_capacity = {STACK_CAPACITY};\n" + _NATIVE_POINT_GEOMETRY + r"""
    out_radius.data[qi] = NAN;
    out_status.data[qi] = -1;
    if (!finite3(p) || cap != cap || cap < 0.0f) return;
    if (node_count == 0 || cap == 0.0f) {
        out_radius.data[qi] = cap;
        out_status.data[qi] = 1;
        return;
    }
    const auto node_distance_sq = [&](int node) {
        return box_distance_sq(node_lower.data[node], node_upper.data[node]);
    };
""" + _NATIVE_TRIANGLE_CLOSEST + r"""
    const auto radius_bound_sq = [](float h, float mu, float distance2) {
        const float scale = mu * distance2;
        const float denominator = scale == INFINITY ? 1.0f : (scale - h) / scale;
        return distance2 / (denominator * denominator) * 0.999996f;
    };
    float best2 = cap * cap;
    constexpr uint32_t invalid_node = 0xffffffffu;
    constexpr uint32_t no_silhouette = 0x80000000u;
    const auto visit = [&](int node, float& priority) -> uint32_t {
        const float box2 = node_distance_sq(node);
        priority = box2;
        if (box2 > best2) return invalid_node;
        const wp::vec2 mu = node_robin_bounds.data[node];
        if (mu[0] == INFINITY) return uint32_t(node) | no_silhouette;
        const wp::vec2 sc = node_cone_sin_cos.data[node];
        if (sc[0] < 0.0f) return mu[1] > 0.0f ? uint32_t(node) | no_silhouette : invalid_node;
        if (sc[1] <= 0.0f) return uint32_t(node);
        const wp::vec3 lower = node_lower.data[node];
        const wp::vec3 upper = node_upper.data[node];
        const wp::vec3 view = 0.5f * (lower + upper) - p;
        const wp::vec3 extent = 0.5f * (upper - lower);
        const wp::vec3 axis = node_cone_axis.data[node];
        const float projection = wp::dot(axis, view);
        const float support = wp::dot(wp::abs(axis), extent);
        const float gap = wp::abs(projection) - support;
        if (!(gap > 0.0f)) return uint32_t(node);
        const float radius = node_cone_radius.data[node];
        const wp::vec3 perpendicular = view - projection * axis;
        const float slack = (mu[1] == 0.0f ? 2.0e-6f : 4.0e-6f)
                          * (wp::length(view) + radius);
        // Lower bound on |normal dot (surface point - query)|.
        const float h_min = gap * sc[1] - sc[0] * (wp::length(perpendicular) + radius) - slack;
        if (!(h_min > 0.0f) || !(h_min < INFINITY)) return uint32_t(node);
        if (mu[1] == 0.0f) return invalid_node;
        const wp::vec3 far = wp::max(wp::abs(lower - p), wp::abs(upper - p));
        const float far2 = wp::dot(far, far);
        const float scale = mu[1] * far2;
        if (scale <= h_min) return invalid_node;
        const float distance2 = wp::clamp(3.0f * (h_min / mu[1]), box2, far2);
        const float optimal_scale = mu[1] * distance2;
        if (optimal_scale > h_min && optimal_scale < INFINITY) {
            priority = radius_bound_sq(h_min, mu[1], distance2);
            if (priority > best2) return invalid_node;
        }
        const float upper_scale = mu[0] * wp::sqrt(box2);
        if (upper_scale > 1.0f) {
            const float denominator = upper_scale == INFINITY ? 1.0f
                : (upper_scale - 1.0f) / upper_scale;
            best2 = wp::min(best2, far2 / (denominator * denominator) * 1.000008f);
        }
        return uint32_t(node) | no_silhouette;
    };
    // Bit 31 carries the cone test through deferred traversal.
    uint32_t stack[stack_capacity];
    int size = 0;
    bool failed = false;
    float root_priority;
    uint32_t work = visit(0, root_priority);
    while (work != invalid_node || size > 0) {
        if (work == invalid_node) work = stack[--size];
        const int node = static_cast<int>(work & ~no_silhouette);
        if (node_distance_sq(node) > best2) { work = invalid_node; continue; }
        if (node_left.data[node] < 0) {
            if ((work & no_silhouette) == 0) {
                const int offset = node_silhouette_offset.data[node];
                const int count = node_silhouette_count.data[node];
                for (int j = 0; j < count; ++j) {
                    const int edge = silhouette_refs.data[offset + j];
                    const wp::vec3 a = silhouette_v0.data[edge];
                    const wp::vec3 view = p - a;
                    const float s0 = wp::dot(view, silhouette_n0.data[edge]);
                    const float s1 = wp::dot(view, silhouette_n1.data[edge]);
                    if ((s0 <= 0.0f && s1 >= 0.0f) || (s0 >= 0.0f && s1 <= 0.0f)) {
                        const wp::vec3 delta = closest_segment(a, silhouette_v1.data[edge]) - p;
                        best2 = wp::min(best2, wp::dot(delta, delta));
                    }
                }
            }
            const int offset = node_leaf_offset.data[node];
            const int count = node_leaf_count.data[node];
            if (node_robin_bounds.data[node][1] > 0.0f) {
                for (int k = offset; k < offset + count; ++k) {
                    const float mu = tri_robin_bounds.data[k][1];
                    if (mu == 0.0f) continue;
                    const wp::vec3 a = tri_v0.data[k];
                    float h = 0.0f, q = 0.0f, farthest2 = INFINITY;
                    if (mu < INFINITY) {
                        const wp::vec3 normal = tri_normal.data[k];
                        const wp::vec3 view = p - a;
                        const float normal_length = wp::length(normal);
                        h = normal_length > 0.0f ? wp::abs(wp::dot(view, normal)) / normal_length : 0.0f;
                        if (!(h < INFINITY)) { failed = true; break; }
                        q = mu * h;
                        const float error = 4.0e-6f * (wp::abs(view[0]) + wp::abs(view[1]) + wp::abs(view[2]));
                        if (h > error) {
                            const float lower_h = h - error;
                            if (lower_h * lower_h > best2 || 6.75f * (lower_h / mu) * 0.999996f > best2)
                                continue;
                            if (q < 3.0f) {
                                const wp::vec3 b = a + tri_e1.data[k];
                                const wp::vec3 c = a + tri_e2.data[k];
                                const float da2 = wp::dot(view, view);
                                const float db2 = wp::dot(b - p, b - p);
                                const float dc2 = wp::dot(c - p, c - p);
                                farthest2 = wp::max(da2, wp::max(db2, dc2));
                                const float scale = mu * farthest2;
                                if (farthest2 < INFINITY) {
                                    if (scale <= lower_h) continue;
                                    // The minimum is at r*=sqrt(3h/mu), or an interval endpoint.
                                    if (3.0f * (lower_h / mu) >= farthest2 && scale > h && scale < INFINITY) {
                                        best2 = wp::min(best2, radius_bound_sq(h, mu, farthest2));
                                        continue;
                                    }
                                    const float near_vertex2 = wp::min(da2, wp::min(db2, dc2));
                                    if (near_vertex2 * 1.000004f <= 3.0f * (lower_h / mu)
                                        && 3.0f * ((h + error) / mu) <= farthest2 * 0.999996f) {
                                        best2 = wp::min(best2, 6.75f * (h / mu) * 0.999996f);
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    const wp::vec3 e1 = tri_e1.data[k];
                    const wp::vec3 e2 = tri_e2.data[k];
                    const wp::vec3 delta = closest_triangle(a, e1, e2) - p;
                    const float closest2 = wp::dot(delta, delta);
                    if (!(closest2 < INFINITY)) { failed = true; break; }
                    if (closest2 > best2) continue;
                    if (mu == INFINITY || closest2 == 0.0f || h == 0.0f || q == 0.0f || q == INFINITY) {
                        best2 = closest2;
                        continue;
                    }
                    float distance2 = closest2;
                    if (q < 3.0f) {
                        if (farthest2 == INFINITY) {
                            const wp::vec3 da = a - p;
                            const wp::vec3 db = a + e1 - p;
                            const wp::vec3 dc = a + e2 - p;
                            farthest2 = wp::max(wp::dot(da, da),
                                wp::max(wp::dot(db, db), wp::dot(dc, dc)));
                        }
                        if (!(farthest2 < INFINITY)) { failed = true; break; }
                        distance2 = wp::clamp(3.0f * (h / mu), closest2, farthest2);
                    }
                    const float scale = mu * distance2;
                    if (scale > h) {
                        best2 = wp::min(best2, wp::max(closest2, radius_bound_sq(h, mu, distance2)));
                    }
                }
            }
            if (failed || best2 == 0.0f) break;
            work = invalid_node;
            continue;
        }
        const int left = node_left.data[node];
        const int right = node_right.data[node];
        float left_priority, right_priority;
        const uint32_t left_work = visit(left, left_priority);
        const uint32_t right_work = visit(right, right_priority);
        const bool visit_left = left_work != invalid_node && left_priority <= best2;
        const bool visit_right = right_work != invalid_node && right_priority <= best2;
        if (visit_left && visit_right) {
            const bool right_first = right_priority < left_priority;
            if (size >= stack_capacity) { failed = true; break; }
            stack[size++] = right_first ? left_work : right_work;
            work = right_first ? right_work : left_work;
        } else {
            work = visit_left ? left_work : (visit_right ? right_work : invalid_node);
        }
    }
    if (!failed && best2 == best2) {
        out_radius.data[qi] = wp::min(cap, wp::sqrt(best2));
        out_status.data[qi] = 1;
    }
"""


@wp.func_native(_NATIVE_ROBIN)
def native_robin_radius(
    qi: int, query: wp.vec3, cap: float, node_count: int,
    node_lower: wp.array(dtype=wp.vec3), node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32), node_right: wp.array(dtype=wp.int32),
    node_leaf_offset: wp.array(dtype=wp.int32), node_leaf_count: wp.array(dtype=wp.int32),
    node_robin_bounds: wp.array(dtype=wp.vec2),
    node_silhouette_offset: wp.array(dtype=wp.int32), node_silhouette_count: wp.array(dtype=wp.int32),
    silhouette_refs: wp.array(dtype=wp.int32),
    silhouette_v0: wp.array(dtype=wp.vec3), silhouette_v1: wp.array(dtype=wp.vec3),
    silhouette_n0: wp.array(dtype=wp.vec3), silhouette_n1: wp.array(dtype=wp.vec3),
    node_cone_axis: wp.array(dtype=wp.vec3), node_cone_sin_cos: wp.array(dtype=wp.vec2),
    node_cone_radius: wp.array(dtype=wp.float32),
    tri_v0: wp.array(dtype=wp.vec3), tri_e1: wp.array(dtype=wp.vec3), tri_e2: wp.array(dtype=wp.vec3),
    tri_normal: wp.array(dtype=wp.vec3), tri_robin_bounds: wp.array(dtype=wp.vec2),
    out_radius: wp.array(dtype=wp.float32), out_status: wp.array(dtype=wp.int32),
): ...


@wp.kernel
def robin_radius_kernel(
    points: wp.array(dtype=wp.vec3), caps: wp.array(dtype=wp.float32),
    node_lower: wp.array(dtype=wp.vec3), node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32), node_right: wp.array(dtype=wp.int32),
    node_leaf_offset: wp.array(dtype=wp.int32), node_leaf_count: wp.array(dtype=wp.int32),
    node_robin_bounds: wp.array(dtype=wp.vec2),
    node_silhouette_offset: wp.array(dtype=wp.int32), node_silhouette_count: wp.array(dtype=wp.int32),
    silhouette_refs: wp.array(dtype=wp.int32),
    silhouette_v0: wp.array(dtype=wp.vec3), silhouette_v1: wp.array(dtype=wp.vec3),
    silhouette_n0: wp.array(dtype=wp.vec3), silhouette_n1: wp.array(dtype=wp.vec3),
    node_cone_axis: wp.array(dtype=wp.vec3), node_cone_sin_cos: wp.array(dtype=wp.vec2),
    node_cone_radius: wp.array(dtype=wp.float32),
    tri_v0: wp.array(dtype=wp.vec3), tri_e1: wp.array(dtype=wp.vec3), tri_e2: wp.array(dtype=wp.vec3),
    tri_normal: wp.array(dtype=wp.vec3), tri_robin_bounds: wp.array(dtype=wp.vec2),
    out_radius: wp.array(dtype=wp.float32), out_status: wp.array(dtype=wp.int32),
):
    qi = wp.tid()
    native_robin_radius(
        qi, points[qi], caps[qi], wp.len(node_left),
        node_lower, node_upper, node_left, node_right,
        node_leaf_offset, node_leaf_count, node_robin_bounds,
        node_silhouette_offset, node_silhouette_count, silhouette_refs,
        silhouette_v0, silhouette_v1, silhouette_n0, silhouette_n1,
        node_cone_axis, node_cone_sin_cos, node_cone_radius,
        tri_v0, tri_e1, tri_e2, tri_normal, tri_robin_bounds,
        out_radius, out_status,
    )
