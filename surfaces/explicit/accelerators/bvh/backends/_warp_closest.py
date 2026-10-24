"""Closest-point traversal with cached node bounds."""
from __future__ import annotations

import warp as wp

from ._warp_specs import STACK_CAPACITY


# The failure flag preserves CUDA throughput; avoid early returns in traversal.
_NATIVE_CLOSEST_POINT = f"constexpr int stack_capacity = {STACK_CAPACITY};\n" + r"""
    const wp::vec3 p = query;
    const auto finite3 = [](const wp::vec3& v) {
        const float limit = 3.4028234e38f;
        return v[0] == v[0] && v[1] == v[1] && v[2] == v[2]
            && wp::abs(v[0]) < limit && wp::abs(v[1]) < limit
            && wp::abs(v[2]) < limit;
    };
    const auto box_distance_sq = [&](const wp::vec3& lower,
                                     const wp::vec3& upper) {
        const float dx = wp::max(wp::max(lower[0] - p[0], 0.0f), p[0] - upper[0]);
        const float dy = wp::max(wp::max(lower[1] - p[1], 0.0f), p[1] - upper[1]);
        const float dz = wp::max(wp::max(lower[2] - p[2], 0.0f), p[2] - upper[2]);
        return dx * dx + dy * dy + dz * dz;
    };
    const auto closest_segment = [&](const wp::vec3& a, const wp::vec3& b) {
        const wp::vec3 ab = b - a;
        const float denom = wp::dot(ab, ab);
        float t = 0.0f;
        if (denom > 0.0f)
            t = wp::clamp(wp::dot(p - a, ab) / denom, 0.0f, 1.0f);
        return a + t * ab;
    };
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

    out_point.data[qi] = wp::vec3(0.0f);
    out_distance.data[qi] = INFINITY;
    out_primitive_id.data[qi] = -1;
    out_status.data[qi] = 0;
    if (!finite3(p)) {
        out_status.data[qi] = -1;
        return;
    }
    if (node_count == 0) return;

    struct StackEntry { int node; float lower_d2; };
    StackEntry stack[stack_capacity];
    int stack_size = 0;
    int node = 0;
    const float root_lower_d2 = box_distance_sq(node_lower.data[0], node_upper.data[0]);
    float best_d2 = 3.4028234e38f;
    wp::vec3 best_point(0.0f);
    int best_id = -1;
    bool failed = false;
    if (root_lower_d2 <= best_d2) {
        while (true) {
            // Accepted near children need no retest.
            while (true) {
                const int count = node_leaf_count.data[node];
                if (count > 0) {
                    const int offset = node_leaf_offset.data[node];
                    for (int j = 0; j < count; ++j) {
                        const int k = offset + j;
                        const wp::vec3 a = tri_v0.data[k];
                        const wp::vec3 e1 = tri_e1.data[k];
                        const wp::vec3 e2 = tri_e2.data[k];
                        const wp::vec3 b = a + e1;
                        const wp::vec3 c = a + e2;
                        const wp::vec3 tri_lower(
                            wp::min(a[0], wp::min(b[0], c[0])),
                            wp::min(a[1], wp::min(b[1], c[1])),
                            wp::min(a[2], wp::min(b[2], c[2])));
                        const wp::vec3 tri_upper(
                            wp::max(a[0], wp::max(b[0], c[0])),
                            wp::max(a[1], wp::max(b[1], c[1])),
                            wp::max(a[2], wp::max(b[2], c[2])));
                        const float cull_limit = best_d2 * 1.000002f
                            + 1.0e-12f * wp::dot(p, p);
                        if (box_distance_sq(tri_lower, tri_upper) <= cull_limit) {
                            const wp::vec3 candidate = closest_triangle(a, e1, e2);
                            const wp::vec3 delta = p - candidate;
                            const float d2 = wp::dot(delta, delta);
                            if (d2 <= best_d2) {
                                const int primitive_id = prim_ids.data[k];
                                if (d2 < best_d2 || best_id < 0 || primitive_id < best_id) {
                                    best_d2 = d2;
                                    best_point = candidate;
                                    best_id = primitive_id;
                                }
                            }
                        }
                    }
                    break;
                }

                const int left = node_left.data[node];
                const int right = node_right.data[node];
                const float left_d2 = box_distance_sq(node_lower.data[left], node_upper.data[left]);
                const float right_d2 = box_distance_sq(node_lower.data[right], node_upper.data[right]);
                const bool visit_left = left_d2 <= best_d2;
                const bool visit_right = right_d2 <= best_d2;
                if (!visit_left && !visit_right) break;
                if (visit_left && visit_right) {
                    int near_node = left;
                    int far_node = right;
                    float far_d2 = right_d2;
                    if (right_d2 < left_d2) {
                        near_node = right;
                        far_node = left;
                        far_d2 = left_d2;
                    }
                    if (stack_size >= stack_capacity) {
                        failed = true;
                        break;
                    }
                    stack[stack_size++] = StackEntry{far_node, far_d2};
                    node = near_node;
                } else {
                    node = visit_left ? left : right;
                }
            }
            if (failed) break;

            // Retain equal-distance nodes for face-ID ties.
            bool found_next = false;
            while (stack_size > 0) {
                const StackEntry entry = stack[--stack_size];
                if (entry.lower_d2 > best_d2) continue;
                node = entry.node;
                found_next = true;
                break;
            }
            if (!found_next) break;
        }
    }
    if (failed || best_id < 0) {
        out_status.data[qi] = -1;
    } else {
        out_point.data[qi] = best_point;
        out_distance.data[qi] = wp::sqrt(best_d2);
        out_primitive_id.data[qi] = best_id;
        out_status.data[qi] = 1;
    }
"""


@wp.func_native(_NATIVE_CLOSEST_POINT)
def native_closest_point(
    qi: int,
    query: wp.vec3,
    node_count: int,
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_leaf_offset: wp.array(dtype=wp.int32),
    node_leaf_count: wp.array(dtype=wp.int32),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_primitive_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
): ...


@wp.kernel
def closest_point_kernel(
    query: wp.array(dtype=wp.vec3),
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_leaf_offset: wp.array(dtype=wp.int32),
    node_leaf_count: wp.array(dtype=wp.int32),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_primitive_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    qi = wp.tid()
    native_closest_point(
        qi, query[qi], wp.len(node_left), node_lower, node_upper, node_left, node_right,
        node_leaf_offset, node_leaf_count, prim_ids, tri_v0, tri_e1, tri_e2,
        out_point, out_distance, out_primitive_id, out_status,
    )
