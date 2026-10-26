"""Closest-point traversal with cached node bounds."""
from __future__ import annotations

import warp as wp

from ._warp_specs import STACK_CAPACITY
from ._warp_geometry import _NATIVE_POINT_GEOMETRY, _NATIVE_TRIANGLE_CLOSEST


_NATIVE_CLOSEST_POINT = (f"constexpr int stack_capacity = {STACK_CAPACITY};\n"
                         + _NATIVE_POINT_GEOMETRY + _NATIVE_TRIANGLE_CLOSEST + r"""

    out_point.data[qi] = wp::vec3(0.0f);
    out_distance.data[qi] = INFINITY;
    out_primitive_id.data[qi] = -1;
    out_status.data[qi] = 0;
    if (!finite3(p) || cap != cap || cap < 0.0f) {
        out_status.data[qi] = -1;
        return;
    }
    if (node_count == 0) return;

    struct StackEntry { int node; float lower_d2; };
    StackEntry stack[stack_capacity];
    int stack_size = 0;
    int node = 0;
    const float root_lower_d2 = box_distance_sq(node_lower.data[0], node_upper.data[0]);
    float best_d2 = wp::min(3.4028234e38f, cap*cap);
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
    if (failed || (best_id < 0 && cap*cap >= 3.4028234e38f)) {
        out_status.data[qi] = -1;
    } else if (best_id >= 0) {
        out_point.data[qi] = best_point;
        out_distance.data[qi] = wp::sqrt(best_d2);
        out_primitive_id.data[qi] = best_id;
        out_status.data[qi] = 1;
    }
""")


@wp.func_native(_NATIVE_CLOSEST_POINT)
def native_closest_point(
    qi: int,
    query: wp.vec3,
    cap: wp.float32,
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
