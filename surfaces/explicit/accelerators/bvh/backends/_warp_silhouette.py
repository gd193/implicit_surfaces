"""Bounded closest silhouette with box-support cone pruning."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import _NATIVE_POINT_GEOMETRY, _NATIVE_SEGMENT_CLOSEST
from ._warp_specs import STACK_CAPACITY


_NATIVE_SILHOUETTE = (f"constexpr int stack_capacity = {STACK_CAPACITY};\n"
                      + _NATIVE_POINT_GEOMETRY + _NATIVE_SEGMENT_CLOSEST + r"""
    out_point.data[qi] = wp::vec3(0.0f);
    out_distance.data[qi] = INFINITY;
    out_edge_id.data[qi] = -1;
    out_status.data[qi] = 0;
    if (!finite3(p) || cap != cap || cap < 0.0f) {
        out_status.data[qi] = -1;
        return;
    }
    if (node_count == 0) return;

    const auto node_distance_sq = [&](int node) {
        return box_distance_sq(node_lower.data[node], node_upper.data[node]);
    };
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

    int stack[stack_capacity];
    int stack_size = 0;
    int node = 0;
    float best_d2 = cap * cap;
    wp::vec3 best_point(0.0f);
    int best_edge = -1;
    bool failed = false;
    if (node_distance_sq(0) <= best_d2 && cone_may_have_silhouette(0)) {
        while (node >= 0 || stack_size > 0) {
            if (node < 0) node = stack[--stack_size];
            if (node_distance_sq(node) > best_d2) {
                node = -1;
                continue;
            }
            if (node_left.data[node] < 0) {
                const int offset = node_silhouette_offset.data[node];
                const int count = node_silhouette_count.data[node];
                for (int j = 0; j < count; ++j) {
                    const int edge = silhouette_refs.data[offset + j];
                    const wp::vec3 a = silhouette_v0.data[edge];
                    const wp::vec3 view = p - a;
                    const float side0 = wp::dot(view, silhouette_n0.data[edge]);
                    const float side1 = wp::dot(view, silhouette_n1.data[edge]);
                    // Include tangencies and open edges (n1=-n0).
                    if ((side0 <= 0.0f && side1 >= 0.0f) || (side0 >= 0.0f && side1 <= 0.0f)) {
                        const wp::vec3 candidate = closest_segment(a, silhouette_v1.data[edge]);
                        const wp::vec3 delta = candidate - p;
                        const float d2 = wp::dot(delta, delta);
                        if (d2 <= best_d2 && (d2 < best_d2 || best_edge < 0 || edge < best_edge)) {
                            best_d2 = d2;
                            best_point = candidate;
                            best_edge = edge;
                        }
                    }
                }
                node = -1;
                continue;
            }
            const int left = node_left.data[node];
            const int right = node_right.data[node];
            const float left_d2 = node_distance_sq(left);
            const float right_d2 = node_distance_sq(right);
            bool visit_left = left_d2 <= best_d2;
            bool visit_right = right_d2 <= best_d2;
            if (visit_left) visit_left = cone_may_have_silhouette(left);
            if (visit_right) visit_right = cone_may_have_silhouette(right);
            if (!visit_left && !visit_right) {
                node = -1;
                continue;
            }
            if (visit_left && visit_right) {
                int near_node = left;
                int far_node = right;
                if (right_d2 < left_d2) {
                    near_node = right;
                    far_node = left;
                }
                if (stack_size >= stack_capacity) {
                    failed = true;
                    break;
                }
                stack[stack_size++] = far_node;
                node = near_node;
            } else {
                node = visit_left ? left : right;
            }
        }
    }
    if (failed) {
        out_status.data[qi] = -1;
    } else if (best_edge >= 0) {
        out_point.data[qi] = best_point;
        out_distance.data[qi] = wp::sqrt(best_d2);
        out_edge_id.data[qi] = best_edge;
        out_status.data[qi] = 1;
    }
""")


@wp.func_native(_NATIVE_SILHOUETTE)
def native_closest_silhouette(
    qi: int,
    query: wp.vec3,
    cap: float,
    node_count: int,
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_silhouette_offset: wp.array(dtype=wp.int32),
    node_silhouette_count: wp.array(dtype=wp.int32),
    silhouette_refs: wp.array(dtype=wp.int32),
    silhouette_v0: wp.array(dtype=wp.vec3),
    silhouette_v1: wp.array(dtype=wp.vec3),
    silhouette_n0: wp.array(dtype=wp.vec3),
    silhouette_n1: wp.array(dtype=wp.vec3),
    node_cone_axis: wp.array(dtype=wp.vec3),
    node_cone_sin_cos: wp.array(dtype=wp.vec2),
    node_cone_radius: wp.array(dtype=wp.float32),
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_edge_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
): ...


@wp.kernel
def closest_silhouette_kernel(
    points: wp.array(dtype=wp.vec3),
    max_distance: wp.array(dtype=wp.float32),
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_silhouette_offset: wp.array(dtype=wp.int32),
    node_silhouette_count: wp.array(dtype=wp.int32),
    silhouette_refs: wp.array(dtype=wp.int32),
    silhouette_v0: wp.array(dtype=wp.vec3),
    silhouette_v1: wp.array(dtype=wp.vec3),
    silhouette_n0: wp.array(dtype=wp.vec3),
    silhouette_n1: wp.array(dtype=wp.vec3),
    node_cone_axis: wp.array(dtype=wp.vec3),
    node_cone_sin_cos: wp.array(dtype=wp.vec2),
    node_cone_radius: wp.array(dtype=wp.float32),
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_edge_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    q = wp.tid()
    native_closest_silhouette(
        q, points[q], max_distance[q], wp.len(node_left),
        node_lower, node_upper, node_left, node_right,
        node_silhouette_offset, node_silhouette_count, silhouette_refs,
        silhouette_v0, silhouette_v1, silhouette_n0, silhouette_n1,
        node_cone_axis, node_cone_sin_cos, node_cone_radius,
        out_point, out_distance, out_edge_id, out_status,
    )
