"""Packed ray traversal; float4 nodes require 16-byte alignment."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import _NATIVE_FINITE3, _NATIVE_PACKED_NODES
from ._warp_specs import STACK_CAPACITY


# Literal modes specialize first-hit/any-hit; IDs stay int32.
_NATIVE_QUERY_COMMON = (
    f"constexpr int stack_capacity = {STACK_CAPACITY};\n"
    + _NATIVE_FINITE3 + _NATIVE_PACKED_NODES + r"""
    const auto run_query = [&](float* result_t, int* result_id,
                               int* result_hit, int* result_status,
                               bool any_hit) {
        const wp::vec3 o = origin;
        const wp::vec3 d = direction;
        const float lo_t = min_t;
        const float hi_t = max_t;
        if (result_t) *result_t = hi_t;
        if (result_id) *result_id = -1;
        *result_hit = 0;
        *result_status = 0;

        if (!finite3(o) || !finite3(d) || lo_t != lo_t || hi_t != hi_t
            || wp::isinf(lo_t) || lo_t < 0.0f || hi_t < lo_t
            || wp::dot(d, d) <= 1.0e-30f) {
            *result_status = -1;
            return;
        }
        if (node_count == 0) return;

        const bool fast_aabb = wp::abs(d[0]) >= 1.0e-20f
            && wp::abs(d[1]) >= 1.0e-20f && wp::abs(d[2]) >= 1.0e-20f;
        wp::vec3 inv_dir(0.0f);
        if (fast_aabb) {
            inv_dir = wp::vec3(1.0f / d[0], 1.0f / d[1], 1.0f / d[2]);
        }

        const auto box_t = [&](const wp::vec3& lower, const wp::vec3& upper,
                               float tmin, float tmax) {
            float enter = tmin;
            float leave = tmax;
            if (fast_aabb) {
                float t0 = (lower[0] - o[0]) * inv_dir[0];
                float t1 = (upper[0] - o[0]) * inv_dir[0];
                enter = wp::max(enter, wp::min(t0, t1));
                leave = wp::min(leave, wp::max(t0, t1));
                t0 = (lower[1] - o[1]) * inv_dir[1];
                t1 = (upper[1] - o[1]) * inv_dir[1];
                enter = wp::max(enter, wp::min(t0, t1));
                leave = wp::min(leave, wp::max(t0, t1));
                t0 = (lower[2] - o[2]) * inv_dir[2];
                t1 = (upper[2] - o[2]) * inv_dir[2];
                enter = wp::max(enter, wp::min(t0, t1));
                leave = wp::min(leave, wp::max(t0, t1));
            } else {
                for (int axis = 0; axis < 3; ++axis) {
                    const float component = d[axis];
                    const float origin_component = o[axis];
                    if (component == 0.0f) {
                        if (origin_component < lower[axis]
                            || origin_component > upper[axis]) return -1.0f;
                    } else {
                        const float t0 = (lower[axis] - origin_component) / component;
                        const float t1 = (upper[axis] - origin_component) / component;
                        enter = wp::max(enter, wp::min(t0, t1));
                        leave = wp::min(leave, wp::max(t0, t1));
                    }
                }
            }
            return enter <= leave ? enter : -1.0f;
        };

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

        const wp::vec4* lower_data = node_ray_lower.data;
        const wp::vec4* upper_data = node_ray_upper.data;
        uint64_t current_node = pack_node(load_vec4_nc(lower_data, 0),
                                          load_vec4_nc(upper_data, 0));
        // Each slot is written by a push before a pop can read it. Keep this
        // array uninitialized: clearing every slot per query is costly.
        uint64_t stack[stack_capacity];
        int stack_size = 0;
        float best_t = hi_t;
        int best_id = -1;
        bool failed = false;
        while (true) {
            if (is_leaf(current_node)) {
                const int primitive_begin = lower_payload(current_node);
                const int primitive_end = upper_payload(current_node);
                for (int k = primitive_begin; k < primitive_end; ++k) {
                    float t;
                    const bool hit = triangle_t(tri_v0.data[k], tri_e1.data[k],
                        tri_e2.data[k], lo_t, any_hit ? hi_t : best_t, t);
                    if (any_hit) {
                        if (hit) {
                            *result_hit = 1;
                            *result_status = 1;
                            return;
                        }
                    } else {
                        const int primitive_id = prim_ids.data[k];
                        if (hit && (t < best_t
                            || (t == best_t && (best_id < 0 || primitive_id < best_id)))) {
                            best_t = t;
                            best_id = primitive_id;
                        }
                    }
                }
                if (stack_size == 0) break;
                current_node = stack[--stack_size];
                continue;
            } else {
                const int left_index = lower_payload(current_node);
                const int right_index = upper_payload(current_node);
                const wp::vec4 left_lower = load_vec4_nc(lower_data, left_index);
                const wp::vec4 left_upper = load_vec4_nc(upper_data, left_index);
                const wp::vec4 right_lower = load_vec4_nc(lower_data, right_index);
                const wp::vec4 right_upper = load_vec4_nc(upper_data, right_index);
                const uint64_t left_node = pack_node(left_lower, left_upper);
                const uint64_t right_node = pack_node(right_lower, right_upper);
                const float bound_t = any_hit ? hi_t : best_t;
                const float tl = box_t(wp::vec3(left_lower[0], left_lower[1], left_lower[2]),
                                       wp::vec3(left_upper[0], left_upper[1], left_upper[2]),
                                       lo_t, bound_t);
                const float tr = box_t(wp::vec3(right_lower[0], right_lower[1], right_lower[2]),
                                       wp::vec3(right_upper[0], right_upper[1], right_upper[2]),
                                       lo_t, bound_t);
                const bool hit_left = tl >= 0.0f;
                const bool hit_right = tr >= 0.0f;
                if (hit_left && hit_right) {
                    uint64_t near_node = left_node;
                    uint64_t far_node = right_node;
                    if (tr < tl) {
                        near_node = right_node;
                        far_node = left_node;
                    }
                    if (stack_size >= stack_capacity) {
                        failed = true;
                        break;
                    } else {
                        stack[stack_size++] = far_node;
                        current_node = near_node;
                    }
                } else if (hit_left) {
                    current_node = left_node;
                } else if (hit_right) {
                    current_node = right_node;
                } else {
                    if (stack_size == 0) break;
                    current_node = stack[--stack_size];
                }
            }
        }
        if (failed) {
            *result_status = -1;
        } else if (!any_hit && best_id >= 0) {
            *result_t = best_t;
            *result_id = best_id;
            *result_hit = 1;
            *result_status = 1;
        }
    };
""")


@wp.func_native(_NATIVE_QUERY_COMMON + r"""
    run_query(out_t.data + qi, out_primitive_id.data + qi,
              out_hit.data + qi, out_status.data + qi, false);
""")
def native_first_hit(
    qi: int,
    origin: wp.vec3,
    direction: wp.vec3,
    min_t: wp.float32,
    max_t: wp.float32,
    node_count: int,
    node_ray_lower: wp.array(dtype=wp.vec4),
    node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_t: wp.array(dtype=wp.float32),
    out_primitive_id: wp.array(dtype=wp.int32),
    out_hit: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
): ...


@wp.func_native(_NATIVE_QUERY_COMMON + r"""
    run_query(nullptr, nullptr, out_hit.data + qi, out_status.data + qi, true);
""")
def native_any_hit(
    qi: int,
    origin: wp.vec3,
    direction: wp.vec3,
    min_t: wp.float32,
    max_t: wp.float32,
    node_count: int,
    node_ray_lower: wp.array(dtype=wp.vec4),
    node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_hit: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
): ...


@wp.kernel
def first_hit_kernel(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=wp.float32),
    max_t: wp.array(dtype=wp.float32),
    node_ray_lower: wp.array(dtype=wp.vec4),
    node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_t: wp.array(dtype=wp.float32),
    out_primitive_id: wp.array(dtype=wp.int32),
    out_hit: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    qi = wp.tid()
    native_first_hit(qi, origins[qi], directions[qi], min_t[qi], max_t[qi],
        wp.len(node_ray_lower), node_ray_lower, node_ray_upper, prim_ids,
        tri_v0, tri_e1, tri_e2,
        out_t, out_primitive_id, out_hit, out_status)


@wp.kernel
def any_hit_kernel(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=wp.float32),
    max_t: wp.array(dtype=wp.float32),
    node_ray_lower: wp.array(dtype=wp.vec4),
    node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_hit: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    qi = wp.tid()
    native_any_hit(qi, origins[qi], directions[qi], min_t[qi], max_t[qi],
        wp.len(node_ray_lower), node_ray_lower, node_ray_upper, prim_ids,
        tri_v0, tri_e1, tri_e2,
        out_hit, out_status)
