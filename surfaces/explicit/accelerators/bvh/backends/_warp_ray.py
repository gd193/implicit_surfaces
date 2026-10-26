"""Packed ray traversal; float4 nodes require 16-byte alignment."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import _NATIVE_FINITE3, _NATIVE_PACKED_TRAVERSAL, _NATIVE_RAY_BOX


# Literal modes specialize first-hit/any-hit; IDs stay int32.
_NATIVE_QUERY_COMMON = (
    _NATIVE_FINITE3 + _NATIVE_PACKED_TRAVERSAL + r"""
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

""" + _NATIVE_RAY_BOX + r"""
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

        float best_t = hi_t;
        int best_id = -1;
        const auto bounds = [&](const wp::vec4& lower, const wp::vec4& upper, float& enter) {
            return box_interval(wp::vec3(lower[0], lower[1], lower[2]),
                wp::vec3(upper[0], upper[1], upper[2]), lo_t, any_hit ? hi_t : best_t, enter);
        };
        const int traversal = traverse(bounds, [&](int k) {
            float t;
            const bool hit = triangle_t(tri_v0.data[k], tri_e1.data[k], tri_e2.data[k],
                                         lo_t, any_hit ? hi_t : best_t, t);
            if (any_hit) return hit ? 1 : 0;
            const int id = prim_ids.data[k];
            if (hit && (t < best_t || (t == best_t && (best_id < 0 || id < best_id)))) {
                best_t = t;
                best_id = id;
            }
            return 0;
        }, true, false);
        if (traversal < 0) {
            *result_status = -1;
        } else if (any_hit && traversal) {
            *result_hit = 1;
            *result_status = 1;
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
