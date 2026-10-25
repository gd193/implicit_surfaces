"""Half-open ray/line crossings with optional bounded hit output."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import (
    _NATIVE_EDGE_PREDICATES, _NATIVE_FINITE3, _NATIVE_PACKED_TRAVERSAL, _NATIVE_RAY_BOX,
)


_NATIVE_CROSSINGS = (
    _NATIVE_FINITE3 + _NATIVE_PACKED_TRAVERSAL + _NATIVE_EDGE_PREDICATES + r"""
    const auto run_query = [&](float* times, int* faces, int capacity,
                               int* result_count, int* status) {
        *result_count = -1;
        *status = -1;
        for (int i = 0; i < capacity; ++i) { times[i] = INFINITY; faces[i] = -1; }
        const wp::vec3 o = origin, d = direction;
        const float lo_t = min_t, hi_t = max_t;
        if (!finite3(o) || !finite3(d) || (d[0] == 0 && d[1] == 0 && d[2] == 0)
            || lo_t != lo_t || hi_t != hi_t || lo_t > hi_t
            || lo_t == INFINITY || hi_t == -INFINITY
            || exclude_face < -1 || exclude_face >= prim_ids.shape[0]) return;
        *result_count = 0;
        *status = 0;
        if (!node_count) return;
""" + _NATIVE_RAY_BOX + r"""
        int kz = 0;
        if (wp::abs(d[1]) > wp::abs(d[kz])) kz = 1;
        if (wp::abs(d[2]) > wp::abs(d[kz])) kz = 2;
        const int kx = (kz+1)%3, ky = (kz+2)%3;
        const auto triangle_t = [&](int slot, float& result) {
            const wp::vec3i ids = tri_indices.data[slot];
            // Reuse one predicate at both precisions. -1 asks for the double path.
            const auto solve = [&](auto zero) {
                using S = decltype(zero);
                S x[3], y[3], z[3], uncertainty[3];
                const S sx = static_cast<S>(d[kx]) / static_cast<S>(d[kz]);
                const S sy = static_cast<S>(d[ky]) / static_cast<S>(d[kz]);
                constexpr float rounding = 8.0f * 1.1920928955078125e-7f;
                for (int i = 0; i < 3; ++i) {
                    const wp::vec3 v = mesh_vertices.data[ids[i]];
                    const S a = static_cast<S>(v[kx])-static_cast<S>(o[kx]);
                    const S b = static_cast<S>(v[ky])-static_cast<S>(o[ky]);
                    z[i] = static_cast<S>(v[kz])-static_cast<S>(o[kz]);
                    x[i] = a - sx*z[i];
                    y[i] = b - sy*z[i];
                    uncertainty[i] = rounding * wp::max(wp::abs(a)+wp::abs(sx*z[i]),
                        wp::abs(b)+wp::abs(sy*z[i])) + static_cast<S>(1.e-37f);
                }
                S e[3], error[3];
                bool positive = false, negative = false, uncertain = false;
                for (int i = 0; i < 3; ++i) {
                    const int a = (i+1)%3, b = (i+2)%3;
                    if constexpr (sizeof(S) == sizeof(float)) {
                        e[i] = x[a]*y[b] - y[a]*x[b];
                        error[i] = rounding*(wp::abs(x[a]*y[b])+wp::abs(y[a]*x[b]))
                            + uncertainty[a]*(wp::abs(x[b])+wp::abs(y[b])+2*uncertainty[b])
                            + uncertainty[b]*(wp::abs(x[a])+wp::abs(y[a]));
                    } else {
                        e[i] = cross2(x[a], y[a], x[b], y[b]);
                        error[i] = 0;
                    }
                    positive |= e[i] > error[i];
                    negative |= e[i] < -error[i];
                    uncertain |= !(wp::abs(e[i]) > error[i]);
                }
                if (positive && negative) return 0;
                if constexpr (sizeof(S) == sizeof(float)) {
                    if (uncertain) return -1;
                }
                const S det = e[0]+e[1]+e[2];
                if (det == 0) return 0; // Parallel/coplanar faces have no isolated crossing.
                const S numerator = e[0]*z[0]+e[1]*z[1]+e[2]*z[2];
                const S denominator = det*static_cast<S>(d[kz]);
                S t = numerator / denominator;
                if constexpr (sizeof(S) == sizeof(float)) {
                    const S num_error = rounding*(wp::abs(e[0]*z[0])+wp::abs(e[1]*z[1])
                        + wp::abs(e[2]*z[2])) + error[0]*wp::abs(z[0])
                        + error[1]*wp::abs(z[1]) + error[2]*wp::abs(z[2]);
                    const S den_error = wp::abs(d[kz])*(error[0]+error[1]+error[2])
                        + rounding*wp::abs(denominator);
                    if (!(wp::abs(denominator) > den_error) || !wp::isfinite(t)) return -1;
                    const S t_error = (num_error+wp::abs(t)*den_error)
                        / (wp::abs(denominator)-den_error);
                    if (!wp::isfinite(t_error)) return -1;
                    if (t+t_error < lo_t || t-t_error > hi_t) return 0;
                    if (t-t_error < lo_t || t+t_error > hi_t) return -1;
                } else {
                    // Recompute from the original plane at ambiguous interval endpoints.
                    const wp::vec3 a = mesh_vertices.data[ids[0]];
                    const wp::vec3 b = mesh_vertices.data[ids[1]];
                    const wp::vec3 c = mesh_vertices.data[ids[2]];
                    S ab[3], ac[3], normal[3];
                    for (int i = 0; i < 3; ++i) {
                        ab[i] = static_cast<S>(b[i])-static_cast<S>(a[i]);
                        ac[i] = static_cast<S>(c[i])-static_cast<S>(a[i]);
                    }
                    S plane = 0, velocity = 0;
                    for (int i = 0; i < 3; ++i) {
                        const int u = (i+1)%3, v = (i+2)%3;
                        normal[i] = cross2(ab[u],ab[v],ac[u],ac[v]);
                        plane += normal[i]*(static_cast<S>(a[i])-static_cast<S>(o[i]));
                        velocity += normal[i]*static_cast<S>(d[i]);
                    }
                    if (velocity == 0) return 0;
                    t = plane/velocity;
                    if (t < lo_t || t > hi_t) return 0;
                    const S orientation = det > 0 ? 1 : -1;
                    for (int i = 0; i < 3; ++i) {
                        if (e[i] == 0) {
                            const int a = (i+1)%3, b = (i+2)%3;
                            if (!owns_edge(x[a], y[a], x[b], y[b], orientation)) return 0;
                        }
                    }
                }
                result = static_cast<float>(t);
                return times && !wp::isfinite(result) ? -2 : 1;
            };
            const int fast = solve(0.0f);
            return fast == -1 ? solve(0.0) : fast;
        };
        const auto overlaps = [&](const wp::vec4& lower, const wp::vec4& upper, float& enter) {
            return box_interval(wp::vec3(lower[0],lower[1],lower[2]),
                                wp::vec3(upper[0],upper[1],upper[2]), lo_t, hi_t, enter, true);
        };
        int count = 0;
        const int traversal = traverse(overlaps, [&](int k) {
            const int face = prim_ids.data[k];
            if (face == exclude_face) return 0;
            float t;
            const int hit = triangle_t(k, t);
            if (hit < 0) return -1;
            if (hit) {
                if (count < capacity) { times[count] = t; faces[count] = face; }
                ++count;
            }
            return 0;
        });
        if (traversal < 0) {
            *result_count = -1;
            *status = -1;
            for (int i = 0; i < capacity; ++i) { times[i] = INFINITY; faces[i] = -1; }
        } else {
            *result_count = count;
            *status = count ? (times && count > capacity ? -2 : 1) : 0;
        }
    };
""")


@wp.func_native(_NATIVE_CROSSINGS + r"""
    run_query(nullptr, nullptr, 0, out_count.data+qi, out_status.data+qi);
""")
def native_count_hits(
    qi: int, origin: wp.vec3, direction: wp.vec3, min_t: float, max_t: float,
    exclude_face: int, node_count: int,
    node_ray_lower: wp.array(dtype=wp.vec4), node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32), mesh_vertices: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.vec3i), out_count: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
): ...


@wp.func_native(_NATIVE_CROSSINGS + r"""
    const int capacity = out_t.shape[1];
    const uint64_t offset = uint64_t(qi)*uint64_t(capacity);
    run_query(out_t.data+offset, out_face.data+offset, capacity,
              out_count.data+qi, out_status.data+qi);
""")
def native_all_hits(
    qi: int, origin: wp.vec3, direction: wp.vec3, min_t: float, max_t: float,
    exclude_face: int, node_count: int,
    node_ray_lower: wp.array(dtype=wp.vec4), node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32), mesh_vertices: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.vec3i), out_t: wp.array2d(dtype=wp.float32),
    out_face: wp.array2d(dtype=wp.int32), out_count: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
): ...


@wp.kernel
def count_hits_kernel(
    origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=wp.float32), max_t: wp.array(dtype=wp.float32),
    exclude_face: wp.array(dtype=wp.int32),
    node_ray_lower: wp.array(dtype=wp.vec4), node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32), mesh_vertices: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.vec3i), out_count: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    native_count_hits(i, origins[i], directions[i], min_t[i], max_t[i], exclude_face[i],
        wp.len(node_ray_lower), node_ray_lower, node_ray_upper, prim_ids,
        mesh_vertices, tri_indices, out_count, out_status)


@wp.kernel
def all_hits_kernel(
    origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=wp.float32), max_t: wp.array(dtype=wp.float32),
    exclude_face: wp.array(dtype=wp.int32),
    node_ray_lower: wp.array(dtype=wp.vec4), node_ray_upper: wp.array(dtype=wp.vec4),
    prim_ids: wp.array(dtype=wp.int32), mesh_vertices: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.vec3i), out_t: wp.array2d(dtype=wp.float32),
    out_face: wp.array2d(dtype=wp.int32), out_count: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    native_all_hits(i, origins[i], directions[i], min_t[i], max_t[i], exclude_face[i],
        wp.len(node_ray_lower), node_ray_lower, node_ray_upper, prim_ids,
        mesh_vertices, tri_indices, out_t, out_face, out_count, out_status)
