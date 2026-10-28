#pragma once
#include "geometry.h"

namespace bvh {
CUDA_CALLABLE inline int triangle_event(const wp::vec3 &p, const wp::array_t<wp::vec3> &vertices,
                                        const wp::vec3i &ids) {

    const wp::vec3 a = vertices.data[ids[0]] - p;
    const wp::vec3 b = vertices.data[ids[1]] - p;
    const wp::vec3 c = vertices.data[ids[2]] - p;
    const wp::vec3 e_fast(b[1] * c[2] - b[2] * c[1], c[1] * a[2] - c[2] * a[1],
                          a[1] * b[2] - a[2] * b[1]);
    // Bounds include coordinate subtraction, products and summation.
    constexpr float rounding = 8.0f * 1.1920928955078125e-7f;
    const wp::vec3 error(
        wp::max(rounding * (wp::abs(b[1] * c[2]) + wp::abs(b[2] * c[1])), 1.e-37f),
        wp::max(rounding * (wp::abs(c[1] * a[2]) + wp::abs(c[2] * a[1])), 1.e-37f),
        wp::max(rounding * (wp::abs(a[1] * b[2]) + wp::abs(a[2] * b[1])), 1.e-37f));
    bool positive = false, negative = false, uncertain = false;
    for (int i = 0; i < 3; ++i) {
        positive |= e_fast[i] > error[i];
        negative |= e_fast[i] < -error[i];
        uncertain |= !(wp::abs(e_fast[i]) > error[i]);
    }
    if (positive && negative)
        return 0;
    if (!uncertain) {
        const wp::vec3 terms(e_fast[0] * a[0], e_fast[1] * b[0], e_fast[2] * c[0]);
        const float t = terms[0] + terms[1] + terms[2];
        const float t_error =
            rounding * (wp::abs(terms[0]) + wp::abs(terms[1]) + wp::abs(terms[2])) +
            error[0] * wp::abs(a[0]) + error[1] * wp::abs(b[0]) + error[2] * wp::abs(c[0]);
        if (wp::abs(t) > t_error)
            return (t > 0.0f) == positive ? 1 : 0;
    }
    double v[3][3];
    for (int i = 0; i < 3; ++i) {
        const wp::vec3 vertex = vertices.data[ids[i]];
        for (int j = 0; j < 3; ++j)
            v[i][j] = static_cast<double>(vertex[j]) - static_cast<double>(p[j]);
    }
    double e[3];
    for (int i = 0; i < 3; ++i)
        e[i] = cross2(v[(i + 1) % 3][1], v[(i + 1) % 3][2], v[(i + 2) % 3][1], v[(i + 2) % 3][2]);
    double det = e[0] + e[1] + e[2];
    const bool parallel = det == 0.0;
    int axis = 0;
    if (parallel) {
        // A parallel face can contain the query without crossing the ray.
        double normal[3];
        for (int j = 0; j < 3; ++j) {
            const int u = (j + 1) % 3, w = (j + 2) % 3;
            normal[j] =
                cross2(v[1][u] - v[0][u], v[1][w] - v[0][w], v[2][u] - v[0][u], v[2][w] - v[0][w]);
        }
        axis = wp::abs(normal[1]) > wp::abs(normal[2]) ? 1 : 2;
        const int u = (axis + 1) % 3, w = (axis + 2) % 3;
        for (int i = 0; i < 3; ++i)
            e[i] =
                cross2(v[(i + 1) % 3][u], v[(i + 1) % 3][w], v[(i + 2) % 3][u], v[(i + 2) % 3][w]);
        det = e[0] + e[1] + e[2];
        if (det == 0.0)
            return 0;
    }
    const double orientation = det > 0.0 ? 1.0 : -1.0;
    for (int i = 0; i < 3; ++i) {
        e[i] *= orientation;
        if (e[i] < 0.0)
            return 0;
    }
    const double t = e[0] * v[0][axis] + e[1] * v[1][axis] + e[2] * v[2][axis];
    if (t == 0.0)
        return 2;
    if (parallel || t < 0.0)
        return 0;
    // Top-left ownership counts a shared edge or vertex exactly once.
    for (int i = 0; i < 3; ++i) {
        if (e[i] == 0.0) {
            const int a = (i + 1) % 3, b = (i + 2) % 3;
            if (!owns_edge(v[a][1], v[a][2], v[b][1], v[b][2], orientation))
                return 0;
        }
    }
    return 1;
}

template <class Ray>
CUDA_CALLABLE inline wp::vec2 triangle_crossing(const Ray &ray, float min_t, float max_t, int slot,
                                                const wp::array_t<wp::vec3> &mesh_vertices,
                                                const wp::array_t<wp::vec3i> &tri_indices,
                                                bool half_open, bool store_times) {
    const wp::vec3 o = ray.origin, d = ray.direction;
    const float lo_t = min_t, hi_t = max_t;

    const int kx = ray.axes[0], ky = ray.axes[1], kz = ray.axes[2];
    const auto triangle_t = [&](int slot, float &result) {
        const wp::vec3i ids = tri_indices.data[slot];
        // Reuse one predicate at both precisions. -1 asks for the double path.
        const auto solve = [&](auto zero) {
            using S = decltype(zero);
            S x[3], y[3], z[3], uncertainty[3];
            const S sx = sizeof(S) == sizeof(float) ? static_cast<S>(ray.shear[0])
                                                    : static_cast<S>(ray.frame_direction[0]) /
                                                          static_cast<S>(ray.frame_direction[2]);
            const S sy = sizeof(S) == sizeof(float) ? static_cast<S>(ray.shear[1])
                                                    : static_cast<S>(ray.frame_direction[1]) /
                                                          static_cast<S>(ray.frame_direction[2]);
            constexpr float rounding = 8.0f * 1.1920928955078125e-7f;
            for (int i = 0; i < 3; ++i) {
                const wp::vec3 v = mesh_vertices.data[ids[i]];
                const S a = static_cast<S>(v[kx]) - static_cast<S>(ray.frame_origin[0]);
                const S b = static_cast<S>(v[ky]) - static_cast<S>(ray.frame_origin[1]);
                z[i] = static_cast<S>(v[kz]) - static_cast<S>(ray.frame_origin[2]);
                x[i] = a - sx * z[i];
                y[i] = b - sy * z[i];
                uncertainty[i] = rounding * wp::max(wp::abs(a) + wp::abs(sx * z[i]),
                                                    wp::abs(b) + wp::abs(sy * z[i])) +
                                 static_cast<S>(1.e-37f);
            }
            S e[3], error[3];
            bool positive = false, negative = false, uncertain = false;
            for (int i = 0; i < 3; ++i) {
                const int a = (i + 1) % 3, b = (i + 2) % 3;
                if constexpr (sizeof(S) == sizeof(float)) {
                    e[i] = x[a] * y[b] - y[a] * x[b];
                    error[i] =
                        rounding * (wp::abs(x[a] * y[b]) + wp::abs(y[a] * x[b])) +
                        uncertainty[a] * (wp::abs(x[b]) + wp::abs(y[b]) + 2 * uncertainty[b]) +
                        uncertainty[b] * (wp::abs(x[a]) + wp::abs(y[a]));
                } else {
                    e[i] = cross2(x[a], y[a], x[b], y[b]);
                    error[i] = 0;
                }
                positive |= e[i] > error[i];
                negative |= e[i] < -error[i];
                uncertain |= !(wp::abs(e[i]) > error[i]);
            }
            if (positive && negative)
                return 0;
            if constexpr (sizeof(S) == sizeof(float)) {
                if (uncertain)
                    return -1;
            }
            const S det = e[0] + e[1] + e[2];
            if (det == 0)
                return 0; // Parallel/coplanar faces have no isolated crossing.
            const S numerator = e[0] * z[0] + e[1] * z[1] + e[2] * z[2];
            const S denominator = det * static_cast<S>(ray.frame_direction[2]);
            S t = numerator / denominator;
            if constexpr (sizeof(S) == sizeof(float)) {
                const S num_error =
                    rounding *
                        (wp::abs(e[0] * z[0]) + wp::abs(e[1] * z[1]) + wp::abs(e[2] * z[2])) +
                    error[0] * wp::abs(z[0]) + error[1] * wp::abs(z[1]) + error[2] * wp::abs(z[2]);
                const S den_error =
                    wp::abs(ray.frame_direction[2]) * (error[0] + error[1] + error[2]) +
                    rounding * wp::abs(denominator);
                if (!(wp::abs(denominator) > den_error) || !wp::isfinite(t))
                    return -1;
                const S t_error =
                    (num_error + wp::abs(t) * den_error) / (wp::abs(denominator) - den_error);
                if (!wp::isfinite(t_error))
                    return -1;
                if (t + t_error < lo_t || t - t_error > hi_t)
                    return 0;
                if (t - t_error < lo_t || t + t_error > hi_t)
                    return -1;
            } else {
                // Recompute from the original plane at ambiguous interval endpoints.
                const wp::vec3 a = mesh_vertices.data[ids[0]];
                const wp::vec3 b = mesh_vertices.data[ids[1]];
                const wp::vec3 c = mesh_vertices.data[ids[2]];
                S ab[3], ac[3], normal[3];
                for (int i = 0; i < 3; ++i) {
                    ab[i] = static_cast<S>(b[i]) - static_cast<S>(a[i]);
                    ac[i] = static_cast<S>(c[i]) - static_cast<S>(a[i]);
                }
                S plane = 0, velocity = 0;
                for (int i = 0; i < 3; ++i) {
                    const int u = (i + 1) % 3, v = (i + 2) % 3;
                    normal[i] = cross2(ab[u], ab[v], ac[u], ac[v]);
                    plane += normal[i] * (static_cast<S>(a[i]) - static_cast<S>(o[i]));
                    velocity += normal[i] * static_cast<S>(d[i]);
                }
                if (velocity == 0)
                    return 0;
                t = plane / velocity;
                if (t < lo_t || t > hi_t)
                    return 0;
                const S orientation = det > 0 ? 1 : -1;
                for (int i = 0; i < 3; ++i) {
                    if (half_open && e[i] == 0) {
                        const int a = (i + 1) % 3, b = (i + 2) % 3;
                        if (!owns_edge(x[a], y[a], x[b], y[b], orientation))
                            return 0;
                    }
                }
            }
            result = static_cast<float>(t);
            return store_times && !wp::isfinite(result) ? -2 : 1;
        };
        const int fast = solve(0.0f);
        return fast == -1 ? solve(0.0) : fast;
    };

    float t = 0;
    const int hit = triangle_t(slot, t);
    return wp::vec2(t, hit < 0 ? -1.f : static_cast<float>(hit));
}

template <unsigned Dimension, class Meta>
CUDA_CALLABLE inline int classification_event(const wp::vec_t<Dimension, float> &point,
                                              const Meta &meta, int slot) {
    if constexpr (Dimension == 2) {

        const wp::vec3 a = meta.vertices.data[meta.indices.data[2 * slot]];
        const wp::vec3 b = meta.vertices.data[meta.indices.data[2 * slot + 1]];
        const double ax = double(a[0]) - double(point[0]), ay = double(a[1]) - double(point[1]);
        const double bx = double(b[0]) - double(point[0]), by = double(b[1]) - double(point[1]);
        const double cross = cross2(ax, ay, bx, by);
        if (cross == 0.0 && wp::min(ax, bx) <= 0 && wp::max(ax, bx) >= 0 && wp::min(ay, by) <= 0 &&
            wp::max(ay, by) >= 0)
            return 2;
        if ((ay <= 0 && by > 0) || (by <= 0 && ay > 0))
            return cross / (by - ay) > 0 ? 1 : 0;
        return 0;

    } else {
        const wp::vec3i ids(meta.indices.data[3 * slot], meta.indices.data[3 * slot + 1],
                            meta.indices.data[3 * slot + 2]);
        return triangle_event(point, meta.vertices, ids);
    }
}

template <class Vec>
CUDA_CALLABLE inline bool valid_ray(const Vec &origin, const Vec &direction, float minimum,
                                    float maximum, int excluded, int count, bool line,
                                    bool exclusion) {

    for (float v : origin.c)
        if (!(v == v && wp::abs(v) < 3.4028234e38f))
            return false;
    bool nonzero = false;
    for (float v : direction.c) {
        if (!(v == v && wp::abs(v) < 3.4028234e38f))
            return false;
        nonzero |= v != 0.f;
    }
    if (!(minimum <= maximum) || minimum == INFINITY || maximum == -INFINITY)
        return false;
    if (exclusion && (excluded < -1 || excluded >= count))
        return false;
    return line ? nonzero : minimum >= 0.f && wp::dot(direction, direction) > 1.e-30f;
}

template <class Ray>
CUDA_CALLABLE inline wp::vec2 ray_box(const Ray &ray, const wp::vec3 &lower, const wp::vec3 &upper,
                                      float minimum, float maximum, int dimension) {
    float enter;
    const bool hit = box_interval(ray.origin, ray.direction, ray.inverse, dimension, ray.fast,
                                  lower, upper, minimum, maximum, enter);
    return wp::vec2(enter, hit ? 1.f : 0.f);
}

} // namespace bvh
