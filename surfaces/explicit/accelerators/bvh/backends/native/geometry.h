#pragma once
#include "builtin.h"

namespace bvh {

CUDA_CALLABLE inline float box_distance_sq(const wp::vec3 &p, const wp::vec3 &lower,
                                           const wp::vec3 &upper) {
    const float dx = wp::max(wp::max(lower[0] - p[0], 0.0f), p[0] - upper[0]);
    const float dy = wp::max(wp::max(lower[1] - p[1], 0.0f), p[1] - upper[1]);
    const float dz = wp::max(wp::max(lower[2] - p[2], 0.0f), p[2] - upper[2]);
    return dx * dx + dy * dy + dz * dz;
}

template <typename Vec>
CUDA_CALLABLE inline Vec closest_segment(const Vec &p, const Vec &a, const Vec &b) {
    const auto ab = b - a;
    const float denom = wp::dot(ab, ab);
    float t = 0.0f;
    if (denom > 0.0f)
        t = wp::clamp(wp::dot(p - a, ab) / denom, 0.0f, 1.0f);
    return a + t * ab;
}

CUDA_CALLABLE inline wp::vec3 closest_triangle(const wp::vec3 &p, const wp::vec3 &a,
                                               const wp::vec3 &e1, const wp::vec3 &e2) {
    const wp::vec3 b = a + e1;
    const wp::vec3 c = a + e2;
    const wp::vec3 n = wp::cross(e1, e2);
    const float n2 = wp::dot(n, n);
    const wp::vec3 e3 = e2 - e1;
    const float edge_scale = wp::max(wp::dot(e3, e3), wp::max(wp::dot(e1, e1), wp::dot(e2, e2)));
    // Keep thin and degenerate triangles on the robust path.
    if (n2 > 1.e-2f * edge_scale * edge_scale) {
        const wp::vec2 uv = wp::closest_point_to_triangle(a, b, c, p);
        return a + uv[1] * e1 + (1.0f - uv[0] - uv[1]) * e2;
    }
    if (n2 > 0.0f) {
        const wp::vec3 projected = p - (wp::dot(p - a, n) / n2) * n;
        const wp::vec3 v = projected - a;
        const float u = wp::dot(wp::cross(v, e2), n) / n2;
        const float w = wp::dot(wp::cross(e1, v), n) / n2;
        if (u >= 0.0f && w >= 0.0f && u + w <= 1.0f)
            return projected;
    }
    wp::vec3 best = closest_segment(p, a, b);
    float best_d2 = wp::dot(p - best, p - best);
    wp::vec3 candidate = closest_segment(p, b, c);
    float candidate_d2 = wp::dot(p - candidate, p - candidate);
    if (candidate_d2 < best_d2) {
        best = candidate;
        best_d2 = candidate_d2;
    }
    candidate = closest_segment(p, c, a);
    candidate_d2 = wp::dot(p - candidate, p - candidate);
    if (candidate_d2 < best_d2)
        best = candidate;
    return best;
}

template <typename Vec>
CUDA_CALLABLE inline bool
box_interval(const Vec &o, const Vec &d, const Vec &inv_dir, int dimension, bool fast_aabb,
             const wp::vec3 &lower, const wp::vec3 &upper, float tmin, float tmax, float &enter) {
    enter = tmin;
    float leave = tmax;
    const auto update = [&](float t0, float t1) {
        float near_t = wp::min(t0, t1), far_t = wp::max(t0, t1);
        enter = wp::max(enter, near_t);
        leave = wp::min(leave, far_t);
    };
    if (fast_aabb) {
        for (int axis = 0; axis < dimension; ++axis) {
            const float t0 = (lower[axis] - o[axis]) * inv_dir[axis];
            const float t1 = (upper[axis] - o[axis]) * inv_dir[axis];
            update(t0, t1);
        }
    } else {
        for (int axis = 0; axis < dimension; ++axis) {
            const float component = d[axis];
            const float origin_component = o[axis];
            if (component == 0.0f) {
                if (origin_component < lower[axis] || origin_component > upper[axis])
                    return false;
            } else {
                const float t0 = (lower[axis] - origin_component) / component;
                const float t1 = (upper[axis] - origin_component) / component;
                update(t0, t1);
            }
        }
    }
    // Expand the reduced interval; narrow slabs can invert through rounding.
    const float padding = 9.5367431640625e-7f * wp::max(wp::abs(enter), wp::abs(leave)) + 1.e-37f;
    if (wp::isfinite(padding)) {
        enter -= padding;
        return enter <= leave + padding;
    }
    enter = tmin;
    return true;
}

CUDA_CALLABLE inline double cross2(double ax, double ay, double bx, double by) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(__dmul_rn(ax, by), -__dmul_rn(ay, bx));
#else
    volatile double a = ax * by;
    volatile double b = ay * bx;
    return a - b;
#endif
}

CUDA_CALLABLE inline bool owns_edge(double ax, double ay, double bx, double by,
                                    double orientation) {
    const double dx = orientation * (bx - ax), dy = orientation * (by - ay);
    return dy > 0 || (dy == 0 && dx < 0);
}

template <typename Lower, typename Upper, typename Angles, typename Axes, typename Radii>
CUDA_CALLABLE inline bool
cone_may_have_silhouette(const wp::vec3 &p, int node, const Lower &node_lower,
                         const Upper &node_upper, const Angles &node_cone_sin_cos,
                         const Axes &node_cone_axis, const Radii &node_cone_radius) {
    const wp::vec2 angles = node_cone_sin_cos.data[node];
    const float sin_alpha = angles[0];
    const float cos_alpha = angles[1];
    if (sin_alpha < 0.0f)
        return false;
    if (cos_alpha <= 0.0f)
        return true;
    const wp::vec3 lower = node_lower.data[node];
    const wp::vec3 upper = node_upper.data[node];
    const wp::vec3 view = 0.5f * (lower + upper) - p;
    const wp::vec3 extent = 0.5f * (upper - lower);
    const wp::vec3 axis = node_cone_axis.data[node];
    const float projection = wp::dot(axis, view);
    // Box support along the cone axis bounds every edge endpoint.
    const float support =
        wp::abs(axis[0]) * extent[0] + wp::abs(axis[1]) * extent[1] + wp::abs(axis[2]) * extent[2];
    const float gap = wp::abs(projection) - support;
    if (gap <= 0.0f)
        return true;
    const wp::vec3 perpendicular = view - projection * axis;
    const float radius = node_cone_radius.data[node];
    const float limit = sin_alpha * (wp::sqrt(wp::dot(perpendicular, perpendicular)) + radius);
    const float slack = 2.0e-6f * (wp::sqrt(wp::dot(view, view)) + radius);
    return gap * cos_alpha <= limit + slack;
}

CUDA_CALLABLE inline bool triangle_t(const wp::vec3 &o, const wp::vec3 &d, const wp::vec3 &a,
                                     const wp::vec3 &e1, const wp::vec3 &e2, float tmin, float tmax,
                                     float &result) {
    const wp::vec3 pvec = wp::cross(d, e2);
    const float det = wp::dot(e1, pvec);
    const float scale = wp::sqrt(wp::dot(e1, e1) * wp::dot(e2, e2) * wp::dot(d, d));
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
        return u >= -slack && v >= -slack && u + v <= 1.0f + slack && t >= tmin && t <= tmax;
    }
    return false;
}

template <typename Ray, typename Data>
CUDA_CALLABLE inline wp::vec2 segment_intersection(const Ray &ray, float minimum, float maximum,
                                                   const Data &data, int slot, bool half_open,
                                                   bool store_times) {
    const auto origin = ray.origin, direction = ray.direction;
    const wp::vec2 a = data.a.data[slot], b = data.b.data[slot];
    // Filter endpoint signs; uncertain signs retain the double predicate.
    // Four rounded operations per term are bounded by 16u times the sum
    // of product magnitudes; padding also covers gradual underflow/FTZ.
    const float fax = a[0] - origin[0], fay = a[1] - origin[1];
    const float fbx = b[0] - origin[0], fby = b[1] - origin[1];
    const float ap = fax * direction[1], aq = fay * direction[0];
    const float bp = fbx * direction[1], bq = fby * direction[0];
    const float fa = ap - aq, fb = bp - bq;
    const float padding =
        4.70197740328915e-38f * (wp::abs(direction[0]) + wp::abs(direction[1]) + wp::abs(fax) +
                                 wp::abs(fay) + wp::abs(fbx) + wp::abs(fby) + 1.f);
    const float ea = 9.5367431640625e-7f * (wp::abs(ap) + wp::abs(aq)) + padding;
    const float eb = 9.5367431640625e-7f * (wp::abs(bp) + wp::abs(bq)) + padding;
    const float shared_error = ea + eb;
    if ((fa > shared_error && fb > shared_error) || (fa < -shared_error && fb < -shared_error))
        return wp::vec2(0.f, 0.f);
    const bool interior =
        (fa > shared_error && fb < -shared_error) || (fa < -shared_error && fb > shared_error);
    // Certify the original interval expressions; uncertain signs,
    // bounds and overflow retain the double predicate.
    bool certified = false;
    if (interior) {
        const float ex = b[0] - a[0], ey = b[1] - a[1];
        const float dp = direction[0] * ey, dq = direction[1] * ex;
        const float np = fax * ey, nq = fay * ex;
        const float sign = fa < 0.f ? -1.f : 1.f;
        const float scale = sign * (dp - dq), numerator = sign * (np - nq);
        const float ed = 9.5367431640625e-7f * (wp::abs(dp) + wp::abs(dq)) +
                         4.70197740328915e-38f * (wp::abs(direction[0]) + wp::abs(direction[1]) +
                                                  wp::abs(ex) + wp::abs(ey) + 1.f);
        const float en =
            9.5367431640625e-7f * (wp::abs(np) + wp::abs(nq)) +
            4.70197740328915e-38f * (wp::abs(fax) + wp::abs(fay) + wp::abs(ex) + wp::abs(ey) + 1.f);
        if (scale > ed) {
            const float low = minimum * scale, high = maximum * scale;
            const float el = en + wp::abs(minimum) * ed +
                             2.384185791015625e-7f * (wp::abs(numerator) + wp::abs(low)) +
                             4.70197740328915e-38f * (wp::abs(minimum) + scale + 1.f);
            const float eh = en + wp::abs(maximum) * ed +
                             2.384185791015625e-7f * (wp::abs(numerator) + wp::abs(high)) +
                             4.70197740328915e-38f * (wp::abs(maximum) + scale + 1.f);
            certified = (minimum == -INFINITY || numerator - low > el) &&
                        (maximum == INFINITY || high - numerator > eh);
            if (certified && !store_times)
                return wp::vec2(0.f, 1.f);
        }
    }
    if (certified && store_times) {
        const bool vertical = a[0] == b[0], horizontal = a[1] == b[1];
        if (vertical || horizontal) {
            const double edge = double(vertical ? b[1] : b[0]) - double(vertical ? a[1] : a[0]);
            // Only exact FP64 power-of-two factors cancel without rounding.
            uint64_t bits;
            memcpy(&bits, &edge, 8);
            if ((bits & 0x7fffffffffffffffull) != 0 && (bits & 0x000fffffffffffffull) == 0) {
                const double offset =
                    double(vertical ? a[0] : a[1]) - double(vertical ? origin[0] : origin[1]);
                const float component = vertical ? direction[0] : direction[1];
                const float offset32 = static_cast<float>(offset);
                const float result =
                    double(offset32) == offset && wp::abs(component) >= 1.1754943508222875e-38f
                        ? offset32 / component
                        : static_cast<float>(offset / double(component));
                if (wp::isfinite(result) && wp::abs(result) >= 1.1754943508222875e-38f)
                    return wp::vec2(result, 1.f);
            }
        }
    }
    const double ex = double(b[0]) - double(a[0]), ey = double(b[1]) - double(a[1]);
    const double dx = direction[0], dy = direction[1];
    const double ax = double(a[0]) - double(origin[0]), ay = double(a[1]) - double(origin[1]);
    const double denominator = dx * ey - dy * ex;
    double sign = 1.0, scale = denominator;
    if (!certified) {
        if (denominator == 0)
            return wp::vec2(0.f, 0.f);
        sign = denominator < 0 ? -1.0 : 1.0;
        scale = sign * denominator;
    }
    // Strict opposite signs certify interior and half-open ownership.
    if (!interior) {
        const double sa = ax * dy - ay * dx, u = sign * sa;
        if (u < 0 || u > scale)
            return wp::vec2(0.f, 0.f);
        if (half_open) {
            const double sb =
                (double(b[0]) - double(origin[0])) * dy - (double(b[1]) - double(origin[1])) * dx;
            if (!((sa <= 0 && sb > 0) || (sb <= 0 && sa > 0)))
                return wp::vec2(0.f, 0.f);
        }
    }
    double numerator = ax * ey - ay * ex;
    if (!certified) {
        numerator = sign * numerator;
        if (numerator < double(minimum) * scale || numerator > double(maximum) * scale)
            return wp::vec2(0.f, 0.f);
    }
    const float result = store_times ? static_cast<float>(numerator / scale) : 0.f;
    return wp::vec2(result, store_times && !wp::isfinite(result) ? -1.f : 1.f);
}

template <typename Ray, typename Data>
CUDA_CALLABLE inline wp::vec2 first_triangle_intersection(const Ray &ray, float minimum,
                                                          float maximum, const Data &data,
                                                          int slot) {
    float t;
    const bool hit = triangle_t(ray.origin, ray.direction, data.a.data[slot], data.e1.data[slot],
                                data.e2.data[slot], minimum, maximum, t);
    return wp::vec2(t, hit ? 1.f : 0.f);
}

} // namespace bvh
