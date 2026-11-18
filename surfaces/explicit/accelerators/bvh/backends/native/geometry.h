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

using TriangleVector = wp::vec_t<3, double>;

CUDA_CALLABLE inline TriangleVector closest_triangle_edge(const TriangleVector &a,
                                                           const TriangleVector &b,
                                                           const TriangleVector &edge,
                                                           const TriangleVector &pa,
                                                           const TriangleVector &pb,
                                                           double length2) {
    const double from_a = wp::dot(pa, edge);
    const double from_b = -wp::dot(pb, edge);
    if (from_a <= 0.0)
        return a;
    if (from_b <= 0.0)
        return b;
    // Interpolate from the nearer endpoint to limit cancellation.
    return from_a < from_b ? a + (from_a / length2) * edge
                           : b - (from_b / length2) * edge;
}

CUDA_CALLABLE inline wp::vec3 closest_triangle_precise(const wp::vec3 &p32, const wp::vec3 &a32,
                                                       const wp::vec3 &b32, const wp::vec3 &c32) {
    const TriangleVector p(p32), a(a32), b(b32), c(c32);
    const TriangleVector ab = b - a, bc = c - b, ca = a - c;
    const TriangleVector pa = p - a, pb = p - b, pc = p - c;
    const double ab2 = wp::dot(ab, ab), bc2 = wp::dot(bc, bc), ca2 = wp::dot(ca, ca);
    // Include the shortest edge in the normal to avoid subtracting products
    // of nearly equal long edges when two vertices are close together.
    const TriangleVector n = ca2 < ab2 && ca2 < bc2 ? wp::cross(bc, ca) : wp::cross(ab, bc);
    const double n2 = wp::dot(n, n);
    if (n2 > 0.0 && wp::dot(wp::cross(ab, pa), n) >= 0.0 &&
        wp::dot(wp::cross(bc, pb), n) >= 0.0 && wp::dot(wp::cross(ca, pc), n) >= 0.0) {
        const double da2 = wp::dot(pa, pa), db2 = wp::dot(pb, pb), dc2 = wp::dot(pc, pc);
        const TriangleVector offset = da2 <= db2 && da2 <= dc2 ? pa : db2 <= dc2 ? pb : pc;
        const wp::vec3 projected(p - (wp::dot(offset, n) / n2) * n);
        // Projection can round past the AABB, including on a flat axis.
        const wp::vec3 lower = wp::min(a32, wp::min(b32, c32));
        const wp::vec3 upper = wp::max(a32, wp::max(b32, c32));
        return wp::min(wp::max(projected, lower), upper);
    }

    const TriangleVector q_ab = closest_triangle_edge(a, b, ab, pa, pb, ab2);
    const TriangleVector q_bc = closest_triangle_edge(b, c, bc, pb, pc, bc2);
    const TriangleVector q_ca = closest_triangle_edge(c, a, ca, pc, pa, ca2);
    const double d_ab = wp::dot(p - q_ab, p - q_ab);
    const double d_bc = wp::dot(p - q_bc, p - q_bc);
    const double d_ca = wp::dot(p - q_ca, p - q_ca);
    return wp::vec3(d_ab <= d_bc && d_ab <= d_ca ? q_ab : d_bc <= d_ca ? q_bc : q_ca);
}

CUDA_CALLABLE inline wp::vec3 closest_triangle(const wp::vec3 &p, const wp::vec3 &a,
                                               const wp::vec3 &b, const wp::vec3 &c) {
    if (p == a || p == b || p == c)
        return p;

    const wp::vec3 ab = b - a, ac = c - a, bc = c - b;
    const wp::vec3 n = wp::cross(ab, ac);
    const float edge2 = wp::max(wp::dot(ab, ab), wp::max(wp::dot(ac, ac), wp::dot(bc, bc)));
    const wp::vec3 view = p - a;
    const float view2 = wp::dot(view, view);
    // Use single precision only on well-conditioned, round-trip-safe edges.
    // The range checks also keep the barycentric products in float range.
    if (edge2 >= 1.e-16f && edge2 <= 1.e16f && view2 <= 1.e4f * edge2 &&
        wp::dot(n, n) > 1.e-2f * edge2 * edge2 &&
        a + ab == b && b - ab == a && a + ac == c && c - ac == a &&
        b + bc == c && c - bc == b) {
        const wp::vec2 uv = wp::closest_point_to_triangle(a, b, c, p);
        const float u = uv[0], v = uv[1], w = 1.0f - (u + v);
        constexpr float weight_slack = 2.e-6f;
        if ((u == 0.0f || u >= weight_slack) && (v == 0.0f || v >= weight_slack) &&
            (w == 0.0f || w >= weight_slack)) {
            // Anchor the point at the vertex with the greatest weight.
            if (u >= v && u >= w)
                return a + v * ab + w * ac;
            if (v >= w)
                return b - u * ab + w * bc;
            return c - u * ac - v * bc;
        }
    }
    return closest_triangle_precise(p, a, b, c);
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

} // namespace bvh
