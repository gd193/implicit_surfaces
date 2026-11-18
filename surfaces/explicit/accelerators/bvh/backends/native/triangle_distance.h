#pragma once
#include "geometry.h"

namespace bvh {

constexpr int triangle_frame_words = 16;
constexpr int triangle_frame_permutation = 14;

CUDA_CALLABLE inline double triangle_plane_dot(const TriangleVector &n, const TriangleVector &p) {
    // Recover low product and addition terms when the plane equation cancels.
#if defined(__CUDA_ARCH__)
    const double x = __dmul_rn(n[0], p[0]), y = __dmul_rn(n[1], p[1]), z = __dmul_rn(n[2], p[2]);
#else
    const volatile double x = n[0] * p[0], y = n[1] * p[1], z = n[2] * p[2];
#endif
    const double xy = x + y, sum = xy + z;
    const double y_hi = xy - x, z_hi = sum - xy;
    const double add_lo = ((x - (xy - y_hi)) + (y - y_hi)) +
                          ((xy - (sum - z_hi)) + (z - z_hi));
    const double mul_lo = fma(n[0], p[0], -x) + fma(n[1], p[1], -y) + fma(n[2], p[2], -z);
    return sum + (add_lo + mul_lo);
}

template <bool ReturnPoint>
CUDA_CALLABLE inline auto triangle_edge_result(const TriangleVector &a, const TriangleVector &b,
                                               const TriangleVector &edge,
                                               const TriangleVector &pa, const TriangleVector &pb,
                                               double from_a, double from_b) {
    if constexpr (ReturnPoint) {
        if (from_a <= 0.)
            return wp::vec3(a);
        if (from_b <= 0.)
            return wp::vec3(b);
        const double length2 = wp::dot(edge, edge);
        return wp::vec3(from_a < from_b ? a + (from_a / length2) * edge
                                        : b - (from_b / length2) * edge);
    } else {
        if (from_a <= 0.)
            return wp::dot(pa, pa);
        if (from_b <= 0.)
            return wp::dot(pb, pb);
        const TriangleVector p = from_a < from_b ? pa : pb;
        const TriangleVector perpendicular(cross2(edge[1], edge[2], p[1], p[2]),
                                            cross2(edge[2], edge[0], p[2], p[0]),
                                            cross2(edge[0], edge[1], p[0], p[1]));
        return wp::dot(perpendicular, perpendicular) / wp::dot(edge, edge);
    }
}

CUDA_CALLABLE inline TriangleVector triangle_cross(const TriangleVector &a, const TriangleVector &b) {
    return TriangleVector(cross2(a[1], a[2], b[1], b[2]),
                          cross2(a[2], a[0], b[2], b[0]),
                          cross2(a[0], a[1], b[0], b[1]));
}

template <bool ReturnPoint>
#if defined(__CUDA_ARCH__)
__device__ __noinline__
#else
inline
#endif
auto triangle_precise_result(const wp::vec3 &query, const wp::vec3 &a32,
                             const wp::vec3 &b32, const wp::vec3 &c32) {
    if (query == a32 || query == b32 || query == c32) {
        if constexpr (ReturnPoint) return query;
        else return 0.;
    }
    const TriangleVector p(query), a(a32), b(b32), c(c32);
    const TriangleVector ab = b - a, bc = c - b, ca = a - c;
    const TriangleVector pa = p - a, pb = p - b, pc = p - c;
    const double ab2 = wp::dot(ab, ab), bc2 = wp::dot(bc, bc), ca2 = wp::dot(ca, ca);
    const TriangleVector n = ca2 < ab2 && ca2 < bc2 ? triangle_cross(bc, ca) : triangle_cross(ab, bc);
    const double n2 = wp::dot(n, n);
    if (n2 == 0.) {
        if (ab2 >= bc2 && ab2 >= ca2)
            return triangle_edge_result<ReturnPoint>(a, b, ab, pa, pb, wp::dot(pa, ab), -wp::dot(pb, ab));
        if (bc2 >= ca2)
            return triangle_edge_result<ReturnPoint>(b, c, bc, pb, pc, wp::dot(pb, bc), -wp::dot(pc, bc));
        return triangle_edge_result<ReturnPoint>(c, a, ca, pc, pa, wp::dot(pc, ca), -wp::dot(pa, ca));
    }
    const bool outside_ab = wp::dot(triangle_cross(ab, pa), n) < 0.;
    const bool outside_bc = wp::dot(triangle_cross(bc, pb), n) < 0.;
    const bool outside_ca = wp::dot(triangle_cross(ca, pc), n) < 0.;
    if (!(outside_ab || outside_bc || outside_ca)) {
        const double da2 = wp::dot(pa, pa), db2 = wp::dot(pb, pb), dc2 = wp::dot(pc, pc);
        const TriangleVector offset = da2 <= db2 && da2 <= dc2 ? pa : db2 <= dc2 ? pb : pc;
        const double height = triangle_plane_dot(n, offset);
        if constexpr (ReturnPoint) {
            const wp::vec3 projected(p - (height / n2) * n);
            return wp::min(wp::max(projected, wp::min(a32, wp::min(b32, c32))),
                           wp::max(a32, wp::max(b32, c32)));
        } else {
            return height * height / n2;
        }
    }
    if (outside_ab) {
        const double from_a = wp::dot(pa, ab), from_b = -wp::dot(pb, ab);
        if (from_a > 0. && from_b > 0.)
            return triangle_edge_result<ReturnPoint>(a, b, ab, pa, pb, from_a, from_b);
    }
    if (outside_bc) {
        const double from_b = wp::dot(pb, bc), from_c = -wp::dot(pc, bc);
        if (from_b > 0. && from_c > 0.)
            return triangle_edge_result<ReturnPoint>(b, c, bc, pb, pc, from_b, from_c);
    }
    if (outside_ca) {
        const double from_c = wp::dot(pc, ca), from_a = -wp::dot(pa, ca);
        if (from_c > 0. && from_a > 0.)
            return triangle_edge_result<ReturnPoint>(c, a, ca, pc, pa, from_c, from_a);
    }
    if (wp::dot(pa + pb, ab) <= 0.) {
        if (wp::dot(pc + pa, ca) >= 0.) {
            if constexpr (ReturnPoint) return a32;
            else return wp::dot(pa, pa);
        }
    } else if (wp::dot(pb + pc, bc) <= 0.) {
        if constexpr (ReturnPoint) return b32;
        else return wp::dot(pb, pb);
    }
    if constexpr (ReturnPoint) return c32;
    else return wp::dot(pc, pc);
}

CUDA_CALLABLE inline float triangle_frame_scalar(const uint32_t *words, int column) {
    float value;
    memcpy(&value, words + column, sizeof(value));
    return value;
}

CUDA_CALLABLE inline double triangle_frame_double(const uint32_t *words, int column) {
    double value;
    memcpy(&value, words + 2 * column, sizeof(value));
    return value;
}

struct TriangleFloatPair {
    float hi, lo;
};

CUDA_CALLABLE inline TriangleFloatPair triangle_float_sum(float a, float b) {
    const float sum = a + b, b_hi = sum - a;
    return {sum, (a - (sum - b_hi)) + (b - b_hi)};
}

CUDA_CALLABLE inline TriangleFloatPair triangle_float_cross(const TriangleFloatPair &x,
                                                             const TriangleFloatPair &y,
                                                             const TriangleFloatPair &u,
                                                             const TriangleFloatPair &v) {
    const float a = x.hi * v.hi, b = y.hi * u.hi;
    const auto difference = triangle_float_sum(a, -b);
    const float residual = (fmaf(x.hi, v.hi, -a) - fmaf(y.hi, u.hi, -b))
                           + (x.hi * v.lo - y.hi * u.lo)
                           + (x.lo * v.hi - y.lo * u.hi);
    return triangle_float_sum(difference.hi, difference.lo + residual);
}

CUDA_CALLABLE inline TriangleFloatPair triangle_float_norm(const TriangleFloatPair &x,
                                                            const TriangleFloatPair &y,
                                                            const TriangleFloatPair &z) {
    const float xx = x.hi * x.hi, yy = y.hi * y.hi, zz = z.hi * z.hi;
    const auto xy = triangle_float_sum(xx, yy), xyz = triangle_float_sum(xy.hi, zz);
    const float error = fmaf(x.hi, x.hi, -xx) + fmaf(y.hi, y.hi, -yy) + fmaf(z.hi, z.hi, -zz)
                        + 2.f * (x.hi * x.lo + y.hi * y.lo + z.hi * z.lo);
    return {xyz.hi, xyz.lo + xy.lo + error};
}

CUDA_CALLABLE inline double triangle_frame_edge_score(const wp::vec3 &a, const wp::vec3 &b,
                                                       const wp::vec3 &p, bool from_a) {
    const wp::vec3 u = from_a ? a : b;
    const auto ex = triangle_float_sum(b[0], -a[0]), ey = triangle_float_sum(b[1], -a[1]),
               ez = triangle_float_sum(b[2], -a[2]);
    const auto px = triangle_float_sum(p[0], -u[0]), py = triangle_float_sum(p[1], -u[1]),
               pz = triangle_float_sum(p[2], -u[2]);
    const auto cx = triangle_float_cross(ey, ez, py, pz);
    const auto cy = triangle_float_cross(ez, ex, pz, px);
    const auto cz = triangle_float_cross(ex, ey, px, py);
    const auto num = triangle_float_norm(cx, cy, cz), denom = triangle_float_norm(ex, ey, ez);
    const float quotient = num.hi / denom.hi;
    const float residual = fmaf(-quotient, denom.hi, num.hi) + fmaf(-quotient, denom.lo, num.lo);
    // Correct the quotient before rounding to the float32 score type.
    return double(quotient + residual / denom.hi);
}

CUDA_CALLABLE inline uint32_t triangle_vertex_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

CUDA_CALLABLE inline float triangle_vertex_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// Each float is a signed 24-bit significand times 2^(shift - 149).
// Accumulate the nine terms of sum(p² + v² - 2pv) in units of 2^-298.
// 576 bits cover even the sum of three squared differences of opposite
// maximum finite coordinates. Unsigned carries and borrows are modulo 2^576.
CUDA_CALLABLE inline void triangle_vertex_accumulate(uint64_t *sum, uint64_t term,
                                                     int shift, bool subtract) {
    int word = shift / 64;
    const int offset = shift % 64;
    uint64_t low = term << offset;
    uint64_t high = offset ? term >> (64 - offset) : 0;
    while ((low || high) && word < 9) {
        const uint64_t old = sum[word];
        const uint64_t next = subtract ? old - low : old + low;
        const bool carry = subtract ? old < low : next < old;
        sum[word++] = next;
        low = high + carry;
        high = 0;
    }
}

#if defined(__CUDA_ARCH__)
__device__ __noinline__
#else
inline
#endif
double triangle_frame_vertex_exact(const wp::vec3 &p, const wp::vec3 &v) {
    uint64_t sum[9] = {};
    for (int i = 0; i < 3; ++i) {
        const uint32_t p_bits = triangle_vertex_bits(p[i]), v_bits = triangle_vertex_bits(v[i]);
        const int p_exp = (p_bits >> 23) & 255, v_exp = (v_bits >> 23) & 255;
        if (p_exp == 255 || v_exp == 255) {
            const TriangleVector delta = TriangleVector(p) - TriangleVector(v);
            return wp::dot(delta, delta);
        }
        const uint64_t pi = (p_bits & 0x7fffff) | (p_exp ? 0x800000 : 0);
        const uint64_t vi = (v_bits & 0x7fffff) | (v_exp ? 0x800000 : 0);
        const int ps = p_exp ? p_exp - 1 : 0, vs = v_exp ? v_exp - 1 : 0;
        triangle_vertex_accumulate(sum, pi * pi, 2 * ps, false);
        triangle_vertex_accumulate(sum, vi * vi, 2 * vs, false);
        triangle_vertex_accumulate(sum, pi * vi, ps + vs + 1, (p_bits ^ v_bits) < 0x80000000u);
    }
    int top = 8;
    while (top >= 0 && !sum[top])
        --top;
    // Half the smallest subnormal is bit 148 in this fixed-point representation.
    if (top < 2)
        return 0.;
#if defined(__CUDA_ARCH__)
    const int highest = 64 * top + 63 - __clzll(sum[top]);
#else
    const int highest = 64 * top + 63 - __builtin_clzll(sum[top]);
#endif
    const int shift = highest > 172 ? highest - 23 : 149;
    const int word = shift / 64, offset = shift % 64;
    uint64_t rounded = sum[word] >> offset;
    if (offset && word < 8)
        rounded |= sum[word + 1] << (64 - offset);
    const int round_word = (shift - 1) / 64;
    const uint64_t round_bit = uint64_t(1) << ((shift - 1) % 64);
    bool sticky = (sum[round_word] & (round_bit - 1)) != 0;
    for (int i = 0; i < round_word; ++i)
        sticky |= sum[i] != 0;
    rounded += (sum[round_word] & round_bit) && (sticky || (rounded & 1));
    if (highest > 425 || (highest == 425 && rounded == 0x1000000))
        return double(triangle_vertex_float(0x7f800000u));
    // Construct the rounded result in double, including float subnormals,
    // without a float operation that could flush a subnormal result to zero.
    return ldexp(double(rounded), shift - 298);
}

CUDA_CALLABLE inline float triangle_vertex_product(float value) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(value, value);
#else
    const volatile float square = value * value;
    return square;
#endif
}

CUDA_CALLABLE inline double triangle_frame_vertex_score(const wp::vec3 &p, const wp::vec3 &v) {
    const auto x = triangle_float_sum(p[0], -v[0]), y = triangle_float_sum(p[1], -v[1]),
               z = triangle_float_sum(p[2], -v[2]);
    const float xx = triangle_vertex_product(x.hi), yy = triangle_vertex_product(y.hi),
                zz = triangle_vertex_product(z.hi);
    const auto xy = triangle_float_sum(xx, yy), xyz = triangle_float_sum(xy.hi, zz);
    if (xyz.hi > 0x1p-60f && xyz.hi < 0x1p+60f) {
        const float product_error = fmaf(x.hi, x.hi, -xx) + fmaf(y.hi, y.hi, -yy)
                                  + fmaf(z.hi, z.hi, -zz);
        const float cross_error = 2.f * (x.hi * x.lo + y.hi * y.lo + z.hi * z.lo);
        const float low = ((xyz.lo + xy.lo) + product_error) + cross_error;
        const float rounded = xyz.hi + low;
        const float residual = low - (rounded - xyz.hi);
        const uint32_t bits = triangle_vertex_bits(rounded);
        const uint32_t exponent = bits & 0x7f800000;
        const float upper_half = triangle_vertex_float(exponent - (24u << 23));
        const float lower_half = bits == exponent ? 0.5f * upper_half : upper_half;
        // Bound omitted low squares and all correction roundoffs. The
        // subtraction between the high and rounded scores is exact (Sterbenz).
        const float error = xyz.hi * 0x1p-42f;
        if (residual > error - lower_half && residual < upper_half - error)
            return double(rounded);
    }
    return triangle_frame_vertex_exact(p, v);
}

template <typename Data>
CUDA_CALLABLE inline double triangle_score(const wp::vec3 &query, const Data &data, int slot) {
    const uint32_t *frame = &data.frame.data[triangle_frame_words * int64_t(slot)];
    const wp::vec3 p0 = data.a.data[slot], p1 = data.b.data[slot], p2 = data.c.data[slot];
    const int start = frame[triangle_frame_permutation];
    if (start == 3)
        return triangle_precise_result<false>(query, p0, p1, p2);

    const wp::vec3 a = start == 0 ? p0 : start == 1 ? p1 : p2;
    const wp::vec3 b = start == 0 ? p1 : start == 1 ? p2 : p0;
    const wp::vec3 c = start == 0 ? p2 : start == 1 ? p0 : p1;
    const wp::vec3 ab = b - a, pa = query - a, pb = query - b;
    const wp::vec3 v(triangle_frame_scalar(frame, 8), triangle_frame_scalar(frame, 9),
                      triangle_frame_scalar(frame, 10));
    const float x_a = wp::dot(pa, ab), x_b = wp::dot(pb, ab);
    const float y_a = wp::dot(pa, v), y_b = wp::dot(pb, v);
    const float h = triangle_frame_scalar(frame, 13);
    const float cx_a = triangle_frame_scalar(frame, 11), cx_b = triangle_frame_scalar(frame, 12);
    const bool outside_ab = y_a < 0.f;
    const bool outside_bc = cx_b * y_b - h * x_b < 0.f;
    const bool outside_ca = cx_a * y_a - h * x_a > 0.f;

    double score;
    if (!(outside_ab || outside_bc || outside_ca)) {
        const TriangleVector normal(triangle_frame_double(frame, 0), triangle_frame_double(frame, 1),
                                    triangle_frame_double(frame, 2));
        const TriangleVector offset = TriangleVector(query) - TriangleVector(x_a < -x_b ? a : b);
        const double height = wp::dot(normal, offset);
        score = height * height * triangle_frame_double(frame, 3);
    } else {
        const wp::vec3 bc = c - b, ca = a - c, pc = query - c;
        const float b_bc = wp::dot(pb, bc), c_bc = -wp::dot(pc, bc);
        const float c_ca = wp::dot(pc, ca), a_ca = -wp::dot(pa, ca);
        const bool edge_ab = outside_ab && x_a > 0.f && x_b < 0.f;
        const bool edge_bc = outside_bc && b_bc > 0.f && c_bc > 0.f;
        const bool edge_ca = outside_ca && c_ca > 0.f && a_ca > 0.f;
        if (edge_ab || edge_bc || edge_ca) {
            const wp::vec3 u = edge_ab ? a : edge_bc ? b : c;
            const wp::vec3 v = edge_ab ? b : edge_bc ? c : a;
            const bool from_u = edge_ab ? x_a < -x_b : edge_bc ? b_bc < c_bc : c_ca < a_ca;
            score = triangle_frame_edge_score(u, v, query, from_u);
        } else if (wp::dot(pa + pb, ab) <= 0.f) {
            score = triangle_frame_vertex_score(query, wp::dot(pc + pa, ca) >= 0.f ? a : c);
        } else {
            score = triangle_frame_vertex_score(query, wp::dot(pb + pc, bc) <= 0.f ? b : c);
        }
    }
    // Refine near contact, where the float frame's angular error can hide
    // small distances, and when the float intermediates exceed their range.
    constexpr float contact_error2 = 0x1p-38f;
    if (!(score > contact_error2 * wp::dot(pa, pa) && score < 1.e16f))
        return triangle_precise_result<false>(query, p0, p1, p2);
    return score;
}

template <typename Data>
CUDA_CALLABLE inline wp::vec3 triangle_closest(const wp::vec3 &query, const Data &data, int slot) {
    return triangle_precise_result<true>(query, data.a.data[slot], data.b.data[slot], data.c.data[slot]);
}

} // namespace bvh
