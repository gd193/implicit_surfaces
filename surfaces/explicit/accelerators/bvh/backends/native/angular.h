#pragma once
#include "builtin.h"

namespace bvh {
using AngularVector = wp::vec_t<3, double>;

template <unsigned Dimension>
CUDA_CALLABLE inline AngularVector angular_coordinates(const wp::vec_t<Dimension, float> &point) {
    if constexpr (Dimension == 2)
        return AngularVector(point[0], point[1], 0.0);
    else
        return AngularVector(point[0], point[1], point[2]);
}

template <unsigned Dimension>
CUDA_CALLABLE inline AngularVector angular_query_row(const wp::mat_t<2, Dimension, float> &query,
                                                     int row) {
    if constexpr (Dimension == 2)
        return AngularVector(query.data[row][0], query.data[row][1], 0.0);
    else
        return AngularVector(query.data[row][0], query.data[row][1], query.data[row][2]);
}

CUDA_CALLABLE inline double angle(const AngularVector &view, const AngularVector &direction) {
    return wp::dot(view, view) > 0
               ? atan2(wp::length(wp::cross(view, direction)), wp::dot(view, direction))
               : INFINITY;
}

template <unsigned Dimension>
CUDA_CALLABLE inline float angular_bound(const wp::mat_t<2, Dimension, float> &query,
                                         const wp::vec3 &lower, const wp::vec3 &upper) {
    const AngularVector origin = angular_query_row(query, 0);
    const AngularVector direction = angular_query_row(query, 1);
    const AngularVector view = .5 * (AngularVector(lower) + AngularVector(upper)) - origin;
    const AngularVector extent = .5 * (AngularVector(upper) - AngularVector(lower));
    const double length2 = wp::dot(view, view), radius2 = wp::dot(extent, extent);
    if (length2 <= radius2)
        return 0.f;
    const double lower_angle = angle(view, direction) - asin(sqrt(radius2 / length2));
    return static_cast<float>(wp::max(0.0, lower_angle - 4.e-7));
}

template <unsigned Dimension>
CUDA_CALLABLE inline wp::vec4 angular_segment(const wp::mat_t<2, Dimension, float> &query,
                                              const wp::vec_t<Dimension, float> &a,
                                              const wp::vec_t<Dimension, float> &b) {
    const AngularVector origin = angular_query_row(query, 0);
    const AngularVector direction = angular_query_row(query, 1);
    const AngularVector va = angular_coordinates(a), vb = angular_coordinates(b);
    wp::vec3 best(va);
    float score = static_cast<float>(angle(va - origin, direction));
    float other = static_cast<float>(angle(vb - origin, direction));
    if (other < score) {
        best = wp::vec3(vb);
        score = other;
    }
    const AngularVector view = va - origin, edge = vb - va;
    const double alpha = wp::dot(direction, view), beta = wp::dot(direction, edge);
    const double c = wp::dot(view, view), d = wp::dot(view, edge), f = wp::dot(edge, edge);
    const double denominator = beta * d - alpha * f;
    if (denominator != 0) {
        const double t = (alpha * d - beta * c) / denominator;
        if (t > 0 && t < 1) {
            const wp::vec3 point(va + t * edge);
            other = static_cast<float>(angle(AngularVector(point) - origin, direction));
            if (other < score) {
                best = point;
                score = other;
            }
        }
    }
    return wp::vec4(best[0], best[1], best[2], score);
}
} // namespace bvh
