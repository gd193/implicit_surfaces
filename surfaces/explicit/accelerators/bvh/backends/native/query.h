#pragma once
#include "traversal.h"
#include "geometry.h"
#include "triangle_distance.h"
#include "angular.h"
#include "events.h"
#include "boundary.h"

namespace bvh {
CUDA_CALLABLE inline float conservative_box_score(const wp::vec3 &query, const wp::vec3 &lower,
                                                  const wp::vec3 &upper) {
    const wp::vec3 point = wp::min(wp::max(query, lower), upper);
#if defined(__CUDA_ARCH__)
    const float x = __fsub_rz(query[0], point[0]);
    const float y = __fsub_rz(query[1], point[1]);
    const float z = __fsub_rz(query[2], point[2]);
    return __fmaf_rd(z, z, __fmaf_rd(y, y, __fmul_rd(x, x)));
#else
    const double x = double(query[0]) - double(point[0]);
    const double y = double(query[1]) - double(point[1]);
    const double z = double(query[2]) - double(point[2]);
    const double score = x*x + y*y + z*z;
    float rounded = float(score);
    // Positive float bit patterns are ordered by value. Step down when
    // rounding to nearest crossed above the double-precision box score.
    if (double(rounded) > score) {
        uint32_t bits;
        memcpy(&bits, &rounded, sizeof(bits));
        --bits;
        memcpy(&rounded, &bits, sizeof(rounded));
    }
    return rounded;
#endif
}
} // namespace bvh
