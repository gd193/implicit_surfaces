// Match bounded JAX coordinates: CUDA rounds mul/add separately; CPU contracts them.
#pragma once
namespace dual_grid {
CUDA_CALLABLE inline wp::vec3 position(wp::vec3i c, wp::vec3 lower, wp::vec3 spacing) {
    wp::vec3 result;
    for (int axis = 0; axis < 3; ++axis) {
#ifdef __CUDA_ARCH__
        result[axis] = __fadd_rn(lower[axis], __fmul_rn(float(c[axis]), spacing[axis]));
#else
        result[axis] = __builtin_fmaf(float(c[axis]), spacing[axis], lower[axis]);
#endif
    }
    return result;
}
}
