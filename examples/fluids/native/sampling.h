#pragma once

namespace fluid {

CUDA_CALLABLE inline void sin_cos(float phi, float& s, float& c) {
    sincosf(phi, &s, &c);
}
CUDA_CALLABLE inline void sin_cos(double phi, double& s, double& c) {
    sincos(phi, &s, &c);
}

template <typename T>
CUDA_CALLABLE inline wp::vec_t<3, T> sphere_direction(T rr, T phi, T z) {
    T s, c;
    sin_cos(phi, s, c);
    return wp::vec_t<3, T>(rr*c, rr*s, z);
}

}  // namespace fluid
