#pragma once
#include "builtin.h"

namespace bvh {
CUDA_CALLABLE inline float quiet_nan() {
#if defined(__CUDA_ARCH__)
    return __int_as_float(0x7fc00000);
#else
    const uint32_t bits = 0x7fc00000u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// Copy storage without reading unwritten slots; the first store activates data.
template <typename T, int N> union Stack {
    T data[N];
    CUDA_CALLABLE Stack() {}
};

template <typename T, int N>
CUDA_CALLABLE inline void push(Stack<T, N> &stack, int index, T value) {
    stack.data[index] = value;
}
template <typename T, int N> CUDA_CALLABLE inline T pop(const Stack<T, N> &stack, int index) {
    return stack.data[index];
}

template <typename Array> CUDA_CALLABLE inline auto load_array(const Array &values, int index) {
    return values.data[index];
}
template <typename Array, typename Index, typename Value>
CUDA_CALLABLE inline void store_array(Array &values, Index index, Value value) {
    values.data[index] = value;
}

// Packed layout adapted from NVIDIA Warp bvh.h; Apache-2.0.
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
CUDA_CALLABLE inline wp::vec4 load_node(const wp::array_t<wp::vec4> &values, int index) {
#if defined(__CUDA_ARCH__)
    const float4 value = __ldg(reinterpret_cast<const float4 *>(values.data + index));
    return wp::vec4(value.x, value.y, value.z, value.w);
#else
    return values.data[index];
#endif
}

CUDA_CALLABLE inline uint64_t pack_node(const wp::vec4 &lower, const wp::vec4 &upper) {
    uint32_t a, b;
    memcpy(&a, &lower.c[3], 4);
    memcpy(&b, &upper.c[3], 4);
    return uint64_t(a) | (uint64_t(b) << 32u);
}
CUDA_CALLABLE inline bool node_is_leaf(uint64_t node) { return (node & 0x80000000ull) != 0; }
CUDA_CALLABLE inline int node_lower(uint64_t node) {
    return static_cast<int>(node & 0x7fffffffull);
}
CUDA_CALLABLE inline int node_upper(uint64_t node) {
    return static_cast<int>((node >> 32u) & 0x7fffffffull);
}
CUDA_CALLABLE inline uint32_t pack_entry(int index, bool tag) {
    return uint32_t(index) | (uint32_t(tag) << 31u);
}
CUDA_CALLABLE inline int entry_index(uint32_t entry) {
    return static_cast<int>(entry & 0x7fffffffull);
}
CUDA_CALLABLE inline bool entry_tag(uint32_t entry) { return (entry & 0x80000000ull) != 0; }
CUDA_CALLABLE inline uint64_t cache_entry(uint32_t entry, float score) {
    uint32_t bits;
    memcpy(&bits, &score, 4);
    return (uint64_t(bits) << 32u) | entry;
}
CUDA_CALLABLE inline float entry_score(uint64_t entry) {
    uint32_t bits = uint32_t(entry >> 32u);
    float value;
    memcpy(&value, &bits, 4);
    return value;
}
} // namespace bvh
// Warp's default NAN divides doubles in generated query code.
#undef NAN
#define NAN (::bvh::quiet_nan())
