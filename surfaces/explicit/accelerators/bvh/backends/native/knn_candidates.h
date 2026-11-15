#pragma once
#include "traversal.h"

namespace bvh {
// The host specialization selects register-list or heap storage.
template<int K, bool Small> struct KnnCandidates;

template<int K> struct KnnCandidates<K, true> {
    float scores[K];
    int32_t ids[K];
    int32_t size;
    CUDA_CALLABLE KnnCandidates() : size(0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scores[i] = INFINITY;
            ids[i] = 0x7fffffff;
        }
    }
};

template<int K> struct KnnCandidates<K, false> {
    uint64_t entries[K];
    int32_t size;
    CUDA_CALLABLE KnnCandidates() : size(0) {}
};

CUDA_CALLABLE inline bool knn_worse(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai > bi);
}

template<int K, bool Small> CUDA_CALLABLE inline void knn_insert(KnnCandidates<K, Small>& c, float score, int id) {
    const int n = c.size;
    if constexpr (Small) {
        // Constant indices let CUDA keep these candidate lists in registers.
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            const float previous_score = c.scores[i];
            const int previous_id = c.ids[i];
            const bool replace = knn_worse(previous_score, previous_id, score, id);
            c.scores[i] = replace ? score : previous_score;
            c.ids[i] = replace ? id : previous_id;
            score = replace ? previous_score : score;
            id = replace ? previous_id : id;
        }
        if (n < K) c.size = n+1;
    } else {
        // Nonnegative float bits preserve score order; the low word breaks ties.
        // IEEE signed zeros compare equal, so normalize before encoding.
        const uint64_t value = cache_entry(uint32_t(id), score == 0.0f ? 0.0f : score);
        if (n < K) {
            int p = n;
            c.size = n+1;
            while (p > 0) {
                const int parent = (p-1)/2;
                if (value <= c.entries[parent]) break;
                c.entries[p] = c.entries[parent];
                p = parent;
            }
            c.entries[p] = value;
        } else {
            if (value >= c.entries[0]) return;
            int p = 0;
            while (2*p+1 < K) {
                int child = 2*p+1;
                if (child+1 < K && c.entries[child+1] > c.entries[child]) ++child;
                if (c.entries[child] <= value) break;
                c.entries[p] = c.entries[child];
                p = child;
            }
            c.entries[p] = value;
        }
    }
}

template<int K, bool Small> CUDA_CALLABLE inline float knn_score(const KnnCandidates<K, Small>& c, int i) {
    if constexpr (Small) return c.scores[i];
    else return entry_score(c.entries[i]);
}
template<int K, bool Small> CUDA_CALLABLE inline int knn_id(const KnnCandidates<K, Small>& c, int i) {
    if constexpr (Small) return c.ids[i];
    else return static_cast<int32_t>(uint32_t(c.entries[i]));
}
template<int K, bool Small> CUDA_CALLABLE inline float knn_limit(const KnnCandidates<K, Small>& c, float initial) {
    return c.size < K ? initial : knn_score(c, Small ? K-1 : 0);
}
template<int K, bool Small> CUDA_CALLABLE inline void knn_sort(KnnCandidates<K, Small>& c) {
    if constexpr (!Small) {
        for (int end = c.size-1; end > 0; --end) {
            const uint64_t value = c.entries[end];
            c.entries[end] = c.entries[0];
            int p = 0;
            while (2*p+1 < end) {
                int child = 2*p+1;
                if (child+1 < end && c.entries[child+1] > c.entries[child]) ++child;
                if (c.entries[child] <= value) break;
                c.entries[p] = c.entries[child];
                p = child;
            }
            c.entries[p] = value;
        }
    }
}
} // namespace bvh
