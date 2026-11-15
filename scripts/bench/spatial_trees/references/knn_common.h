#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>

inline void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}

template<class T> std::vector<T> read(const std::string& p) {
    auto n = std::filesystem::file_size(p);
    if (n % sizeof(T))
        throw std::runtime_error("invalid array " + p);
    std::vector<T> v(n / sizeof(T));
    if (!std::ifstream(p, std::ios::binary).read((char*)v.data(), n))
        throw std::runtime_error("read " + p);
    return v;
}

template<class T> T* upload(const std::vector<T>& v) {
    T* p;
    check(cudaMalloc(&p, std::max(size_t(1), v.size()) * sizeof(T)));
    if (v.size())
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

template<class T> void save(const std::string& p, T* d, size_t n) {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::ofstream(p, std::ios::binary).write((char*)v.data(), n * sizeof(T));
}

// Upstream sorted lists emit directly. Heap sorts its existing encoded entries in place.
template<int K> __device__ void sift_entries(uint64_t (&entry)[K], int root, int n) {
    while (2 * root + 1 < n) {
        int child = 2 * root + 1;
        if (child + 1 < n && entry[child] < entry[child + 1])
            ++child;
        if (entry[root] >= entry[child])
            break;
        uint64_t a = entry[root];
        entry[root] = entry[child];
        entry[child] = a;
        root = child;
    }
}

template<int K> __device__ void sort_entries(uint64_t (&entry)[K]) {
    // Upstream HeapCandidateList already supplies a max heap.
    for (int end = K - 1; end > 0; --end) {
        uint64_t a = entry[0];
        entry[0] = entry[end];
        entry[end] = a;
        sift_entries(entry, 0, end);
    }
}

template<class Launch>
void measure(Launch launch, int launches, int repeats, cudaStream_t stream,
             const std::string& prefix) {
    // Warm ordinary launches before capturing the resident query workload.
    for (int i = 0; i < 5; ++i)
        launch();
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));

    cudaGraph_t g;
    cudaGraphExec_t x;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < launches; ++i)
        launch();
    check(cudaStreamEndCapture(stream, &g));
    check(cudaGraphInstantiate(&x, g, nullptr, nullptr, 0));
    check(cudaGraphLaunch(x, stream));
    check(cudaStreamSynchronize(stream));

    // Only graph execution lies between the timing events.
    cudaEvent_t a, b;
    check(cudaEventCreate(&a));
    check(cudaEventCreate(&b));
    std::ofstream f(prefix + ".ms");
    f.precision(12);
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(a, stream));
        check(cudaGraphLaunch(x, stream));
        check(cudaEventRecord(b, stream));
        check(cudaEventSynchronize(b));
        float ms;
        check(cudaEventElapsedTime(&ms, a, b));
        f << ms / launches << '\n';
    }
    check(cudaEventDestroy(a));
    check(cudaEventDestroy(b));
    check(cudaGraphExecDestroy(x));
    check(cudaGraphDestroy(g));
}
