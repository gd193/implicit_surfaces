#include "knn_common.h"
#include <cuBQL/queries/pointData/knn.h>

using Vec = cuBQL::vec_t<float, BENCH_DIM>;
using Box = cuBQL::box_t<float, BENCH_DIM>;
using Tree = cuBQL::BinaryBVH<float, BENCH_DIM>;

__global__ void query(Tree t, const Vec* p, const Vec* q, int nq, float cap,
                      float* scores, int* ids, int* counts, int* status) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nq)
        return;
    cuBQL::knn::Candidate c[BENCH_K];
    auto r = cuBQL::points::findKNN(c, BENCH_K, t, p, q[row], cap);
    for (int j = 0; j < BENCH_K; ++j) {
        scores[row * BENCH_K + j] = j < r.numFound ? c[j].sqrDist : INFINITY;
        ids[row * BENCH_K + j] = j < r.numFound ? c[j].primID : -1;
    }
    counts[row] = r.numFound;
    status[row] = r.numFound ? 1 : 0;
}

int main(int argc, char** argv) try {
    if (argc != 8)
        throw std::runtime_error("usage: binary FIXTURE BLOCK LAUNCHES REPEATS PREFIX CAP_SQUARED LEAF");
    std::string folder = argv[1], prefix = argv[5];
    int block = std::stoi(argv[2]), launches = std::stoi(argv[3]),
        repeats = std::stoi(argv[4]), leaf = std::stoi(argv[7]);
    float cap = std::stof(argv[6]);
    if (block < 1 || block > 1024 || launches < 1 || repeats < 1 || leaf < 1)
        throw std::runtime_error("invalid settings");

    auto hp = read<Vec>(folder + "/data.f32"), hq = read<Vec>(folder + "/queries.f32");
    if (hp.empty() || hq.empty())
        throw std::runtime_error("nonempty fixture required");
    auto p = upload(hp), q = upload(hq);

    // Build the independent reference BVH before timing.
    std::vector<Box> hb;
    for (auto v : hp)
        hb.emplace_back(v, v);
    auto boxes = upload(hb);
    Tree tree;
    cuBQL::DeviceMemoryResource memory;
    cuBQL::BuildConfig cfg;
    cfg.makeLeafThreshold = leaf;
    if constexpr (BENCH_DIM == 3)
        cfg.enableSAH();
    cuBQL::gpuBuilder(tree, boxes, hb.size(), cfg, 0, memory);
    check(cudaDeviceSynchronize());

    float* scores;
    int *ids, *counts, *status;
    size_t n = hq.size();
    check(cudaMalloc(&scores, n * BENCH_K * 4));
    check(cudaMalloc(&ids, n * BENCH_K * 4));
    check(cudaMalloc(&counts, n * 4));
    check(cudaMalloc(&status, n * 4));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Query and output stores are inside the measured kernel.
    measure([&] {
        query<<<(n + block - 1) / block, block, 0, stream>>>(
            tree, p, q, n, cap, scores, ids, counts, status);
    }, launches, repeats, stream, prefix);

    save(prefix + ".scores", scores, n * BENCH_K);
    save(prefix + ".ids", ids, n * BENCH_K);
    save(prefix + ".count", counts, n);
    save(prefix + ".status", status, n);
    cudaFuncAttributes a;
    check(cudaFuncGetAttributes(&a, query));
    std::ofstream(prefix + ".json")
        << "{\"registers\":" << a.numRegs
        << ",\"local_bytes\":" << a.localSizeBytes
        << ",\"nodes\":" << tree.numNodes << "}";
    return 0;
} catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
