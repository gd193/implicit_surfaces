#include "knn_common.h"
#include <cukd/builder.h>
#include <cukd/knn.h>

#if BENCH_DIM == 2
using Vec = float2;
#else
using Vec = float3;
#endif

struct Item {
    Vec p;
    int id;
};
struct Traits : cukd::default_data_traits<Vec> {
    using data_t = Item;
    static __host__ __device__ int get_dim(const Item&) { return 0; }
    static __host__ __device__ void set_dim(Item&, int) {}
    static __host__ __device__ const Vec& get_point(const Item& a) { return a.p; }
    static __host__ __device__ float get_coord(const Item& a, int d) {
        return cukd::get_coord(a.p, d);
    }
};

#if BENCH_HEAP
using List = cukd::HeapCandidateList<BENCH_K>;
#else
using List = cukd::FixedCandidateList<BENCH_K>;
#endif

__global__ void query(cukd::box_t<Vec> bounds, const Item* p, int np,
                      const Vec* q, int nq, float cap, float* scores,
                      int* ids, int* counts, int* status) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nq)
        return;
    List c(sqrtf(cap));
#if BENCH_STACK
    cukd::stackBased::knn<List, Item, Traits>(c, q[row], p, np);
#else
    cukd::cct::knn<List, Item, Traits>(c, q[row], bounds, p, np);
#endif
#if BENCH_HEAP
    sort_entries(c.entry);
#endif
    // Sorting and original-ID remapping remain inside the timed kernel.
    int found = 0;
    for (int j = 0; j < BENCH_K; ++j) {
        int id = c.get_pointID(j);
        scores[row * BENCH_K + j] = id >= 0 ? c.get_dist2(j) : INFINITY;
        ids[row * BENCH_K + j] = id >= 0 ? p[id].id : -1;
        found += id >= 0;
    }
    counts[row] = found;
    status[row] = found ? 1 : 0;
}

int main(int argc, char** argv) try {
    if (argc != 8)
        throw std::runtime_error("usage: binary FIXTURE BLOCK LAUNCHES REPEATS PREFIX CAP_SQUARED LEAF_UNUSED");
    std::string folder = argv[1], prefix = argv[5];
    int block = std::stoi(argv[2]), launches = std::stoi(argv[3]),
        repeats = std::stoi(argv[4]);
    float cap = std::stof(argv[6]);
    if (block < 1 || block > 1024 || launches < 1 || repeats < 1)
        throw std::runtime_error("invalid settings");

    auto hp = read<Vec>(folder + "/data.f32"), hq = read<Vec>(folder + "/queries.f32");
    if (hp.empty() || hq.empty())
        throw std::runtime_error("nonempty fixture required");
    // Keep original IDs alongside points through the builder's permutation.
    std::vector<Item> hi;
    for (int i = 0; i < (int)hp.size(); ++i)
        hi.push_back({hp[i], i});
    auto p = upload(hi);
    auto q = upload(hq);

    cukd::box_t<Vec>* db;
    check(cudaMalloc(&db, sizeof(*db)));
    cukd::buildTree<Item, Traits>(p, hi.size(), db);
    cukd::box_t<Vec> bounds;
    check(cudaMemcpy(&bounds, db, sizeof(bounds), cudaMemcpyDeviceToHost));
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

    measure([&] {
        query<<<(n + block - 1) / block, block, 0, stream>>>(
            bounds, p, hi.size(), q, n, cap, scores, ids, counts, status);
    }, launches, repeats, stream, prefix);

    save(prefix + ".scores", scores, n * BENCH_K);
    save(prefix + ".ids", ids, n * BENCH_K);
    save(prefix + ".count", counts, n);
    save(prefix + ".status", status, n);
    cudaFuncAttributes a;
    check(cudaFuncGetAttributes(&a, query));
    std::ofstream(prefix + ".json")
        << "{\"registers\":" << a.numRegs
        << ",\"local_bytes\":" << a.localSizeBytes << "}";
    return 0;
} catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
