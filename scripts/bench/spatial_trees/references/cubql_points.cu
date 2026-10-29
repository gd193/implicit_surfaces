#include <cuda_runtime.h>
#include <cuBQL/queries/pointData/findClosest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using Vec = cuBQL::vec_t<float, BENCH_DIM>;
using Box = cuBQL::box_t<float, BENCH_DIM>;
using Tree = cuBQL::BinaryBVH<float, BENCH_DIM>;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class T> std::vector<T> read(const std::string& path) {
    auto size = std::filesystem::file_size(path);
    if (size%sizeof(T)) throw std::runtime_error("invalid array: " + path);
    std::vector<T> values(size/sizeof(T));
    if (!std::ifstream(path,std::ios::binary).read((char*)values.data(), size)) throw std::runtime_error("cannot read " + path);
    return values;
}
template<class T> T* upload(const std::vector<T>& values) {
    T* result;
    check(cudaMalloc(&result, values.size()*sizeof(T)));
    check(cudaMemcpy(result, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
    return result;
}
__global__ void nearest(Tree tree, const Vec* data, const Vec* queries, int count, Vec* points, float* distances, int* ids) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int id = cuBQL::points::findClosest(tree, data, queries[i], .75f*.75f);
    ids[i] = id;
    points[i] = id >= 0 ? data[id] : Vec(0.f);
    distances[i] = id >= 0 ? sqrtf(cuBQL::sqrLength(data[id]-queries[i])) : INFINITY;
}
int main(int argc, char** argv) try {
    if (argc != 6 && argc != 7) throw std::runtime_error("usage: cubql_points FIXTURE BLOCK LAUNCHES REPEATS PREFIX [LEAF]");
    const std::string fixture=argv[1], prefix=argv[5];
    const int block=std::stoi(argv[2]), launches=std::stoi(argv[3]), repeats=std::stoi(argv[4]);
    const int leaf=argc == 7 ? std::stoi(argv[6]) : 4;
    if (block<1 || block>1024 || launches<1 || repeats<1 || leaf<1) throw std::runtime_error("invalid timing configuration");
    auto host=read<Vec>(fixture+"/vertices.f32"), queries=read<Vec>(fixture+"/points.f32");
    std::vector<Box> bounds;
    for (auto point:host) bounds.push_back(Box(point,point));
    auto data=upload(host), q=upload(queries);
    auto boxes=upload(bounds);
    Tree tree;
    cuBQL::BuildConfig config;
    config.makeLeafThreshold=leaf;
    if constexpr (BENCH_DIM == 3) config.enableSAH();
    cuBQL::DeviceMemoryResource memory;
    cuBQL::gpuBuilder(tree,boxes,bounds.size(),config,0,memory);
    check(cudaDeviceSynchronize());
    Vec* points; float* distances; int* ids;
    const int count=queries.size();
    check(cudaMalloc(&points,count*sizeof(Vec)));
    check(cudaMalloc(&distances,count*sizeof(float)));
    check(cudaMalloc(&ids,count*sizeof(int)));
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto launch=[&] { nearest<<<(count+block-1)/block,block,0,stream>>>(tree,data,q,count,points,distances,ids); };
    for (int i=0;i<5;++i) launch();
    check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for (int i=0;i<launches;++i) launch();
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    check(cudaGraphLaunch(executable,stream)); check(cudaStreamSynchronize(stream));
    cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    std::ofstream samples(prefix+".ms"); samples.precision(12);
    for (int i=0;i<repeats;++i) {
        check(cudaEventRecord(start,stream)); check(cudaGraphLaunch(executable,stream)); check(cudaEventRecord(stop,stream));
        check(cudaEventSynchronize(stop)); float elapsed; check(cudaEventElapsedTime(&elapsed,start,stop));
        samples << elapsed/launches << '\n';
    }
    std::vector<Vec> hp(count); std::vector<float> hd(count); std::vector<int> hi(count);
    check(cudaMemcpy(hp.data(),points,count*sizeof(Vec),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hd.data(),distances,count*sizeof(float),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hi.data(),ids,count*sizeof(int),cudaMemcpyDeviceToHost));
    std::ofstream output(prefix+".bin",std::ios::binary);
    for (int i=0;i<count;++i) {
        output.write((char*)&hp[i],sizeof(Vec)); output.write((char*)&hd[i],sizeof(float)); output.write((char*)&hi[i],sizeof(int));
    }
    cudaFuncAttributes properties; check(cudaFuncGetAttributes(&properties,nearest));
    std::ofstream(prefix+".json") << "{\"registers\":" << properties.numRegs << ",\"local_bytes\":" << properties.localSizeBytes << ",\"nodes\":" << tree.numNodes << ",\"leaf_threshold\":" << leaf << "}\n";
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop)); check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    cuBQL::cuda::free(tree,0,memory);
    for (void* p:{(void*)data,(void*)q,(void*)boxes,(void*)points,(void*)distances,(void*)ids}) check(cudaFree(p));
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
