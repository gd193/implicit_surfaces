#pragma once
#include <cuda.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Slang-RHI exports API pointer variables; resolve native functions explicitly.

#define CUDA_NAME_IMPL(name) #name
#define CUDA_NAME(name) CUDA_NAME_IMPL(name)
struct CudaDriver {
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    template<class T> T load(const char* name) {
        if (!library) throw std::runtime_error("cannot load CUDA driver");
        void* symbol = dlsym(library, name);
        if (!symbol) throw std::runtime_error(std::string("missing CUDA symbol ") + name);
        return reinterpret_cast<T>(symbol);
    }
    decltype(&cuCtxPopCurrent) cuCtxPopCurrent_fn = load<decltype(&cuCtxPopCurrent)>(CUDA_NAME(cuCtxPopCurrent));
    decltype(&cuCtxPushCurrent) cuCtxPushCurrent_fn = load<decltype(&cuCtxPushCurrent)>(CUDA_NAME(cuCtxPushCurrent));
    decltype(&cuEventCreate) cuEventCreate_fn = load<decltype(&cuEventCreate)>(CUDA_NAME(cuEventCreate));
    decltype(&cuEventDestroy) cuEventDestroy_fn = load<decltype(&cuEventDestroy)>(CUDA_NAME(cuEventDestroy));
    decltype(&cuEventElapsedTime) cuEventElapsedTime_fn = load<decltype(&cuEventElapsedTime)>(CUDA_NAME(cuEventElapsedTime));
    decltype(&cuEventRecord) cuEventRecord_fn = load<decltype(&cuEventRecord)>(CUDA_NAME(cuEventRecord));
    decltype(&cuEventSynchronize) cuEventSynchronize_fn = load<decltype(&cuEventSynchronize)>(CUDA_NAME(cuEventSynchronize));
    decltype(&cuFuncGetAttribute) cuFuncGetAttribute_fn = load<decltype(&cuFuncGetAttribute)>(CUDA_NAME(cuFuncGetAttribute));
    decltype(&cuGetErrorString) cuGetErrorString_fn = load<decltype(&cuGetErrorString)>(CUDA_NAME(cuGetErrorString));
    decltype(&cuGraphAddKernelNode) cuGraphAddKernelNode_fn = load<decltype(&cuGraphAddKernelNode)>(CUDA_NAME(cuGraphAddKernelNode));
    decltype(&cuGraphCreate) cuGraphCreate_fn = load<decltype(&cuGraphCreate)>(CUDA_NAME(cuGraphCreate));
    decltype(&cuGraphDestroy) cuGraphDestroy_fn = load<decltype(&cuGraphDestroy)>(CUDA_NAME(cuGraphDestroy));
    decltype(&cuGraphExecDestroy) cuGraphExecDestroy_fn = load<decltype(&cuGraphExecDestroy)>(CUDA_NAME(cuGraphExecDestroy));
    decltype(&cuGraphGetNodes) cuGraphGetNodes_fn = load<decltype(&cuGraphGetNodes)>(CUDA_NAME(cuGraphGetNodes));
    decltype(&cuGraphInstantiateWithFlags) cuGraphInstantiateWithFlags_fn = load<decltype(&cuGraphInstantiateWithFlags)>(CUDA_NAME(cuGraphInstantiateWithFlags));
    decltype(&cuGraphKernelNodeGetParams) cuGraphKernelNodeGetParams_fn = load<decltype(&cuGraphKernelNodeGetParams)>(CUDA_NAME(cuGraphKernelNodeGetParams));
    decltype(&cuGraphLaunch) cuGraphLaunch_fn = load<decltype(&cuGraphLaunch)>(CUDA_NAME(cuGraphLaunch));
    decltype(&cuGraphNodeGetType) cuGraphNodeGetType_fn = load<decltype(&cuGraphNodeGetType)>(CUDA_NAME(cuGraphNodeGetType));
    decltype(&cuStreamBeginCapture) cuStreamBeginCapture_fn = load<decltype(&cuStreamBeginCapture)>(CUDA_NAME(cuStreamBeginCapture));
    decltype(&cuStreamCreate) cuStreamCreate_fn = load<decltype(&cuStreamCreate)>(CUDA_NAME(cuStreamCreate));
    decltype(&cuStreamDestroy) cuStreamDestroy_fn = load<decltype(&cuStreamDestroy)>(CUDA_NAME(cuStreamDestroy));
    decltype(&cuStreamEndCapture) cuStreamEndCapture_fn = load<decltype(&cuStreamEndCapture)>(CUDA_NAME(cuStreamEndCapture));
    decltype(&cuStreamGetCtx) cuStreamGetCtx_fn = load<decltype(&cuStreamGetCtx)>(CUDA_NAME(cuStreamGetCtx));
    decltype(&cuStreamSynchronize) cuStreamSynchronize_fn = load<decltype(&cuStreamSynchronize)>(CUDA_NAME(cuStreamSynchronize));
};
#undef CUDA_NAME
#undef CUDA_NAME_IMPL
inline CudaDriver& driver() { static CudaDriver api; return api; }

extern "C" void* switchBenchmarkQueueStream(rhi::ICommandQueue*, void*);
extern "C" void completeBenchmarkHeapEvents(rhi::IDevice*, void*);

struct KernelTiming {
    std::vector<double> samples_ms;
    int regs = 0, local_bytes = 0, captured_kernels = 0;
};

inline void cudaCheck(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    driver().cuGetErrorString_fn(result, &message);
    throw std::runtime_error(std::string(operation) + ": " + (message ? message : "CUDA error"));
}

struct NativeGraphResources {
    CUgraph captured = nullptr, kernels = nullptr;
    CUgraphExec full_exec = nullptr, kernel_exec = nullptr;
    CUevent start = nullptr, stop = nullptr;
    ~NativeGraphResources() {
        if (kernel_exec) driver().cuGraphExecDestroy_fn(kernel_exec);
        if (full_exec) driver().cuGraphExecDestroy_fn(full_exec);
        if (kernels) driver().cuGraphDestroy_fn(kernels);
        if (captured) driver().cuGraphDestroy_fn(captured);
        if (start) driver().cuEventDestroy_fn(start);
        if (stop) driver().cuEventDestroy_fn(stop);
    }
};

template<class Query>
KernelTiming timeKernelOnly(
    fcpw::GPUContext& context, const fcpw::ComputeShader& shader,
    const Query& query,
    const std::function<void(const fcpw::ComputeShader&, const fcpw::ShaderCursor&)>& bindBvh,
    uint32_t groups, uint32_t launches, uint32_t repeats) {
    using namespace fcpw;
    NativeHandle native;
    if (context.queue->getNativeHandle(&native) != SLANG_OK || native.type != NativeHandleType::CUstream)
        throw std::runtime_error("CUDA queue stream unavailable");
    CUstream queue_stream = reinterpret_cast<CUstream>(native.value);
    CUcontext cuda_context = nullptr;
    cudaCheck(driver().cuStreamGetCtx_fn(queue_stream, &cuda_context), "cuStreamGetCtx");
    cudaCheck(driver().cuCtxPushCurrent_fn(cuda_context), "cuCtxPushCurrent");
    struct PopContext { ~PopContext() { CUcontext popped; driver().cuCtxPopCurrent_fn(&popped); } } pop;
    struct StreamGuard {
        CUstream value = nullptr;
        ~StreamGuard() { if (value) driver().cuStreamDestroy_fn(value); }
    } captured_stream;
    cudaCheck(driver().cuStreamCreate_fn(&captured_stream.value, CU_STREAM_NON_BLOCKING), "create capture stream");
    CUstream stream = captured_stream.value;
    struct QueueStreamGuard {
        rhi::ICommandQueue* queue;
        void* previous;
        ~QueueStreamGuard() { switchBenchmarkQueueStream(queue, previous); }
    } queue_stream_guard{context.queue.get(), switchBenchmarkQueueStream(context.queue.get(), stream)};
    auto makeCommand = [&]() {
        auto encoder = context.queue->createCommandEncoder();
        auto pass = encoder->beginComputePass();
        auto root = pass->bindPipeline(shader.pipeline);
        ShaderCursor cursor(root);
        bindBvh(shader, cursor);
        ShaderCursor args(root->getEntryPoint(0));
        query.setResources(args, false);
        pass->dispatchCompute(groups, 1, 1);
        pass->end();
        return encoder->finish();
    };
    for (int i = 0; i < 5; ++i) {
        auto command = makeCommand();
        if (context.queue->submit(command) != SLANG_OK) throw std::runtime_error("warmup submit failed");
        if (context.queue->waitOnHost() != SLANG_OK) throw std::runtime_error("RHI warmup synchronization failed");
    }
    std::vector<GPUInteraction> reference;
    query.read(context, reference);
    KernelTiming timing;
    {
        NativeGraphResources native_graph;
        // RHI waitOnHost retires bindings, so keep this CB in flight during graph use.
        auto command = makeCommand();
        switchBenchmarkQueueStream(context.queue.get(), stream);
        cudaCheck(driver().cuStreamBeginCapture_fn(stream, CU_STREAM_CAPTURE_MODE_GLOBAL), "cuStreamBeginCapture");
        auto submit_result = context.queue->submit(command);
        auto capture_result = driver().cuStreamEndCapture_fn(stream, &native_graph.captured);
        if (submit_result != SLANG_OK) throw std::runtime_error("capture submit failed");
        cudaCheck(capture_result, "cuStreamEndCapture");
        cudaCheck(driver().cuGraphInstantiateWithFlags_fn(&native_graph.full_exec, native_graph.captured, 0), "instantiate captured graph");
        cudaCheck(driver().cuGraphLaunch_fn(native_graph.full_exec, stream), "initialize captured bindings");
        cudaCheck(driver().cuStreamSynchronize_fn(stream), "binding initialization sync");
        size_t count = 0;
        cudaCheck(driver().cuGraphGetNodes_fn(native_graph.captured, nullptr, &count), "get captured node count");
        std::vector<CUgraphNode> nodes(count);
        cudaCheck(driver().cuGraphGetNodes_fn(native_graph.captured, nodes.data(), &count), "get captured nodes");
        CUDA_KERNEL_NODE_PARAMS params{};
        for (auto node : nodes) {
            CUgraphNodeType type;
            cudaCheck(driver().cuGraphNodeGetType_fn(node, &type), "get node type");
            if (type != CU_GRAPH_NODE_TYPE_KERNEL) continue;
            ++timing.captured_kernels;
            cudaCheck(driver().cuGraphKernelNodeGetParams_fn(node, &params), "get kernel parameters");
        }
        if (timing.captured_kernels != 1) throw std::runtime_error("expected exactly one FCPW kernel");
        cudaCheck(driver().cuFuncGetAttribute_fn(&timing.regs, CU_FUNC_ATTRIBUTE_NUM_REGS, params.func), "get register count");
        cudaCheck(driver().cuFuncGetAttribute_fn(&timing.local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, params.func), "get local memory");
        cudaCheck(driver().cuGraphCreate_fn(&native_graph.kernels, 0), "create kernel-only graph");
        CUgraphNode previous = nullptr;
        for (uint32_t i = 0; i < launches; ++i) {
            CUgraphNode current;
            cudaCheck(driver().cuGraphAddKernelNode_fn(&current, native_graph.kernels, previous ? &previous : nullptr,
                                          previous ? 1 : 0, &params), "add FCPW kernel node");
            previous = current;
        }
        cudaCheck(driver().cuGraphInstantiateWithFlags_fn(&native_graph.kernel_exec, native_graph.kernels, 0), "instantiate kernel-only graph");
        cudaCheck(driver().cuGraphLaunch_fn(native_graph.kernel_exec, stream), "warm kernel-only graph");
        cudaCheck(driver().cuStreamSynchronize_fn(stream), "warm graph sync");
        cudaCheck(driver().cuEventCreate_fn(&native_graph.start, 0), "create start event");
        cudaCheck(driver().cuEventCreate_fn(&native_graph.stop, 0), "create stop event");
        for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            cudaCheck(driver().cuEventRecord_fn(native_graph.start, stream), "record start");
            cudaCheck(driver().cuGraphLaunch_fn(native_graph.kernel_exec, stream), "launch kernel-only graph");
            cudaCheck(driver().cuEventRecord_fn(native_graph.stop, stream), "record stop");
            cudaCheck(driver().cuEventSynchronize_fn(native_graph.stop), "wait for stop");
            float ms;
            cudaCheck(driver().cuEventElapsedTime_fn(&ms, native_graph.start, native_graph.stop), "elapsed event time");
            timing.samples_ms.push_back(double(ms) / launches);
        }
        completeBenchmarkHeapEvents(context.device.get(), stream);
    }
    std::vector<GPUInteraction> result;
    query.read(context, result);
    if (result.size() != reference.size()) throw std::runtime_error("graph output size mismatch");
    for (size_t i = 0; i < result.size(); ++i) {
        const auto& a = reference[i];
        const auto& b = result[i];
        if (a.index != b.index || (a.index != FCPW_GPU_UINT_MAX &&
            (a.d != b.d || std::memcmp(&a.p, &b.p, sizeof(a.p)) != 0)))
            throw std::runtime_error("kernel-only graph differs from RHI at query " + std::to_string(i));
    }
    if (context.queue->waitOnHost() != SLANG_OK) throw std::runtime_error("RHI completion synchronization failed");
    return timing;
}
