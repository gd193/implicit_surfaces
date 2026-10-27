#include "cuda/cuda-command.h"

extern "C" void* switchBenchmarkQueueStream(rhi::ICommandQueue* queue, void* stream) {
    auto* native = static_cast<rhi::cuda::CommandQueueImpl*>(queue);
    void* previous = native->m_stream;
    native->m_stream = static_cast<CUstream>(stream);
    native->m_submitsSinceEvent = 0;
    return previous;
}

#include "cuda/cuda-device.h"
#include "cuda/cuda-heap.h"
#include <unordered_set>

extern "C" void completeBenchmarkHeapEvents(rhi::IDevice* device, void* stream) {
    auto* native = static_cast<rhi::cuda::DeviceImpl*>(device);
    std::unordered_set<rhi::cuda::HeapImpl::PageImpl*> pages;
    for (auto* heap : {native->m_deviceMemHeap.get(), native->m_hostMemHeap.get()}) {
        if (!heap) continue;
        for (auto* page : heap->m_pages)
            if (page) pages.insert(static_cast<rhi::cuda::HeapImpl::PageImpl*>(page));
        for (auto* page : heap->m_pageCache.m_cachedPages) pages.insert(page);
    }
    for (auto* page : pages) {
        for (auto& pending : page->m_pendingEvents) {
            if (pending.stream == stream && cuEventRecord(pending.event, static_cast<CUstream>(stream)) != CUDA_SUCCESS)
                throw std::runtime_error("cannot complete captured heap event");
        }
    }
    if (cuStreamSynchronize(static_cast<CUstream>(stream)) != CUDA_SUCCESS)
        throw std::runtime_error("heap event synchronization failed");
}
