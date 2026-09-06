#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#if defined(_WIN32)
#define SIDEDGPU_API __declspec(dllexport)
#define SIDEDGPU_HIDDEN
#else
#define SIDEDGPU_API __attribute__((visibility("default")))
#define SIDEDGPU_HIDDEN __attribute__((visibility("hidden")))
#endif

class SIDEDGPU_HIDDEN SidedGreenContext {
  public:
    SIDEDGPU_API static cudaError_t create_stream(int side, cudaStream_t *stream);

  private:
    SidedGreenContext(const SidedGreenContext &)            = delete;
    SidedGreenContext &operator=(const SidedGreenContext &) = delete;
    SidedGreenContext();
    ~SidedGreenContext();
};

SIDEDGPU_API cudaError_t cudaMallocSided(size_t bytes, int side,
                                         std::vector<size_t> &page_ids);
SIDEDGPU_API cudaError_t cudaFreeSided(const std::vector<size_t> &page_ids);
SIDEDGPU_API cudaError_t cudaGetSidedPoolBase(void **base);

#undef SIDEDGPU_API
#undef SIDEDGPU_HIDDEN
