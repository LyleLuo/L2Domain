#include "sidedgpu.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdint.h>
#include <string>
#include <vector>

namespace {

__device__ __forceinline__ uint32_t __smid() {
  uint32_t ret;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
  return ret;
}

__global__ void k_get_smid(uint32_t *smids) {
  int bidx = blockIdx.x;
  smids[bidx] = __smid();
}

constexpr size_t kSidedPageSize = 4096ULL;
constexpr size_t kSidedAttrBlockBytes = 2ULL * 1024ULL * 1024ULL;
constexpr size_t kSidedPagesPerAttrBlock =
    kSidedAttrBlockBytes / kSidedPageSize;
constexpr size_t kSidedPoolGrowBytes = 1ULL << 30;
constexpr size_t kSidedPoolReserveBytes = 64ULL << 30;
constexpr int kMemoryAttrExp2StartBit = 7;
constexpr int kMemoryAttrExp2StopBit = 21;
constexpr int kMemoryAttrExp2NumProbeOffsets =
    kMemoryAttrExp2StopBit - kMemoryAttrExp2StartBit;

struct SidedPage {
  int side = -1;
  bool allocated = false;
};

struct SidedChunk {
  CUdeviceptr addr = 0;
  size_t bytes = 0;
  CUmemGenericAllocationHandle handle = 0;
};

struct RuntimeState {
  cudaError_t status = cudaSuccess;
  int device_id = 0;

  std::array<std::vector<cudaDevResource>, 2> side_resources;
  std::array<cudaDevResourceDesc_t, 2> resource_descs{};
  std::array<cudaExecutionContext_t, 2> side_contexts{};
};

struct AllocatorState {
  std::mutex mutex;
  cudaError_t status = cudaSuccess;
  bool initialized = false;
  int device_id = 0;
  CUdevice device = 0;
  CUdeviceptr base = 0;
  size_t reserved_bytes = 0;
  size_t mapped_bytes = 0;
  size_t vmm_granularity = 0;
  uint32_t mask = 0;
  int num_sm = 0;
  int target_side1_smid = -1;
  std::vector<SidedPage> pages;
  std::vector<SidedChunk> chunks;

  ~AllocatorState() {
    for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
      if (it->addr != 0 && it->bytes != 0) {
        cuMemUnmap(it->addr, it->bytes);
      }
      if (it->handle != 0) {
        cuMemRelease(it->handle);
      }
    }
    if (base != 0 && reserved_bytes != 0) {
      cuMemAddressFree(base, reserved_bytes);
    }
  }
};

cudaError_t from_cu(CUresult result) {
  switch (result) {
  case CUDA_SUCCESS:
    return cudaSuccess;
  case CUDA_ERROR_OUT_OF_MEMORY:
    return cudaErrorMemoryAllocation;
  case CUDA_ERROR_INVALID_VALUE:
  case CUDA_ERROR_INVALID_HANDLE:
    return cudaErrorInvalidValue;
  case CUDA_ERROR_INVALID_DEVICE:
    return cudaErrorInvalidDevice;
  case CUDA_ERROR_NOT_INITIALIZED:
  case CUDA_ERROR_DEINITIALIZED:
    return cudaErrorInitializationError;
  case CUDA_ERROR_NOT_PERMITTED:
    return cudaErrorNotPermitted;
  case CUDA_ERROR_NOT_SUPPORTED:
    return cudaErrorNotSupported;
  default:
    return cudaErrorUnknown;
  }
}

size_t round_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

std::string sidedgpu_config_path(const char *filename) {
  const char *dir = std::getenv("SIDEDGPU_CONFIG_DIR");
  if (dir == nullptr || dir[0] == '\0') {
    return {};
  }

  std::string path(dir);
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  path += filename;
  return path;
}

__device__ __forceinline__ int
predict_page_side_device(uintptr_t addr, uint32_t mask, int attr) {
  auto const parity =
      static_cast<int>(__popc(static_cast<uint32_t>(addr) & mask) & 0x1);
  return parity ^ attr;
}

__device__ __forceinline__ void sided_signal_set(int *signal, int value) {
  atomicExch(signal, value);
}

__device__ __forceinline__ void sided_signal_wait(int *signal, int expected) {
  while (atomicCAS(signal, expected, expected) != expected) {
    __nanosleep(2000);
  }
}

__device__ __forceinline__ void access_pair(const uint8_t *base,
                                            volatile uint64_t *sink) {
  auto ptr = reinterpret_cast<const uint64_t *>(base);
  sink[0] = *ptr;
}

__device__ __forceinline__ void
measure_pair(const uint8_t *base, uint32_t *latency, volatile uint64_t *sink) {
  auto ptr = reinterpret_cast<const uint64_t *>(base);
  auto start = clock();
  sink[0] = *ptr;
  auto stop = clock();
  latency[0] = static_cast<uint32_t>(stop - start);
}

__device__ __forceinline__ void access_exp2_series(const uint8_t *base,
                                                   volatile uint64_t *sink) {
  sink[0] = base[128];
  for (int bit = kMemoryAttrExp2StartBit; bit < kMemoryAttrExp2StopBit; ++bit) {
    auto ptr = reinterpret_cast<const uint64_t *>(base + (size_t{1} << bit));
    sink[0] = *ptr;
  }
}

__device__ __forceinline__ void measure_exp2_series(const uint8_t *base,
                                                    uint32_t *latencies,
                                                    volatile uint64_t *sink) {
  sink[0] = base[32];
  for (int bit = kMemoryAttrExp2StartBit; bit < kMemoryAttrExp2StopBit; ++bit) {
    auto ptr = reinterpret_cast<const uint64_t *>(base + (size_t{1} << bit));
    auto start = clock();
    sink[0] = *ptr;
    auto stop = clock();
    latencies[bit - kMemoryAttrExp2StartBit] =
        static_cast<uint32_t>(stop - start);
  }
}

__global__ void k_classify_pages(uint8_t *base, size_t start_page,
                                 size_t num_pages, uint32_t mask,
                                 int target_smid, int *signal,
                                 int *page_sides) {
  if (threadIdx.x != 0) {
    return;
  }

  auto const smid = __smid();
  if (smid != 0 && smid != static_cast<uint32_t>(target_smid)) {
    return;
  }

  __shared__ volatile uint64_t sink;
  __shared__ uint32_t pair_latency[1];
  __shared__ uint32_t exp2_latencies[kMemoryAttrExp2NumProbeOffsets];
  auto const select_id = [](int selected_smid) { return -selected_smid; };

  sink = 0;

  if (smid == 0) {
    sided_signal_set(signal, 0);
    sided_signal_set(signal, select_id(target_smid));

    size_t const num_attr_blocks =
        (num_pages + kSidedPagesPerAttrBlock - 1) / kSidedPagesPerAttrBlock;
    float exp2_avg = 0.0f;
    for (size_t block_id = 0; block_id < num_attr_blocks; ++block_id) {
      sided_signal_wait(signal, static_cast<int>(block_id + 1));
      auto const block_base = base + block_id * kSidedAttrBlockBytes;

      if (block_id == 0) {
        measure_exp2_series(block_base, exp2_latencies, &sink);
      }
      measure_pair(block_base, pair_latency, &sink);

      auto latency = pair_latency[0];
      if (block_id == 0) {
        for (int i = 0; i < kMemoryAttrExp2NumProbeOffsets; ++i) {
          exp2_avg += static_cast<float>(exp2_latencies[i]);
        }
        exp2_avg /= static_cast<float>(kMemoryAttrExp2NumProbeOffsets);
        latency = exp2_latencies[0];
      }

      int const attr = static_cast<float>(latency) > exp2_avg;
      size_t const first_page = block_id * kSidedPagesPerAttrBlock;
      size_t const last_page =
          min(first_page + kSidedPagesPerAttrBlock, num_pages);
      for (size_t page_id = first_page; page_id < last_page; ++page_id) {
        page_sides[page_id] = predict_page_side_device(
            reinterpret_cast<uintptr_t>(base) + page_id * kSidedPageSize, mask,
            attr);
      }

      if (block_id + 1 < num_attr_blocks) {
        sided_signal_set(signal, select_id(target_smid));
      }
    }

    sided_signal_set(signal, 0);
    return;
  }

  while (true) {
    auto const observed = atomicCAS(signal, 0, 0);
    if (observed == 0) {
      __nanosleep(2000);
      continue;
    }
    if (observed != select_id(static_cast<int>(smid))) {
      return;
    }
    break;
  }

  size_t const num_attr_blocks =
      (num_pages + kSidedPagesPerAttrBlock - 1) / kSidedPagesPerAttrBlock;
  for (size_t block_id = 0; block_id < num_attr_blocks; ++block_id) {
    auto const block_base = base + block_id * kSidedAttrBlockBytes;
    access_pair(block_base, &sink);
    if (block_id == 0) {
      access_exp2_series(block_base, &sink);
    }
    sided_signal_set(signal, static_cast<int>(block_id + 1));
    if (block_id + 1 < num_attr_blocks) {
      sided_signal_wait(signal, select_id(static_cast<int>(smid)));
    }
  }
}

cudaError_t read_smid2side(std::vector<int> *smid2side) {
  if (smid2side == nullptr) {
    return cudaErrorInvalidValue;
  }

  std::ifstream file(sidedgpu_config_path("smid2side.csv"));
  if (!file.is_open()) {
    return cudaErrorFileNotFound;
  }

  int side = 0;
  while (file >> side) {
    if (side != 0 && side != 1) {
      return cudaErrorInvalidValue;
    }
    smid2side->push_back(side);
    if (file.peek() == ',') {
      file.ignore();
    }
  }

  return smid2side->empty() ? cudaErrorInvalidValue : cudaSuccess;
}

cudaError_t create_probe_context(int device_id, cudaDevResource *resource,
                                 cudaExecutionContext_t *ctx) {
  cudaDevResourceDesc_t desc = {};
  cudaError_t err = cudaDevResourceGenerateDesc(&desc, resource, 1);
  if (err != cudaSuccess) {
    return err;
  }
  return cudaGreenCtxCreate(ctx, desc, device_id, 0);
}

cudaError_t create_side_context(RuntimeState *state, int side) {
  if (state == nullptr || side < 0 || side > 1 ||
      state->side_resources[side].empty()) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaDevResourceGenerateDesc(
      &state->resource_descs[side], state->side_resources[side].data(),
      static_cast<unsigned int>(state->side_resources[side].size()));
  if (err != cudaSuccess) {
    return err;
  }

  return cudaGreenCtxCreate(&state->side_contexts[side],
                            state->resource_descs[side], state->device_id, 0);
}

cudaError_t create_grouped_resources(RuntimeState *state) {
  if (state == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaGetDevice(&state->device_id);
  if (err != cudaSuccess) {
    return err;
  }

  std::vector<int> smid2side;
  err = read_smid2side(&smid2side);
  if (err != cudaSuccess) {
    return err;
  }

  int num_sm = 0;
  err = cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount,
                               state->device_id);
  if (err != cudaSuccess) {
    return err;
  }
  if (smid2side.size() != static_cast<size_t>(num_sm) || num_sm <= 0 ||
      (num_sm % 2) != 0) {
    return cudaErrorInvalidValue;
  }

  uint32_t *d_smids = nullptr;
  err = cudaMalloc(&d_smids, sizeof(uint32_t) * static_cast<size_t>(num_sm));
  if (err != cudaSuccess) {
    return err;
  }

  cudaDevResource sm_pool{};
  err = cudaDeviceGetDevResource(state->device_id, &sm_pool,
                                 cudaDevResourceTypeSm);
  if (err != cudaSuccess) {
    cudaFree(d_smids);
    return err;
  }

  unsigned int const num_tpcs = sm_pool.sm.smCount / 2;
  if (num_tpcs == 0 || num_tpcs * 2 != static_cast<unsigned int>(num_sm)) {
    cudaFree(d_smids);
    return cudaErrorInvalidValue;
  }

  std::vector<cudaDevResource> tpc_resources(num_tpcs);
  std::vector<cudaDevSmResourceGroupParams> params(num_tpcs);
  for (auto &param : params) {
    param.smCount = 2;
    param.coscheduledSmCount = 2;
    param.preferredCoscheduledSmCount = 0;
    param.flags = 0;
  }

  cudaDevResource remainder{};
  err = cudaDevSmResourceSplit(tpc_resources.data(), num_tpcs, &sm_pool,
                               &remainder, 0u, params.data());
  if (err != cudaSuccess) {
    cudaFree(d_smids);
    return err;
  }
  if (remainder.type != cudaDevResourceTypeInvalid ||
      num_tpcs * 2 != static_cast<unsigned int>(num_sm)) {
    cudaFree(d_smids);
    return cudaErrorInvalidValue;
  }

  for (unsigned int i = 0; i < num_tpcs; ++i) {
    cudaExecutionContext_t probe_ctx = {};
    err = create_probe_context(state->device_id, &tpc_resources[i], &probe_ctx);
    if (err != cudaSuccess) {
      cudaFree(d_smids);
      return err;
    }

    cudaStream_t probe_stream = nullptr;
    err = cudaExecutionCtxStreamCreate(&probe_stream, probe_ctx,
                                       cudaStreamDefault, 0);
    if (err == cudaSuccess) {
      k_get_smid<<<2, 1, 0, probe_stream>>>(d_smids + 2 * i);
      err = cudaStreamSynchronize(probe_stream);
    }
    cudaError_t destroy_stream_err =
        probe_stream == nullptr ? cudaSuccess : cudaStreamDestroy(probe_stream);
    cudaError_t destroy_ctx_err = cudaExecutionCtxDestroy(probe_ctx);
    if (err == cudaSuccess) {
      err = destroy_stream_err;
    }
    if (err == cudaSuccess) {
      err = destroy_ctx_err;
    }
    if (err == cudaSuccess) {
      err = cudaGetLastError();
    }
    if (err != cudaSuccess) {
      cudaFree(d_smids);
      return err;
    }
  }

  std::vector<uint32_t> h_smids(num_sm);
  err = cudaMemcpy(h_smids.data(), d_smids,
                   sizeof(uint32_t) * static_cast<size_t>(num_sm),
                   cudaMemcpyDeviceToHost);
  cudaError_t free_err = cudaFree(d_smids);
  if (err == cudaSuccess) {
    err = free_err;
  }
  if (err != cudaSuccess) {
    return err;
  }

  for (unsigned int i = 0; i < num_tpcs; ++i) {
    uint32_t const first_smid = h_smids[2 * i];
    uint32_t const second_smid = h_smids[2 * i + 1];
    if (first_smid >= static_cast<uint32_t>(num_sm) ||
        second_smid >= static_cast<uint32_t>(num_sm)) {
      return cudaErrorInvalidValue;
    }

    int const side = smid2side[first_smid];
    if (side != smid2side[second_smid]) {
      return cudaErrorInvalidValue;
    }
    state->side_resources[side].push_back(tpc_resources[i]);
  }

  if (state->side_resources[0].empty() || state->side_resources[1].empty()) {
    return cudaErrorInvalidValue;
  }

  err = create_side_context(state, 0);
  if (err != cudaSuccess) {
    return err;
  }

  err = create_side_context(state, 1);
  if (err != cudaSuccess) {
    cudaExecutionCtxDestroy(state->side_contexts[0]);
    state->side_contexts[0] = {};
    return err;
  }

  return cudaSuccess;
}

RuntimeState &state() {
  static RuntimeState runtime_state;
  return runtime_state;
}

AllocatorState &allocator_state() {
  static AllocatorState state;
  return state;
}

cudaError_t read_mask(uint32_t *mask) {
  if (mask == nullptr) {
    return cudaErrorInvalidValue;
  }

  auto const path = sidedgpu_config_path("mask.csv");
  FILE *file = std::fopen(path.c_str(), "r");
  if (file == nullptr) {
    return cudaErrorFileNotFound;
  }

  unsigned int value = 0;
  int matched = std::fscanf(file, "%x", &value);
  std::fclose(file);
  if (matched != 1) {
    return cudaErrorInvalidValue;
  }

  *mask = static_cast<uint32_t>(value);
  return cudaSuccess;
}

cudaError_t init_allocator_locked(AllocatorState *state) {
  if (state == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (state->initialized) {
    return state->status;
  }

  cudaError_t err = cudaGetDevice(&state->device_id);
  if (err != cudaSuccess) {
    state->status = err;
    state->initialized = true;
    return err;
  }

  CUresult cu_result = cuInit(0);
  if (cu_result != CUDA_SUCCESS) {
    state->status = from_cu(cu_result);
    state->initialized = true;
    return state->status;
  }

  cu_result = cuDeviceGet(&state->device, state->device_id);
  if (cu_result != CUDA_SUCCESS) {
    state->status = from_cu(cu_result);
    state->initialized = true;
    return state->status;
  }

  int vmm_supported = 0;
  cu_result = cuDeviceGetAttribute(
      &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      state->device);
  if (cu_result != CUDA_SUCCESS) {
    state->status = from_cu(cu_result);
    state->initialized = true;
    return state->status;
  }
  if (!vmm_supported) {
    state->status = cudaErrorNotSupported;
    state->initialized = true;
    return state->status;
  }

  err = read_mask(&state->mask);
  if (err != cudaSuccess) {
    state->status = err;
    state->initialized = true;
    return err;
  }

  std::vector<int> smid2side;
  err = read_smid2side(&smid2side);
  if (err != cudaSuccess) {
    state->status = err;
    state->initialized = true;
    return err;
  }

  err = cudaDeviceGetAttribute(&state->num_sm, cudaDevAttrMultiProcessorCount,
                               state->device_id);
  if (err != cudaSuccess) {
    state->status = err;
    state->initialized = true;
    return err;
  }
  if (smid2side.size() != static_cast<size_t>(state->num_sm)) {
    state->status = cudaErrorInvalidValue;
    state->initialized = true;
    return state->status;
  }
  for (int smid = 1; smid < state->num_sm; ++smid) {
    if (smid2side[smid] == 1) {
      state->target_side1_smid = smid;
      break;
    }
  }
  if (state->target_side1_smid < 0) {
    state->status = cudaErrorInvalidValue;
    state->initialized = true;
    return state->status;
  }

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = state->device_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  cu_result = cuMemGetAllocationGranularity(&state->vmm_granularity, &prop,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (cu_result != CUDA_SUCCESS) {
    state->status = from_cu(cu_result);
    state->initialized = true;
    return state->status;
  }

  state->reserved_bytes =
      round_up(kSidedPoolReserveBytes, state->vmm_granularity);
  cu_result = cuMemAddressReserve(&state->base, state->reserved_bytes,
                                  state->vmm_granularity, 0, 0);
  if (cu_result != CUDA_SUCCESS) {
    state->status = from_cu(cu_result);
    state->initialized = true;
    return state->status;
  }

  state->status = cudaSuccess;
  state->initialized = true;
  return cudaSuccess;
}

cudaError_t classify_new_pages(AllocatorState *state, size_t start_page,
                               size_t num_pages) {
  if (state == nullptr || num_pages == 0) {
    return cudaErrorInvalidValue;
  }

  int *h_sides = nullptr;
  cudaError_t err =
      cudaHostAlloc(&h_sides, num_pages * sizeof(int), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    return err;
  }

  int *d_sides = nullptr;
  err = cudaHostGetDevicePointer(&d_sides, h_sides, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(h_sides);
    return err;
  }

  int *d_signal = nullptr;
  err = cudaMalloc(&d_signal, sizeof(int));
  if (err != cudaSuccess) {
    cudaFreeHost(h_sides);
    return err;
  }
  err = cudaMemset(d_signal, 0, sizeof(int));
  if (err != cudaSuccess) {
    cudaFree(d_signal);
    cudaFreeHost(h_sides);
    return err;
  }

  auto *chunk_base =
      reinterpret_cast<uint8_t *>(state->base + start_page * kSidedPageSize);
  k_classify_pages<<<state->num_sm, 1>>>(chunk_base, start_page, num_pages,
                                         state->mask, state->target_side1_smid,
                                         d_signal, d_sides);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err != cudaSuccess) {
    cudaFree(d_signal);
    cudaFreeHost(h_sides);
    return err;
  }

  size_t const old_size = state->pages.size();
  state->pages.resize(old_size + num_pages);
  for (size_t i = 0; i < num_pages; ++i) {
    int side = h_sides[i] & 0x1;
    state->pages[old_size + i].side = side;
    state->pages[old_size + i].allocated = false;
  }

  cudaError_t free_signal_err = cudaFree(d_signal);
  err = cudaFreeHost(h_sides);
  if (err == cudaSuccess) {
    err = free_signal_err;
  }
  return err;
}

cudaError_t grow_allocator_locked(AllocatorState *state) {
  if (state == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = init_allocator_locked(state);
  if (err != cudaSuccess) {
    return err;
  }

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = state->device_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  size_t grow_bytes = round_up(kSidedPoolGrowBytes, state->vmm_granularity);
  if (state->mapped_bytes + grow_bytes > state->reserved_bytes) {
    return cudaErrorMemoryAllocation;
  }

  CUmemGenericAllocationHandle handle = 0;
  CUresult cu_result = cuMemCreate(&handle, grow_bytes, &prop, 0);
  if (cu_result != CUDA_SUCCESS) {
    return from_cu(cu_result);
  }

  CUdeviceptr map_addr = state->base + state->mapped_bytes;
  cu_result = cuMemMap(map_addr, grow_bytes, 0, handle, 0);
  if (cu_result != CUDA_SUCCESS) {
    cuMemRelease(handle);
    return from_cu(cu_result);
  }

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = state->device_id;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cu_result = cuMemSetAccess(map_addr, grow_bytes, &access, 1);
  if (cu_result != CUDA_SUCCESS) {
    cuMemUnmap(map_addr, grow_bytes);
    cuMemRelease(handle);
    return from_cu(cu_result);
  }

  size_t const start_page = state->mapped_bytes / kSidedPageSize;
  size_t const num_pages = grow_bytes / kSidedPageSize;
  state->chunks.push_back(SidedChunk{map_addr, grow_bytes, handle});
  state->mapped_bytes += grow_bytes;

  err = classify_new_pages(state, start_page, num_pages);
  if (err != cudaSuccess) {
    state->mapped_bytes -= grow_bytes;
    state->chunks.pop_back();
    cuMemUnmap(map_addr, grow_bytes);
    cuMemRelease(handle);
    return err;
  }

  return cudaSuccess;
}

size_t count_free_pages(const AllocatorState &state, int side) {
  size_t count = 0;
  for (auto const &page : state.pages) {
    if (!page.allocated && page.side == side) {
      ++count;
    }
  }
  return count;
}

} // namespace

SidedGreenContext::SidedGreenContext() {
  auto &runtime_state = state();
  runtime_state.status = create_grouped_resources(&runtime_state);
}

SidedGreenContext::~SidedGreenContext() {
  auto &runtime_state = state();
  for (auto &ctx : runtime_state.side_contexts) {
    if (ctx != nullptr) {
      cudaExecutionCtxDestroy(ctx);
      ctx = {};
    }
  }
}

cudaError_t SidedGreenContext::create_stream(int side, cudaStream_t *stream) {
  if (stream == nullptr || side < 0 || side > 1) {
    return cudaErrorInvalidValue;
  }

  *stream = nullptr;
  static SidedGreenContext singleton;

  auto &runtime_state = state();
  if (runtime_state.status != cudaSuccess) {
    return runtime_state.status;
  }

  return cudaExecutionCtxStreamCreate(stream, runtime_state.side_contexts[side],
                                      cudaStreamDefault, 0);
}

cudaError_t cudaMallocSided(size_t bytes, int side,
                            std::vector<size_t> &page_ids) {
  if (side < 0 || side > 1) {
    return cudaErrorInvalidValue;
  }
  if (bytes == 0) {
    page_ids.clear();
    return cudaSuccess;
  }

  auto &state = allocator_state();
  std::lock_guard<std::mutex> lock(state.mutex);

  cudaError_t err = init_allocator_locked(&state);
  if (err != cudaSuccess) {
    return err;
  }

  size_t const needed_pages = (bytes + kSidedPageSize - 1) / kSidedPageSize;
  while (count_free_pages(state, side) < needed_pages) {
    err = grow_allocator_locked(&state);
    if (err != cudaSuccess) {
      return err;
    }
  }

  std::vector<size_t> selected;
  selected.reserve(needed_pages);
  for (size_t page_id = 0;
       page_id < state.pages.size() && selected.size() < needed_pages;
       ++page_id) {
    auto &page = state.pages[page_id];
    if (!page.allocated && page.side == side) {
      selected.push_back(page_id);
    }
  }

  if (selected.size() != needed_pages) {
    return cudaErrorMemoryAllocation;
  }

  for (size_t page_id : selected) {
    state.pages[page_id].allocated = true;
  }

  page_ids = selected;
  return cudaSuccess;
}

cudaError_t cudaFreeSided(const std::vector<size_t> &page_ids) {
  auto &state = allocator_state();
  std::lock_guard<std::mutex> lock(state.mutex);

  cudaError_t err = init_allocator_locked(&state);
  if (err != cudaSuccess) {
    return err;
  }

  for (size_t page_id : page_ids) {
    if (page_id >= state.pages.size()) {
      return cudaErrorInvalidValue;
    }
    auto const &page = state.pages[page_id];
    if (!page.allocated) {
      return cudaErrorInvalidValue;
    }
  }

  for (size_t page_id : page_ids) {
    state.pages[page_id].allocated = false;
  }

  return cudaSuccess;
}

cudaError_t cudaGetSidedPoolBase(void **base) {
  if (base == nullptr) {
    return cudaErrorInvalidValue;
  }

  auto &state = allocator_state();
  std::lock_guard<std::mutex> lock(state.mutex);

  cudaError_t err = init_allocator_locked(&state);
  if (err != cudaSuccess) {
    *base = nullptr;
    return err;
  }

  *base = reinterpret_cast<void *>(state.base);
  return cudaSuccess;
}
