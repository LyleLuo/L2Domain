#include "latopt.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace latopt;

namespace {

constexpr int K = 1024;
#ifndef LANES_PER_ROW
#define LANES_PER_ROW 32
#endif
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 16
#endif
constexpr int kLanesPerRow = LANES_PER_ROW;
constexpr int kWarpsPerBlock = WARPS_PER_BLOCK;
constexpr int kThreads = kWarpsPerBlock * 32;
constexpr int kRowsPerWarp = 32 / kLanesPerRow;
constexpr int kRowsPerBlock = kWarpsPerBlock * kRowsPerWarp;

static_assert(kLanesPerRow == 8 || kLanesPerRow == 16 || kLanesPerRow == 32, "LANES_PER_ROW must be 8, 16, or 32");

template <typename T>
struct AlignedAllocResult {
    T *raw = nullptr;
    T *ptr = nullptr;
    size_t aligned_bytes = 0;
};

template <typename T>
AlignedAllocResult<T> alloc_aligned(size_t elems) {
    AlignedAllocResult<T> mem;
    size_t bytes = elems * sizeof(T);
    mem.aligned_bytes = ceil_div(bytes, kMemoryAllocGran) * kMemoryAllocGran;
    CHECK_CUDA(cudaMalloc(&mem.raw, mem.aligned_bytes + kMemoryAllocGran));
    uintptr_t base = (reinterpret_cast<uintptr_t>(mem.raw) + kMemoryAllocGran - 1) & ~(uintptr_t)(kMemoryAllocGran - 1);
    mem.ptr = reinterpret_cast<T *>(base);
    CHECK_CUDA(cudaMemset(mem.raw, 0, mem.aligned_bytes + kMemoryAllocGran));
    return mem;
}

int argi(int argc, char **argv, int idx, int fallback) { return argc > idx ? std::atoi(argv[idx]) : fallback; }

float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct SideInfo {
    uint32_t mask = 0;
    std::vector<int> attr;
    std::vector<uint8_t> side_index;
    int side0_sms = 0;
    uint8_t *d_side_index = nullptr;
};

void build_side_info(SideInfo &si, void *base, size_t aligned_bytes) {
    int *d_sm = nullptr;
    size_t sm_bytes = 0;
    get_map_smid2side<int>(nullptr, sm_bytes);
    CHECK_CUDA(cudaMallocManaged(&d_sm, sm_bytes));
    get_map_smid2side(d_sm, sm_bytes);
    int num_sm = static_cast<int>(sm_bytes / sizeof(int));
    si.side_index.resize(num_sm);
    int c0 = 0, c1 = 0;
    for (int i = 0; i < num_sm; ++i) {
        int side = d_sm[i] & 1;
        int idx = side ? c1++ : c0++;
        si.side_index[i] = static_cast<uint8_t>((idx << 1) | side);
    }
    si.side0_sms = c0;
    CHECK_CUDA(cudaMalloc(&si.d_side_index, si.side_index.size() * sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(si.d_side_index, si.side_index.data(), si.side_index.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    int *d_attr = nullptr;
    uint32_t *d_mask = nullptr;
    size_t attr_bytes = 0;
    get_map_memory_attr<int>(base, aligned_bytes, nullptr, attr_bytes, nullptr, d_sm);
    CHECK_CUDA(cudaMallocManaged(&d_attr, attr_bytes));
    CHECK_CUDA(cudaMallocManaged(&d_mask, sizeof(uint32_t)));
    get_map_memory_attr<int>(base, aligned_bytes, d_attr, attr_bytes, d_mask, d_sm);
    si.mask = *d_mask;
    si.attr.assign(d_attr, d_attr + attr_bytes / sizeof(int));
    CHECK_CUDA(cudaFree(d_attr));
    CHECK_CUDA(cudaFree(d_mask));
    CHECK_CUDA(cudaFree(d_sm));
}

uint8_t predict_addr_side(uintptr_t base, uint32_t mask, const std::vector<int> &attr, uintptr_t addr) {
    size_t page = (addr - base) / kMemoryAllocGran;
    return static_cast<uint8_t>(predict_map_memory_attr_far(0, mask, addr, attr[page]) & 1);
}

template <typename T>
T *copy_to_device(const std::vector<T> &h) {
    T *d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty()) CHECK_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

__global__ void init_data(float *A, float *x, float *y0, float *y1, float *y2, int M, unsigned seed) {
    size_t total = static_cast<size_t>(M) * K;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t h = mix32(static_cast<uint32_t>(i) ^ seed);
#ifdef INTEGER_FLOAT_INPUTS
        A[i] = static_cast<float>(static_cast<int>(h % 17u) - 8);
#else
        A[i] = static_cast<float>(h & 0xffu) / 255.0f - 0.5f;
#endif
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += gridDim.x * blockDim.x) {
        uint32_t h = mix32(static_cast<uint32_t>(i) ^ (seed + 0x9e37u));
#ifdef INTEGER_FLOAT_INPUTS
        x[i] = static_cast<float>(static_cast<int>(h % 7u) - 3);
#else
        x[i] = static_cast<float>(h & 0xffu) / 255.0f - 0.5f;
#endif
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x) {
        y0[i] = y1[i] = y2[i] = 0.0f;
    }
}

__device__ __forceinline__ float subgroup_reduce(float v, int lane, int group_start) {
    unsigned mask = ((kLanesPerRow == 32 ? 0xffffffffu : ((1u << kLanesPerRow) - 1u)) << group_start);
    for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 32);
    }
    return v;
}

__device__ __forceinline__ float dot_subwarp(const float *A, const float *x, int row, int sub_lane) {
    const float4 *row4 = reinterpret_cast<const float4 *>(A + static_cast<size_t>(row) * K);
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    float sum = 0.0f;
    constexpr int K4 = K / 4;
#pragma unroll
    for (int k4 = sub_lane; k4 < K4; k4 += kLanesPerRow) {
        float4 a = row4[k4];
        float4 xv = x4[k4];
        sum += a.x * xv.x + a.y * xv.y + a.z * xv.z + a.w * xv.w;
    }
    return sum;
}

__global__ void gemv_partitioned_kernel(const float *A,
                                        const float *x,
                                        float *y,
                                        const int *rows0,
                                        int n0,
                                        const int *rows1,
                                        int n1,
                                        const uint8_t *side_index,
                                        int num_side0,
                                        int swapped) {
    uint8_t packed = side_index[__smid()];
    int sm_side = packed & 1;
    int sm_idx = packed >> 1;
    int logical_side = sm_side ^ (swapped & 1);
    const int *rows = logical_side ? rows1 : rows0;
    int n = logical_side ? n1 : n0;
    int my_sms = sm_side ? (static_cast<int>(gridDim.x) - num_side0) : num_side0;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int group_in_warp = lane / kLanesPerRow;
    int sub_lane = lane - group_in_warp * kLanesPerRow;
    int row_slot = warp * kRowsPerWarp + group_in_warp;

    for (int base = sm_idx * kRowsPerBlock; base < n; base += my_sms * kRowsPerBlock) {
        int pos = base + row_slot;
        if (pos < n) {
            int row = rows[pos];
            float sum = dot_subwarp(A, x, row, sub_lane);
            sum = subgroup_reduce(sum, lane, group_in_warp * kLanesPerRow);
            if (sub_lane == 0) y[row] = sum;
        }
    }
}

__global__ void gemv_baseline_kernel(const float *A,
                                     const float *x,
                                     float *y,
                                     const int *rows0,
                                     int n0,
                                     const int *rows1,
                                     int n1,
                                     const uint8_t *side_index,
                                     int num_side0) {
    uint8_t packed = side_index[__smid()];
    int sm_side = packed & 1;
    int sm_idx = packed >> 1;
    const int *rows = sm_side ? rows1 : rows0;
    int n = sm_side ? n1 : n0;
    int my_sms = sm_side ? (static_cast<int>(gridDim.x) - num_side0) : num_side0;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int group_in_warp = lane / kLanesPerRow;
    int sub_lane = lane - group_in_warp * kLanesPerRow;
    int row_slot = warp * kRowsPerWarp + group_in_warp;

    for (int base = sm_idx * kRowsPerBlock; base < n; base += my_sms * kRowsPerBlock) {
        int pos = base + row_slot;
        if (pos < n) {
            int row = rows[pos];
            float sum = dot_subwarp(A, x, row, sub_lane);
            sum = subgroup_reduce(sum, lane, group_in_warp * kLanesPerRow);
            if (sub_lane == 0) y[row] = sum;
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    int M = argi(argc, argv, 1, 2048);
    int runs = argi(argc, argv, 2, 100);
    int seed = argi(argc, argv, 3, 4242);
    int flush_l2 = argi(argc, argv, 4, 0);
    M = std::max(M, 1024);
    runs = std::max(runs, 1);
    flush_l2 = std::max(0, std::min(flush_l2, 1));

    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    auto A = alloc_aligned<float>(static_cast<size_t>(M) * K);
    auto x = alloc_aligned<float>(K);
    auto y_baseline = alloc_aligned<float>(M);
    auto y_aware = alloc_aligned<float>(M);
    auto y_swapped = alloc_aligned<float>(M);
    init_data<<<std::min(prop.multiProcessorCount * 8, 4096), 256>>>(
        A.ptr, x.ptr, y_baseline.ptr, y_aware.ptr, y_swapped.ptr, M, seed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    SideInfo si;
    build_side_info(si, A.ptr, A.aligned_bytes);
    uintptr_t base = reinterpret_cast<uintptr_t>(A.ptr);
    std::vector<int> side_rows[2];
    std::vector<int> baseline_rows[2];
    for (int row = 0; row < M; ++row) {
        uintptr_t addr = base + static_cast<uintptr_t>(row) * K * sizeof(float);
        int side = predict_addr_side(base, si.mask, si.attr, addr);
        side_rows[side].push_back(row);
        baseline_rows[row & 1].push_back(row);
    }
    int *d_side0 = copy_to_device(side_rows[0]);
    int *d_side1 = copy_to_device(side_rows[1]);
    int *d_baseline0 = copy_to_device(baseline_rows[0]);
    int *d_baseline1 = copy_to_device(baseline_rows[1]);

    int blocks = static_cast<int>(si.side_index.size());
    int side0_blocks = si.side0_sms;
    int side1_blocks = blocks - side0_blocks;

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto finish_prepare = [&]() {
        if (flush_l2) {
            GlobalL2Cache::reset();
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto prepare_baseline = [&]() {
        CHECK_CUDA(cudaMemset(y_baseline.ptr, 0, M * sizeof(float)));
        finish_prepare();
    };
    auto prepare_aware = [&]() {
        CHECK_CUDA(cudaMemset(y_aware.ptr, 0, M * sizeof(float)));
        finish_prepare();
    };
    auto prepare_swapped = [&]() {
        CHECK_CUDA(cudaMemset(y_swapped.ptr, 0, M * sizeof(float)));
        finish_prepare();
    };
    auto time_baseline = [&]() {
        CHECK_CUDA(cudaEventRecord(start));
        gemv_baseline_kernel<<<blocks, kThreads>>>(
            A.ptr,
            x.ptr,
            y_baseline.ptr,
            d_baseline0,
            static_cast<int>(baseline_rows[0].size()),
            d_baseline1,
            static_cast<int>(baseline_rows[1].size()),
            si.d_side_index,
            si.side0_sms);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        return event_ms(start, stop);
    };
    auto time_partitioned = [&](bool swapped) {
        float *y = swapped ? y_swapped.ptr : y_aware.ptr;
        CHECK_CUDA(cudaEventRecord(start));
        gemv_partitioned_kernel<<<blocks, kThreads>>>(
            A.ptr,
            x.ptr,
            y,
            d_side0,
            static_cast<int>(side_rows[0].size()),
            d_side1,
            static_cast<int>(side_rows[1].size()),
            si.d_side_index,
            si.side0_sms,
            swapped ? 1 : 0);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        return event_ms(start, stop);
    };

    for (int i = 0; i < 5; ++i) {
        prepare_baseline();
        time_baseline();
        prepare_aware();
        time_partitioned(false);
        prepare_swapped();
        time_partitioned(true);
    }
    float baseline = 0, aware = 0, swapped = 0;
    for (int r = 0; r < runs; ++r) {
        prepare_baseline();
        baseline += time_baseline();
        prepare_aware();
        aware += time_partitioned(false);
        prepare_swapped();
        swapped += time_partitioned(true);
    }
    baseline /= runs;
    aware /= runs;
    swapped /= runs;
    double flops = 2.0 * M * K;
    std::printf("=== Figure 11: K=1024 GEMV ===\n");
    std::printf(
        "gpu=%s M=%d lanes/row=%d warps/block=%d rows/block=%d flush_l2=%d side_counts=%zu/%zu side_sms=%d/%d\n",
        prop.name,
        M,
        kLanesPerRow,
        kWarpsPerBlock,
        kRowsPerBlock,
        flush_l2,
        side_rows[0].size(),
        side_rows[1].size(),
        side0_blocks,
        side1_blocks);
    std::printf("baseline_ms %.6f ms %.2f GFLOP/s\n", baseline, flops / (baseline * 1e6));
    std::printf("aware_ms %.6f ms aware_speedup=%.3fx %.2f GFLOP/s\n", aware, baseline / aware, flops / (aware * 1e6));
    std::printf("swapped_ms %.6f ms swapped_speedup=%.3fx %.2f GFLOP/s\n", swapped, baseline / swapped, flops / (swapped * 1e6));

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_x(K);
    std::vector<float> h_ref(M, 0.0f);
    std::vector<float> h_baseline(M), h_aware(M), h_swapped(M);
    CHECK_CUDA(cudaMemcpy(h_A.data(), A.ptr, h_A.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_x.data(), x.ptr, h_x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_baseline.data(), y_baseline.ptr, M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_aware.data(), y_aware.ptr, M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_swapped.data(), y_swapped.ptr, M * sizeof(float), cudaMemcpyDeviceToHost));
    for (int row = 0; row < M; ++row) {
        float sum = 0.0f;
        const float *a = h_A.data() + static_cast<size_t>(row) * K;
        for (int k = 0; k < K; ++k) {
            sum += a[k] * h_x[k];
        }
        h_ref[row] = sum;
    }
    auto max_abs_error = [&](const std::vector<float> &y) {
        float err = 0.0f;
        for (int i = 0; i < M; ++i) {
            err = std::max(err, std::fabs(y[i] - h_ref[i]));
        }
        return err;
    };
    std::printf(
        "correctness max_abs_error baseline=%.6g aware=%.6g swapped=%.6g\n",
        max_abs_error(h_baseline),
        max_abs_error(h_aware),
        max_abs_error(h_swapped));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_side0));
    CHECK_CUDA(cudaFree(d_side1));
    CHECK_CUDA(cudaFree(d_baseline0));
    CHECK_CUDA(cudaFree(d_baseline1));
    CHECK_CUDA(cudaFree(si.d_side_index));
    CHECK_CUDA(cudaFree(A.raw));
    CHECK_CUDA(cudaFree(x.raw));
    CHECK_CUDA(cudaFree(y_baseline.raw));
    CHECK_CUDA(cudaFree(y_aware.raw));
    CHECK_CUDA(cudaFree(y_swapped.raw));
    return 0;
}
