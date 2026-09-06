#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "latopt.cuh"

#include <cutlass/array.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/scale_type.h>

using CutlassRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    float,
    4,
    float,
    float,
    cutlass::epilogue::thread::ScaleType::Nothing>;

__device__ __forceinline__ float generated_value(size_t index, unsigned seed) {
    unsigned x = static_cast<unsigned>(index) ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<float>(x & 0xffffu) / 32768.0f - 1.0f;
}

__global__ void init_data(float *data, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        data[i] = generated_value(i, seed);
    }
}

__global__ void touch_k(const float *data, size_t n) {
    float sink = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        sink += data[i];
    }
    if (sink == -999.0f) {
        asm volatile("" ::: "memory");
    }
}

__global__ void cutlass_epilogue_relu_kernel(float *__restrict__ data, int n_float4) {
    float4 *ptr4 = reinterpret_cast<float4 *>(data);
    CutlassRelu op(CutlassRelu::Params{});
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_float4; idx += gridDim.x * blockDim.x) {
        float4 v = ptr4[idx];
        cutlass::Array<float, 4> frag;
        frag[0] = v.x;
        frag[1] = v.y;
        frag[2] = v.z;
        frag[3] = v.w;
        cutlass::Array<float, 4> out = op(frag);
        ptr4[idx] = make_float4(out[0], out[1], out[2], out[3]);
    }
}

__global__ void validate_relu(const float *data, size_t n, unsigned seed, unsigned long long *mismatches) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float input = generated_value(i, seed);
        float expected = input > 0.0f ? input : 0.0f;
        if (data[i] != expected) atomicAdd(mismatches, 1ULL);
    }
}

int main(int argc, char **argv) {
    int data_mb = argc > 1 ? std::atoi(argv[1]) : 1024;
    int runs = argc > 2 ? std::atoi(argv[2]) : 20;
    int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    int blocks_arg = argc > 4 ? std::atoi(argv[4]) : 0;
    int warmups = argc > 5 ? std::atoi(argv[5]) : 5;
    int cache_mode = argc > 6 ? std::atoi(argv[6]) : 1; // 0=cold, 1=warm after flush

    if (data_mb <= 0 || runs <= 0 || threads <= 0 || warmups < 0) {
        std::fprintf(stderr, "usage: %s <data_mb> <runs> <threads> <blocks|0=auto> <warmups> <cache_mode:0=cold,1=warm>\n", argv[0]);
        return 2;
    }
    cache_mode = std::max(0, std::min(cache_mode, 1));

    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    size_t n = static_cast<size_t>(data_mb) * 1024ULL * 1024ULL / sizeof(float);
    if (n % 4 != 0) {
        std::fprintf(stderr, "number of floats must be divisible by 4\n");
        return 2;
    }
    int n_float4 = static_cast<int>(n / 4);

    float *d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, n * sizeof(float)));

    int init_blocks = std::min(prop.multiProcessorCount * 8, 4096);
    int blocks = blocks_arg > 0 ? blocks_arg : std::min((n_float4 + threads - 1) / threads, prop.multiProcessorCount * 8);

    auto restore = [&](unsigned seed) {
        init_data<<<init_blocks, 256>>>(d, n, seed);
        CHECK_CUDA(cudaGetLastError());
        latopt::GlobalL2Cache::reset();
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto warm = [&]() {
        touch_k<<<blocks, threads>>>(d, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    auto run_relu = [&]() {
        cutlass_epilogue_relu_kernel<<<blocks, threads>>>(d, n_float4);
    };

    constexpr unsigned correctness_seed = 4242;
    unsigned long long *d_mismatches = nullptr;
    CHECK_CUDA(cudaMalloc(&d_mismatches, sizeof(*d_mismatches)));
    CHECK_CUDA(cudaMemset(d_mismatches, 0, sizeof(*d_mismatches)));
    restore(correctness_seed);
    run_relu();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    validate_relu<<<init_blocks, 256>>>(d, n, correctness_seed, d_mismatches);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long mismatches = 0;
    CHECK_CUDA(cudaMemcpy(&mismatches, d_mismatches, sizeof(mismatches), cudaMemcpyDeviceToHost));
    bool correct = mismatches == 0;

    for (int i = 0; i < warmups; ++i) {
        restore(1000u + static_cast<unsigned>(i));
        if (cache_mode) warm();
        run_relu();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        restore(2000u + static_cast<unsigned>(i));
        if (cache_mode) warm();
        CHECK_CUDA(cudaEventRecord(start));
        run_relu();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaGetLastError());
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    float avg_ms = total_ms / static_cast<float>(runs);
    double gb = static_cast<double>(data_mb) / 1024.0;
    double effective_gbs = (2.0 * gb) / (avg_ms / 1000.0);

    std::printf("=== Standalone epilogue-style elementwise ReLU ===\n");
    std::printf("implementation=standalone_elementwise_kernel activation=cutlass::epilogue::thread::LinearCombinationRelu full_cutlass_device_kernel=no\n");
    std::printf("gpu=%s data=%dMB runs=%d threads=%d blocks=%d cache_mode=%s elems=%zu\n", prop.name, data_mb, runs, threads, blocks, cache_mode ? "warm" : "cold", n);
    std::printf("correct=%s\n", correct ? "PASS" : "FAIL");
    std::printf("cutlass_epilogue_relu %.6f ms effective_bw=%.2f GB/s\n", avg_ms, effective_gbs);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_mismatches));
    CHECK_CUDA(cudaFree(d));
    return correct ? 0 : 3;
}
