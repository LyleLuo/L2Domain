#include "latopt.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t st = (call);                                               \
        if (st != CUBLAS_STATUS_SUCCESS) {                                        \
            std::fprintf(stderr, "[cuBLAS] %s:%d failed: %d\n", __FILE__, __LINE__, static_cast<int>(st)); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void init_matrix_vector(float *A, float *x, float *y, int M, int K, unsigned seed) {
    size_t total = static_cast<size_t>(M) * K;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t h = mix32(static_cast<uint32_t>(i) ^ seed);
        A[i] = (static_cast<float>(h & 0xffu) / 255.0f) - 0.5f;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += gridDim.x * blockDim.x) {
        uint32_t h = mix32(static_cast<uint32_t>(i) ^ (seed + 0x9e37u));
        x[i] = (static_cast<float>(h & 0xffu) / 255.0f) - 0.5f;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x) {
        y[i] = 0.0f;
    }
}

int main(int argc, char **argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 65536;
    int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    int runs = argc > 3 ? std::atoi(argv[3]) : 20;
    int warmups = argc > 4 ? std::atoi(argv[4]) : 5;
    int flush_l2 = argc > 5 ? std::atoi(argv[5]) : 0;
    M = std::max(M, 1);
    K = std::max(K, 1);
    runs = std::max(runs, 1);
    warmups = std::max(warmups, 0);
    flush_l2 = std::max(0, std::min(flush_l2, 1));

    cudaDeviceProp prop{};
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    float *A = nullptr, *x = nullptr, *y = nullptr;
    CHECK_CUDA(cudaMalloc(&A, static_cast<size_t>(M) * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&x, K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y, M * sizeof(float)));
    init_matrix_vector<<<std::min(prop.multiProcessorCount * 8, 4096), 256>>>(A, x, y, M, K, 4242);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;

    // A is row-major MxK. cuBLAS expects column-major, so row-major A times x is
    // equivalent to y = A_col^T x where A_col is KxM column-major with the same memory.
    for (int i = 0; i < warmups; ++i) {
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, K, M, &alpha, A, K, x, 1, &beta, y, 1));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        if (flush_l2) {
            latopt::GlobalL2Cache::reset();
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, K, M, &alpha, A, K, x, 1, &beta, y, 1));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    float avg_ms = total_ms / static_cast<float>(runs);
    double flops = 2.0 * static_cast<double>(M) * K;

    std::printf("=== cuBLAS SGEMV ===\n");
    std::printf("gpu=%s M=%d K=%d runs=%d flush_l2=%d matrix=%.1f MiB\n", prop.name, M, K, runs, flush_l2, static_cast<double>(M) * K * sizeof(float) / (1024.0 * 1024.0));
    std::printf("cublas_ms %.6f ms %.2f GFLOP/s\n", avg_ms, flops / (avg_ms * 1.0e6));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaFree(y));
    return 0;
}
