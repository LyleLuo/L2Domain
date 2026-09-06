#include "sidedgpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

constexpr size_t kPageBytes = 4096;
constexpr int kKernelCtas = 32;

struct SidedAllocation {
    std::vector<size_t> pages;
    size_t bytes = 0;
};

struct CaseResult {
    float library_ms = 0.0f;
    float aware_ms = 0.0f;
    float swapped_ms = 0.0f;
};

__device__ __forceinline__ unsigned char *page_ptr(unsigned char *pool, const size_t *pages, size_t offset) {
    size_t page_id     = offset / kPageBytes;
    size_t page_offset = offset % kPageBytes;
    return pool + pages[page_id] * kPageBytes + page_offset;
}

__global__ void init_chunks_kernel(unsigned char *pool, const size_t *src_pages, const size_t *dst_pages, int chunks, size_t chunk_bytes) {
    int tid   = threadIdx.x;

    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        size_t base = static_cast<size_t>(chunk) * chunk_bytes;
        for (size_t offset = tid; offset < chunk_bytes; offset += blockDim.x) {
            unsigned char value = static_cast<unsigned char>((chunk * 131 + offset) & 0xff);
            *page_ptr(pool, src_pages, base + offset) = value;
            *page_ptr(pool, dst_pages, base + offset) = 0;
        }
    }
}

__global__ void copy_chunks_kernel(unsigned char *pool, const size_t *src_pages, const size_t *dst_pages, int chunks, size_t chunk_bytes) {
    int tid   = threadIdx.x;

    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        size_t base = static_cast<size_t>(chunk) * chunk_bytes;
        for (size_t offset = tid; offset < chunk_bytes; offset += blockDim.x) {
            *page_ptr(pool, dst_pages, base + offset) = *page_ptr(pool, src_pages, base + offset);
        }
    }
}

__global__ void validate_chunks_kernel(unsigned char *pool, const size_t *src_pages, const size_t *dst_pages, int chunks, size_t chunk_bytes, int *errors) {
    int tid   = threadIdx.x;

    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        size_t base = static_cast<size_t>(chunk) * chunk_bytes;
        for (size_t offset = tid; offset < chunk_bytes; offset += blockDim.x) {
            unsigned char s = *page_ptr(pool, src_pages, base + offset);
            unsigned char d = *page_ptr(pool, dst_pages, base + offset);
            if (s != d) {
                atomicAdd(errors, 1);
            }
        }
    }
}

__global__ void zero_chunks_kernel(unsigned char *pool, const size_t *dst_pages, int chunks, size_t chunk_bytes) {
    int tid   = threadIdx.x;

    for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        size_t base = static_cast<size_t>(chunk) * chunk_bytes;
        for (size_t offset = tid; offset < chunk_bytes; offset += blockDim.x) {
            *page_ptr(pool, dst_pages, base + offset) = 0;
        }
    }
}

__global__ void flush_l2_kernel(unsigned char *buf, size_t bytes) {
    size_t idx    = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * 128;
    unsigned int acc = 0;
    for (size_t offset = idx; offset < bytes; offset += stride) {
        acc += buf[offset];
    }
    if (idx == 0) {
        buf[0] = static_cast<unsigned char>(acc);
    }
}

SidedAllocation sided_alloc(size_t bytes, int side) {
    SidedAllocation allocation;
    allocation.bytes = bytes;
    CHECK_CUDA(cudaMallocSided(bytes, side, allocation.pages));
    return allocation;
}

void copy_page_ids(size_t **device_ids, const std::vector<size_t> &host_ids) {
    CHECK_CUDA(cudaMalloc(device_ids, host_ids.size() * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(*device_ids, host_ids.data(), host_ids.size() * sizeof(size_t), cudaMemcpyHostToDevice));
}

std::vector<void *> make_ptrs(unsigned char *pool, const std::vector<size_t> &pages, int chunks, size_t chunk_bytes) {
    std::vector<void *> ptrs(chunks);
    for (int i = 0; i < chunks; ++i) {
        size_t offset = static_cast<size_t>(i) * chunk_bytes;
        size_t page = offset / kPageBytes;
        size_t page_offset = offset % kPageBytes;
        ptrs[i] = pool + pages[page] * kPageBytes + page_offset;
    }
    return ptrs;
}

float time_kernel_copy(cudaStream_t stream,
                       unsigned char *pool,
                       size_t *src_pages,
                       size_t *dst_pages,
                       int chunks,
                       size_t chunk_bytes,
                       unsigned char *l2_flush,
                       size_t l2_flush_bytes,
                       int timed_runs) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float elapsed = 0.0f;
    for (int run = 0; run < timed_runs; ++run) {
        zero_chunks_kernel<<<kKernelCtas, 256, 0, stream>>>(pool, dst_pages, chunks, chunk_bytes);
        CHECK_CUDA(cudaGetLastError());
        flush_l2_kernel<<<256, 256, 0, stream>>>(l2_flush, l2_flush_bytes);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(start, stream));
        copy_chunks_kernel<<<kKernelCtas, 256, 0, stream>>>(pool, src_pages, dst_pages, chunks, chunk_bytes);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float run_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&run_ms, start, stop));
        elapsed += run_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsed / static_cast<float>(timed_runs);
}

float time_batch_copy(cudaStream_t stream,
                      const std::vector<void *> &dst_ptrs,
                      const std::vector<void *> &src_ptrs,
                      const std::vector<size_t> &sizes,
                      unsigned char *pool,
                      size_t *dst_pages,
                      int chunks,
                      size_t chunk_bytes,
                      unsigned char *l2_flush,
                      size_t l2_flush_bytes,
                      int timed_runs) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

    float elapsed = 0.0f;
    for (int run = 0; run < timed_runs; ++run) {
        zero_chunks_kernel<<<kKernelCtas, 256, 0, stream>>>(pool, dst_pages, chunks, chunk_bytes);
        CHECK_CUDA(cudaGetLastError());
        flush_l2_kernel<<<256, 256, 0, stream>>>(l2_flush, l2_flush_bytes);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(start, stream));
        CHECK_CUDA(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), dst_ptrs.size(), attr, stream));
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float run_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&run_ms, start, stop));
        elapsed += run_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return elapsed / static_cast<float>(timed_runs);
}

void validate_copy(unsigned char *pool, size_t *src_pages, size_t *dst_pages, int chunks, size_t chunk_bytes) {
    int *d_errors = nullptr;
    CHECK_CUDA(cudaMalloc(&d_errors, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_errors, 0, sizeof(int)));
    validate_chunks_kernel<<<kKernelCtas, 256>>>(pool, src_pages, dst_pages, chunks, chunk_bytes, d_errors);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    int h_errors = 0;
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_errors));
    if (h_errors != 0) {
        std::fprintf(stderr, "validation failed with %d mismatches\n", h_errors);
        std::exit(EXIT_FAILURE);
    }
}

CaseResult run_case(int chunks,
                    size_t chunk_bytes,
                    int timed_runs,
                    unsigned char *pool,
                    unsigned char *l2_flush,
                    size_t l2_flush_bytes,
                    cudaStream_t batch_stream,
                    cudaStream_t side0_stream,
                    cudaStream_t side1_stream) {
    size_t total_bytes = static_cast<size_t>(chunks) * chunk_bytes;
    auto src = sided_alloc(total_bytes, 0);
    auto dst = sided_alloc(total_bytes, 0);

    size_t *d_src_pages = nullptr;
    size_t *d_dst_pages = nullptr;
    copy_page_ids(&d_src_pages, src.pages);
    copy_page_ids(&d_dst_pages, dst.pages);

    init_chunks_kernel<<<kKernelCtas, 256>>>(pool, d_src_pages, d_dst_pages, chunks, chunk_bytes);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    auto src_ptrs = make_ptrs(pool, src.pages, chunks, chunk_bytes);
    auto dst_ptrs = make_ptrs(pool, dst.pages, chunks, chunk_bytes);
    std::vector<size_t> sizes(chunks, chunk_bytes);

    CaseResult result;
    result.library_ms = time_batch_copy(batch_stream, dst_ptrs, src_ptrs, sizes, pool, d_dst_pages, chunks, chunk_bytes, l2_flush, l2_flush_bytes, timed_runs);
    validate_copy(pool, d_src_pages, d_dst_pages, chunks, chunk_bytes);
    result.aware_ms = time_kernel_copy(side0_stream, pool, d_src_pages, d_dst_pages, chunks, chunk_bytes, l2_flush, l2_flush_bytes, timed_runs);
    validate_copy(pool, d_src_pages, d_dst_pages, chunks, chunk_bytes);
    result.swapped_ms = time_kernel_copy(side1_stream, pool, d_src_pages, d_dst_pages, chunks, chunk_bytes, l2_flush, l2_flush_bytes, timed_runs);
    validate_copy(pool, d_src_pages, d_dst_pages, chunks, chunk_bytes);

    CHECK_CUDA(cudaFree(d_src_pages));
    CHECK_CUDA(cudaFree(d_dst_pages));
    CHECK_CUDA(cudaFreeSided(src.pages));
    CHECK_CUDA(cudaFreeSided(dst.pages));
    return result;
}

int parse_positive(const char *value, int fallback) {
    if (value == nullptr) {
        return fallback;
    }
    int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

int main(int argc, char **argv) {
    int const chunks = argc > 1 ? parse_positive(argv[1], 4096) : 4096;
    int const chunk_bytes_arg = argc > 2 ? parse_positive(argv[2], 4096) : 4096;
    int const timed_runs = argc > 3 ? parse_positive(argv[3], 20) : 20;
    size_t const chunk_bytes = static_cast<size_t>(chunk_bytes_arg);
    if (chunk_bytes % kPageBytes != 0) {
        std::fprintf(stderr, "chunk_bytes must be a positive multiple of 4096\n");
        return EXIT_FAILURE;
    }

    void *pool_base = nullptr;
    CHECK_CUDA(cudaGetSidedPoolBase(&pool_base));
    auto *pool = static_cast<unsigned char *>(pool_base);

    cudaStream_t batch_stream = nullptr;
    cudaStream_t side0_stream = nullptr;
    cudaStream_t side1_stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&batch_stream));
    CHECK_CUDA(SidedGreenContext::create_stream(0, &side0_stream));
    CHECK_CUDA(SidedGreenContext::create_stream(1, &side1_stream));

    cudaDeviceProp prop{};
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    size_t const l2_flush_bytes = static_cast<size_t>(prop.l2CacheSize) * 2;
    unsigned char *l2_flush = nullptr;
    CHECK_CUDA(cudaMalloc(&l2_flush, l2_flush_bytes));
    CHECK_CUDA(cudaMemset(l2_flush, 1, l2_flush_bytes));

    CaseResult result = run_case(chunks, chunk_bytes, timed_runs, pool, l2_flush, l2_flush_bytes, batch_stream, side0_stream, side1_stream);

    std::printf("batched memcpy: chunks=%d chunk_bytes=%zu total=%.2f MiB kernel_ctas=%d timed_runs=%d\n",
                chunks,
                chunk_bytes,
                static_cast<double>(chunks) * static_cast<double>(chunk_bytes) / (1024.0 * 1024.0),
                kKernelCtas,
                timed_runs);
    std::printf("| allocation side | cudaMemcpyBatchAsync (ms) | aware (ms) | swapped (ms) |\n");
    std::printf("|---:|---:|---:|---:|\n");
    std::printf("| 0 | %.6f | %.6f | %.6f |\n", result.library_ms, result.aware_ms, result.swapped_ms);
    std::printf("RESULT,0,%d,%.6f,%.6f,%.6f\n", kKernelCtas, result.library_ms, result.aware_ms, result.swapped_ms);

    CHECK_CUDA(cudaFree(l2_flush));
    CHECK_CUDA(cudaStreamDestroy(batch_stream));
    CHECK_CUDA(cudaStreamDestroy(side0_stream));
    CHECK_CUDA(cudaStreamDestroy(side1_stream));
    return EXIT_SUCCESS;
}
