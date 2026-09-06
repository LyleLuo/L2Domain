#include "latopt.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace latopt;

namespace {

constexpr size_t kChunkBytes = 4096;
constexpr int kChunksPerPage = kMemoryAllocGran / kChunkBytes;
constexpr int kSideChunksPerPage = kChunksPerPage / 2;
constexpr int kFloatsPerChunk = static_cast<int>(kChunkBytes / sizeof(float));
constexpr int kFloat4PerChunk = kFloatsPerChunk / 4;
constexpr int kWarmups = 5;

__constant__ uint16_t c_offsets_by_parity[2 * kSideChunksPerPage];

std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

__device__ __forceinline__ float4 relu4(float4 value) {
    value.x = value.x > 0.0f ? value.x : 0.0f;
    value.y = value.y > 0.0f ? value.y : 0.0f;
    value.z = value.z > 0.0f ? value.z : 0.0f;
    value.w = value.w > 0.0f ? value.w : 0.0f;
    return value;
}

__global__ void touch_k(const float *data, int n) {
    float sink = 0.0f;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += gridDim.x * blockDim.x) {
        sink += data[idx];
    }
    if (sink == -999.0f) asm volatile("" ::: "memory");
}

__global__ void relu_contiguous_vec(float *__restrict__ data, int n_float4) {
    float4 *ptr4 = reinterpret_cast<float4 *>(data);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_float4; idx += gridDim.x * blockDim.x) {
        ptr4[idx] = relu4(ptr4[idx]);
    }
}

__global__ void relu_persistent_predicate_vec(float *__restrict__ data,
                                               const uint8_t *__restrict__ attr,
                                               int num_pages,
                                               const uint8_t *__restrict__ side_index,
                                               int num_side0) {
    float4 *ptr4 = reinterpret_cast<float4 *>(data);
    uint8_t packed = side_index[__smid()];
    int side = packed & 1;
    int side_rank = packed >> 1;
    int side_sms = side == 0 ? num_side0 : static_cast<int>(gridDim.x) - num_side0;
    int total_side_chunks = num_pages * kSideChunksPerPage;
    int chunks_per_block = blockDim.x / kFloat4PerChunk;

    if (chunks_per_block > 1) {
        int lane = threadIdx.x / kFloat4PerChunk;
        int elem4 = threadIdx.x - lane * kFloat4PerChunk;
        for (int ord_base = side_rank * chunks_per_block;
             ord_base < total_side_chunks;
             ord_base += side_sms * chunks_per_block) {
            int ord = ord_base + lane;
            if (ord < total_side_chunks) {
                int page = ord >> 8;
                int slot = ord & (kSideChunksPerPage - 1);
                int parity = side ^ (attr[page] & 1);
                int chunk_in_page = c_offsets_by_parity[parity * kSideChunksPerPage + slot];
                int base4 = (page * kChunksPerPage + chunk_in_page) * kFloat4PerChunk;
                ptr4[base4 + elem4] = relu4(ptr4[base4 + elem4]);
            }
        }
        return;
    }

    for (int ord = side_rank; ord < total_side_chunks; ord += side_sms) {
        int page = ord >> 8;
        int slot = ord & (kSideChunksPerPage - 1);
        int parity = side ^ (attr[page] & 1);
        int chunk_in_page = c_offsets_by_parity[parity * kSideChunksPerPage + slot];
        int base4 = (page * kChunksPerPage + chunk_in_page) * kFloat4PerChunk;
        for (int elem4 = threadIdx.x; elem4 < kFloat4PerChunk; elem4 += blockDim.x) {
            ptr4[base4 + elem4] = relu4(ptr4[base4 + elem4]);
        }
    }
}

struct AlignedMem {
    uint8_t *raw = nullptr;
    uintptr_t base = 0;
    size_t aligned_size = 0;
    float *data() { return reinterpret_cast<float *>(base); }
};

struct SideInfo {
    int num_sm = 0;
    int num_side0 = 0;
    size_t num_pages = 0;
    uint32_t mask = 0;
    uint8_t *d_side_index = nullptr;
    uint8_t *d_attr = nullptr;
};

AlignedMem alloc_aligned(size_t bytes) {
    AlignedMem mem;
    mem.aligned_size = ceil_div(bytes, kMemoryAllocGran) * kMemoryAllocGran;
    CHECK_CUDA(cudaMalloc(&mem.raw, mem.aligned_size + kMemoryAllocGran));
    CHECK_CUDA(cudaMemset(mem.raw, 0, mem.aligned_size + kMemoryAllocGran));
    mem.base = (reinterpret_cast<uintptr_t>(mem.raw) + kMemoryAllocGran - 1) & ~(uintptr_t)(kMemoryAllocGran - 1);
    return mem;
}

void build_side_info(SideInfo &info, uintptr_t base, size_t bytes) {
    int *d_smid2side = nullptr;
    size_t smid2side_bytes = 0;
    get_map_smid2side<int>(nullptr, smid2side_bytes);
    CHECK_CUDA(cudaMallocManaged(&d_smid2side, smid2side_bytes));
    get_map_smid2side(d_smid2side, smid2side_bytes);

    info.num_sm = static_cast<int>(smid2side_bytes / sizeof(int));
    std::vector<uint8_t> side_index(info.num_sm);
    int counts[2] = {0, 0};
    for (int sm = 0; sm < info.num_sm; ++sm) {
        int side = d_smid2side[sm] & 1;
        side_index[sm] = static_cast<uint8_t>((counts[side]++ << 1) | side);
    }
    info.num_side0 = counts[0];
    CHECK_CUDA(cudaMalloc(&info.d_side_index, side_index.size() * sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(info.d_side_index, side_index.data(), side_index.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    int *attr = nullptr;
    uint32_t *mask = nullptr;
    size_t attr_bytes = 0;
    get_map_memory_attr<int>(reinterpret_cast<void *>(base), bytes, nullptr, attr_bytes, nullptr, d_smid2side);
    CHECK_CUDA(cudaMallocManaged(&attr, attr_bytes));
    CHECK_CUDA(cudaMallocManaged(&mask, sizeof(uint32_t)));
    get_map_memory_attr<int>(reinterpret_cast<void *>(base), bytes, attr, attr_bytes, mask, d_smid2side);
    info.mask = *mask;
    info.num_pages = attr_bytes / sizeof(int);

    std::vector<uint8_t> attr_u8(info.num_pages);
    for (size_t page = 0; page < info.num_pages; ++page) attr_u8[page] = static_cast<uint8_t>(attr[page] & 1);
    CHECK_CUDA(cudaMalloc(&info.d_attr, attr_u8.size() * sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(info.d_attr, attr_u8.data(), attr_u8.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    std::vector<uint16_t> offsets(2 * kSideChunksPerPage);
    int parity_counts[2] = {0, 0};
    for (int chunk = 0; chunk < kChunksPerPage; ++chunk) {
        uintptr_t address = base + static_cast<uintptr_t>(chunk) * kChunkBytes;
        int parity = __builtin_popcountll(static_cast<unsigned long long>(address & info.mask)) & 1;
        offsets[parity * kSideChunksPerPage + parity_counts[parity]++] = static_cast<uint16_t>(chunk);
    }
    LATOPT_CHECK(parity_counts[0] == kSideChunksPerPage && parity_counts[1] == kSideChunksPerPage,
                 "expected balanced page offsets");
    CHECK_CUDA(cudaMemcpyToSymbol(c_offsets_by_parity, offsets.data(), offsets.size() * sizeof(uint16_t)));

    CHECK_CUDA(cudaFree(d_smid2side));
    CHECK_CUDA(cudaFree(attr));
    CHECK_CUDA(cudaFree(mask));
}

} // namespace

int main(int argc, char **argv) {
    int data_mb = argc > 1 ? std::atoi(argv[1]) : 256;
    int runs = argc > 2 ? std::atoi(argv[2]) : 5;
    int threads = argc > 3 ? std::atoi(argv[3]) : 1024;
    int cache_mode = argc > 4 ? std::atoi(argv[4]) : 1;
    if (argc > 5 || data_mb <= 0 || runs <= 0 || threads <= 0 || threads > 1024 || (cache_mode != 0 && cache_mode != 1)) {
        std::fprintf(stderr, "usage: %s <data_mb> <runs> <threads> <cache_mode:0=cold,1=warm>\n", argv[0]);
        return 2;
    }

    size_t data_elems = static_cast<size_t>(data_mb) * 1024ULL * 1024ULL / sizeof(float);
    LATOPT_CHECK(data_elems % 4 == 0, "data elements must be divisible by 4");
    size_t data_bytes = data_elems * sizeof(float);
    int n_float4 = static_cast<int>(data_elems / 4);

    AlignedMem mem = alloc_aligned(data_bytes);
    std::vector<float> input(data_elems);
    for (float &value : input) value = dist(rng);
    CHECK_CUDA(cudaMemcpy(mem.data(), input.data(), data_bytes, cudaMemcpyHostToDevice));

    SideInfo side;
    build_side_info(side, mem.base, mem.aligned_size);
    int persistent_blocks = side.num_sm;

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto restore = [&]() {
        CHECK_CUDA(cudaMemcpy(mem.data(), input.data(), data_bytes, cudaMemcpyHostToDevice));
        GlobalL2Cache::reset();
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto warm = [&]() {
        touch_k<<<persistent_blocks, threads>>>(mem.data(), static_cast<int>(data_elems));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto run_baseline = [&]() {
        relu_contiguous_vec<<<persistent_blocks, threads>>>(mem.data(), n_float4);
        CHECK_CUDA(cudaGetLastError());
    };
    auto run_ours = [&]() {
        relu_persistent_predicate_vec<<<persistent_blocks, threads>>>(mem.data(),
                                                                      side.d_attr,
                                                                      static_cast<int>(side.num_pages),
                                                                      side.d_side_index,
                                                                      side.num_side0);
        CHECK_CUDA(cudaGetLastError());
    };
    auto time = [&](auto launch) {
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        return elapsed_ms;
    };

    bool correct = false;
    {
        restore();
        run_baseline();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> baseline_output(data_elems);
        CHECK_CUDA(cudaMemcpy(baseline_output.data(), mem.data(), data_bytes, cudaMemcpyDeviceToHost));

        restore();
        run_ours();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<float> our_output(data_elems);
        CHECK_CUDA(cudaMemcpy(our_output.data(), mem.data(), data_bytes, cudaMemcpyDeviceToHost));
        correct = baseline_output == our_output;
    }

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        restore(); if (cache_mode) warm(); run_ours();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float our_ms = 0.0f;
    for (int run = 0; run < runs; ++run) {
        restore(); if (cache_mode) warm(); our_ms += time(run_ours);
    }
    our_ms /= static_cast<float>(runs);

    std::printf("config data_mb=%d runs=%d threads=%d persistent_blocks=%d cache_mode=%d\n",
                data_mb, runs, threads, persistent_blocks, cache_mode);
    std::printf("correct=%s\n", correct ? "PASS" : "FAIL");
    std::printf("our_ms=%.6f\n", our_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(side.d_side_index));
    CHECK_CUDA(cudaFree(side.d_attr));
    CHECK_CUDA(cudaFree(mem.raw));
    return correct ? 0 : 3;
}
