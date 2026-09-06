#include "latopt.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

using namespace latopt;

namespace {

constexpr size_t kChunkBytes = 4096;
constexpr int kWordsPerChunk = static_cast<int>(kChunkBytes / sizeof(uint32_t));
constexpr int kBins = 256;
constexpr int kMergeThreads = 256;

__device__ __forceinline__ void add_word(unsigned int *hist, uint32_t word) {
    atomicAdd(hist + ((word >> 0) & 0xffu), 1);
    atomicAdd(hist + ((word >> 8) & 0xffu), 1);
    atomicAdd(hist + ((word >> 16) & 0xffu), 1);
    atomicAdd(hist + ((word >> 24) & 0xffu), 1);
}

__global__ void touch_k(const uint32_t *data, size_t words) {
    uint32_t sink = 0;
    for (size_t idx = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
         idx < words;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        sink += data[idx];
    }
    if (sink == 0xdeadbeefu) {
        asm volatile("" ::: "memory");
    }
}

__global__ void flush_l2_k(uint8_t *buf, size_t bytes) {
    size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * 128;
    uint32_t sink = 0;
    for (size_t offset = idx; offset < bytes; offset += stride) {
        sink += buf[offset];
    }
    if (idx == 0 && sink == 0x12345678u) {
        buf[0] = static_cast<uint8_t>(sink);
    }
}

__global__ void histogram256_chunked_kernel(const uint32_t *__restrict__ data,
                                            unsigned int *__restrict__ partials,
                                            const int *__restrict__ chunks,
                                            int n_chunks,
                                            size_t total_words) {
    extern __shared__ unsigned int shared_hist[];
    int warp = threadIdx.x >> 5;
    int lane_hist_offset = warp * kBins;
    int warp_count = (blockDim.x + 31) >> 5;

    for (int i = threadIdx.x; i < warp_count * kBins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    for (int chunk_idx = blockIdx.x; chunk_idx < n_chunks; chunk_idx += gridDim.x) {
        size_t base_word = static_cast<size_t>(chunks[chunk_idx]) * kWordsPerChunk;
        if (base_word >= total_words) continue;
        size_t remaining = total_words - base_word;
        int words = static_cast<int>(remaining < static_cast<size_t>(kWordsPerChunk) ? remaining : static_cast<size_t>(kWordsPerChunk));
        for (int elem = threadIdx.x; elem < words; elem += blockDim.x) {
            add_word(shared_hist + lane_hist_offset, data[base_word + elem]);
        }
    }
    __syncthreads();

    unsigned int *out = partials + static_cast<size_t>(blockIdx.x) * kBins;
    for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warp_count; ++w) {
            sum += shared_hist[w * kBins + bin];
        }
        out[bin] = sum;
    }
}

__global__ void histogram256_persistent_kernel(const uint32_t *__restrict__ data,
                                               unsigned int *__restrict__ partials,
                                               const int *__restrict__ near_chunks,
                                               int n_near,
                                               const int *__restrict__ far_chunks,
                                               int n_far,
                                               const uint16_t *__restrict__ side_index,
                                               int num_side0,
                                               size_t total_words,
                                               int reverse) {
    extern __shared__ unsigned int shared_hist[];
    int warp = threadIdx.x >> 5;
    int lane_hist_offset = warp * kBins;
    int warp_count = (blockDim.x + 31) >> 5;

    for (int i = threadIdx.x; i < warp_count * kBins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    uint16_t packed = side_index[__smid()];
    int physical_side = packed & 1;
    int side_rank = packed >> 1;
    bool use_far = (physical_side ^ (reverse & 1)) != 0;
    const int *chunks = use_far ? far_chunks : near_chunks;
    int n_chunks = use_far ? n_far : n_near;
    int side_sms = physical_side == 0 ? num_side0 : (static_cast<int>(gridDim.x) - num_side0);

    for (int chunk_idx = side_rank; chunk_idx < n_chunks; chunk_idx += side_sms) {
        size_t base_word = static_cast<size_t>(chunks[chunk_idx]) * kWordsPerChunk;
        if (base_word >= total_words) continue;
        size_t remaining = total_words - base_word;
        int words = static_cast<int>(remaining < static_cast<size_t>(kWordsPerChunk) ? remaining : static_cast<size_t>(kWordsPerChunk));
        for (int elem = threadIdx.x; elem < words; elem += blockDim.x) {
            add_word(shared_hist + lane_hist_offset, data[base_word + elem]);
        }
    }
    __syncthreads();

    unsigned int *out = partials + static_cast<size_t>(blockIdx.x) * kBins;
    for (int bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
        unsigned int sum = 0;
        for (int w = 0; w < warp_count; ++w) {
            sum += shared_hist[w * kBins + bin];
        }
        out[bin] = sum;
    }
}

__global__ void merge_histogram256_kernel(const unsigned int *__restrict__ partials,
                                          unsigned int *__restrict__ hist,
                                          int partial_count) {
    __shared__ unsigned int scratch[kMergeThreads];
    int bin = blockIdx.x;
    unsigned int sum = 0;
    for (int partial = threadIdx.x; partial < partial_count; partial += blockDim.x) {
        sum += partials[static_cast<size_t>(partial) * kBins + bin];
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        hist[bin] = scratch[0];
    }
}

struct AlignedMem {
    uint8_t *raw = nullptr;
    uintptr_t base = 0;
    size_t aligned_sz = 0;
    uint32_t *u32ptr() { return reinterpret_cast<uint32_t *>(base); }
};

struct SideInfo {
    int num_sm = 0;
    int num_side0 = 0;
    uint32_t mask = 0;
    size_t num_chunks = 0;
    std::vector<uint16_t> side_index;
    std::vector<int> all_chunks;
    std::vector<int> near_chunks;
    std::vector<int> far_chunks;
    uint16_t *d_side_index = nullptr;
    int *d_all_chunks = nullptr;
    int *d_near_chunks = nullptr;
    int *d_far_chunks = nullptr;
};

#if CUDART_VERSION >= 13010
struct GreenPartition {
    std::span<cudaDevResource> resources;
    cudaExecutionContext_t ctx = {};
    cudaStream_t stream = {};

    GreenPartition() = default;
    GreenPartition(std::span<cudaDevResource> res, int device_id) : resources(res) {
        cudaDevResourceDesc_t desc{};
        CHECK_CUDA(cudaDevResourceGenerateDesc(&desc, resources.data(), static_cast<unsigned int>(resources.size())));
        CHECK_CUDA(cudaGreenCtxCreate(&ctx, desc, device_id, 0));
        CHECK_CUDA(cudaExecutionCtxStreamCreate(&stream, ctx, cudaStreamDefault, 0));
    }

    int get_num_sms() const {
        int total = 0;
        for (const auto &res : resources) total += res.sm.smCount;
        return total;
    }

    ~GreenPartition() {
        if (stream) CHECK_CUDA(cudaStreamDestroy(stream));
        if (ctx) CHECK_CUDA(cudaExecutionCtxDestroy(ctx));
    }
};
#endif

template <typename T>
T *copy_to_device(const std::vector<T> &host) {
    if (host.empty()) return nullptr;
    T *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

AlignedMem alloc_aligned(size_t bytes) {
    AlignedMem mem;
    size_t pages = ceil_div(bytes, kMemoryAllocGran);
    mem.aligned_sz = pages * kMemoryAllocGran;
    CHECK_CUDA(cudaMalloc(&mem.raw, mem.aligned_sz + kMemoryAllocGran));
    CHECK_CUDA(cudaMemset(mem.raw, 0, mem.aligned_sz + kMemoryAllocGran));
    mem.base = (reinterpret_cast<uintptr_t>(mem.raw) + kMemoryAllocGran - 1) & ~(uintptr_t)(kMemoryAllocGran - 1);
    return mem;
}

void build_side_info(SideInfo &si, uintptr_t base, size_t aligned_sz) {
    int *d_smid2side = nullptr;
    size_t smid2side_bytes = 0;
    get_map_smid2side<int>(nullptr, smid2side_bytes);
    CHECK_CUDA(cudaMallocManaged(&d_smid2side, smid2side_bytes));
    get_map_smid2side(d_smid2side, smid2side_bytes);

    si.num_sm = static_cast<int>(smid2side_bytes / sizeof(int));
    si.side_index.resize(si.num_sm);
    int c0 = 0, c1 = 0;
    for (int sm = 0; sm < si.num_sm; ++sm) {
        int side = d_smid2side[sm] & 1;
        int rank = side == 0 ? c0++ : c1++;
        si.side_index[sm] = static_cast<uint16_t>((rank << 1) | side);
    }
    si.num_side0 = c0;

    int *attr = nullptr;
    uint32_t *mask = nullptr;
    size_t attr_bytes = 0;
    get_map_memory_attr<int>(reinterpret_cast<void *>(base), aligned_sz, nullptr, attr_bytes, nullptr, d_smid2side);
    CHECK_CUDA(cudaMallocManaged(&attr, attr_bytes));
    CHECK_CUDA(cudaMallocManaged(&mask, sizeof(uint32_t)));
    get_map_memory_attr<int>(reinterpret_cast<void *>(base), aligned_sz, attr, attr_bytes, mask, d_smid2side);
    si.mask = *mask;

    si.num_chunks = aligned_sz / kChunkBytes;
    si.all_chunks.reserve(si.num_chunks);
    si.near_chunks.reserve(si.num_chunks / 2);
    si.far_chunks.reserve(si.num_chunks / 2);
    for (size_t chunk = 0; chunk < si.num_chunks; ++chunk) {
        uintptr_t addr = base + chunk * kChunkBytes;
        size_t page = (chunk * kChunkBytes) / kMemoryAllocGran;
        int side = predict_map_memory_attr_far(0, si.mask, addr, attr[page]);
        si.all_chunks.push_back(static_cast<int>(chunk));
        if (side == 0) {
            si.near_chunks.push_back(static_cast<int>(chunk));
        } else {
            si.far_chunks.push_back(static_cast<int>(chunk));
        }
    }

    si.d_side_index = copy_to_device(si.side_index);
    si.d_all_chunks = copy_to_device(si.all_chunks);
    si.d_near_chunks = copy_to_device(si.near_chunks);
    si.d_far_chunks = copy_to_device(si.far_chunks);

    CHECK_CUDA(cudaFree(d_smid2side));
    CHECK_CUDA(cudaFree(attr));
    CHECK_CUDA(cudaFree(mask));
}

std::vector<unsigned int> compute_reference_hist(const std::vector<uint32_t> &words) {
    std::vector<unsigned int> hist(kBins, 0);
    for (uint32_t word : words) {
        hist[(word >> 0) & 0xffu]++;
        hist[(word >> 8) & 0xffu]++;
        hist[(word >> 16) & 0xffu]++;
        hist[(word >> 24) & 0xffu]++;
    }
    return hist;
}

bool compare_hist(const std::vector<unsigned int> &lhs, const std::vector<unsigned int> &rhs, size_t &bad) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            bad = i;
            return false;
        }
    }
    bad = lhs.size();
    return true;
}

std::vector<unsigned int> fetch_hist(unsigned int *d_hist) {
    std::vector<unsigned int> hist(kBins);
    CHECK_CUDA(cudaMemcpy(hist.data(), d_hist, hist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return hist;
}

} // namespace

int main(int argc, char **argv) {
    int data_mb = argc > 1 ? std::atoi(argv[1]) : 256;
    int runs = argc > 2 ? std::atoi(argv[2]) : 10;
    int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    int blocks_per_sm = argc > 4 ? std::atoi(argv[4]) : 1;
    int cache_mode = argc > 5 ? std::atoi(argv[5]) : 1;
    blocks_per_sm = std::max(1, blocks_per_sm);
    cache_mode = std::max(0, std::min(1, cache_mode));

    LATOPT_CHECK(data_mb > 0, "data_mb must be positive");
    LATOPT_CHECK(runs > 0, "runs must be positive");
    LATOPT_CHECK(threads > 0 && threads <= 1024, "threads must be in (0, 1024]");

#if CUDART_VERSION < 13010
    std::fprintf(stderr, "Green Context APIs require CUDART_VERSION >= 13010\n");
    return 1;
#else
    size_t data_bytes = static_cast<size_t>(data_mb) * 1024ULL * 1024ULL;
    LATOPT_CHECK(data_bytes % sizeof(uint32_t) == 0, "data bytes must be divisible by 4");
    size_t total_words = data_bytes / sizeof(uint32_t);

    AlignedMem mem = alloc_aligned(data_bytes);
    std::vector<uint32_t> h_words(total_words);
    for (size_t i = 0; i < h_words.size(); ++i) {
        uint32_t x = static_cast<uint32_t>(i);
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        h_words[i] = x;
    }
    CHECK_CUDA(cudaMemcpy(mem.u32ptr(), h_words.data(), h_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    SideInfo si;
    build_side_info(si, mem.base, mem.aligned_sz);

    int device_id = 0;
    CHECK_CUDA(cudaGetDevice(&device_id));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    size_t flush_bytes = static_cast<size_t>(prop.l2CacheSize) * 2;
    uint8_t *d_l2_flush = nullptr;
    CHECK_CUDA(cudaMalloc(&d_l2_flush, flush_bytes));
    CHECK_CUDA(cudaMemset(d_l2_flush, 1, flush_bytes));

    std::vector<cudaDevResource> resources;
    std::vector<int> group_cnts;
    create_sm_resource_groups(resources, group_cnts, device_id);
    auto resource_view = std::span(resources);
    GreenPartition green0(resource_view.subspan(0, static_cast<size_t>(group_cnts[0])), device_id);
    GreenPartition green1(resource_view.subspan(static_cast<size_t>(group_cnts[0])), device_id);

    int side0_blocks = green0.get_num_sms() * blocks_per_sm;
    int side1_blocks = green1.get_num_sms() * blocks_per_sm;
    int baseline_blocks = side0_blocks + side1_blocks;
    int persistent_blocks = si.num_sm;
    int max_partials = std::max(baseline_blocks, persistent_blocks);
    size_t shared_bytes = static_cast<size_t>(((threads + 31) >> 5) * kBins) * sizeof(unsigned int);

    unsigned int *d_partials = nullptr;
    unsigned int *d_hist = nullptr;
    CHECK_CUDA(cudaMalloc(&d_partials, static_cast<size_t>(max_partials) * kBins * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_hist, kBins * sizeof(unsigned int)));

    cudaEvent_t start{}, stop{}, side1_done{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&side1_done));

    auto warm = [&]() {
        touch_k<<<baseline_blocks, 256>>>(mem.u32ptr(), total_words);
        CHECK_CUDA(cudaDeviceSynchronize());
    };
    auto prepare_cache = [&]() {
        if (cache_mode) {
            warm();
        } else {
            flush_l2_k<<<256, 256>>>(d_l2_flush, flush_bytes);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    };
    auto merge_on_stream = [&](int partial_count, cudaStream_t stream) {
        merge_histogram256_kernel<<<kBins, kMergeThreads, 0, stream>>>(d_partials, d_hist, partial_count);
    };
    auto run_baseline = [&]() {
        histogram256_chunked_kernel<<<baseline_blocks, threads, shared_bytes>>>(mem.u32ptr(), d_partials, si.d_all_chunks, static_cast<int>(si.num_chunks), total_words);
    };
    auto run_baseline_full = [&]() {
        run_baseline();
        merge_on_stream(baseline_blocks, 0);
    };
    auto run_side = [&](bool reverse) {
        int *side0_chunks = reverse ? si.d_far_chunks : si.d_near_chunks;
        int side0_count = reverse ? static_cast<int>(si.far_chunks.size()) : static_cast<int>(si.near_chunks.size());
        int *side1_chunks = reverse ? si.d_near_chunks : si.d_far_chunks;
        int side1_count = reverse ? static_cast<int>(si.near_chunks.size()) : static_cast<int>(si.far_chunks.size());

        histogram256_chunked_kernel<<<side0_blocks, threads, shared_bytes, green0.stream>>>(
            mem.u32ptr(), d_partials, side0_chunks, side0_count, total_words);
        histogram256_chunked_kernel<<<side1_blocks, threads, shared_bytes, green1.stream>>>(
            mem.u32ptr(), d_partials + static_cast<size_t>(side0_blocks) * kBins, side1_chunks, side1_count, total_words);
    };
    auto run_persistent = [&](bool reverse) {
        histogram256_persistent_kernel<<<persistent_blocks, threads, shared_bytes>>>(
            mem.u32ptr(),
            d_partials,
            si.d_near_chunks,
            static_cast<int>(si.near_chunks.size()),
            si.d_far_chunks,
            static_cast<int>(si.far_chunks.size()),
            si.d_side_index,
            si.num_side0,
            total_words,
            reverse ? 1 : 0);
    };
    auto run_persistent_full = [&](bool reverse) {
        run_persistent(reverse);
        merge_on_stream(persistent_blocks, 0);
    };

    auto time_baseline = [&]() {
        prepare_cache();
        CHECK_CUDA(cudaEventRecord(start));
        run_baseline();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    auto time_side = [&](bool reverse) {
        prepare_cache();
        CHECK_CUDA(cudaEventRecord(start, green0.stream));
        CHECK_CUDA(cudaStreamWaitEvent(green1.stream, start, 0));
        run_side(reverse);
        CHECK_CUDA(cudaEventRecord(side1_done, green1.stream));
        CHECK_CUDA(cudaStreamWaitEvent(green0.stream, side1_done, 0));
        CHECK_CUDA(cudaEventRecord(stop, green0.stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };
    auto time_persistent = [&](bool reverse) {
        prepare_cache();
        CHECK_CUDA(cudaEventRecord(start));
        run_persistent(reverse);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    std::vector<unsigned int> reference = compute_reference_hist(h_words);
    size_t bad = 0;

    run_baseline_full();
    CHECK_CUDA(cudaDeviceSynchronize());
    bool baseline_ok = compare_hist(fetch_hist(d_hist), reference, bad);
    run_side(false);
    CHECK_CUDA(cudaDeviceSynchronize());
    merge_histogram256_kernel<<<kBins, kMergeThreads>>>(d_partials, d_hist, baseline_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    bool green_aware_ok = compare_hist(fetch_hist(d_hist), reference, bad);
    run_side(true);
    CHECK_CUDA(cudaDeviceSynchronize());
    merge_histogram256_kernel<<<kBins, kMergeThreads>>>(d_partials, d_hist, baseline_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());
    bool green_swapped_ok = compare_hist(fetch_hist(d_hist), reference, bad);
    run_persistent_full(false);
    CHECK_CUDA(cudaDeviceSynchronize());
    bool persistent_aware_ok = compare_hist(fetch_hist(d_hist), reference, bad);
    run_persistent_full(true);
    CHECK_CUDA(cudaDeviceSynchronize());
    bool persistent_swapped_ok = compare_hist(fetch_hist(d_hist), reference, bad);

    for (int i = 0; i < 5; ++i) {
        time_baseline();
        time_side(false);
        time_side(true);
        time_persistent(false);
        time_persistent(true);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float baseline_ms = 0.0f;
    float green_aware_ms = 0.0f, green_swapped_ms = 0.0f;
    float persistent_aware_ms = 0.0f, persistent_swapped_ms = 0.0f;
    for (int run = 0; run < runs; ++run) {
        baseline_ms += time_baseline();
        green_aware_ms += time_side(false);
        green_swapped_ms += time_side(true);
        persistent_aware_ms += time_persistent(false);
        persistent_swapped_ms += time_persistent(true);
    }
    baseline_ms /= runs;
    green_aware_ms /= runs;
    green_swapped_ms /= runs;
    persistent_aware_ms /= runs;
    persistent_swapped_ms /= runs;

    std::printf("=== Histogram256 (Figure 10) ===\n");
    std::printf("data=%dMB words=%zu runs=%d threads=%d blocks_per_sm=%d cache_mode=%s baseline_blocks=%d persistent_blocks=%d side_blocks[%d,%d]\n",
                data_mb,
                total_words,
                runs,
                threads,
                blocks_per_sm,
                cache_mode ? "warm" : "cold",
                baseline_blocks,
                persistent_blocks,
                side0_blocks,
                side1_blocks);
    std::printf("device_sms=%d green_sms[%d,%d] sm_side[%d,%d] mask=0x%08x chunks=%zu near=%zu far=%zu\n\n",
                si.num_sm,
                green0.get_num_sms(),
                green1.get_num_sms(),
                si.num_side0,
                si.num_sm - si.num_side0,
                si.mask,
                si.num_chunks,
                si.near_chunks.size(),
                si.far_chunks.size());
    std::printf("correct baseline=%s green_aware=%s green_swapped=%s persistent_aware=%s persistent_swapped=%s\n\n",
                baseline_ok ? "PASS" : "FAIL",
                green_aware_ok ? "PASS" : "FAIL",
                green_swapped_ok ? "PASS" : "FAIL",
                persistent_aware_ok ? "PASS" : "FAIL",
                persistent_swapped_ok ? "PASS" : "FAIL");
    std::printf("baseline_kernel_ms %.6f ms\n", baseline_ms);
    std::printf("green_aware_kernel_ms %.6f ms kernel_speedup_vs_baseline=%.3fx\n", green_aware_ms, baseline_ms / green_aware_ms);
    std::printf("green_swapped_kernel_ms %.6f ms kernel_speedup_vs_baseline=%.3fx\n", green_swapped_ms, baseline_ms / green_swapped_ms);
    std::printf("persistent_aware_kernel_ms %.6f ms kernel_speedup_vs_baseline=%.3fx\n", persistent_aware_ms, baseline_ms / persistent_aware_ms);
    std::printf("persistent_swapped_kernel_ms %.6f ms kernel_speedup_vs_baseline=%.3fx\n", persistent_swapped_ms, baseline_ms / persistent_swapped_ms);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(side1_done));
    CHECK_CUDA(cudaFree(d_partials));
    CHECK_CUDA(cudaFree(d_hist));
    CHECK_CUDA(cudaFree(d_l2_flush));
    CHECK_CUDA(cudaFree(mem.raw));
    CHECK_CUDA(cudaFree(si.d_side_index));
    CHECK_CUDA(cudaFree(si.d_all_chunks));
    CHECK_CUDA(cudaFree(si.d_near_chunks));
    CHECK_CUDA(cudaFree(si.d_far_chunks));
    return (baseline_ok && green_aware_ok && green_swapped_ok &&
            persistent_aware_ok && persistent_swapped_ok) ? 0 : 3;
#endif
}
