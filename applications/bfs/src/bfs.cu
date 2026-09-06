#include "latopt.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <span>
#include <vector>

using namespace latopt;

namespace {

#ifndef LUO_BLK
#define LUO_BLK 128
#endif
#ifndef LUO_NUM_BIN
#define LUO_NUM_BIN 16
#endif
#ifndef LUO_W_QUEUE_SIZE
#define LUO_W_QUEUE_SIZE 256
#endif

constexpr int BLK = LUO_BLK;
constexpr int NUM_BIN = LUO_NUM_BIN;
constexpr int W_QUEUE_SIZE = LUO_W_QUEUE_SIZE;
constexpr int THREADS_PER_BIN = (BLK + NUM_BIN - 1) / NUM_BIN;
constexpr int FLUSH_INTERVAL = (W_QUEUE_SIZE / THREADS_PER_BIN) > 0 ? (W_QUEUE_SIZE / THREADS_PER_BIN) : 1;
constexpr uint32_t kUnvisited = 0xffffffffu;

template <typename T>
struct AlignedAllocResult {
    T *raw = nullptr;
    T *ptr = nullptr;
    size_t elems = 0;
    size_t aligned_bytes = 0;
};

template <typename T>
AlignedAllocResult<T> alloc_aligned(size_t elems) {
    AlignedAllocResult<T> mem;
    mem.elems = elems;
    size_t bytes = elems * sizeof(T);
    mem.aligned_bytes = ceil_div(bytes, kMemoryAllocGran) * kMemoryAllocGran;
    CHECK_CUDA(cudaMalloc(&mem.raw, mem.aligned_bytes + kMemoryAllocGran));
    uintptr_t base = (reinterpret_cast<uintptr_t>(mem.raw) + kMemoryAllocGran - 1) & ~(uintptr_t)(kMemoryAllocGran - 1);
    mem.ptr = reinterpret_cast<T *>(base);
    CHECK_CUDA(cudaMemset(mem.raw, 0, mem.aligned_bytes + kMemoryAllocGran));
    return mem;
}

struct SideInfo {
    int num_sm = 0;
    int side0_sms = 0;
    uint32_t mask = 0;
    std::vector<int> attr;
};

double summarize_median(std::vector<float> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return (samples.size() & 1) ? samples[samples.size() / 2] :
                                  0.5 * (samples[samples.size() / 2 - 1] + samples[samples.size() / 2]);
}

void build_side_info(SideInfo &si, void *base, size_t aligned_bytes) {
    int *d_sm = nullptr;
    size_t sm_bytes = 0;
    get_map_smid2side<int>(nullptr, sm_bytes);
    CHECK_CUDA(cudaMallocManaged(&d_sm, sm_bytes));
    get_map_smid2side(d_sm, sm_bytes);
    si.num_sm = static_cast<int>(sm_bytes / sizeof(int));
    si.side0_sms = 0;
    for (int i = 0; i < si.num_sm; ++i) si.side0_sms += d_sm[i] == 0;

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

uint8_t predict_side(uintptr_t base, uint32_t mask, const std::vector<int> &attr, int vertex) {
    uintptr_t addr = base + static_cast<uintptr_t>(vertex) * sizeof(uint32_t);
    size_t page = (addr - base) / kMemoryAllocGran;
    return static_cast<uint8_t>(predict_map_memory_attr_far(0, mask, addr, attr[page]) & 1);
}

uint8_t predict_row_side(uintptr_t base, uint32_t mask, const std::vector<int> &attr, int vertex, int degree) {
    uintptr_t addr = base + static_cast<uintptr_t>(vertex) * static_cast<uintptr_t>(degree) * sizeof(int);
    size_t page = (addr - base) / kMemoryAllocGran;
    return static_cast<uint8_t>(predict_map_memory_attr_far(0, mask, addr, attr[page]) & 1);
}

template <typename T>
static T *copy_to_device(const std::vector<T> &h) {
    T *d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty()) CHECK_CUDA(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static int argi(int argc, char **argv, int idx, int fallback) {
    return (argc > idx) ? std::atoi(argv[idx]) : fallback;
}

static float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

#if CUDART_VERSION >= 13010
struct GreenPartition {
    std::span<cudaDevResource> resources;
    cudaExecutionContext_t ctx = {};
    cudaStream_t stream = {};

    GreenPartition(std::span<cudaDevResource> res, int device_id) : resources(res) {
        cudaDevResourceDesc_t desc{};
        CHECK_CUDA(cudaDevResourceGenerateDesc(&desc, resources.data(), static_cast<unsigned int>(resources.size())));
        CHECK_CUDA(cudaGreenCtxCreate(&ctx, desc, device_id, 0));
        CHECK_CUDA(cudaExecutionCtxStreamCreate(&stream, ctx, cudaStreamDefault, 0));
    }

    int sms() const {
        int total = 0;
        for (auto const &r : resources) total += r.sm.smCount;
        return total;
    }

    ~GreenPartition() {
        if (stream) CHECK_CUDA(cudaStreamDestroy(stream));
        if (ctx) CHECK_CUDA(cudaExecutionCtxDestroy(ctx));
    }
};
#endif

__device__ __forceinline__ int ld_i(const int *p) {
    return __ldcs(p);
}

__global__ void init_dist_kernel(uint32_t *dist, int vertices) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vertices; i += gridDim.x * blockDim.x) {
        dist[i] = kUnvisited;
    }
}

__global__ void seed_baseline_kernel(uint32_t *dist, int *frontier, int *count, const int *sources, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int v = sources[i];
        dist[v] = 0;
        int pos = atomicAdd(count, 1);
        frontier[pos] = v;
    }
}

__global__ void seed_side_kernel(uint32_t *dist, int *frontier, int *counts, const int *sources, const uint8_t *work_side, int n, int capacity) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int v = sources[i];
        dist[v] = 0;
        int side = work_side[v] & 1;
        int pos = atomicAdd(counts + side, 1);
        frontier[side * capacity + pos] = v;
    }
}

__device__ __forceinline__ void local_append(int *q, int *tails, int bin, int value, int *overflow) {
    int pos = atomicAdd(&tails[bin], 1);
    if (pos < W_QUEUE_SIZE) {
        q[bin * W_QUEUE_SIZE + pos] = value;
    } else {
        *overflow = 1;
    }
}

__device__ __forceinline__ int local_prefix_sum(int tails[NUM_BIN], int prefix[NUM_BIN]) {
    int total = 0;
    for (int i = 0; i < NUM_BIN; ++i) {
        prefix[i] = total;
        total += tails[i];
    }
    return total;
}

__device__ __forceinline__ int flush_local_queue(int *q, int *tails, int *prefix, int *global, int *global_count) {
    __shared__ int base;
    __shared__ int total;
    if (threadIdx.x == 0) {
        total = local_prefix_sum(tails, prefix);
        base = total ? atomicAdd(global_count, total) : 0;
    }
    __syncthreads();
    if (total == 0) {
        __syncthreads();
        return 0;
    }
    for (int bin = 0; bin < NUM_BIN; ++bin) {
        for (int i = threadIdx.x; i < tails[bin]; i += blockDim.x) {
            if (i < W_QUEUE_SIZE) global[base + prefix[bin] + i] = q[bin * W_QUEUE_SIZE + i];
        }
    }
    __syncthreads();
    return total;
}

__global__ void luo_baseline_level_kernel(const int *row_offsets, const int *cols, const int *frontier, const int *frontier_count,
                                          uint32_t *dist, int *next, int *next_count, int depth, int *overflow) {
    __shared__ int q[NUM_BIN * W_QUEUE_SIZE];
    __shared__ int tails[NUM_BIN];
    __shared__ int prefix[NUM_BIN];
    int total = *frontier_count;
    for (int base_idx = blockIdx.x * blockDim.x; base_idx < total; base_idx += gridDim.x * blockDim.x) {
        if (threadIdx.x < NUM_BIN) tails[threadIdx.x] = 0;
        __syncthreads();
        int idx = base_idx + threadIdx.x;
        int beg = 0;
        int len = 0;
        if (idx < total) {
            int v = frontier[idx];
            beg = ld_i(row_offsets + v);
            len = ld_i(row_offsets + v + 1) - beg;
        }
        __shared__ int max_len;
        if (threadIdx.x == 0) max_len = 0;
        __syncthreads();
        if (len > 0) atomicMax(&max_len, len);
        __syncthreads();
        for (int j = 0; j < max_len; ++j) {
            if (j < len) {
                int dst = ld_i(cols + beg + j);
                if (atomicCAS(dist + dst, kUnvisited, static_cast<uint32_t>(depth + 1)) == kUnvisited) {
                    local_append(q, tails, threadIdx.x & (NUM_BIN - 1), dst, overflow);
                }
            }
            __syncthreads();
            if (((j + 1) % FLUSH_INTERVAL == 0) || (j + 1 == max_len)) {
                flush_local_queue(q, tails, prefix, next, next_count);
                if (threadIdx.x < NUM_BIN) tails[threadIdx.x] = 0;
                __syncthreads();
            }
        }
    }
}

__global__ void luo_side_level_kernel(const int *row_offsets, const int *cols, const uint8_t *work_side,
                                      const int *frontier, const int *frontier_counts, int capacity,
                                      uint32_t *dist, int *next, int *next_counts, int depth, int target_side, int invert, int *overflow) {
    __shared__ int q0[NUM_BIN * W_QUEUE_SIZE];
    __shared__ int q1[NUM_BIN * W_QUEUE_SIZE];
    __shared__ int tails0[NUM_BIN];
    __shared__ int tails1[NUM_BIN];
    __shared__ int prefix[NUM_BIN];
    int consume_side = target_side ^ (invert & 1);
    int total = frontier_counts[consume_side];
    const int *my_frontier = frontier + consume_side * capacity;
    for (int base_idx = blockIdx.x * blockDim.x; base_idx < total; base_idx += gridDim.x * blockDim.x) {
        if (threadIdx.x < NUM_BIN) {
            tails0[threadIdx.x] = 0;
            tails1[threadIdx.x] = 0;
        }
        __syncthreads();
        int idx = base_idx + threadIdx.x;
        int beg = 0;
        int len = 0;
        if (idx < total) {
            int v = my_frontier[idx];
            beg = ld_i(row_offsets + v);
            len = ld_i(row_offsets + v + 1) - beg;
        }
        __shared__ int max_len;
        if (threadIdx.x == 0) max_len = 0;
        __syncthreads();
        if (len > 0) atomicMax(&max_len, len);
        __syncthreads();
        for (int j = 0; j < max_len; ++j) {
            if (j < len) {
                int dst = ld_i(cols + beg + j);
                if (atomicCAS(dist + dst, kUnvisited, static_cast<uint32_t>(depth + 1)) == kUnvisited) {
                    int side = work_side[dst] & 1;
                    if (side) {
                        local_append(q1, tails1, threadIdx.x & (NUM_BIN - 1), dst, overflow);
                    } else {
                        local_append(q0, tails0, threadIdx.x & (NUM_BIN - 1), dst, overflow);
                    }
                }
            }
            __syncthreads();
            if (((j + 1) % FLUSH_INTERVAL == 0) || (j + 1 == max_len)) {
                flush_local_queue(q0, tails0, prefix, next, next_counts);
                flush_local_queue(q1, tails1, prefix, next + capacity, next_counts + 1);
                if (threadIdx.x < NUM_BIN) {
                    tails0[threadIdx.x] = 0;
                    tails1[threadIdx.x] = 0;
                }
                __syncthreads();
            }
        }
    }
}

__global__ void count_visited_kernel(const uint32_t *dist, int n, int *out) {
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) local += dist[i] != kUnvisited;
    if (local) atomicAdd(out, local);
}

__global__ void diff_dist_kernel(const uint32_t *a, const uint32_t *b, int n, int *bad) {
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) local += a[i] != b[i];
    if (local) atomicAdd(bad, local);
}

} // namespace

int main(int argc, char **argv) {
#if CUDART_VERSION < 13010
    std::fprintf(stderr, "Green Context APIs require CUDART_VERSION >= 13010\n");
    return 1;
#else
    int vertices = argi(argc, argv, 1, 2000000);
    int degree = argi(argc, argv, 2, 64);
    int source_count = argi(argc, argv, 3, 4096);
    int max_depth = argi(argc, argv, 4, 4);
    int dst_local_pct = argi(argc, argv, 5, 100);
    int hot_per_side = argi(argc, argv, 6, 200000);
    int runs = argi(argc, argv, 7, 5);
    int seed = argi(argc, argv, 8, 8201);

    vertices = std::max(vertices, 4096);
    degree = std::max(degree, 1);
    source_count = std::max(1, std::min(source_count, vertices / 4));
    max_depth = std::max(1, max_depth);
    dst_local_pct = std::max(0, std::min(dst_local_pct, 100));
    hot_per_side = std::max(1024, std::min(hot_per_side, vertices / 3));
    runs = std::max(runs, 1);

    cudaDeviceProp prop{};
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::printf("=== Locality-aware BFS (Table 10) ===\n");
    std::printf("gpu=%s vertices=%d degree=%d sources=%d depth=%d dst_local=%d%% hot/side=%d runs=%d seed=%d\n",
                prop.name, vertices, degree, source_count, max_depth, dst_local_pct, hot_per_side, runs, seed);

    auto row_mem = alloc_aligned<int>((size_t)vertices + 1);
    auto cols_mem = alloc_aligned<int>((size_t)vertices * degree);
    auto dist_mem = alloc_aligned<uint32_t>(vertices);
    auto dist_ref_mem = alloc_aligned<uint32_t>(vertices);
    SideInfo dist_side_info, row_side_info;
    build_side_info(dist_side_info, dist_mem.ptr, dist_mem.aligned_bytes);
    build_side_info(row_side_info, cols_mem.ptr, cols_mem.aligned_bytes);

    std::vector<uint8_t> h_vertex_side(vertices);
    std::vector<int> hot0, hot1;
    hot0.reserve(hot_per_side);
    hot1.reserve(hot_per_side);
    uintptr_t cols_base = reinterpret_cast<uintptr_t>(cols_mem.ptr);
    uintptr_t dist_base = reinterpret_cast<uintptr_t>(dist_mem.ptr);
    for (int v = 0; v < vertices; ++v) {
        uint8_t row_side = predict_row_side(cols_base, row_side_info.mask, row_side_info.attr, v, degree);
        uint8_t dist_side = predict_side(dist_base, dist_side_info.mask, dist_side_info.attr, v);
        h_vertex_side[v] = row_side;
        if (row_side != dist_side) continue;
        if (row_side == 0 && (int)hot0.size() < hot_per_side) hot0.push_back(v);
        if (row_side == 1 && (int)hot1.size() < hot_per_side) hot1.push_back(v);
    }
    if ((int)hot0.size() < hot_per_side || (int)hot1.size() < hot_per_side) {
        std::fprintf(stderr, "failed hot pools side0=%zu side1=%zu requested=%d\n", hot0.size(), hot1.size(), hot_per_side);
        return 2;
    }

    std::vector<uint8_t> h_work_side = h_vertex_side;
    std::vector<uint8_t> h_cross_source(vertices, 0);
    int const cross_per_side = hot_per_side * (100 - dst_local_pct) / 100;
    for (int i = 0; i < cross_per_side; ++i) {
        int const v0 = hot0[hot_per_side - 1 - i];
        int const v1 = hot1[hot_per_side - 1 - i];
        h_cross_source[v0] = 1;
        h_cross_source[v1] = 1;
        h_work_side[v0] = 1;
        h_work_side[v1] = 0;
    }

    std::vector<int> h_rows((size_t)vertices + 1);
    std::vector<int> h_cols((size_t)vertices * degree);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> hot0d(0, hot_per_side - 1);
    std::uniform_int_distribution<int> hot1d(0, hot_per_side - 1);
    for (int v = 0; v < vertices; ++v) {
        h_rows[v] = v * degree;
        int side = h_vertex_side[v] & 1;
        int const dst_side = h_cross_source[v] ? (side ^ 1) : side;
        for (int j = 0; j < degree; ++j) {
            int dst = dst_side ? hot1[hot1d(rng)] : hot0[hot0d(rng)];
            h_cols[(size_t)v * degree + j] = dst;
        }
    }
    h_rows[vertices] = vertices * degree;
    std::vector<int> h_sources(source_count);
    for (int i = 0; i < source_count; ++i) h_sources[i] = (i & 1) ? hot1[(size_t)(i / 2) % hot1.size()] : hot0[(size_t)(i / 2) % hot0.size()];

    CHECK_CUDA(cudaMemcpy(row_mem.ptr, h_rows.data(), h_rows.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cols_mem.ptr, h_cols.data(), h_cols.size() * sizeof(int), cudaMemcpyHostToDevice));
    int *d_sources = copy_to_device(h_sources);
    uint8_t *d_work_side = copy_to_device(h_work_side);

    int *frontier_a = nullptr, *frontier_b = nullptr, *count_a = nullptr, *count_b = nullptr, *overflow = nullptr, *bad = nullptr, *d_count = nullptr;
    CHECK_CUDA(cudaMalloc(&frontier_a, sizeof(int) * (size_t)2 * vertices));
    CHECK_CUDA(cudaMalloc(&frontier_b, sizeof(int) * (size_t)2 * vertices));
    CHECK_CUDA(cudaMalloc(&count_a, sizeof(int) * 2));
    CHECK_CUDA(cudaMalloc(&count_b, sizeof(int) * 2));
    CHECK_CUDA(cudaMalloc(&overflow, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&bad, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));

    std::vector<cudaDevResource> resources;
    std::vector<int> group_cnts;
    create_sm_resource_groups(resources, group_cnts, device);
    auto all_view = std::span(resources);
    GreenPartition baseline_green(all_view, device);
    GreenPartition green0(all_view.subspan(0, (size_t)group_cnts[0]), device);
    GreenPartition green1(all_view.subspan((size_t)group_cnts[0], (size_t)group_cnts[1]), device);
    int baseline_blocks = green0.sms() + green1.sms();
    std::printf("config BLK=%d NUM_BIN=%d W_QUEUE_SIZE=%d\n", BLK, NUM_BIN, W_QUEUE_SIZE);

    cudaEvent_t e0{}, e1{};
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    auto init_baseline = [&](uint32_t *dist, int *frontier, int *count, cudaStream_t stream) {
        init_dist_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK, 0, stream>>>(dist, vertices);
        CHECK_CUDA(cudaMemsetAsync(count, 0, sizeof(int), stream));
        seed_baseline_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK, 0, stream>>>(dist, frontier, count, d_sources, source_count);
    };
    auto init_side = [&](uint32_t *dist, int *frontier, int *counts, const uint8_t *side_map, cudaStream_t stream) {
        init_dist_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK, 0, stream>>>(dist, vertices);
        CHECK_CUDA(cudaMemsetAsync(counts, 0, sizeof(int) * 2, stream));
        seed_side_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK, 0, stream>>>(dist, frontier, counts, d_sources, side_map, source_count, vertices);
    };
    auto run_baseline = [&](uint32_t *dist, cudaStream_t stream) {
        int *cur = frontier_a, *next = frontier_b, *cur_count = count_a, *next_count = count_b;
        init_baseline(dist, cur, cur_count, stream);
        CHECK_CUDA(cudaMemsetAsync(overflow, 0, sizeof(int), stream));
        for (int depth = 0; depth < max_depth; ++depth) {
            CHECK_CUDA(cudaMemsetAsync(next_count, 0, sizeof(int), stream));
            luo_baseline_level_kernel<<<baseline_blocks, BLK, 0, stream>>>(row_mem.ptr, cols_mem.ptr, cur, cur_count, dist, next, next_count, depth, overflow);
            std::swap(cur, next);
            std::swap(cur_count, next_count);
        }
    };
    auto run_side = [&](uint32_t *dist, const uint8_t *side_map, int invert) {
        int *cur = frontier_a, *next = frontier_b, *cur_counts = count_a, *next_counts = count_b;
        init_side(dist, cur, cur_counts, side_map, baseline_green.stream);
        CHECK_CUDA(cudaMemsetAsync(overflow, 0, sizeof(int), baseline_green.stream));
        std::vector<cudaEvent_t> zero_done(max_depth), done0(max_depth), done1(max_depth);
        for (int d = 0; d < max_depth; ++d) {
            CHECK_CUDA(cudaEventCreateWithFlags(&zero_done[d], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&done0[d], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(&done1[d], cudaEventDisableTiming));
        }
        for (int depth = 0; depth < max_depth; ++depth) {
            CHECK_CUDA(cudaMemsetAsync(next_counts, 0, sizeof(int) * 2, baseline_green.stream));
            CHECK_CUDA(cudaEventRecord(zero_done[depth], baseline_green.stream));
            CHECK_CUDA(cudaStreamWaitEvent(green0.stream, zero_done[depth], 0));
            CHECK_CUDA(cudaStreamWaitEvent(green1.stream, zero_done[depth], 0));
            luo_side_level_kernel<<<green0.sms(), BLK, 0, green0.stream>>>(row_mem.ptr, cols_mem.ptr, side_map, cur, cur_counts, vertices, dist, next, next_counts, depth, 0, invert, overflow);
            luo_side_level_kernel<<<green1.sms(), BLK, 0, green1.stream>>>(row_mem.ptr, cols_mem.ptr, side_map, cur, cur_counts, vertices, dist, next, next_counts, depth, 1, invert, overflow);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(done0[depth], green0.stream));
            CHECK_CUDA(cudaEventRecord(done1[depth], green1.stream));
            CHECK_CUDA(cudaStreamWaitEvent(baseline_green.stream, done0[depth], 0));
            CHECK_CUDA(cudaStreamWaitEvent(baseline_green.stream, done1[depth], 0));
            std::swap(cur, next);
            std::swap(cur_counts, next_counts);
        }
        CHECK_CUDA(cudaStreamSynchronize(baseline_green.stream));
        for (int d = 0; d < max_depth; ++d) {
            CHECK_CUDA(cudaEventDestroy(zero_done[d]));
            CHECK_CUDA(cudaEventDestroy(done0[d]));
            CHECK_CUDA(cudaEventDestroy(done1[d]));
        }
    };

    run_baseline(dist_ref_mem.ptr, baseline_green.stream);
    CHECK_CUDA(cudaStreamSynchronize(baseline_green.stream));
    run_side(dist_mem.ptr, d_work_side, 0);
    CHECK_CUDA(cudaMemset(bad, 0, sizeof(int)));
    diff_dist_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK>>>(dist_ref_mem.ptr, dist_mem.ptr, vertices, bad);
    int aware_bad = 0;
    CHECK_CUDA(cudaMemcpy(&aware_bad, bad, sizeof(int), cudaMemcpyDeviceToHost));
    run_side(dist_mem.ptr, d_work_side, 1);
    CHECK_CUDA(cudaMemset(bad, 0, sizeof(int)));
    diff_dist_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK>>>(dist_ref_mem.ptr, dist_mem.ptr, vertices, bad);
    int swapped_bad = 0;
    CHECK_CUDA(cudaMemcpy(&swapped_bad, bad, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));
    count_visited_kernel<<<std::min(prop.multiProcessorCount * 8, 4096), BLK>>>(dist_ref_mem.ptr, vertices, d_count);
    int visited = 0;
    CHECK_CUDA(cudaMemcpy(&visited, d_count, sizeof(int), cudaMemcpyDeviceToHost));

    auto time_baseline = [&]() {
        CHECK_CUDA(cudaEventRecord(e0, baseline_green.stream));
        run_baseline(dist_ref_mem.ptr, baseline_green.stream);
        CHECK_CUDA(cudaEventRecord(e1, baseline_green.stream));
        CHECK_CUDA(cudaEventSynchronize(e1));
        return event_ms(e0, e1);
    };
    auto time_side = [&](const uint8_t *side_map, int invert) {
        CHECK_CUDA(cudaEventRecord(e0, baseline_green.stream));
        CHECK_CUDA(cudaEventSynchronize(e0));
        run_side(dist_mem.ptr, side_map, invert);
        CHECK_CUDA(cudaEventRecord(e1, baseline_green.stream));
        CHECK_CUDA(cudaEventSynchronize(e1));
        return event_ms(e0, e1);
    };
    std::vector<float> base_samples, aware_samples, swapped_samples;
    for (int r = 0; r < runs; ++r) {
        base_samples.push_back(time_baseline());
        aware_samples.push_back(time_side(d_work_side, 0));
        swapped_samples.push_back(time_side(d_work_side, 1));
    }
    double const base = summarize_median(base_samples);
    double const aware = summarize_median(aware_samples);
    double const swapped = summarize_median(swapped_samples);
    int h_overflow = 0;
    CHECK_CUDA(cudaMemcpy(&h_overflow, overflow, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("visited=%d aware_bad=%d swapped_bad=%d overflow=%d groups=%d/%d\n", visited, aware_bad, swapped_bad, h_overflow, group_cnts[0], group_cnts[1]);
    std::printf("| mode | median_ms | speedup_vs_baseline |\n");
    std::printf("|---|---:|---:|\n");
    std::printf("| baseline | %.6f | 1.000000 |\n", base);
    std::printf("| aware | %.6f | %.6f |\n", aware, base / aware);
    std::printf("| swapped | %.6f | %.6f |\n", swapped, base / swapped);

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaFree(d_count));
    CHECK_CUDA(cudaFree(bad));
    CHECK_CUDA(cudaFree(overflow));
    CHECK_CUDA(cudaFree(count_b));
    CHECK_CUDA(cudaFree(count_a));
    CHECK_CUDA(cudaFree(frontier_b));
    CHECK_CUDA(cudaFree(frontier_a));
    CHECK_CUDA(cudaFree(d_work_side));
    CHECK_CUDA(cudaFree(d_sources));
    CHECK_CUDA(cudaFree(dist_ref_mem.raw));
    CHECK_CUDA(cudaFree(dist_mem.raw));
    CHECK_CUDA(cudaFree(cols_mem.raw));
    CHECK_CUDA(cudaFree(row_mem.raw));
    return (aware_bad == 0 && swapped_bad == 0 && h_overflow == 0) ? 0 : 3;
#endif
}
