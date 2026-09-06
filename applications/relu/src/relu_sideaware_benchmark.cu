#include "latopt.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <span>
#include <vector>

using namespace latopt;

namespace {

std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

constexpr size_t kChunkBytes  = 4096;
constexpr int kFloatsPerChunk = static_cast<int>(kChunkBytes / sizeof(float));

__global__ void warmup_k(const float *data, int n) {
    float sink = 0.0f;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += gridDim.x * blockDim.x) {
        sink += data[idx];
    }
    if (sink == -999.0f) {
        asm volatile("" ::: "memory");
    }
}

__global__ void relu_chunked(const float *__restrict__ input, float *__restrict__ output, const int *__restrict__ chunks, int n_chunks) {
    for (int chunk_idx = blockIdx.x; chunk_idx < n_chunks; chunk_idx += gridDim.x) {
        int base = chunks[chunk_idx] * kFloatsPerChunk;
        for (int elem = threadIdx.x; elem < kFloatsPerChunk; elem += blockDim.x) {
            float value         = __ldcs(input + base + elem);
            output[base + elem] = value > 0.0f ? value : 0.0f;
        }
    }
}

__global__ void relu_persistent(const float *__restrict__ input,
                                float *__restrict__ output,
                                const int *__restrict__ near_chunks,
                                int n_near,
                                const int *__restrict__ far_chunks,
                                int n_far,
                                const uint8_t *__restrict__ side_index,
                                int num_side0) {
    uint8_t si  = side_index[__smid()];
    int my_side = si & 1, my_idx = si >> 1;
    const int *my_chunks = (my_side == 0) ? near_chunks : far_chunks;
    int my_n             = (my_side == 0) ? n_near : n_far;
    int my_sms           = (my_side == 0) ? num_side0 : ((int)gridDim.x - num_side0);
    for (int ci = my_idx; ci < my_n; ci += my_sms) {
        int base = my_chunks[ci] * kFloatsPerChunk;
        for (int elem = threadIdx.x; elem < kFloatsPerChunk; elem += blockDim.x) {
            float value         = __ldcs(input + base + elem);
            output[base + elem] = value > 0.0f ? value : 0.0f;
        }
    }
}

struct SideInfo {
    int num_sm        = 0;
    int num_side0     = 0;
    uint32_t mask     = 0;
    size_t num_chunks = 0;
    std::vector<uint8_t> side_index;
    std::vector<int8_t> chunk_table;
    std::vector<int> near_chunks;
    std::vector<int> far_chunks;
    uint8_t *d_side_index = nullptr;
    int *d_all_chunks     = nullptr;
    int *d_near_chunks    = nullptr;
    int *d_far_chunks     = nullptr;
};

struct AlignedMem {
    uint8_t *raw      = nullptr;
    uintptr_t base    = 0;
    size_t aligned_sz = 0;

    float *fptr() { return reinterpret_cast<float *>(base); }
};

#if CUDART_VERSION >= 13010
struct GreenPartition {
    std::span<cudaDevResource> resources_;
    cudaExecutionContext_t ctx = {};
    cudaStream_t stream        = {};

    GreenPartition()           = default;
    GreenPartition(std::span<cudaDevResource> resources, int device_id) {
        resources_ = resources;
        cudaDevResourceDesc_t resource_desc{};
        CHECK_CUDA(cudaDevResourceGenerateDesc(&resource_desc, resources_.data(), static_cast<unsigned int>(resources_.size())));
        CHECK_CUDA(cudaGreenCtxCreate(&ctx, resource_desc, device_id, 0));
        CHECK_CUDA(cudaExecutionCtxStreamCreate(&stream, ctx, cudaStreamDefault, 0));
    }
    int get_num_sms() const {
        int num_sms = 0;
        for (const auto &res : resources_) {
            num_sms += res.sm.smCount;
        }
        return num_sms;
    }
    ~GreenPartition() {
        if (stream) {
            CHECK_CUDA(cudaStreamDestroy(stream));
        }
        if (ctx) {
            CHECK_CUDA(cudaExecutionCtxDestroy(ctx));
        }
    }
};
#endif

template <typename T> T *copy_to_device(const std::vector<T> &host_vec) {
    if (host_vec.empty()) {
        return nullptr;
    }
    T *device_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&device_ptr, host_vec.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device_ptr, host_vec.data(), host_vec.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}

AlignedMem alloc_aligned(size_t bytes) {
    AlignedMem mem;
    size_t pages   = ceil_div(bytes, kMemoryAllocGran);
    mem.aligned_sz = pages * kMemoryAllocGran;
    CHECK_CUDA(cudaMalloc(&mem.raw, mem.aligned_sz + kMemoryAllocGran));
    CHECK_CUDA(cudaMemset(mem.raw, 0, mem.aligned_sz + kMemoryAllocGran));
    mem.base = (reinterpret_cast<uintptr_t>(mem.raw) + kMemoryAllocGran - 1) & ~(uintptr_t)(kMemoryAllocGran - 1);
    return mem;
}

void build_side_info(SideInfo &si, uintptr_t base, size_t aligned_sz) {
    int *d_smid2side       = nullptr;
    size_t smid2side_bytes = 0;
    get_map_smid2side<int>(nullptr, smid2side_bytes);
    CHECK_CUDA(cudaMallocManaged(&d_smid2side, smid2side_bytes));
    get_map_smid2side(d_smid2side, smid2side_bytes);

    // Build side_index: pack (idx_within_side << 1) | side per SM
    si.num_sm = static_cast<int>(smid2side_bytes / sizeof(int));
    si.side_index.resize(si.num_sm);
    int c0 = 0, c1 = 0;
    for (int i = 0; i < si.num_sm; ++i) {
        int side         = d_smid2side[i];
        int idx          = (side == 0) ? c0++ : c1++;
        si.side_index[i] = static_cast<uint8_t>((idx << 1) | side);
    }
    si.num_side0      = c0;

    int *attr         = nullptr;
    uint32_t *mask    = nullptr;
    size_t attr_bytes = 0;
    get_map_memory_attr<int>(reinterpret_cast<void *>(base), aligned_sz, nullptr, attr_bytes, nullptr, d_smid2side);
    CHECK_CUDA(cudaMallocManaged(&attr, attr_bytes));
    CHECK_CUDA(cudaMallocManaged(&mask, sizeof(uint32_t)));
    get_map_memory_attr<int>(reinterpret_cast<void *>(base), aligned_sz, attr, attr_bytes, mask, d_smid2side);
    si.mask       = *mask;

    si.num_chunks = aligned_sz / kChunkBytes;
    // ! Predict chunk locality once, reused.
    si.chunk_table.resize(si.num_chunks);
    std::vector<int> all_chunks;
    for (size_t chunk_id = 0; chunk_id < si.num_chunks; ++chunk_id) {
        uintptr_t addr           = base + chunk_id * kChunkBytes;
        size_t page_id           = (chunk_id * kChunkBytes) / kMemoryAllocGran;
        si.chunk_table[chunk_id] = static_cast<int8_t>(predict_map_memory_attr_far(0, si.mask, addr, attr[page_id]));
        all_chunks.push_back(static_cast<int>(chunk_id));
        if (si.chunk_table[chunk_id] == 0) {
            si.near_chunks.push_back(static_cast<int>(chunk_id));
        } else {
            si.far_chunks.push_back(static_cast<int>(chunk_id));
        }
    }

    si.d_all_chunks  = copy_to_device(all_chunks);
    si.d_near_chunks = copy_to_device(si.near_chunks);
    si.d_far_chunks  = copy_to_device(si.far_chunks);

    si.d_side_index  = copy_to_device(si.side_index);

    CHECK_CUDA(cudaFree(d_smid2side));
    CHECK_CUDA(cudaFree(attr));
    CHECK_CUDA(cudaFree(mask));
}
} // namespace

int main(int argc, char **argv) {
    int data_mb           = (argc > 1) ? std::atoi(argv[1]) : 24;
    int runs              = (argc > 2) ? std::atoi(argv[2]) : 20;
    int threads_per_block = (argc > 3) ? std::atoi(argv[3]) : 128;
    int run_persistent    = (argc > 4) ? std::atoi(argv[4]) : 0;
    int swapped           = (argc > 5) ? std::atoi(argv[5]) : 0;
    int blocks_per_sm     = (argc > 6) ? std::atoi(argv[6]) : 1;

    LATOPT_CHECK(threads_per_block > 0, "threads_per_block must be positive");
    LATOPT_CHECK(argc <= 7, "usage: <data_mb> <runs> <threads> <persistent:0|1> <swapped:0|1> <blocks_per_sm>");
    LATOPT_CHECK(run_persistent == 0 || run_persistent == 1, "persistent must be 0 or 1");
    LATOPT_CHECK(swapped == 0 || swapped == 1, "swapped must be 0 or 1");
    LATOPT_CHECK(blocks_per_sm > 0, "blocks_per_sm must be positive");
    LATOPT_CHECK(!run_persistent || blocks_per_sm == 1, "persistent workflow requires blocks_per_sm=1");

    int baseline_blocks = 0;
    size_t data_elems   = static_cast<size_t>(data_mb) * 1024ULL * 1024ULL / sizeof(float);
    size_t data_bytes   = data_elems * sizeof(float);

#if CUDART_VERSION < 13010
    std::fprintf(stderr, "Green Context APIs require CUDART_VERSION >= 13010\n");
    return 1;
#else
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    int num_tpc    = prop.multiProcessorCount / 2;

    AlignedMem mem = alloc_aligned(data_bytes);
    std::vector<float> h_input(data_elems);
    for (float &value : h_input) {
        value = dist(rng);
    }
    CHECK_CUDA(cudaMemcpy(mem.fptr(), h_input.data(), data_bytes, cudaMemcpyHostToDevice));

    SideInfo si;
    build_side_info(si, mem.base, mem.aligned_sz);
    baseline_blocks = run_persistent ? si.num_sm : 0;

    int device_id = 0;
    CHECK_CUDA(cudaGetDevice(&device_id));

    std::vector<cudaDevResource> resources;
    std::vector<int> group_cnts;
    std::unique_ptr<GreenPartition> green_side0;
    std::unique_ptr<GreenPartition> green_side1;
    int green_side0_blocks = 0;
    int green_side1_blocks = 0;
    if (!run_persistent) {
        create_sm_resource_groups(resources, group_cnts, device_id);
        LATOPT_CHECK(group_cnts.size() == 2, "expected two green resource groups");
        LATOPT_CHECK(group_cnts[0] > 0 && group_cnts[1] > 0, "green resource groups must be non-empty");
        auto resource_view = std::span(resources);

        green_side0 = std::make_unique<GreenPartition>(resource_view.subspan(0, static_cast<size_t>(group_cnts[0])), device_id);
        green_side1 = std::make_unique<GreenPartition>(resource_view.subspan(static_cast<size_t>(group_cnts[0])), device_id);
        green_side0_blocks = green_side0->get_num_sms() * blocks_per_sm;
        green_side1_blocks = green_side1->get_num_sms() * blocks_per_sm;
        baseline_blocks = green_side0_blocks + green_side1_blocks;
        printf("Device has %d SMs; Green Context grids use %d and %d blocks\n", si.num_sm, green_side0_blocks, green_side1_blocks);
        printf("num_side0=%d num_side1=%d\n", si.num_side0, si.num_sm - si.num_side0);
    }
    LATOPT_CHECK(baseline_blocks > 0, "baseline_blocks must be positive");

    cudaEvent_t start      = {};
    cudaEvent_t stop       = {};
    cudaEvent_t side1_done = {};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    if (!run_persistent) {
        CHECK_CUDA(cudaEventCreate(&side1_done));
    }

    auto restore_input = [&]() {
        CHECK_CUDA(cudaMemcpy(mem.fptr(), h_input.data(), data_bytes, cudaMemcpyHostToDevice));
        GlobalL2Cache::reset(); // flush L2
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    auto warmup = [&]() {
        warmup_k<<<baseline_blocks, threads_per_block>>>(mem.fptr(), static_cast<int>(data_elems));
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    auto run_baseline = [&]() {
        relu_chunked<<<baseline_blocks, threads_per_block>>>(mem.fptr(), mem.fptr(), si.d_all_chunks, static_cast<int>(si.num_chunks));
    };

    int *ours_side0_chunks = swapped ? si.d_far_chunks : si.d_near_chunks;
    int ours_side0_count   = swapped ? static_cast<int>(si.far_chunks.size()) : static_cast<int>(si.near_chunks.size());
    int *ours_side1_chunks = swapped ? si.d_near_chunks : si.d_far_chunks;
    int ours_side1_count   = swapped ? static_cast<int>(si.near_chunks.size()) : static_cast<int>(si.far_chunks.size());

    auto run_ours = [&]() {
        if (run_persistent) {
            relu_persistent<<<si.num_sm, threads_per_block>>>(mem.fptr(),
                                                              mem.fptr(),
                                                              ours_side0_chunks,
                                                              ours_side0_count,
                                                              ours_side1_chunks,
                                                              ours_side1_count,
                                                              si.d_side_index,
                                                              si.num_side0);
        } else {
            relu_chunked<<<green_side0_blocks, threads_per_block, 0, green_side0->stream>>>(mem.fptr(),
                                                                                             mem.fptr(),
                                                                                             ours_side0_chunks,
                                                                                             ours_side0_count);
            relu_chunked<<<green_side1_blocks, threads_per_block, 0, green_side1->stream>>>(mem.fptr(),
                                                                                             mem.fptr(),
                                                                                             ours_side1_chunks,
                                                                                             ours_side1_count);
        }
    };

    auto time_ours = [&](float &elapsed_ms) {
        if (run_persistent) {
            CHECK_CUDA(cudaEventRecord(start));
            run_ours();
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        } else {
            CHECK_CUDA(cudaEventRecord(start, green_side0->stream));
            CHECK_CUDA(cudaStreamWaitEvent(green_side1->stream, start, 0));
            run_ours();
            CHECK_CUDA(cudaEventRecord(side1_done, green_side1->stream));
            CHECK_CUDA(cudaStreamWaitEvent(green_side0->stream, side1_done, 0));
            CHECK_CUDA(cudaEventRecord(stop, green_side0->stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        }
    };

    const char *mode_name = run_persistent ? "persistent" : "green_context";
    std::printf("=== Dense ReLU Benchmark ===\n");
    std::printf("data=%dMB runs=%d threads/block=%d blocks_per_sm=%d num_tpc=%d workflow=%s assignment=%s green_blocks[%d,%d] baseline_blocks=%d\n",
                data_mb,
                runs,
                threads_per_block,
                blocks_per_sm,
                num_tpc,
                mode_name,
                swapped ? "swapped" : "aware",
                green_side0_blocks,
                green_side1_blocks,
                baseline_blocks);
    std::printf("mask=0x%08x chunks=%zu near=%zu far=%zu\n\n", si.mask, si.num_chunks, si.near_chunks.size(), si.far_chunks.size());

    std::vector<float> baseline_out(data_elems);
    std::vector<float> our_out(data_elems);
    restore_input();
    run_baseline();
    CHECK_CUDA(cudaMemcpy(baseline_out.data(), mem.fptr(), data_bytes, cudaMemcpyDeviceToHost));

    restore_input();
    run_ours();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(our_out.data(), mem.fptr(), data_bytes, cudaMemcpyDeviceToHost));

    bool correct = (baseline_out == our_out);
    std::printf("  %-16s Baseline vs Ours=%s\n", "ReLU", correct ? "PASS" : "FAIL");

    float baseline_ms = 0.0f;
    float our_ms      = 0.0f;

    for (int i = 0; i < 5; ++i) {
        restore_input();
        warmup();
        run_baseline();

        restore_input();
        warmup();
        run_ours();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    for (int run = 0; run < runs; ++run) {
        float elapsed_ms = 0.0f;

        restore_input();
        warmup();

        CHECK_CUDA(cudaEventRecord(start));
        run_baseline();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
        baseline_ms += elapsed_ms;
        CHECK_CUDA(cudaDeviceSynchronize());
        restore_input();
        warmup();
        time_ours(elapsed_ms);
        our_ms += elapsed_ms;
    }
    baseline_ms /= static_cast<float>(runs);
    our_ms /= static_cast<float>(runs);
    float speedup = baseline_ms / our_ms;
    std::printf("  %-16s Baseline=%.6f ms  Ours=%.6f ms  Speedup=%.6fx\n", "ReLU", baseline_ms, our_ms, speedup);

    CHECK_CUDA(cudaFree(mem.raw));
    CHECK_CUDA(cudaFree(si.d_all_chunks));
    CHECK_CUDA(cudaFree(si.d_side_index));
    CHECK_CUDA(cudaFree(si.d_near_chunks));
    CHECK_CUDA(cudaFree(si.d_far_chunks));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    if (!run_persistent) {
        CHECK_CUDA(cudaEventDestroy(side1_done));
    }
    return correct ? 0 : 3;
#endif
}
