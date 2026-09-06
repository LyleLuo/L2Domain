/* Recover the SM-to-side map in SIDEDGPU_CONFIG_DIR. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

static constexpr size_t kMemoryAllocGran = 2ULL * 1024ULL * 1024ULL; // 2MB

std::string sidedgpu_config_path(const char *filename) {
    const char *dir = getenv("SIDEDGPU_CONFIG_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        fprintf(stderr, "SIDEDGPU_CONFIG_DIR is required; run this tool through a repository script\n");
        exit(EXIT_FAILURE);
    }

    std::string path(dir);
    if (!path.empty() && path.back() != '/') {
        path += '/';
    }
    path += filename;
    return path;
}

template <typename T> struct Signal {
    T *ptr;
    __device__ __forceinline__ Signal(T *signal_ptr) : ptr(signal_ptr) {}
    __device__ __forceinline__ void set(T value) { atomicExch(ptr, value); }
    __device__ __forceinline__ void wait(T expected, int64_t cycles = 100) {
        while (atomicCAS(ptr, expected, expected) != expected) {
            __nanosleep(cycles);
        }
    }
};

__device__ __forceinline__ uint32_t __smid() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ void serial_access_mem(const uint8_t *slice_base, uint64_t *sink) {
    for (int bit = 5; bit < 21; ++bit) {
        auto ptr = reinterpret_cast<const uint64_t *>(slice_base + (size_t{1} << bit));
        sink[0] += *ptr;
    }
}

__device__ __forceinline__ void serial_measure_access_mem(const uint8_t *slice_base, uint32_t *latencies, uint64_t *sink) {
    // ! prevent measuring parameter load latency
    sink[0] = slice_base[0];
    for (int bit = 5; bit < 21; ++bit) {
        auto ptr           = reinterpret_cast<const uint64_t *>(slice_base + (size_t{1} << bit));
        auto start_clk     = clock();
        sink[0]            = *ptr;
        auto end_clk       = clock();
        auto latency       = static_cast<uint32_t>(end_clk - start_clk);
        latencies[bit - 5] = latency;
    }
    return;
}

template <typename T> __global__ void k_get_map_smid2side(int num_sm, const uint8_t *workspace, T *map, uint64_t *sink, int *signal) {
    __shared__ uint32_t shm_latencies[64];
    __shared__ uint64_t _sink;

    auto const smid = __smid();
    if (smid >= static_cast<uint32_t>(num_sm)) {
        return;
    }

    auto sig = Signal(signal);
    enum : int { Idle = static_cast<int>(-1) };

    if (smid == 0) {
        sig.set(Idle);
        for (int target_smid = 1; target_smid < num_sm; ++target_smid) {
            sig.wait(Idle);
            auto const target_base = workspace + static_cast<size_t>(target_smid) * kMemoryAllocGran;
            serial_access_mem(target_base, &_sink);
            sig.set(target_smid);
            sig.wait(Idle);
        }
        sink[0]   = _sink;
        map[smid] = 0;
        return;
    }

    sig.wait(smid);
    auto const slice_base = workspace + static_cast<size_t>(smid) * kMemoryAllocGran;
    serial_measure_access_mem(slice_base, shm_latencies, &_sink);
    uint32_t max_latency = 0;
    uint32_t min_latency = 1024;
    for (int i = 0; i < 21 - 5; ++i) {
        auto cur    = shm_latencies[i];
        max_latency = cur > max_latency ? cur : max_latency;
        min_latency = cur < min_latency ? cur : min_latency;
    }
    sink[0]   = _sink;
    map[smid] = (max_latency - min_latency) > 100;
    sig.set(Idle);
}

int main(int argc, char *argv[]) {
    if (argc != 1) {
        fprintf(stderr, "usage: %s\n", argv[0]);
        return EXIT_FAILURE;
    }

    int device_id;
    CHECK_CUDA(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));
    auto num_sm      = device_prop.multiProcessorCount;
    auto device_name = device_prop.name;
    printf("Device %d: %s has %d SMs\n", device_id, device_name, num_sm);

    size_t const num_bytes = static_cast<size_t>(num_sm) * sizeof(int);
    int *map;
    CHECK_CUDA(cudaMalloc(&map, num_bytes));
    CHECK_CUDA(cudaMemset(map, 0, num_bytes));

    size_t const workspace_bytes = static_cast<size_t>(num_sm) * kMemoryAllocGran;
    uint8_t *ws;
    CHECK_CUDA(cudaMalloc(&ws, workspace_bytes));
    CHECK_CUDA(cudaMemset(ws, 0, workspace_bytes));

    uint64_t *sink;
    CHECK_CUDA(cudaMalloc(&sink, sizeof(uint64_t)));

    int *signal;
    CHECK_CUDA(cudaMalloc(&signal, sizeof(int)));
    CHECK_CUDA(cudaMemset(signal, 0, sizeof(int)));

    // ! flush L2 cache to make sure the latency measurement is not affected by the cache state
    size_t const l2_cache_size = device_prop.l2CacheSize;
    uint8_t *l2_cache;
    CHECK_CUDA(cudaMalloc(&l2_cache, l2_cache_size));
    CHECK_CUDA(cudaMemset(l2_cache, 0, l2_cache_size));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    k_get_map_smid2side<<<num_sm, 1>>>(num_sm, ws, map, sink, signal);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::vector<int> h_map(num_sm);
    CHECK_CUDA(cudaMemcpy(h_map.data(), map, num_bytes, cudaMemcpyDeviceToHost));

    auto const out_path = sidedgpu_config_path("smid2side.csv");
    FILE *out = fopen(out_path.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "failed to open output file '%s'\n", out_path.c_str());
        return EXIT_FAILURE;
    }

    for (int smid = 0; smid < num_sm; ++smid) {
        fprintf(out, "%d,", h_map[smid]);
    }
    fprintf(out, "\n");
    if (fclose(out) != 0) {
        fprintf(stderr, "failed to close output file '%s'\n", out_path.c_str());
        return EXIT_FAILURE;
    }

    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Wrote SMID-to-side map to %s\n", out_path.c_str());

    CHECK_CUDA(cudaFree(map));
    CHECK_CUDA(cudaFree(ws));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(signal));
    CHECK_CUDA(cudaFree(l2_cache));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}
