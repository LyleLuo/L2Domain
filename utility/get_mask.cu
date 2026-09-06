/* Recover the memory-side address mask in SIDEDGPU_CONFIG_DIR. */

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
static constexpr int kExp2StartBit       = 7;
static constexpr int kExp2StopBit        = 21;
static constexpr int kNumProbeOffsets    = kExp2StopBit - kExp2StartBit;

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
    __device__ __forceinline__ void wait(T expected, int64_t cycles = 2000) {
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

__device__ __forceinline__ void access_pair(const uint8_t *base, volatile uint64_t *sink) {
    auto ptr = reinterpret_cast<const uint64_t *>(base);
    sink[0]  = *ptr;
}

__device__ __forceinline__ void measure_pair(const uint8_t *base, uint32_t *latency, volatile uint64_t *sink) {
    auto ptr      = reinterpret_cast<const uint64_t *>(base);
    auto start    = clock();
    sink[0]       = *ptr;
    auto stop     = clock();
    latency[0]    = static_cast<uint32_t>(stop - start);
}

__device__ __forceinline__ void access_exp2_series(const uint8_t *base, volatile uint64_t *sink) {
    sink[0] = base[128];
    for (int bit = kExp2StartBit; bit < kExp2StopBit; ++bit) {
        auto ptr = reinterpret_cast<const uint64_t *>(base + (size_t{1} << bit));
        sink[0]  = *ptr;
    }
}

__device__ __forceinline__ void measure_exp2_series(const uint8_t *base, uint32_t *latencies, volatile uint64_t *sink) {
    sink[0] = base[32];
    for (int bit = kExp2StartBit; bit < kExp2StopBit; ++bit) {
        auto ptr                          = reinterpret_cast<const uint64_t *>(base + (size_t{1} << bit));
        auto start                        = clock();
        sink[0]                           = *ptr;
        auto stop                         = clock();
        latencies[bit - kExp2StartBit]    = static_cast<uint32_t>(stop - start);
    }
}

__global__ void k_get_mask(const uint8_t *mem, uint32_t *mask, int target_smid, uint64_t *sink, int *signal) {
    if (threadIdx.x != 0) {
        return;
    }

    auto const smid = __smid();
    if (smid != 0 && smid != static_cast<uint32_t>(target_smid)) {
        return;
    }

    auto sig = Signal(signal);
    __shared__ volatile uint64_t _sink;
    __shared__ uint32_t pair_latency[1];
    __shared__ uint32_t exp2_latencies[kNumProbeOffsets];
    auto select_id    = [](int selected_smid) { return -selected_smid; };

    _sink = 0;

    if (smid == 0) {
        sig.set(0);
        sig.set(select_id(target_smid));
        sig.wait(1);

        measure_exp2_series(mem, exp2_latencies, &_sink);
        measure_pair(mem, pair_latency, &_sink);

        float exp2_avg = 0.0f;
        for (int i = 0; i < kNumProbeOffsets; ++i) {
            exp2_avg += static_cast<float>(exp2_latencies[i]);
        }
        exp2_avg /= static_cast<float>(kNumProbeOffsets);

        bool const base_attr = static_cast<float>(exp2_latencies[0]) > exp2_avg;
        uint32_t local_mask  = 0;
        for (int i = 0; i < kNumProbeOffsets; ++i) {
            bool const cur_attr = static_cast<float>(exp2_latencies[i]) > exp2_avg;
            if (cur_attr != base_attr) {
                local_mask |= uint32_t{1} << (i + kExp2StartBit);
            }
        }

        mask[0] = local_mask;
        sig.set(0);
        sink[0] = _sink;
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

    access_pair(mem, &_sink);
    access_exp2_series(mem, &_sink);
    sig.set(1);
    sink[0] = _sink;
}

std::vector<int> read_smid2side(const char *path) {
    FILE *file = fopen(path, "r");
    if (file == nullptr) {
        fprintf(stderr, "failed to open SMID-to-side map '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    std::vector<int> values;
    int value = 0;
    while (fscanf(file, "%d", &value) == 1) {
        values.push_back(value);
        int ch = fgetc(file);
        if (ch != EOF && ch != ',') {
            ungetc(ch, file);
        }
    }
    fclose(file);
    return values;
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

    auto const smid2side_path = sidedgpu_config_path("smid2side.csv");
    auto const h_smid2side = read_smid2side(smid2side_path.c_str());
    if (h_smid2side.size() != static_cast<size_t>(num_sm)) {
        fprintf(stderr, "SMID-to-side map has %zu entries, expected %d\n", h_smid2side.size(), num_sm);
        return EXIT_FAILURE;
    }

    int target_smid = -1;
    for (int smid = 1; smid < num_sm; ++smid) {
        if (h_smid2side[smid] == 1) {
            target_smid = smid;
            break;
        }
    }
    if (target_smid < 0) {
        fprintf(stderr, "failed to find side-1 SMID in %s\n", smid2side_path.c_str());
        return EXIT_FAILURE;
    }

    uint8_t *raw_mem;
    CHECK_CUDA(cudaMalloc(&raw_mem, 2 * kMemoryAllocGran));
    CHECK_CUDA(cudaMemset(raw_mem, 0, 2 * kMemoryAllocGran));
    auto const raw_addr     = reinterpret_cast<uintptr_t>(raw_mem);
    auto const aligned_addr = (raw_addr + kMemoryAllocGran - 1) & ~(kMemoryAllocGran - 1);
    auto *mem               = reinterpret_cast<uint8_t *>(aligned_addr);

    uint32_t *mask;
    CHECK_CUDA(cudaMalloc(&mask, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(mask, 0, sizeof(uint32_t)));

    uint64_t *sink;
    CHECK_CUDA(cudaMalloc(&sink, sizeof(uint64_t)));

    int *signal;
    CHECK_CUDA(cudaMalloc(&signal, sizeof(int)));
    CHECK_CUDA(cudaMemset(signal, 0, sizeof(int)));

    size_t const l2_cache_size = device_prop.l2CacheSize;
    uint8_t *l2_cache;
    CHECK_CUDA(cudaMalloc(&l2_cache, l2_cache_size));
    CHECK_CUDA(cudaMemset(l2_cache, 0, l2_cache_size));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    k_get_mask<<<num_sm, 1>>>(mem, mask, target_smid, sink, signal);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    uint32_t h_mask = 0;
    CHECK_CUDA(cudaMemcpy(&h_mask, mask, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    auto const mask_path = sidedgpu_config_path("mask.csv");
    FILE *out = fopen(mask_path.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "failed to open output file '%s'\n", mask_path.c_str());
        return EXIT_FAILURE;
    }
    fprintf(out, "0x%08x\n", h_mask);
    if (fclose(out) != 0) {
        fprintf(stderr, "failed to close output file '%s'\n", mask_path.c_str());
        return EXIT_FAILURE;
    }

    printf("mask = 0x%08x\n", h_mask);
    printf("target_smid = %d\n", target_smid);
    printf("Elapsed time: %.3f ms\n", elapsed_ms);
    printf("Wrote mask to %s\n", mask_path.c_str());

    CHECK_CUDA(cudaFree(raw_mem));
    CHECK_CUDA(cudaFree(mask));
    CHECK_CUDA(cudaFree(sink));
    CHECK_CUDA(cudaFree(signal));
    CHECK_CUDA(cudaFree(l2_cache));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
