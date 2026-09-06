#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

/// ****************** Error Handling ****************** ///
#define CHECK_CUDA(call)                                                                                                           \
    do {                                                                                                                           \
        cudaError_t err = (call);                                                                                                  \
        if (err != cudaSuccess) {                                                                                                  \
            ::fprintf(stderr, "[CUDA] %s:%d: %s failed: %s (%d)\n", __FILE__, __LINE__, #call, cudaGetErrorString(err), (int)err); \
            ::fflush(stderr);                                                                                                      \
            ::abort();                                                                                                             \
        }                                                                                                                          \
    } while (0)

#define LATOPT_CHECK(cond, msg)                                                 \
    do {                                                                        \
        if (!(cond)) {                                                          \
            ::fprintf(stderr, "[LATOPT] %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ::fflush(stderr);                                                   \
            ::abort();                                                          \
        }                                                                       \
    } while (0)

/// ****************** Macro Utilities ****************** ///
#define LATOPT_STRINGIFY_IMPL(x) #x
#define LATOPT_STRINGIFY(x)      LATOPT_STRINGIFY_IMPL(x)
#define _HOST                    __host__
#define _DEVICE                  __device__
#define _FORCE_INLINE            __forceinline__
#define _GLOBAL                  __global__
#define _HOST_DEVICE             _HOST _DEVICE
#define _DEVICE_INLINE           _DEVICE _FORCE_INLINE
#define _NS_BEGIN(name)          namespace name {
#define _NS_END(name)            }

_NS_BEGIN(latopt)
constexpr size_t kMemoryAllocGran = 2ULL * 1024ULL * 1024ULL;

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

/// ****************** Singleton Variables ****************** ///
template <typename T> struct GlobalSingletonDevicePtrResource {
    using Type = T *;

    static Type null() { return nullptr; }
    static bool is_null(Type ptr) { return ptr == nullptr; }

    static Type create() {
        T *ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, sizeof(T)));
        return ptr;
    }

    static void init(Type &ptr) { reset(ptr); }
    static void reset(Type &ptr) { CHECK_CUDA(cudaMemset(ptr, 0, sizeof(T))); }

    static void destroy(Type &ptr) noexcept {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
            ptr = nullptr;
        }
    }
};

struct GlobalSingletonNumSmResource {
    using Type = int;

    static Type null() { return 0; }
    static bool is_null(Type value) { return value == 0; }

    static Type create() { return null(); }

    static void init(Type &value) {
        int device = 0;
        CHECK_CUDA(cudaGetDevice(&device));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        value = prop.multiProcessorCount;
    }

    static void reset(Type &) {}
    static void destroy(Type &value) noexcept { value = null(); }
};

struct GlobalSingletonL2CacheDevicePtrResource {
    using Type = uint8_t *;

    static size_t bytes() {
        int device = 0;
        CHECK_CUDA(cudaGetDevice(&device));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        LATOPT_CHECK(prop.l2CacheSize > 0, "device L2 cache size must be positive");
        return static_cast<size_t>(prop.l2CacheSize);
    }

    static Type null() { return nullptr; }
    static bool is_null(Type ptr) { return ptr == nullptr; }

    static Type create() {
        Type ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes()));
        return ptr;
    }

    static void init(Type &ptr) { reset(ptr); }
    static void reset(Type &ptr) { CHECK_CUDA(cudaMemset(ptr, 0, bytes())); }

    static void destroy(Type &ptr) noexcept {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
            ptr = nullptr;
        }
    }
};

template <typename ResourceTraits> class GlobalSingleton {
  public:
    using Resource = typename ResourceTraits::Type;

    static Resource get() {
        auto &singleton = instance();
        singleton.ensure_init();
        return singleton.resource_;
    }

    static void reset() {
        auto &singleton = instance();
        singleton.ensure_init();
        ResourceTraits::reset(singleton.resource_);
    }

  private:
    Resource resource_ = ResourceTraits::null();
    bool initialized_  = false;

    static GlobalSingleton &instance() {
        static GlobalSingleton singleton;
        return singleton;
    }

    void ensure_init() {
        if (ResourceTraits::is_null(resource_)) {
            resource_ = ResourceTraits::create();
        }
        if (!initialized_) {
            ResourceTraits::init(resource_);
            initialized_ = true;
        }
    }

    void cleanup() noexcept {
        if (ResourceTraits::is_null(resource_)) {
            initialized_ = false;
            return;
        }
        ResourceTraits::destroy(resource_);
        resource_    = ResourceTraits::null();
        initialized_ = false;
    }

    GlobalSingleton() = default;
    ~GlobalSingleton() noexcept { cleanup(); }
    GlobalSingleton(const GlobalSingleton &)            = delete;
    GlobalSingleton &operator=(const GlobalSingleton &) = delete;
};

template <typename T> using GlobalSink   = GlobalSingleton<GlobalSingletonDevicePtrResource<T>>;
template <typename T> using GlobalSignal = GlobalSingleton<GlobalSingletonDevicePtrResource<T>>;
using GlobalNumSm                        = GlobalSingleton<GlobalSingletonNumSmResource>;
using GlobalL2Cache                      = GlobalSingleton<GlobalSingletonL2CacheDevicePtrResource>;

/// ****************** Math Utilities ****************** ///
_HOST_DEVICE auto ceil_div(auto x, auto y) { return (x + y - 1) / y; }

/// ****************** PTX Utilities ****************** ///
_DEVICE_INLINE uint32_t __smid() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

_DEVICE_INLINE void discard_line(void *addr) {
    asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(addr) : "memory");
    return;
}

/// ****************** Device Utilities ****************** ///
_DEVICE_INLINE void block_copy(auto *dst, auto *src, int numel) {
    auto gtid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = gtid; i < numel; i += blockDim.x * blockDim.y * blockDim.z) {
        dst[i] = src[i];
    }
}

_DEVICE_INLINE void init_addr_chain(uint64_t *addrs, int numel, size_t stride) {
    auto tidx        = threadIdx.x;
    uint8_t *addr_u8 = reinterpret_cast<uint8_t *>(addrs);
    for (int i = tidx; i < numel; i += blockDim.x) {
        reinterpret_cast<uint64_t *>(addr_u8 + i * stride)[0] = (uint64_t)(addr_u8 + ((i + 1) % numel) * stride);
    }
    __syncthreads();
    if (tidx == 0) {
        reinterpret_cast<uint64_t *>(addr_u8 + (numel - 1) * stride)[0] = (uint64_t)(addr_u8);
    }
    __syncthreads();
}

template <int64_t ITERS> _DEVICE_INLINE void pchase(uint64_t *base_addr, float *lat, uint64_t *sink) {
    uint64_t *p    = base_addr;
    auto start_clk = clock();
    for (int64_t i = 0; i < ITERS; ++i) {
        p = (uint64_t *)(*p);
    }
    auto end_clk  = clock();
    float latency = static_cast<float>(end_clk - start_clk) / ITERS;
    *lat          = latency;
    sink[0]       = reinterpret_cast<uint64_t>(p);
}

_DEVICE_INLINE void pchase_fine_grained(int n, uint64_t *base_addr, float *latencies, uint64_t *sink) {
    uint64_t *p = base_addr;
    for (int i = 0; i < n; ++i) {
        auto start_clk = clock();
        p              = (uint64_t *)(*p);
        sink[0]        = (uint64_t)p;
        auto end_clk   = clock();
        latencies[i]   = static_cast<float>(end_clk - start_clk);
    }
}

constexpr int kMapSmid2SideProbeStartBit   = 5;
constexpr int kMapSmid2SideProbeStopBit    = 21;
constexpr int kMapSmid2SideNumProbeOffsets = kMapSmid2SideProbeStopBit - kMapSmid2SideProbeStartBit;

template <typename AddrT> _HOST_DEVICE _FORCE_INLINE int predict_map_memory_attr_far(int side, uint32_t mask, AddrT addr, int attr) {
    auto const addr_bits = static_cast<uint32_t>(static_cast<uintptr_t>(addr));
#ifdef __CUDA_ARCH__
    auto const parity = __popc(addr_bits & mask) & 0x1;
#else
    auto const parity = __builtin_popcount(addr_bits & mask) & 0x1;
#endif
    return parity ^ side ^ attr;
}

template <typename SinkT> _DEVICE_INLINE void access_map_smid2side_series(const uint8_t *slice_base, SinkT *sink) {
    for (int bit = kMapSmid2SideProbeStartBit; bit < kMapSmid2SideProbeStopBit; ++bit) {
        auto ptr = reinterpret_cast<const uint64_t *>(slice_base + (size_t{1} << bit));
        sink[0] += static_cast<SinkT>(*ptr);
    }
}

template <typename SinkT> _DEVICE_INLINE void measure_map_smid2side_series(const uint8_t *slice_base, uint32_t *latencies, SinkT *sink) {
    // ! prevent measuring parameter load latency
    sink[0] = slice_base[0];
    for (int bit = kMapSmid2SideProbeStartBit; bit < kMapSmid2SideProbeStopBit; ++bit) {
        auto ptr                                    = reinterpret_cast<const uint64_t *>(slice_base + (size_t{1} << bit));
        auto start_clk                              = clock();
        sink[0]                                     = static_cast<SinkT>(*ptr);
        auto end_clk                                = clock();
        auto latency                                = static_cast<uint32_t>(end_clk - start_clk);
        latencies[bit - kMapSmid2SideProbeStartBit] = latency;
    }
    return;
}

template <typename SinkT> _DEVICE_INLINE void access_map_memory_attr_pair(const uint8_t *base, volatile SinkT *sink) {
    auto ptr0 = reinterpret_cast<const uint64_t *>(base);
    sink[0]   = static_cast<SinkT>(*ptr0);
}

template <typename LatT, typename SinkT>
_DEVICE_INLINE void measure_map_memory_attr_pair(const uint8_t *base, LatT *latencies, int slot, volatile SinkT *sink) {
    auto ptr0       = reinterpret_cast<const uint64_t *>(base);
    auto start_clk  = clock();
    sink[0]         = static_cast<SinkT>(*ptr0);
    auto end_clk    = clock();
    latencies[slot] = static_cast<LatT>(end_clk - start_clk);
}

constexpr int kMapMemoryAttrExp2StartBit        = 7;
constexpr int kMapMemoryAttrExp2StopBit         = 21;
constexpr int kMapMemoryAttrExp2NumProbeOffsets = kMapMemoryAttrExp2StopBit - kMapMemoryAttrExp2StartBit;

template <typename SinkT> _DEVICE_INLINE void access_map_memory_attr_exp2_series(const uint8_t *base, volatile SinkT *sink) {
    sink[0] = static_cast<SinkT>(base[128]);
    for (int bit = kMapMemoryAttrExp2StartBit; bit < kMapMemoryAttrExp2StopBit; ++bit) {
        auto ptr = reinterpret_cast<const uint64_t *>(base + (size_t{1} << bit));
        sink[0]  = static_cast<SinkT>(*ptr);
    }
}

template <typename LatT, typename SinkT> _DEVICE_INLINE void measure_map_memory_attr_exp2_series(const uint8_t *base, LatT *latencies, volatile SinkT *sink) {
    sink[0] = static_cast<SinkT>(base[32]);
    for (int bit = kMapMemoryAttrExp2StartBit; bit < kMapMemoryAttrExp2StopBit; ++bit) {
        auto ptr                                    = reinterpret_cast<const uint64_t *>(base + (size_t{1} << bit));
        auto start_clk                              = clock();
        sink[0]                                     = static_cast<SinkT>(*ptr);
        auto end_clk                                = clock();
        latencies[bit - kMapMemoryAttrExp2StartBit] = static_cast<LatT>(end_clk - start_clk);
    }
}

/// ****************** Kernel & Host Functions ****************** ///
static _GLOBAL void k_init_addr_chain(uint64_t *addrs, int numel, size_t stride) {
    init_addr_chain(addrs, numel, stride);
    return;
}

template <int N_THREADS = 512> _HOST inline void init_data(uint64_t *addrs, int numel, size_t stride) {
    LATOPT_CHECK(addrs != nullptr, "addrs must not be null");
    LATOPT_CHECK(numel > 0, "numel must be positive");
    LATOPT_CHECK(stride > 0, "stride must be positive");
    k_init_addr_chain<<<1, N_THREADS>>>(addrs, numel, stride);
    return;
}

template <typename SinkT>
static _GLOBAL void k_pchase(uint64_t *base_addr, float *lat, SinkT* sink) {
    uint64_t sink_;
    pchase<16 * 1024 * 1024>(base_addr, lat, &sink_);
    *sink = sink_;
    return;
}

template <int N_THREADS = 1> _HOST inline void pchase(uint64_t *base_addr, float *latency) {
    LATOPT_CHECK(base_addr != nullptr, "base_addr must not be null");
    LATOPT_CHECK(latency != nullptr, "latency must not be null");
    GlobalL2Cache::reset();
    auto sink   = GlobalSink<uint64_t>::get();
    k_pchase<<<1, N_THREADS>>>(base_addr, latency, sink);
    return;
}

template <typename LatT, typename SinkT, typename SigT>
static _GLOBAL void k_peer_pchase(int pA, int pB, int numel, uint64_t *base_addr, LatT *latencies, SinkT *sink, SigT *signal) {
    auto const smid = __smid();
    if (smid != pA && smid != pB) {
        return;
    }
    extern __shared__ LatT shm_latencies[];
    __shared__ SinkT _sink;
    auto sig = Signal(signal);
    enum { Idle, Ready };
    if (smid == pA) {
        sig.set(Idle);
        pchase_fine_grained(numel, base_addr, shm_latencies, &_sink);
        block_copy(latencies, shm_latencies, numel);
        *sink = _sink;
        pchase_fine_grained(numel, base_addr, shm_latencies, &_sink);
        block_copy(latencies + numel, shm_latencies, numel);
        *sink = _sink;
        sig.set(Ready);
    }

    if (smid == pB) {
        sig.wait(Ready);
        pchase_fine_grained(numel, base_addr, shm_latencies, &_sink);
        block_copy(latencies + 2 * numel, shm_latencies, numel);
        *sink = _sink;
        pchase_fine_grained(numel, base_addr, shm_latencies, &_sink);
        block_copy(latencies + 3 * numel, shm_latencies, numel);
        *sink = _sink;
        sig.set(Idle);
    }
}

template <int N_THREADS = 1> _HOST inline void peer_pchase(int pA, int pB, int numel, uint64_t *pos_arr, float *latencies) {
    LATOPT_CHECK(pA != pB, "SMs must be different");
    LATOPT_CHECK(numel > 0, "numel must be positive");
    LATOPT_CHECK(pos_arr != nullptr, "pos_arr must not be null");
    LATOPT_CHECK(latencies != nullptr, "latencies must not be null");

    int const num_ctas = GlobalNumSm::get();
    LATOPT_CHECK(num_ctas > 0, "num_ctas must be positive");

    size_t smem_size = static_cast<size_t>(numel) * sizeof(float);
    auto Kernel      = k_peer_pchase<float, uint64_t, int>;
    CHECK_CUDA(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    auto sink   = GlobalSink<uint64_t>::get();
    auto signal = GlobalSignal<int>::get();
    GlobalL2Cache::reset();
    GlobalSink<uint64_t>::reset();
    GlobalSignal<int>::reset();
    Kernel<<<num_ctas, N_THREADS, smem_size>>>(pA, pB, numel, pos_arr, latencies, sink, signal);
    return;
}

template <typename LatT, typename SinkT, typename SigT>
_GLOBAL void k_access_addr_list(int pA, int pB, int group_size, int num_groups, const uint64_t *addrs, LatT *latencies, SinkT *sink, SigT *signal) {
    auto const smid = __smid();
    if (smid != pA && smid != pB) {
        return;
    }

    extern __shared__ uint64_t shm_addrs[];

    auto const num_addrs = group_size * num_groups;
    block_copy(shm_addrs, addrs, num_addrs);
    __syncthreads();

    __shared__ SinkT _sink;
    auto sig = Signal(signal);
    enum { Idle };
    if (smid == pA) {
        sig.set(Idle);
        for (int group_id = 0; group_id < num_groups; ++group_id) {
            sig.wait(Idle);
            auto const group_start = group_id * group_size;
            auto const group_end   = group_start + group_size;
            for (int addr_id = group_start; addr_id < group_end; ++addr_id) {
                auto ptr = (uint64_t *)shm_addrs[addr_id];
                _sink    = *ptr;
                sink[0]  = _sink;
            }
            sig.set(group_id + 1);
        }
    }

    if (smid == pB) {
        for (int group_id = 0; group_id < num_groups; ++group_id) {
            sig.wait(group_id + 1, 2000);
            auto const group_start = group_id * group_size;
            auto const group_end   = group_start + group_size;
            for (int addr_id = group_start; addr_id < group_end; ++addr_id) {
                auto ptr           = (uint64_t *)shm_addrs[addr_id];
                auto start_clk     = clock();
                _sink              = *ptr;
                auto end_clk       = clock();
                latencies[addr_id] = static_cast<LatT>(end_clk - start_clk);
            }
            sig.set(Idle);
        }
        sink[0] = _sink;
    }
}

template <int N_THREADS = 1> _HOST inline void access_addr_list(int pA, int pB, int group_size, size_t num_addrs, const uint64_t *addrs, float *latencies) {
    LATOPT_CHECK(pA != pB, "SMs must be different");
    LATOPT_CHECK(group_size > 0, "group_size must be positive");
    LATOPT_CHECK(num_addrs > 0, "num_addrs must be positive");
    LATOPT_CHECK(addrs != nullptr, "addrs must not be null");
    LATOPT_CHECK(latencies != nullptr, "latencies must not be null");
    LATOPT_CHECK(num_addrs % static_cast<size_t>(group_size) == 0, "num_addrs must be divisible by group_size");

    int const num_ctas = GlobalNumSm::get();
    LATOPT_CHECK(num_ctas > 0, "num_ctas must be positive");

    size_t smem_size = num_addrs * sizeof(uint64_t);
    auto Kernel      = k_access_addr_list<float, uint64_t, int>;
    CHECK_CUDA(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    auto sink   = GlobalSink<uint64_t>::get();
    auto signal = GlobalSignal<int>::get();
    GlobalL2Cache::reset();
    GlobalSink<uint64_t>::reset();
    GlobalSignal<int>::reset();
    Kernel<<<num_ctas, N_THREADS, smem_size>>>(pA,
                                               pB,
                                               group_size,
                                               static_cast<int>(num_addrs / static_cast<size_t>(group_size)),
                                               addrs,
                                               latencies,
                                               sink,
                                               signal);
    return;
}

template <typename T, typename SinkT, typename SigT>
static _GLOBAL void k_get_map_smid2side(int num_sm, const uint8_t *workspace, T *map, SinkT *sink, SigT *signal) {
    if (threadIdx.x != 0) {
        return;
    }

    auto const smid = __smid();
    if (smid >= static_cast<uint32_t>(num_sm)) {
        return;
    }

    __shared__ uint32_t shm_latencies[64];
    auto sig = Signal(signal);
    enum : SigT { Idle = static_cast<SigT>(-1) };
    __shared__ SinkT _sink;

    if (smid == 0) {
        sig.set(Idle);
        for (int target_smid = 1; target_smid < num_sm; ++target_smid) {
            sig.wait(Idle);
            auto const target_base = workspace + static_cast<size_t>(target_smid) * kMemoryAllocGran;
            access_map_smid2side_series(target_base, &_sink);
            sig.set(static_cast<SigT>(target_smid));
            sig.wait(Idle);
        }
        sink[0]   = _sink;
        map[smid] = 0;
        return;
    }

    sig.wait(static_cast<SigT>(smid));
    auto const slice_base = workspace + static_cast<size_t>(smid) * kMemoryAllocGran;
    measure_map_smid2side_series(slice_base, shm_latencies, &_sink);
    uint32_t max_latency = 0;
    uint32_t min_latency = 1024;
    for (int i = 0; i < kMapSmid2SideNumProbeOffsets; ++i) {
        auto cur    = shm_latencies[i];
        max_latency = cur > max_latency ? cur : max_latency;
        min_latency = cur < min_latency ? cur : min_latency;
    }
    sink[0]   = _sink;
    map[smid] = (max_latency - min_latency) > 100;
    sig.set(Idle);
}

template <typename T = int> _HOST inline void get_map_smid2side(T *map, size_t &bytes_needed) {
    int const num_sm = GlobalNumSm::get();
    LATOPT_CHECK(num_sm > 0, "num_sm must be positive");

    size_t const map_bytes = static_cast<size_t>(num_sm) * sizeof(T);
    bytes_needed           = map_bytes;
    if (map == nullptr) {
        return;
    }

    CHECK_CUDA(cudaMemset(map, 0, map_bytes));

    size_t const workspace_bytes = static_cast<size_t>(num_sm) * kMemoryAllocGran;

    uint8_t *workspace           = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUDA(cudaMemset(workspace, 0, workspace_bytes));

    auto sink   = GlobalSink<uint64_t>::get();
    auto signal = GlobalSignal<int>::get();
    GlobalL2Cache::reset();
    GlobalSink<uint64_t>::reset();
    GlobalSignal<int>::reset();

    auto Kernel = k_get_map_smid2side<T, uint64_t, int>;
    Kernel<<<num_sm, 1>>>(num_sm, workspace, map, sink, signal);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(workspace));
}

#if CUDART_VERSION >= 13010
template <typename MapT> static _GLOBAL void k_cta_side_count(int *cnt, MapT *smid2sideMap) {
    auto smid = __smid();
    auto side = smid2sideMap[smid];
    if (side == 1) {
        atomicAdd(cnt, 1);
    }
}

template <typename MapT, int NUM_CTAS = 256>
_HOST inline void tag_resources(cudaDevResource *resources, int *tags, MapT *d_smid2sideMap, int num_resources, int device_id = 0) {
    int *d_counts;
    int *h_counts = (int *)malloc(sizeof(int) * num_resources);
    CHECK_CUDA(cudaMalloc(&d_counts, sizeof(int) * num_resources));
    CHECK_CUDA(cudaMemset(d_counts, 0, sizeof(int) * num_resources));

    for (int i = 0; i < num_resources; i++) {
        cudaDevResourceDesc_t resource_desc;
        CHECK_CUDA(cudaDevResourceGenerateDesc(&resource_desc, &resources[i], 1));

        cudaExecutionContext_t green_ctx = {};
        CHECK_CUDA(cudaGreenCtxCreate(&green_ctx, resource_desc, device_id, 0));
        cudaStream_t green_stream = {};
        CHECK_CUDA(cudaExecutionCtxStreamCreate(&green_stream, green_ctx, cudaStreamDefault, 0));

        k_cta_side_count<<<NUM_CTAS, 1, 0, green_stream>>>(&d_counts[i], d_smid2sideMap);

        CHECK_CUDA(cudaStreamSynchronize(green_stream));
        CHECK_CUDA(cudaStreamDestroy(green_stream));
        CHECK_CUDA(cudaExecutionCtxDestroy(green_ctx));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_counts, d_counts, sizeof(int) * num_resources, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_resources; i++) {
        auto cnt = h_counts[i];
        if (cnt == NUM_CTAS)
            tags[i] = 1; // side 1
        else if (cnt == 0)
            tags[i] = 0; // side 0
        else
            tags[i] = -1; // mixed
    }
    CHECK_CUDA(cudaFree(d_counts));
    free(h_counts);
}

template <int NUM_GROUPS = 2> _HOST inline void group_sm_resources(cudaDevResource *resources, int *tags, int num_resources, int *group_cnts) {
    static_assert(NUM_GROUPS == 2, "group_sm_resources currently supports binary tags only");

    memset(group_cnts, 0, sizeof(int) * NUM_GROUPS);
    for (int i = 0; i < num_resources; i++) {
        LATOPT_CHECK(tags[i] == 0 || tags[i] == 1, "tags must be binary");
        group_cnts[tags[i]]++;
    }

    int left  = 0;
    int right = num_resources - 1;
    while (left < right) {
        while (left < right && tags[left] == 0) {
            left++;
        }
        while (left < right && tags[right] == 1) {
            right--;
        }
        if (left < right) {
            auto tmp_resource = resources[left];
            resources[left]   = resources[right];
            resources[right]  = tmp_resource;

            auto tmp_tag      = tags[left];
            tags[left]        = tags[right];
            tags[right]       = tmp_tag;
            left++;
            right--;
        }
    }
}

_HOST inline void create_sm_resource_groups(std::vector<cudaDevResource> &resources, std::vector<int> &group_cnts, int device_id = 0) {
    /// 0. Reset Resource Vector
    resources.clear();
    /// 1. Obtain SM Resource Pool
    cudaDevResource sm_pool{};
    CHECK_CUDA(cudaDeviceGetDevResource(device_id, &sm_pool, cudaDevResourceTypeSm));
    /// 2. Setting Split Parameters --- Based on TPC
    uint32_t num_groups = sm_pool.sm.smCount / 2; // Number of TPCs
    resources.resize(num_groups);
    std::vector<cudaDevSmResourceGroupParams> params(num_groups);
    for (uint32_t i = 0; i < num_groups; ++i) {
        params[i].smCount                     = 2;
        params[i].coscheduledSmCount          = 2;
        params[i].preferredCoscheduledSmCount = 0;
        params[i].flags                       = 0;
    }
    /// 3. Split SM Resource Pool into Groups
    cudaDevResource remainder{};
    CHECK_CUDA(cudaDevSmResourceSplit(resources.data(), num_groups, &sm_pool, &remainder, 0u, params.data()));
    if (remainder.sm.smCount > 0) {
        resources.push_back(remainder);
    }
    /// 4. Get SMID-to-side mapping
    int *smid2side_map  = nullptr;
    size_t bytes_needed = 0;
    get_map_smid2side(smid2side_map, bytes_needed);
    CHECK_CUDA(cudaMalloc(&smid2side_map, bytes_needed));
    get_map_smid2side(smid2side_map, bytes_needed);
    /// 5. Tag side for each SM resource group
    std::vector<int> tags(resources.size());
    auto num_resources = static_cast<int>(resources.size());
    tag_resources(resources.data(), tags.data(), smid2side_map, num_resources, device_id);
    /// 6. Permute SM resource groups based on tags, making same side groups contiguous
    group_cnts.resize(2);
    group_sm_resources(resources.data(), tags.data(), num_resources, group_cnts.data());
    CHECK_CUDA(cudaFree(smid2side_map));
}
#endif

template <typename T, typename SinkT, typename SigT>
static _GLOBAL void
k_get_map_memory_attr(const uint8_t *mem, size_t num_pages, T *attr_map, uint32_t *mask, const T *smid2side_map, SinkT *sink, SigT *signal) {
    if (threadIdx.x != 0) {
        return;
    }

    auto const smid = __smid();
    if (smid != 0 && smid2side_map[smid] != static_cast<T>(1)) {
        return;
    }

    auto sig = Signal(signal);
    __shared__ volatile SinkT _sink;
    __shared__ uint32_t shm_latency[1];
    __shared__ uint32_t shm_exp2_latencies[kMapMemoryAttrExp2NumProbeOffsets];
    auto const num_sm   = static_cast<int>(gridDim.x);
    float exp2_avg      = 0.0f;
    uint32_t local_mask = 0;
    bool base_attr      = false;

    auto select_id      = [&](int smid) { return -smid; };

    _sink               = static_cast<SinkT>(0);

    if (smid == 0) {
        sig.set(static_cast<SigT>(0));
        int winner_smid = 0;
        for (int candidate_smid = 1; candidate_smid < num_sm; ++candidate_smid) {
            if (smid2side_map[candidate_smid] == static_cast<T>(1)) {
                winner_smid = candidate_smid;
                break;
            }
        }
        if (winner_smid == 0) {
            return;
        }

        sig.set(static_cast<SigT>(select_id(winner_smid)));

        for (size_t page_id = 0; page_id < num_pages; ++page_id) {
            sig.wait(static_cast<SigT>(page_id + 1), 2000);
            auto const base = mem + page_id * kMemoryAllocGran;
            // _sink           = base[128]; // ! warm up TLB
            if (page_id == 0) {
                measure_map_memory_attr_exp2_series<uint32_t>(base, shm_exp2_latencies, &_sink);
            }
            measure_map_memory_attr_pair<uint32_t>(base, shm_latency, 0, &_sink);
            auto const latency = page_id == 0 ? shm_exp2_latencies[0] : shm_latency[0];
            if (page_id == 0) {
                for (int i = 0; i < kMapMemoryAttrExp2NumProbeOffsets; ++i) {
                    exp2_avg += static_cast<float>(shm_exp2_latencies[i]);
                }
                exp2_avg /= static_cast<float>(kMapMemoryAttrExp2NumProbeOffsets);
                base_attr = static_cast<float>(latency) > exp2_avg;
                for (int i = 0; i < kMapMemoryAttrExp2NumProbeOffsets; ++i) {
                    auto cur_attr = static_cast<float>(shm_exp2_latencies[i]) > exp2_avg;
                    if (cur_attr != base_attr) {
                        local_mask |= uint32_t{1} << (i + kMapMemoryAttrExp2StartBit);
                    }
                }
                mask[0] = local_mask;
            }
            attr_map[page_id] = static_cast<T>(static_cast<float>(latency) > exp2_avg);
            if (page_id + 1 < num_pages) {
                sig.set(static_cast<SigT>(select_id(winner_smid)));
            }
        }

        sig.set(static_cast<SigT>(0));
        sink[0] = static_cast<SinkT>(_sink);
        return;
    }

    while (true) {
        auto const observed = atomicCAS(signal, static_cast<SigT>(0), static_cast<SigT>(0));
        if (observed == 0) {
            __nanosleep(2000);
            continue;
        }
        if (observed != static_cast<SigT>(static_cast<int>(-smid))) {
            return;
        }
        break;
    }

    for (size_t page_id = 0; page_id < num_pages; ++page_id) {
        auto const base = mem + page_id * kMemoryAllocGran;
        access_map_memory_attr_pair(base, &_sink);
        if (page_id == 0) {
            access_map_memory_attr_exp2_series(base, &_sink);
        }
        sig.set(static_cast<SigT>(page_id + 1));
        if (page_id + 1 < num_pages) {
            sig.wait(static_cast<SigT>(select_id(static_cast<int>(smid))));
        }
    }

    sink[0] = static_cast<SinkT>(_sink);
}

template <typename T = int>
_HOST inline void get_map_memory_attr(void *mem, size_t bytesOfMem, T *attrMap, size_t &bytesOfMap, uint32_t *mask, T *smid2sideMap) {
    LATOPT_CHECK(mem != nullptr, "mem must not be null");
    LATOPT_CHECK(reinterpret_cast<uintptr_t>(mem) % kMemoryAllocGran == 0, "mem must be 2 MiB aligned");
    LATOPT_CHECK(bytesOfMem % kMemoryAllocGran == 0, "bytesOfMem must be divisible by 2 MiB");

    size_t const num_pages = bytesOfMem / kMemoryAllocGran;
    bytesOfMap             = num_pages * sizeof(T);
    if (attrMap == nullptr || num_pages == 0) {
        return;
    }

    LATOPT_CHECK(smid2sideMap != nullptr, "smid2sideMap must not be null");
    LATOPT_CHECK(mask != nullptr, "mask must not be null");
    CHECK_CUDA(cudaMemset(attrMap, 0, bytesOfMap));
    CHECK_CUDA(cudaMemset(mask, 0, sizeof(*mask)));

    int const num_ctas = GlobalNumSm::get();
    LATOPT_CHECK(num_ctas > 0, "num_ctas must be positive");

    auto sink   = GlobalSink<uint64_t>::get();
    auto signal = GlobalSignal<int>::get();
    GlobalL2Cache::reset();
    GlobalSink<uint64_t>::reset();
    GlobalSignal<int>::reset();

    auto Kernel = k_get_map_memory_attr<T, uint64_t, int>;
    Kernel<<<num_ctas, 1>>>(reinterpret_cast<const uint8_t *>(mem), num_pages, attrMap, mask, smid2sideMap, sink, signal);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

_NS_END(latopt)
