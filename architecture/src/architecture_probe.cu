#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../utility/latopt.cuh"

#define CUDA_CHECK(call)                                                                                              \
    do {                                                                                                              \
        cudaError_t const error__ = (call);                                                                           \
        if (error__ != cudaSuccess) {                                                                                  \
            std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,                 \
                         cudaGetErrorString(error__));                                                                 \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                             \
    } while (0)

namespace {

constexpr std::size_t kCacheLineBytes = 128;
constexpr std::size_t kProbeAlignment = 2ULL * 1024ULL * 1024ULL;
struct Options {
    int figure = 0;
    int device = 0;
    std::string peer = "auto";
    std::string side_map;
    std::string output;
};

struct SmPair {
    int a = -1;
    int b = -1;
};

[[noreturn]] void fail(std::string const &message);

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }

    DeviceBuffer(DeviceBuffer const &) = delete;
    DeviceBuffer &operator=(DeviceBuffer const &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) {
                (void)cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (ptr_ != nullptr) {
            CUDA_CHECK(cudaFree(ptr_));
            ptr_ = nullptr;
        }
        count_ = count;
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count_ * sizeof(T)));
        }
    }

    T *get() { return ptr_; }
    T const *get() const { return ptr_; }
    std::size_t size() const { return count_; }

  private:
    T *ptr_ = nullptr;
    std::size_t count_ = 0;
};

class AlignedWorkspace {
  public:
    explicit AlignedWorkspace(std::size_t bytes,
                              std::size_t alignment = kProbeAlignment)
        : storage_(bytes + alignment - 1), bytes_(bytes) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            fail("workspace alignment must be a power of two");
        }
        auto const raw = reinterpret_cast<std::uintptr_t>(storage_.get());
        auto const aligned = (raw + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
        data_ = reinterpret_cast<std::uint8_t *>(aligned);
    }

    std::uint8_t *data() { return data_; }
    std::uint8_t const *data() const { return data_; }
    std::size_t size() const { return bytes_; }

  private:
    DeviceBuffer<std::uint8_t> storage_;
    std::uint8_t *data_ = nullptr;
    std::size_t bytes_ = 0;
};

struct ProbeContext {
    int device = 0;
    cudaDeviceProp property{};

    explicit ProbeContext(int device_id) : device(device_id) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaGetDeviceProperties(&property, device));
        if (property.multiProcessorCount < 2) {
            std::fprintf(stderr, "device %d has fewer than two SMs\n", device);
            std::exit(EXIT_FAILURE);
        }
    }
};

[[noreturn]] void fail(std::string const &message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    std::exit(EXIT_FAILURE);
}

bool parse_int(std::string_view text, int &value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    long const parsed = std::strtol(std::string(text).c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void print_usage(char const *program) {
    std::fprintf(stderr,
                 "usage: %s --figure N --device N --peer auto|A,B --side-map FILE --output FILE\n"
                 "  --figure N      architecture figure number (1..5)\n"
                 "  --device N      CUDA device ordinal visible to this process\n"
                 "  --peer A,B      explicit SM pair for figures 2, 4, and 5\n"
                 "  --peer auto     discover one representative SM from each locality domain\n"
                 "  --side-map FILE SMID-to-domain CSV used only with --peer auto\n"
                 "  --output FILE   output CSV path\n",
                 program);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string const arg = argv[i];
        auto require_value = [&](char const *name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--figure") {
            auto const value = require_value("--figure");
            if (!parse_int(value, options.figure) || options.figure < 1 || options.figure > 5) {
                fail("--figure must be an integer from 1 through 5");
            }
        } else if (arg == "--device") {
            auto const value = require_value("--device");
            if (!parse_int(value, options.device)) {
                fail("--device must be a non-negative integer");
            }
        } else if (arg == "--peer") {
            options.peer = require_value("--peer");
        } else if (arg == "--side-map") {
            options.side_map = require_value("--side-map");
        } else if (arg == "--output") {
            options.output = require_value("--output");
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            fail("unknown option: " + arg);
        }
    }

    if (options.figure == 0 || options.output.empty()) {
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    return options;
}

SmPair parse_pair(std::string const &text, int num_sms) {
    auto const comma = text.find(',');
    if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos) {
        fail("--peer must be auto or A,B");
    }
    int a = -1;
    int b = -1;
    if (!parse_int(std::string_view(text).substr(0, comma), a) ||
        !parse_int(std::string_view(text).substr(comma + 1), b)) {
        fail("--peer must contain two non-negative SM IDs");
    }
    if (a == b || a >= num_sms || b >= num_sms) {
        std::ostringstream message;
        message << "invalid --peer " << text << " for a device with " << num_sms << " SMs";
        fail(message.str());
    }
    return {a, b};
}

std::string csv_quote(std::string const &text) {
    std::string result = "\"";
    for (char const ch : text) {
        if (ch == '\"') {
            result += "\"\"";
        } else {
            result += ch;
        }
    }
    result += '\"';
    return result;
}

std::ofstream open_csv(std::string const &path) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        fail("failed to open output CSV: " + path);
    }
    output << std::setprecision(10);
    return output;
}

__device__ __forceinline__ std::uint32_t current_smid() {
    std::uint32_t smid = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void initialize_permuted_chain(std::uint8_t *base, int const *permutation,
                                          int num_elements, std::size_t stride) {
    int const thread = static_cast<int>(threadIdx.x);
    for (int position = thread; position < num_elements; position += static_cast<int>(blockDim.x)) {
        int const index = permutation[position];
        int const next_index = permutation[(position + 1) % num_elements];
        auto *slot = reinterpret_cast<std::uint64_t *>(base + static_cast<std::size_t>(index) * stride);
        auto *next = base + static_cast<std::size_t>(next_index) * stride;
        *slot = reinterpret_cast<std::uint64_t>(next);
    }
}

__global__ void record_resource_smids(std::uint32_t *smids) {
    if (threadIdx.x == 0) {
        std::uint32_t const smid = current_smid();
        std::uint32_t const first = atomicCAS(smids, 0xffffffffU, smid);
        if (first != 0xffffffffU && first != smid) {
            (void)atomicCAS(smids + 1, 0xffffffffU, smid);
        }
    }
}

struct TpcRecord {
    int id = -1;
    int sm0 = -1;
    int sm1 = -1;
};

std::vector<TpcRecord> enumerate_tpcs(ProbeContext &context) {
#if CUDART_VERSION < 13010
    (void)context;
    fail("Figure 3 requires CUDA Runtime 13.1 or newer for Green Context TPC enumeration");
#else
    cudaDevResource sm_pool{};
    CUDA_CHECK(cudaDeviceGetDevResource(context.device, &sm_pool, cudaDevResourceTypeSm));
    if (sm_pool.sm.smCount < 2 || sm_pool.sm.smCount % 2 != 0) {
        fail("Figure 3 requires an even SM resource pool that can be split into two-SM TPCs");
    }

    std::uint32_t const num_tpcs = sm_pool.sm.smCount / 2;
    std::vector<cudaDevResource> resources(num_tpcs);
    std::vector<cudaDevSmResourceGroupParams> params(num_tpcs);
    for (auto &param : params) {
        param.smCount = 2;
        param.coscheduledSmCount = 2;
        param.preferredCoscheduledSmCount = 0;
        param.flags = 0;
    }

    cudaDevResource remainder{};
    CUDA_CHECK(cudaDevSmResourceSplit(resources.data(), num_tpcs, &sm_pool, &remainder, 0U,
                                      params.data()));
    if (remainder.sm.smCount != 0) {
        fail("CUDA left unmatched SM resources while enumerating two-SM TPCs");
    }

    DeviceBuffer<std::uint32_t> device_smids(static_cast<std::size_t>(num_tpcs) * 2);
    CUDA_CHECK(cudaMemset(device_smids.get(), 0xff, device_smids.size() * sizeof(std::uint32_t)));
    for (std::uint32_t tpc = 0; tpc < num_tpcs; ++tpc) {
        cudaDevResourceDesc_t description{};
        CUDA_CHECK(cudaDevResourceGenerateDesc(&description, &resources[tpc], 1));
        cudaExecutionContext_t green_context{};
        CUDA_CHECK(cudaGreenCtxCreate(&green_context, description, context.device, 0));
        cudaStream_t stream{};
        CUDA_CHECK(cudaExecutionCtxStreamCreate(&stream, green_context, cudaStreamDefault, 0));
        record_resource_smids<<<64, 1, 0, stream>>>(device_smids.get() + 2 * tpc);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaExecutionCtxDestroy(green_context));
    }

    std::vector<std::uint32_t> host_smids(device_smids.size());
    CUDA_CHECK(cudaMemcpy(host_smids.data(), device_smids.get(), host_smids.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));

    std::vector<TpcRecord> result;
    result.reserve(num_tpcs);
    std::vector<int> seen(static_cast<std::size_t>(context.property.multiProcessorCount), 0);
    for (std::uint32_t tpc = 0; tpc < num_tpcs; ++tpc) {
        int sm0 = static_cast<int>(host_smids[2 * tpc]);
        int sm1 = static_cast<int>(host_smids[2 * tpc + 1]);
        if (sm0 == sm1 || sm0 < 0 || sm1 < 0 || sm0 >= context.property.multiProcessorCount ||
            sm1 >= context.property.multiProcessorCount) {
            fail("Green Context TPC enumeration returned invalid or duplicate SM IDs");
        }
        if (seen[static_cast<std::size_t>(sm0)] != 0 || seen[static_cast<std::size_t>(sm1)] != 0) {
            fail("Green Context TPC enumeration assigned an SM to more than one TPC");
        }
        seen[static_cast<std::size_t>(sm0)] = 1;
        seen[static_cast<std::size_t>(sm1)] = 1;
        if (sm1 < sm0) {
            std::swap(sm0, sm1);
        }
        result.push_back({static_cast<int>(tpc), sm0, sm1});
    }
    if (std::accumulate(seen.begin(), seen.end(), 0) != context.property.multiProcessorCount) {
        fail("Green Context TPC enumeration did not cover every physical SM");
    }
    return result;
#endif
}

std::vector<int> load_side_map(std::string const &path, int num_sms) {
    if (path.empty()) {
        return {};
    }
    std::ifstream input(path);
    if (!input) {
        fail("failed to open SMID-to-domain map: " + path);
    }
    std::vector<int> result;
    int side = -1;
    while (input >> side) {
        if (side != 0 && side != 1) {
            fail("SMID-to-domain map must contain only 0 and 1: " + path);
        }
        result.push_back(side);
        if (input.peek() == ',') {
            input.ignore();
        }
    }
    if (result.size() != static_cast<std::size_t>(num_sms)) {
        std::ostringstream message;
        message << "SMID-to-domain map " << path << " has " << result.size()
                << " entries; device has " << num_sms << " SMs";
        fail(message.str());
    }
    return result;
}

SmPair choose_pair(ProbeContext &context, std::string const &peer,
                   std::vector<int> const &side_map) {
    if (peer != "auto") {
        return parse_pair(peer, context.property.multiProcessorCount);
    }
    if (side_map.empty()) {
        fail("--peer auto requires --side-map from utility/get_smid2side.cu");
    }

    SmPair pair;
    for (int smid = 0; smid < context.property.multiProcessorCount; ++smid) {
        if (side_map[static_cast<std::size_t>(smid)] == 0 && pair.a < 0) {
            pair.a = smid;
        }
        if (side_map[static_cast<std::size_t>(smid)] == 1 && pair.b < 0) {
            pair.b = smid;
        }
    }
    if (pair.a < 0 || pair.b < 0) {
        fail("automatic peer selection did not find two locality domains; pass --peer A,B only if the GPU is known to expose multiple locality domains");
    }
    std::fprintf(stderr, "auto-selected cross-domain SM pair %d,%d\n", pair.a, pair.b);
    return pair;
}

std::vector<float> measure_offsets(ProbeContext const &context, int leader, int follower,
                                   AlignedWorkspace const &workspace,
                                   std::vector<std::uint64_t> const &offsets) {
    if (leader == follower || offsets.empty()) {
        fail("offset measurement requires distinct SMs and at least one offset");
    }
    if (leader < 0 || follower < 0 || leader >= context.property.multiProcessorCount ||
        follower >= context.property.multiProcessorCount) {
        fail("offset measurement received an out-of-range SM ID");
    }
    auto const maximum = *std::max_element(offsets.begin(), offsets.end());
    if (maximum + sizeof(std::uint64_t) > workspace.size()) {
        fail("offset measurement workspace is too small");
    }

    std::vector<std::uint64_t> addresses;
    addresses.reserve(offsets.size());
    auto const base = reinterpret_cast<std::uintptr_t>(workspace.data());
    for (auto const offset : offsets) {
        addresses.push_back(static_cast<std::uint64_t>(base + offset));
    }

    DeviceBuffer<std::uint64_t> device_addresses(addresses.size());
    DeviceBuffer<float> device_latencies(addresses.size());
    CUDA_CHECK(cudaMemcpy(device_addresses.get(), addresses.data(),
                          addresses.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice));

    // Match the paper's DomainScan/MaskProbe protocol: run the complete probe
    // twice and keep only the second result.  Each call performs its own L2
    // reset and copies the address list into shared memory before measuring.
    latopt::access_addr_list(leader, follower, static_cast<int>(addresses.size()),
                             addresses.size(), device_addresses.get(), device_latencies.get());
    latopt::access_addr_list(leader, follower, static_cast<int>(addresses.size()),
                             addresses.size(), device_addresses.get(), device_latencies.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(offsets.size());
    CUDA_CHECK(cudaMemcpy(result.data(), device_latencies.get(), result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

std::vector<float> measure_peer_traces(ProbeContext const &context, SmPair pair,
                                       std::uint64_t *chain, int num_elements,
                                       bool prime_l2_reset = false) {
    if (pair.a == pair.b || pair.a < 0 || pair.b < 0 ||
        pair.a >= context.property.multiProcessorCount ||
        pair.b >= context.property.multiProcessorCount || num_elements <= 0) {
        fail("peer pointer chase received invalid SM IDs or chain length");
    }

    DeviceBuffer<float> device_latencies(static_cast<std::size_t>(4 * num_elements));
    if (prime_l2_reset) {
        // GlobalL2Cache's first use performs an initialization reset plus the
        // requested reset.  Figure 5 has already used the singleton for its
        // mask probe, so reproduce that complete first-round flush explicitly.
        latopt::GlobalL2Cache::reset();
    }
    latopt::peer_pchase(pair.a, pair.b, num_elements, chain, device_latencies.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(static_cast<std::size_t>(4 * num_elements));
    CUDA_CHECK(cudaMemcpy(result.data(), device_latencies.get(), result.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return result;
}

std::vector<std::uint64_t> bit_offsets(int first_bit, int last_bit) {
    std::vector<std::uint64_t> offsets;
    offsets.reserve(static_cast<std::size_t>(last_bit - first_bit + 1));
    for (int bit = first_bit; bit <= last_bit; ++bit) {
        offsets.push_back(std::uint64_t{1} << bit);
    }
    return offsets;
}

std::vector<bool> derive_active_bits(std::vector<float> const &a_follower,
                                     std::vector<float> const &b_follower) {
    if (a_follower.size() != b_follower.size() || a_follower.empty()) {
        fail("active-bit derivation requires two equally sized latency curves");
    }
    auto relation = [](float a, float b) -> int {
        if (a == b) {
            return 0;
        }
        return a > b ? 1 : -1;
    };
    int base_relation = relation(a_follower.front(), b_follower.front());
    if (base_relation == 0) {
        for (std::size_t index = 1; index < a_follower.size() && base_relation == 0; ++index) {
            base_relation = relation(a_follower[index], b_follower[index]);
        }
    }
    if (base_relation == 0) {
        fail("the two directional curves are indistinguishable; cannot derive active address bits");
    }

    std::vector<bool> active(a_follower.size(), false);
    for (std::size_t index = 0; index < active.size(); ++index) {
        int const current = relation(a_follower[index], b_follower[index]);
        active[index] = current != 0 && current != base_relation;
    }
    return active;
}

void run_figure_1(ProbeContext &context, std::string const &output_path) {
    constexpr std::size_t start_bytes = 1024;
    constexpr std::size_t stop_bytes = 1024ULL * 1024ULL * 1024ULL;
    constexpr std::uint64_t iterations = 16ULL * 1024ULL * 1024ULL;
    constexpr std::uint32_t base_seed = 1373;

    std::vector<std::size_t> regions;
    std::size_t region = start_bytes;
    while (region <= stop_bytes) {
        std::size_t const aligned = region - region % kCacheLineBytes;
        if (aligned >= kCacheLineBytes && (regions.empty() || regions.back() != aligned)) {
            regions.push_back(aligned);
        }
        region = static_cast<std::size_t>(static_cast<double>(region) * 1.1 + 32.0);
    }

    DeviceBuffer<std::uint8_t> workspace(regions.back());
    DeviceBuffer<int> device_permutation(regions.back() / kCacheLineBytes);
    DeviceBuffer<float> device_latency(1);
    auto output = open_csv(output_path);
    output << "device_index,device_name,region_bytes,stride_bytes,iterations,base_seed,permutation_seed,latency_cycles,data_source\n";

    for (std::size_t const bytes : regions) {
        int const num_elements = static_cast<int>(bytes / kCacheLineBytes);
        std::uint32_t const permutation_seed = base_seed + static_cast<std::uint32_t>(num_elements);
        std::mt19937 random(permutation_seed);
        std::vector<int> permutation(static_cast<std::size_t>(num_elements));
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), random);
        CUDA_CHECK(cudaMemcpy(device_permutation.get(), permutation.data(),
                              permutation.size() * sizeof(int), cudaMemcpyHostToDevice));
        initialize_permuted_chain<<<1, 256>>>(workspace.get(), device_permutation.get(),
                                              num_elements, kCacheLineBytes);
        CUDA_CHECK(cudaGetLastError());
        // Keep the paper's fixed-seed random chain, but use lat-opt's exact
        // 16M-iteration timer and L2 reset implementation.
        latopt::pchase(reinterpret_cast<std::uint64_t *>(workspace.get()), device_latency.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float latency = 0.0F;
        CUDA_CHECK(cudaMemcpy(&latency, device_latency.get(), sizeof(latency), cudaMemcpyDeviceToHost));
        output << context.device << ',' << csv_quote(context.property.name) << ',' << bytes << ','
               << kCacheLineBytes << ',' << iterations << ',' << base_seed << ','
               << permutation_seed << ',' << latency << ",sidedgpu\n";
        std::fprintf(stderr, "figure 1: %zu bytes -> %.2f cycles\n", bytes, latency);
    }
}

void run_figure_2(ProbeContext &context, std::string const &peer,
                  std::vector<int> const &side_map,
                  std::string const &output_path) {
    constexpr int num_elements = 2048;
    constexpr std::size_t stride = 128;
    SmPair const pair = choose_pair(context, peer, side_map);

    DeviceBuffer<std::uint8_t> workspace(static_cast<std::size_t>(num_elements) * stride);
    latopt::init_data(reinterpret_cast<std::uint64_t *>(workspace.get()), num_elements, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto const latencies = measure_peer_traces(
        context, pair, reinterpret_cast<std::uint64_t *>(workspace.get()), num_elements);

    auto output = open_csv(output_path);
    output << "device_index,device_name,leader_sm,follower_sm,index,offset_bytes,leader_first_cycles,leader_second_cycles,follower_first_cycles,follower_second_cycles\n";
    for (int index = 0; index < num_elements; ++index) {
        output << context.device << ',' << csv_quote(context.property.name) << ',' << pair.a << ','
               << pair.b << ',' << index << ',' << static_cast<std::size_t>(index) * stride << ','
               << latencies[static_cast<std::size_t>(index)] << ','
               << latencies[static_cast<std::size_t>(num_elements + index)] << ','
               << latencies[static_cast<std::size_t>(2 * num_elements + index)] << ','
               << latencies[static_cast<std::size_t>(3 * num_elements + index)] << '\n';
    }
}

void run_figure_3(ProbeContext &context, std::string const &output_path) {
    constexpr int first_bit = 7;
    constexpr int last_bit = 31;
    auto const offsets = bit_offsets(first_bit, last_bit);
    AlignedWorkspace workspace(offsets.back() + sizeof(std::uint64_t));
    CUDA_CHECK(cudaMemset(workspace.data(), 0, workspace.size()));

    auto tpcs = enumerate_tpcs(context);
    std::sort(tpcs.begin(), tpcs.end(), [](TpcRecord const &left, TpcRecord const &right) {
        return left.sm0 < right.sm0 || (left.sm0 == right.sm0 && left.sm1 < right.sm1);
    });
    constexpr int leader = 0;

    auto output = open_csv(output_path);
    output << "device_index,device_name,leader_sm,tpc_id,resource_id,tpc_sm0,tpc_sm1,follower_sm,address_bit,latency_cycles\n";
    for (std::size_t display_tpc = 0; display_tpc < tpcs.size(); ++display_tpc) {
        auto const &tpc = tpcs[display_tpc];
        int follower = tpc.sm1;
        if (follower == leader) {
            follower = tpc.sm0;
        }
        if (follower == leader) {
            continue;
        }

        auto const latencies = measure_offsets(context, leader, follower, workspace, offsets);
        for (std::size_t index = 0; index < offsets.size(); ++index) {
            output << context.device << ',' << csv_quote(context.property.name) << ',' << leader << ','
                   << display_tpc << ',' << tpc.id << ',' << tpc.sm0 << ',' << tpc.sm1 << ',' << follower << ','
                   << first_bit + static_cast<int>(index) << ',' << latencies[index] << '\n';
        }
    }
    std::fprintf(stderr, "figure 3: measured %zu TPC resources\n", tpcs.size());
}

struct MaskProbeResult {
    SmPair pair;
    int first_bit = 5;
    int last_bit = 23;
    std::vector<float> a_follower;
    std::vector<float> b_follower;
    std::vector<bool> observed_flip;
    std::vector<bool> active;
    std::uint64_t active_mask = 0;
};

MaskProbeResult run_mask_probe(ProbeContext &context, std::string const &peer,
                               std::vector<int> const &side_map) {
    MaskProbeResult result;
    result.pair = choose_pair(context, peer, side_map);
    auto const offsets = bit_offsets(result.first_bit, result.last_bit);
    AlignedWorkspace workspace(offsets.back() + sizeof(std::uint64_t));
    CUDA_CHECK(cudaMemset(workspace.data(), 0, workspace.size()));

    result.a_follower = measure_offsets(context, result.pair.b, result.pair.a, workspace, offsets);
    result.b_follower = measure_offsets(context, result.pair.a, result.pair.b, workspace, offsets);
    result.observed_flip = derive_active_bits(result.a_follower, result.b_follower);
    result.active = result.observed_flip;
    for (std::size_t index = 0; index < result.active.size(); ++index) {
        int const bit = result.first_bit + static_cast<int>(index);
        // cudaMalloc guarantees 2 MiB alignment, so only bits 0..20 have the
        // same virtual and physical meaning.  Keep bits 21..23 in the raw
        // curves, but do not claim page-dependent flips as stable mask bits.
        if (bit > 20) {
            result.active[index] = false;
        } else if (result.active[index]) {
            result.active_mask |= std::uint64_t{1} << bit;
        }
    }
    return result;
}

void run_figure_4(ProbeContext &context, std::string const &peer,
                  std::vector<int> const &side_map,
                  std::string const &output_path) {
    auto const probe = run_mask_probe(context, peer, side_map);
    auto output = open_csv(output_path);
    output << "device_index,device_name,sm_a,sm_b,address_bit,sm_a_follower_cycles,sm_b_follower_cycles,observed_flip,page_stable,active\n";
    for (std::size_t index = 0; index < probe.active.size(); ++index) {
        int const bit = probe.first_bit + static_cast<int>(index);
        output << context.device << ',' << csv_quote(context.property.name) << ',' << probe.pair.a << ','
               << probe.pair.b << ',' << bit << ','
               << probe.a_follower[index] << ',' << probe.b_follower[index] << ','
               << (probe.observed_flip[index] ? 1 : 0) << ',' << (bit <= 20 ? 1 : 0) << ','
               << (probe.active[index] ? 1 : 0) << '\n';
    }
    std::fprintf(stderr, "figure 4: active address mask = 0x%llx\n",
                 static_cast<unsigned long long>(probe.active_mask));
}

void run_figure_5(ProbeContext &context, std::string const &peer,
                  std::vector<int> const &side_map,
                  std::string const &output_path) {
    constexpr int rounds = 2;
    constexpr int num_elements = 1024;
    constexpr std::size_t stride = 128;
    constexpr std::size_t round_bytes = static_cast<std::size_t>(num_elements) * stride;
    constexpr std::size_t total_bytes = rounds * round_bytes;
    // Recover the active bits from a fresh bidirectional MaskProbe in this
    // process so Figure 5's boundary labels come from its own measurements.
    auto const mask_probe = run_mask_probe(context, peer, side_map);
    SmPair const pair = mask_probe.pair;
    std::uint64_t const active_mask = mask_probe.active_mask;

    // This is the exact lat-opt/paper protocol: two adjacent 128 KiB chains,
    // each measured as a full four-trace XDom-PChase after its own L2 reset.
    // Figure 5 retains only the follower's first traversal from each round.
    AlignedWorkspace workspace(total_bytes);
    CUDA_CHECK(cudaMemset(workspace.data(), 0, workspace.size()));

    for (int round = 0; round < rounds; ++round) {
        auto *round_base = reinterpret_cast<std::uint64_t *>(workspace.data() + round * round_bytes);
        latopt::init_data(round_base, num_elements, stride);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::vector<float>> round_latencies;
    round_latencies.reserve(rounds);
    for (int round = 0; round < rounds; ++round) {
        auto *round_base = reinterpret_cast<std::uint64_t *>(workspace.data() + round * round_bytes);
        round_latencies.push_back(
            measure_peer_traces(context, pair, round_base, num_elements, round == 0));
    }

    std::ostringstream mask_text;
    mask_text << "0x" << std::hex << active_mask;
    int first_boundary_bit = 18;
    for (int bit = 7; bit <= 17; ++bit) {
        if ((active_mask & (std::uint64_t{1} << bit)) != 0) {
            first_boundary_bit = bit;
            break;
        }
    }
    auto output = open_csv(output_path);
    output << "device_index,device_name,leader_sm,follower_sm,round_id,index,offset_bytes,latency_cycles,active_mask_hex,boundary_bit,boundary_kind\n";
    for (int round = 0; round < rounds; ++round) {
        auto const &latencies = round_latencies[static_cast<std::size_t>(round)];
        for (int index = 0; index < num_elements; ++index) {
            std::uint64_t const offset = static_cast<std::uint64_t>(round) * round_bytes +
                                         static_cast<std::uint64_t>(index) * stride;
            int boundary_bit = -1;
            std::string boundary_kind;
            if (offset != 0 && (offset & (offset - 1)) == 0) {
                boundary_bit = static_cast<int>(std::log2(static_cast<double>(offset)));
                if (boundary_bit >= first_boundary_bit && boundary_bit <= 17) {
                    bool const is_active = (active_mask & (std::uint64_t{1} << boundary_bit)) != 0;
                    boundary_kind = is_active ? "active" : "inactive";
                } else {
                    boundary_bit = -1;
                }
            }
            output << context.device << ',' << csv_quote(context.property.name) << ',' << pair.a << ','
                   << pair.b << ',' << round << ',' << index << ',' << offset << ','
                   << latencies[static_cast<std::size_t>(2 * num_elements + index)] << ','
                   << mask_text.str() << ',';
            if (boundary_bit >= 0) {
                output << boundary_bit << ',' << boundary_kind;
            } else {
                output << ',';
            }
            output << '\n';
        }
    }
    std::fprintf(stderr, "figure 5: recovered active address mask = %s\n", mask_text.str().c_str());
}

} // namespace

int main(int argc, char **argv) {
    Options const options = parse_options(argc, argv);
    ProbeContext context(options.device);
    auto const side_map = load_side_map(options.side_map, context.property.multiProcessorCount);
    std::fprintf(stderr, "device %d: %s, %d SMs, L2 %d bytes\n", context.device,
                 context.property.name, context.property.multiProcessorCount,
                 context.property.l2CacheSize);

    switch (options.figure) {
    case 1:
        run_figure_1(context, options.output);
        break;
    case 2:
        run_figure_2(context, options.peer, side_map, options.output);
        break;
    case 3:
        run_figure_3(context, options.output);
        break;
    case 4:
        run_figure_4(context, options.peer, side_map, options.output);
        break;
    case 5:
        run_figure_5(context, options.peer, side_map, options.output);
        break;
    default:
        fail("unreachable figure number");
    }

    std::fprintf(stderr, "wrote %s\n", options.output.c_str());
    return EXIT_SUCCESS;
}
