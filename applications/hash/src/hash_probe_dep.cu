#include "latopt.cuh"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
using namespace latopt;

constexpr size_t CHUNK_BYTES = 4096;
constexpr int EMPTY_KEY = -1;

struct Entry {
    int key;
    int next;
};

constexpr int ENTRIES_PER_CHUNK = CHUNK_BYTES / (int)sizeof(Entry);

__device__ __forceinline__ int2 ld_cs_i2(const int2 *p) { return __ldcs(p); }
__device__ __forceinline__ Entry load_entry(const Entry *p) {
    int2 v = ld_cs_i2(reinterpret_cast<const int2 *>(p));
    return Entry{v.x, v.y};
}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__global__ void warmup_k(const Entry *d, int n) {
    int s = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
        s += d[i].key;
    if (s == -999)
        asm volatile("" ::: "memory");
}

__device__ __forceinline__ int probe_chain_one(const Entry *__restrict__ table,
                                               int mask,
                                               int key,
                                               int &next_key) {
    int slot = static_cast<int>(mix32(static_cast<uint32_t>(key)) & static_cast<uint32_t>(mask));
    Entry found{EMPTY_KEY, key};
    for (int step = 0; step <= mask; step++) {
        Entry e = load_entry(table + slot);
        if (e.key == key) {
            found = e;
            break;
        }
        if (e.key == EMPTY_KEY)
            break;
        slot = (slot + 1) & mask;
    }
    next_key = found.next;
    return (found.key == key) ? slot : -1;
}

__global__ void probe_dep_base(const Entry *__restrict__ table,
                               int mask,
                               const int *__restrict__ queries,
                               int n,
                               int chain_len,
                               int *__restrict__ out) {
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += gridDim.x * blockDim.x) {
        int key = queries[tid];
        int acc = 0;
#pragma unroll 4
        for (int iter = 0; iter < chain_len; iter++) {
            int next_key = key;
            acc += probe_chain_one(table, mask, key, next_key);
            key = next_key;
        }
        out[tid] = acc ^ key;
    }
}

__global__ void probe_dep_sa(const Entry *__restrict__ table,
                             int mask,
                             const int *__restrict__ near_q,
                             const int *__restrict__ near_id,
                             int near_n,
                             const int *__restrict__ far_q,
                             const int *__restrict__ far_id,
                             int far_n,
                             int chain_len,
                             int *__restrict__ out,
                             const uint8_t *__restrict__ side_index,
                             int num_side0) {
    uint8_t si  = side_index[__smid()];
    int my_side = si & 1, my_idx = si >> 1;
    const int *my_q  = (my_side == 0) ? near_q : far_q;
    const int *my_id = (my_side == 0) ? near_id : far_id;
    int my_n         = (my_side == 0) ? near_n : far_n;
    int my_sms       = (my_side == 0) ? num_side0 : ((int)gridDim.x - num_side0);

    for (int tid = my_idx * blockDim.x + threadIdx.x; tid < my_n; tid += my_sms * blockDim.x) {
        int key = my_q[tid];
        int qid = my_id[tid];
        int acc = 0;
#pragma unroll 4
        for (int iter = 0; iter < chain_len; iter++) {
            int next_key = key;
            acc += probe_chain_one(table, mask, key, next_key);
            key = next_key;
        }
        out[qid] = acc ^ key;
    }
}

int main(int argc, char **argv) {
    if (argc != 1 && argc != 8) {
        std::fprintf(stderr, "Usage: %s K Q C L h H runs\n", argv[0]);
        return 2;
    }

    int num_keys      = (argc > 1) ? atoi(argv[1]) : 32000000;
    int num_queries   = (argc > 2) ? atoi(argv[2]) : 8000000;
    size_t table_slots = (argc > 3) ? static_cast<size_t>(strtoull(argv[3], nullptr, 10)) : (1ULL << 26);
    int chain_len     = (argc > 4) ? atoi(argv[4]) : 4;
    int hot_pct       = (argc > 5) ? atoi(argv[5]) : 80;
    int hot_per_side  = (argc > 6) ? atoi(argv[6]) : 100000;
    int runs          = (argc > 7) ? atoi(argv[7]) : 7;
    int BLK           = 256;

    LATOPT_CHECK(num_keys > 0, "num_keys must be positive");
    LATOPT_CHECK(num_queries > 0, "num_queries must be positive");
    LATOPT_CHECK(table_slots > 1 && table_slots <= static_cast<size_t>(INT_MAX), "C is out of range");
    LATOPT_CHECK((table_slots & (table_slots - 1)) == 0, "C must be a power of two");
    LATOPT_CHECK(static_cast<size_t>(num_keys) < table_slots, "K must be smaller than C");
    LATOPT_CHECK(chain_len > 0, "chain_len must be positive");
    LATOPT_CHECK(hot_pct >= 0 && hot_pct <= 100, "hot_pct must be in [0, 100]");
    LATOPT_CHECK(hot_per_side > 0 && 2LL * hot_per_side <= num_keys, "H must be positive and 2*H must not exceed K");
    LATOPT_CHECK(runs > 0, "runs must be positive");

    int mask = static_cast<int>(table_slots) - 1;
    size_t table_bytes = table_slots * sizeof(Entry);
    double effective_load_pct = 100.0 * static_cast<double>(num_keys) / static_cast<double>(table_slots);

    printf("=== Dependent Hash Probe (Table 8) ===\n");
    printf("CONFIG K=%d Q=%d C=%zu effective_load=%.6f L=%d h=%d H=%d runs=%d\n\n",
           num_keys, num_queries, table_slots, effective_load_pct, chain_len, hot_pct, hot_per_side, runs);

    // Allocate aligned table region so we can build the true chunk table first.
    size_t pages = ceil_div(table_bytes, kMemoryAllocGran);
    size_t aligned_sz = pages * kMemoryAllocGran;
    uint8_t *d_raw = nullptr;
    CHECK_CUDA(cudaMalloc(&d_raw, aligned_sz + kMemoryAllocGran));
    CHECK_CUDA(cudaMemset(d_raw, 0, aligned_sz + kMemoryAllocGran));
    uintptr_t base = ((uintptr_t)d_raw + kMemoryAllocGran - 1) & ~(uintptr_t)(kMemoryAllocGran - 1);
    Entry *d_table = (Entry *)base;

    // SM side detection + chunk table for the entry array.
    int *d_sm = nullptr;
    size_t sb = 0;
    get_map_smid2side<int>(nullptr, sb);
    CHECK_CUDA(cudaMallocManaged(&d_sm, sb));
    get_map_smid2side(d_sm, sb);
    int num_sm = (int)(sb / sizeof(int));
    std::vector<uint8_t> h_side_index(num_sm);
    int c0 = 0, c1 = 0;
    for (int i = 0; i < num_sm; i++) {
        int s = d_sm[i];
        int idx = (s == 0) ? c0++ : c1++;
        h_side_index[i] = (uint8_t)((idx << 1) | s);
    }
    uint8_t *d_side_index = nullptr;
    CHECK_CUDA(cudaMalloc(&d_side_index, num_sm));
    CHECK_CUDA(cudaMemcpy(d_side_index, h_side_index.data(), num_sm, cudaMemcpyHostToDevice));

    int *d_attr = nullptr;
    size_t ab = 0;
    uint32_t *d_mask = nullptr;
    get_map_memory_attr<int>((void *)base, aligned_sz, nullptr, ab, nullptr, d_sm);
    CHECK_CUDA(cudaMallocManaged(&d_attr, ab));
    CHECK_CUDA(cudaMallocManaged(&d_mask, sizeof(uint32_t)));
    get_map_memory_attr<int>((void *)base, aligned_sz, d_attr, ab, d_mask, d_sm);
    std::vector<int> h_attr(ab / sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_attr.data(), d_attr, ab, cudaMemcpyDeviceToHost));
    size_t nc = aligned_sz / CHUNK_BYTES;
    std::vector<int8_t> h_ct_true(nc);
    for (size_t c = 0; c < nc; c++) {
        uintptr_t addr = base + c * CHUNK_BYTES;
        size_t pg = (c * CHUNK_BYTES) / kMemoryAllocGran;
        h_ct_true[c] = (int8_t)predict_map_memory_attr_far(0, *d_mask, addr, h_attr[pg]);
    }
    // Build keys so we get two hot pools whose initial bucket chunk maps to
    // opposite sides under the true chunk table.
    std::vector<int> hot0, hot1, keys, nexts;
    hot0.reserve(hot_per_side);
    hot1.reserve(hot_per_side);
    keys.reserve(num_keys);
    int candidate = 1;
    while ((int)hot0.size() < hot_per_side || (int)hot1.size() < hot_per_side) {
        int key = candidate;
        candidate += 2;
        int slot = static_cast<int>(mix32(static_cast<uint32_t>(key)) & static_cast<uint32_t>(mask));
        int side = h_ct_true[slot / ENTRIES_PER_CHUNK];
        if (side == 0 && (int)hot0.size() < hot_per_side)
            hot0.push_back(key);
        else if (side == 1 && (int)hot1.size() < hot_per_side)
            hot1.push_back(key);
    }
    for (int i = 0; i < hot_per_side; i++) {
        keys.push_back(hot0[i]);
        nexts.push_back(hot0[(i + 1) % hot_per_side]);
    }
    for (int i = 0; i < hot_per_side; i++) {
        keys.push_back(hot1[i]);
        nexts.push_back(hot1[(i + 1) % hot_per_side]);
    }
    while ((int)keys.size() < num_keys) {
        int key = candidate;
        candidate += 2;
        keys.push_back(key);
        nexts.push_back(key); // cold keys loop on themselves
    }

    // Shuffle insertion order while preserving each key's dependent successor.
    std::vector<int> perm(num_keys);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<Entry> h_table(table_slots, Entry{EMPTY_KEY, 0});
    for (int p : perm) {
        int key = keys[p];
        int nxt = nexts[p];
        size_t slot = mix32(static_cast<uint32_t>(key)) & static_cast<uint32_t>(mask);
        while (h_table[slot].key != EMPTY_KEY)
            slot = (slot + 1) & static_cast<size_t>(mask);
        h_table[slot].key = key;
        h_table[slot].next = nxt;
    }
    CHECK_CUDA(cudaMemcpy(d_table, h_table.data(), table_bytes, cudaMemcpyHostToDevice));

    // Build skewed query stream: mostly hot keys, alternating the two side pools.
    std::vector<int> h_queries(num_queries), h_near_q, h_far_q, h_near_id, h_far_id;
    h_near_q.reserve(num_queries);
    h_far_q.reserve(num_queries);
    h_near_id.reserve(num_queries);
    h_far_id.reserve(num_queries);
    std::uniform_int_distribution<int> hot0d(0, hot_per_side - 1);
    std::uniform_int_distribution<int> hot1d(0, hot_per_side - 1);
    std::uniform_int_distribution<int> keyd(0, num_keys - 1);
    int hot_queries = (int)((long long)num_queries * hot_pct / 100);
    for (int i = 0; i < num_queries; i++) {
        int key = 0;
        if (i < hot_queries)
            key = (i & 1) ? hot1[hot1d(rng)] : hot0[hot0d(rng)];
        else
            key = keys[keyd(rng)];
        h_queries[i] = key;
        int slot = static_cast<int>(mix32(static_cast<uint32_t>(key)) & static_cast<uint32_t>(mask));
        int qside = h_ct_true[slot / ENTRIES_PER_CHUNK];
        if (qside == 0) {
            h_near_q.push_back(key);
            h_near_id.push_back(i);
        } else {
            h_far_q.push_back(key);
            h_far_id.push_back(i);
        }
    }

    int near_n = (int)h_near_q.size();
    int far_n  = (int)h_far_q.size();
    double hot_set_mb = ((double)hot_per_side * sizeof(Entry)) / 1048576.0;
    printf("table=%.1f MiB chunks=%zu hot_chain_set_per_side=%.1f MiB aware_q0=%d aware_q1=%d\n\n",
           table_bytes / 1048576.0, nc, hot_set_mb, near_n, far_n);

    int *d_queries = nullptr, *d_out_base = nullptr, *d_out_sa = nullptr;
    int *d_near_q = nullptr, *d_far_q = nullptr, *d_near_id = nullptr, *d_far_id = nullptr;
    CHECK_CUDA(cudaMalloc(&d_queries, num_queries * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_base, num_queries * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_sa, num_queries * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_queries, h_queries.data(), num_queries * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_near_q, std::max(near_n, 1) * (int)sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_far_q, std::max(far_n, 1) * (int)sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_near_id, std::max(near_n, 1) * (int)sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_far_id, std::max(far_n, 1) * (int)sizeof(int)));
    if (near_n > 0) {
        CHECK_CUDA(cudaMemcpy(d_near_q, h_near_q.data(), near_n * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_near_id, h_near_id.data(), near_n * sizeof(int), cudaMemcpyHostToDevice));
    }
    if (far_n > 0) {
        CHECK_CUDA(cudaMemcpy(d_far_q, h_far_q.data(), far_n * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_far_id, h_far_id.data(), far_n * sizeof(int), cudaMemcpyHostToDevice));
    }

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    auto warmup = [&]() {
        warmup_k<<<num_sm, BLK>>>(d_table, (int)table_slots);
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    auto run_base = [&]() -> float {
        warmup();
        CHECK_CUDA(cudaEventRecord(e0));
        probe_dep_base<<<num_sm, BLK>>>(d_table, mask, d_queries, num_queries, chain_len, d_out_base);
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        return ms;
    };

    auto run_routed = [&](bool swapped) -> float {
        const int *q0 = swapped ? d_far_q : d_near_q;
        const int *id0 = swapped ? d_far_id : d_near_id;
        int n0 = swapped ? far_n : near_n;
        const int *q1 = swapped ? d_near_q : d_far_q;
        const int *id1 = swapped ? d_near_id : d_far_id;
        int n1 = swapped ? near_n : far_n;
        warmup();
        CHECK_CUDA(cudaEventRecord(e0));
        probe_dep_sa<<<num_sm, BLK>>>(d_table,
                                      mask,
                                      q0,
                                      id0,
                                      n0,
                                      q1,
                                      id1,
                                      n1,
                                      chain_len,
                                      d_out_sa,
                                      d_side_index,
                                      c0);
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));
        return ms;
    };

    run_base();
    run_routed(false);
    std::vector<int> h_ref(num_queries), h_aware(num_queries), h_swapped(num_queries);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_out_base, num_queries * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_aware.data(), d_out_sa, num_queries * sizeof(int), cudaMemcpyDeviceToHost));
    run_routed(true);
    CHECK_CUDA(cudaMemcpy(h_swapped.data(), d_out_sa, num_queries * sizeof(int), cudaMemcpyDeviceToHost));
    bool pass = (h_ref == h_aware) && (h_ref == h_swapped);
    printf("Correctness: %s\n\n", pass ? "PASS" : "FAIL");

    float tb = 0, ta = 0, ts = 0;
    for (int r = 0; r < runs; r++) {
        tb += run_base();
        ta += run_routed(false);
        ts += run_routed(true);
    }
    float baseline_ms = tb / runs;
    float aware_ms = ta / runs;
    float swapped_ms = ts / runs;
    printf("%-28s %8.3f ms\n", "Persistent baseline", baseline_ms);
    printf("%-28s %8.3f ms  (speedup %.2fx)\n", "Aware", aware_ms, baseline_ms / aware_ms);
    printf("%-28s %8.3f ms  (speedup %.2fx)\n", "Swapped", swapped_ms, baseline_ms / swapped_ms);
    printf("RESULT baseline_ms=%.6f aware_ms=%.6f swapped_ms=%.6f aware_speedup=%.6f swapped_speedup=%.6f pass=%s\n",
           baseline_ms, aware_ms, swapped_ms, baseline_ms / aware_ms, baseline_ms / swapped_ms, pass ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_raw));
    CHECK_CUDA(cudaFree(d_sm));
    CHECK_CUDA(cudaFree(d_attr));
    CHECK_CUDA(cudaFree(d_mask));
    CHECK_CUDA(cudaFree(d_side_index));
    CHECK_CUDA(cudaFree(d_queries));
    CHECK_CUDA(cudaFree(d_out_base));
    CHECK_CUDA(cudaFree(d_out_sa));
    CHECK_CUDA(cudaFree(d_near_q));
    CHECK_CUDA(cudaFree(d_far_q));
    CHECK_CUDA(cudaFree(d_near_id));
    CHECK_CUDA(cudaFree(d_far_id));
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return pass ? 0 : 1;
}
