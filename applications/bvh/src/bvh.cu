#include "sidedgpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  do {                                                                         \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",         \
                   __FILE__, __LINE__, cudaGetErrorString(err));                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace {

constexpr size_t kPageBytes = 4096;
#ifndef BVH_THREADS
#define BVH_THREADS 64
#endif
constexpr int kThreads = BVH_THREADS;

struct alignas(16) AabbNode {
  float4 lo;     // minx, miny, minz, maxx
  float4 hi;     // maxy, maxz, unused, unused
  uint4 refs;    // child0, child1, payload, unused
  uint4 aux;     // padding / future dynamic metadata
};

static_assert(sizeof(AabbNode) == 64);
static_assert(kPageBytes % sizeof(AabbNode) == 0);
constexpr uint32_t kNodesPerPage = kPageBytes / sizeof(AabbNode);

struct SidedAllocation {
  std::vector<size_t> pages;
  size_t bytes = 0;
  int side = 0;
};

struct TimingStats {
  double mean = 0.0;
};

struct PhaseSample {
  float total_ms = 0.0f;
  float traverse_ms = 0.0f;
  float refit_ms = 0.0f;
  float rebuild_ms = 0.0f;
};

struct DeviceData {
  unsigned char *pool = nullptr;
  uint32_t *roots0 = nullptr;
  uint32_t *roots1 = nullptr;
  uint32_t *next0 = nullptr;
  uint32_t *next1 = nullptr;
  uint32_t *init0 = nullptr;
  uint32_t *init1 = nullptr;
  const uint32_t *slots0 = nullptr;
  const uint32_t *slots1 = nullptr;
  const uint8_t *node_side = nullptr;
  int nodes0 = 0;
  int nodes1 = 0;
  int count0 = 0;
  int count1 = 0;
  int total_rays = 0;
  uint64_t *checksum = nullptr;
  int *work_counts = nullptr;
  unsigned char *l2_flush = nullptr;
  size_t l2_flush_bytes = 0;
};

struct TimingObjects {
  cudaStream_t regular = nullptr;
  cudaStream_t timing = nullptr;
  cudaStream_t side0 = nullptr;
  cudaStream_t side1 = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEvent_t done0 = nullptr;
  cudaEvent_t done1 = nullptr;
};

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ AabbNode *node_ptr(unsigned char *pool,
                                              uint32_t slot) {
  return reinterpret_cast<AabbNode *>(pool) + slot;
}

__device__ __forceinline__ const AabbNode *node_ptr(const unsigned char *pool,
                                                    uint32_t slot) {
  return reinterpret_cast<const AabbNode *>(pool) + slot;
}

__device__ __forceinline__ AabbNode ld_node(const AabbNode *ptr) {
  const float4 *p = reinterpret_cast<const float4 *>(ptr);
  float4 lo = __ldcs(p + 0);
  float4 hi = __ldcs(p + 1);
  float4 refs = __ldcs(p + 2);

  AabbNode n;
  n.lo = lo;
  n.hi = hi;
  n.refs = make_uint4(__float_as_uint(refs.x), __float_as_uint(refs.y),
                      __float_as_uint(refs.z), __float_as_uint(refs.w));
  n.aux = make_uint4(0u, 0u, 0u, 0u);
  return n;
}

__device__ __forceinline__ AabbNode
ld_node_from_slot(const unsigned char *pool, uint32_t slot) {
  return ld_node(node_ptr(pool, slot));
}

__device__ __forceinline__ float ray_box_t(const AabbNode &n, float3 org,
                                           float3 inv_dir) {
  float tx1 = (n.lo.x - org.x) * inv_dir.x;
  float tx2 = (n.lo.w - org.x) * inv_dir.x;
  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);

  float ty1 = (n.lo.y - org.y) * inv_dir.y;
  float ty2 = (n.hi.x - org.y) * inv_dir.y;
  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float tz1 = (n.lo.z - org.z) * inv_dir.z;
  float tz2 = (n.hi.y - org.z) * inv_dir.z;
  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));

  return (tmax >= fmaxf(tmin, 0.0f)) ? tmin : 1.0e20f;
}

__device__ __forceinline__ void make_ray(uint32_t root, float3 &org,
                                         float3 &inv_dir) {
  uint32_t h = mix32(root ^ 0x51ed270bu);
  org = make_float3(((h & 255u) / 255.0f) * 0.2f,
                    (((h >> 8) & 255u) / 255.0f) * 0.2f,
                    -1.0f - (((h >> 16) & 255u) / 255.0f));
  float dx = 0.35f + ((h & 31u) / 255.0f);
  float dy = 0.35f + (((h >> 5) & 31u) / 255.0f);
  inv_dir = make_float3(1.0f / dx, 1.0f / dy, 1.0f);
}

__global__ void init_nodes_kernel(unsigned char *pool,
                                  const uint32_t *slots_this,
                                  const uint32_t *slots_other, int side,
                                  int nodes_this, int nodes_other,
                                  int locality_pct, uint32_t seed) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes_this) {
    return;
  }

  uint32_t const base_hash =
      mix32(seed ^ static_cast<uint32_t>(i) ^ (side ? 0x9e3779b9u : 0u));
  auto make_child = [&](uint32_t salt) {
    uint32_t h = mix32(base_hash + salt);
    int child_side =
        (static_cast<int>(h % 100u) < locality_pct || nodes_other <= 0)
            ? side
            : (side ^ 1);
    int child_count = child_side == side ? nodes_this : nodes_other;
    const uint32_t *child_slots =
        child_side == side ? slots_this : slots_other;
    int safe_child_count = child_count > 0 ? child_count : 1;
    uint32_t child_index =
        mix32(h ^ 0xa511e9b3u) % static_cast<uint32_t>(safe_child_count);
    return child_slots[child_index];
  };

  uint32_t shape = mix32(base_hash ^ 0x3c6ef372u);
  float center = static_cast<float>((shape & 2047u) + 1u) / 2048.0f;
  float width = 0.35f + static_cast<float>((shape >> 11) & 127u) / 512.0f;
  float minx = center * 0.5f;
  float miny = center * 0.25f;
  float minz = center * 0.125f;

  AabbNode n;
  n.lo = make_float4(minx, miny, minz, minx + width);
  n.hi = make_float4(miny + width, minz + width, 0.0f, 0.0f);
  n.refs = make_uint4(make_child(0x85ebca6bu), make_child(0xc2b2ae35u),
                      mix32(base_hash ^ 0x27d4eb2fu), 0u);
  n.aux = make_uint4(0u, 0u, 0u, 0u);
  *node_ptr(pool, slots_this[i]) = n;
}

__global__ void flush_l2_kernel(unsigned char *buf, size_t bytes) {
  size_t idx =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * 128;
  unsigned int acc = 0;
  for (size_t offset = idx; offset < bytes; offset += stride) {
    acc += buf[offset];
  }
  if (idx == 0) {
    buf[0] = static_cast<unsigned char>(acc);
  }
}

__device__ __forceinline__ uint32_t
traverse_one_ray(const unsigned char *__restrict__ pool,
                 uint32_t root, int steps, uint64_t &local,
                 uint64_t &local_steps) {
  uint32_t ref = root;
  float3 org;
  float3 inv_dir;
  make_ray(root, org, inv_dir);

  for (int step = 0; step < steps; ++step) {
    AabbNode const n = ld_node_from_slot(pool, ref);
    AabbNode const c0 = ld_node_from_slot(pool, n.refs.x);
    AabbNode const c1 = ld_node_from_slot(pool, n.refs.y);
    float const t0 = ray_box_t(c0, org, inv_dir);
    float const t1 = ray_box_t(c1, org, inv_dir);
    uint32_t const tie = mix32(n.refs.z ^ static_cast<uint32_t>(step));
    ref = (t0 < t1 || (t0 == t1 && (tie & 1u))) ? n.refs.x : n.refs.y;
    local += static_cast<uint64_t>(ref) ^
             static_cast<uint64_t>(__float_as_uint(t0 + t1));
    ++local_steps;
  }
  return ref;
}

__global__ void traverse_regular_kernel(const unsigned char *__restrict__ pool,
                                        const uint32_t *__restrict__ roots0,
                                        const uint32_t *__restrict__ roots1,
                                        int count0, int count1, int steps,
                                        uint64_t *__restrict__ checksum,
                                        uint32_t *__restrict__ out0,
                                        uint32_t *__restrict__ out1) {
  int const total = count0 + count1;
  uint64_t local = 0;
  uint64_t local_steps = 0;
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x; logical < total;
       logical += gridDim.x * blockDim.x) {
    uint32_t root =
        logical < count0 ? roots0[logical] : roots1[logical - count0];
    uint32_t final_root = traverse_one_ray(pool, root, steps, local, local_steps);
    if (out0 != nullptr) {
      if (logical < count0) {
        out0[logical] = final_root;
      } else {
        out1[logical - count0] = final_root;
      }
    }
  }
  if (local != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(checksum),
              static_cast<unsigned long long>(local));
  }
  if (local_steps != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(checksum + 2),
              static_cast<unsigned long long>(local_steps));
  }
}

template <int TargetSide>
__global__ void
traverse_partition_kernel(const unsigned char *__restrict__ pool,
                          const uint32_t *__restrict__ primary_roots,
                          uint32_t *__restrict__ primary_out,
                          int primary_count,
                          const uint32_t *__restrict__ spill_roots,
                          uint32_t *__restrict__ spill_out,
                          int spill_start, int spill_count, int steps,
                          uint64_t *__restrict__ checksum) {
  int const count = primary_count + spill_count;
  uint64_t local = 0;
  uint64_t local_steps = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    uint32_t root = 0;
    if (i < primary_count) {
      root = primary_roots[i];
      uint32_t final_root = traverse_one_ray(pool, root, steps, local, local_steps);
      if (primary_out != nullptr) {
        primary_out[i] = final_root;
      }
    } else {
      int const spill_index = spill_start + (i - primary_count);
      root = spill_roots[spill_index];
      uint32_t final_root = traverse_one_ray(pool, root, steps, local, local_steps);
      if (spill_out != nullptr) {
        spill_out[spill_index] = final_root;
      }
    }
  }
  if (local != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(checksum + TargetSide),
              static_cast<unsigned long long>(local));
  }
  if (local_steps != 0) {
    atomicAdd(reinterpret_cast<unsigned long long *>(checksum + 2 + TargetSide),
              static_cast<unsigned long long>(local_steps));
  }
}

__global__ void refit_nodes_kernel(unsigned char *__restrict__ pool,
                                   const uint32_t *__restrict__ slots,
                                   int nodes, int bounce, uint32_t seed) {
  int const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nodes) {
    return;
  }
  uint32_t const slot = slots[i];
  uint32_t const h = mix32(seed ^ static_cast<uint32_t>(bounce) * 0x9e3779b9u ^
                           slot);
  float const center = static_cast<float>((h & 2047u) + 1u) / 2048.0f;
  float const width = 0.35f + static_cast<float>((h >> 11) & 127u) / 512.0f;
  float const minx = center * 0.5f;
  float const miny = center * 0.25f;
  float const minz = center * 0.125f;
  AabbNode *node = node_ptr(pool, slots[i]);
  AabbNode n = *node;
  n.lo = make_float4(minx, miny, minz, minx + width);
  n.hi = make_float4(miny + width, minz + width, 0.0f, 0.0f);
  n.refs.z = mix32(slot ^ h);
  *node = n;
}

__global__ void rebuild_worklists_kernel(const uint32_t *__restrict__ roots0,
                                         int count0,
                                         const uint32_t *__restrict__ roots1,
                                         int count1,
                                         const uint8_t *__restrict__ node_side,
                                         uint32_t *__restrict__ next0,
                                         uint32_t *__restrict__ next1,
                                         int *__restrict__ counts) {
  int const total = count0 + count1;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    uint32_t const root = i < count0 ? roots0[i] : roots1[i - count0];
    int const side = node_side[root] & 1;
    int const index = atomicAdd(counts + side, 1);
    if (side == 0) {
      next0[index] = root;
    } else {
      next1[index] = root;
    }
  }
}

int parse_positive(const char *value, int fallback) {
  if (value == nullptr) {
    return fallback;
  }
  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

int parse_int(const char *value, int fallback) {
  if (value == nullptr) {
    return fallback;
  }
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  return end == value ? fallback : static_cast<int>(parsed);
}

int clamp_int(int value, int lo, int hi) {
  return std::max(lo, std::min(value, hi));
}

SidedAllocation sided_alloc_nodes(size_t nodes, int side) {
  SidedAllocation allocation;
  allocation.bytes = nodes * sizeof(AabbNode);
  allocation.side = side;
  CHECK_CUDA(cudaMallocSided(allocation.bytes, side, allocation.pages));
  size_t const expected_pages =
      (allocation.bytes + kPageBytes - 1) / kPageBytes;
  if (allocation.pages.size() != expected_pages) {
    std::fprintf(stderr, "side %d node allocation returned %zu pages, expected "
                         "%zu\n",
                 side, allocation.pages.size(), expected_pages);
    std::exit(EXIT_FAILURE);
  }
  return allocation;
}

std::vector<uint32_t> make_node_slots(const std::vector<size_t> &pages,
                                      int nodes) {
  std::vector<uint32_t> slots;
  slots.reserve(static_cast<size_t>(nodes));
  for (size_t page : pages) {
    for (uint32_t offset = 0; offset < kNodesPerPage &&
                            slots.size() < static_cast<size_t>(nodes);
         ++offset) {
      uint64_t slot64 = static_cast<uint64_t>(page) * kNodesPerPage + offset;
      if (slot64 > 0xffffffffull) {
        std::fprintf(stderr, "node slot exceeds 32-bit reference range\n");
        std::exit(EXIT_FAILURE);
      }
      slots.push_back(static_cast<uint32_t>(slot64));
    }
  }
  if (slots.size() != static_cast<size_t>(nodes)) {
    std::fprintf(stderr, "node slot generation produced %zu slots, expected %d\n",
                 slots.size(), nodes);
    std::exit(EXIT_FAILURE);
  }
  return slots;
}

void copy_u32(uint32_t **device_values, const std::vector<uint32_t> &values) {
  CHECK_CUDA(cudaMalloc(device_values, values.size() * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(*device_values, values.data(),
                        values.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
}

std::vector<uint32_t> make_roots(int count, const std::vector<uint32_t> &slots,
                                 uint32_t seed) {
  std::vector<uint32_t> roots(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    uint32_t pick = mix32(seed + static_cast<uint32_t>(i) * 0x9e3779b9u) %
                    static_cast<uint32_t>(slots.size());
    roots[static_cast<size_t>(i)] = slots[pick];
  }
  return roots;
}

TimingStats summarize(const std::vector<float> &samples) {
  TimingStats stats;
  if (samples.empty()) {
    return stats;
  }
  double total = 0.0;
  for (float sample : samples) {
    total += sample;
  }
  stats.mean = total / static_cast<double>(samples.size());
  return stats;
}

TimingStats summarize_phase(const std::vector<PhaseSample> &samples) {
  std::vector<float> total;
  total.reserve(samples.size());
  for (auto const &sample : samples) {
    total.push_back(sample.total_ms);
  }
  return summarize(total);
}

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

void flush_only(const DeviceData &data, cudaStream_t stream) {
  flush_l2_kernel<<<256, kThreads, 0, stream>>>(data.l2_flush,
                                                data.l2_flush_bytes);
  CHECK_CUDA(cudaGetLastError());
}

float time_refit(const DeviceData &data, TimingObjects &timing, int bounce,
                 int seed) {
  CHECK_CUDA(cudaEventRecord(timing.start, timing.timing));
  refit_nodes_kernel<<<(data.nodes0 + kThreads - 1) / kThreads, kThreads, 0,
                       timing.timing>>>(data.pool, data.slots0, data.nodes0,
                                        bounce, static_cast<uint32_t>(seed));
  refit_nodes_kernel<<<(data.nodes1 + kThreads - 1) / kThreads, kThreads, 0,
                       timing.timing>>>(data.pool, data.slots1, data.nodes1,
                                        bounce, static_cast<uint32_t>(seed) ^
                                                    0xda3e39cbu);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(timing.stop, timing.timing));
  CHECK_CUDA(cudaEventSynchronize(timing.stop));
  return elapsed(timing.start, timing.stop);
}

float rebuild_worklists(DeviceData &data, TimingObjects &timing, int &count0,
                        int &count1, uint32_t *&roots0, uint32_t *&roots1,
                        uint32_t *&next0, uint32_t *&next1) {
  CHECK_CUDA(cudaMemsetAsync(data.work_counts, 0, 2 * sizeof(int),
                             timing.timing));
  CHECK_CUDA(cudaEventRecord(timing.start, timing.timing));
  int const total = count0 + count1;
  rebuild_worklists_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0,
                             timing.timing>>>(next0, count0, next1, count1,
                                              data.node_side, roots0, roots1,
                                              data.work_counts);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(timing.stop, timing.timing));
  CHECK_CUDA(cudaEventSynchronize(timing.stop));
  int host_counts[2] = {};
  CHECK_CUDA(cudaMemcpy(host_counts, data.work_counts, sizeof(host_counts),
                        cudaMemcpyDeviceToHost));
  count0 = host_counts[0];
  count1 = host_counts[1];
  return elapsed(timing.start, timing.stop);
}

float launch_traverse_mode(int mode, const DeviceData &data,
                           TimingObjects &timing, int regular_ctas,
                           int side_ctas, int steps, uint32_t *roots0,
                           int count0, uint32_t *roots1, int count1,
                           uint32_t *out0, uint32_t *out1) {
  flush_only(data, mode == 0 ? timing.regular : timing.timing);
  cudaStream_t start_stream = mode == 0 ? timing.regular : timing.timing;
  CHECK_CUDA(cudaEventRecord(timing.start, start_stream));

  if (mode == 0) {
    traverse_regular_kernel<<<regular_ctas, kThreads, 0, timing.regular>>>(
        data.pool, roots0, roots1, count0, count1, steps, data.checksum, out0,
        out1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(timing.stop, timing.regular));
    CHECK_CUDA(cudaEventSynchronize(timing.stop));
    return elapsed(timing.start, timing.stop);
  }

  CHECK_CUDA(cudaStreamWaitEvent(timing.side0, timing.start, 0));
  CHECK_CUDA(cudaStreamWaitEvent(timing.side1, timing.start, 0));
  if (mode == 1) {
    traverse_partition_kernel<0><<<side_ctas, kThreads, 0, timing.side0>>>(
        data.pool, roots0, out0, count0, roots1, out1, 0, 0, steps,
        data.checksum);
    traverse_partition_kernel<1><<<side_ctas, kThreads, 0, timing.side1>>>(
        data.pool, roots1, out1, count1, roots0, out0, 0, 0, steps,
        data.checksum);
  } else if (mode == 2) {
    int const total_roots = count0 + count1;
    int const half = (total_roots + 1) / 2;
    int const side0_take = count0 < half ? count0 : half;
    int const side1_take = count1 < half ? count1 : half;
    int const side0_spill = count1 > half ? count1 - half : 0;
    int const side1_spill = count0 > half ? count0 - half : 0;
    traverse_partition_kernel<0><<<side_ctas, kThreads, 0, timing.side0>>>(
        data.pool, roots0, out0, side0_take, roots1, out1, half, side0_spill,
        steps, data.checksum);
    traverse_partition_kernel<1><<<side_ctas, kThreads, 0, timing.side1>>>(
        data.pool, roots1, out1, side1_take, roots0, out0, half, side1_spill,
        steps, data.checksum);
  } else {
    traverse_partition_kernel<0><<<side_ctas, kThreads, 0, timing.side0>>>(
        data.pool, roots1, out1, count1, roots0, out0, 0, 0, steps,
        data.checksum);
    traverse_partition_kernel<1><<<side_ctas, kThreads, 0, timing.side1>>>(
        data.pool, roots0, out0, count0, roots1, out1, 0, 0, steps,
        data.checksum);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(timing.done0, timing.side0));
  CHECK_CUDA(cudaEventRecord(timing.done1, timing.side1));
  CHECK_CUDA(cudaStreamWaitEvent(timing.timing, timing.done0, 0));
  CHECK_CUDA(cudaStreamWaitEvent(timing.timing, timing.done1, 0));
  CHECK_CUDA(cudaEventRecord(timing.stop, timing.timing));
  CHECK_CUDA(cudaEventSynchronize(timing.stop));
  return elapsed(timing.start, timing.stop);
}

PhaseSample run_bounced_mode(int mode, DeviceData &data, TimingObjects &timing,
                             int regular_ctas, int side_ctas, int steps,
                             int bounces, int frames, int seed) {
  PhaseSample sample;
  CHECK_CUDA(cudaMemset(data.checksum, 0, 4 * sizeof(uint64_t)));

  for (int frame = 0; frame < frames; ++frame) {
    CHECK_CUDA(cudaMemcpy(data.roots0, data.init0,
                          static_cast<size_t>(data.count0) * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(data.roots1, data.init1,
                          static_cast<size_t>(data.count1) * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice));
    int count0 = data.count0;
    int count1 = data.count1;
    uint32_t *roots0 = data.roots0;
    uint32_t *roots1 = data.roots1;
    uint32_t *next0 = data.next0;
    uint32_t *next1 = data.next1;

    sample.refit_ms += time_refit(data, timing, frame, seed);

    for (int bounce = 0; bounce < bounces; ++bounce) {
      bool const need_rebuild = bounce + 1 < bounces;
      sample.traverse_ms += launch_traverse_mode(
          mode, data, timing, regular_ctas, side_ctas, steps, roots0, count0,
          roots1, count1, need_rebuild ? next0 : nullptr,
          need_rebuild ? next1 : nullptr);
      if (need_rebuild) {
        sample.rebuild_ms += rebuild_worklists(data, timing, count0, count1,
                                               roots0, roots1, next0, next1);
      }
    }
  }
  sample.total_ms = sample.traverse_ms + sample.refit_ms + sample.rebuild_ms;
  return sample;
}

uint64_t checksum_total(const DeviceData &data) {
  uint64_t host[4] = {};
  CHECK_CUDA(cudaMemcpy(host, data.checksum, sizeof(host),
                        cudaMemcpyDeviceToHost));
  return host[0] + host[1];
}

uint64_t step_total(const DeviceData &data) {
  uint64_t host[4] = {};
  CHECK_CUDA(cudaMemcpy(host, data.checksum, sizeof(host),
                        cudaMemcpyDeviceToHost));
  return host[2] + host[3];
}

void destroy_timing(TimingObjects &timing) {
  if (timing.done1) {
    CHECK_CUDA(cudaEventDestroy(timing.done1));
  }
  if (timing.done0) {
    CHECK_CUDA(cudaEventDestroy(timing.done0));
  }
  if (timing.stop) {
    CHECK_CUDA(cudaEventDestroy(timing.stop));
  }
  if (timing.start) {
    CHECK_CUDA(cudaEventDestroy(timing.start));
  }
  if (timing.side1) {
    CHECK_CUDA(cudaStreamDestroy(timing.side1));
  }
  if (timing.side0) {
    CHECK_CUDA(cudaStreamDestroy(timing.side0));
  }
  if (timing.timing) {
    CHECK_CUDA(cudaStreamDestroy(timing.timing));
  }
  if (timing.regular) {
    CHECK_CUDA(cudaStreamDestroy(timing.regular));
  }
}

} // namespace

int main(int argc, char **argv) {
  int total_nodes = argc > 1 ? parse_positive(argv[1], 1 << 20) : 1 << 20;
  int rays = argc > 2 ? parse_positive(argv[2], 1 << 20) : 1 << 20;
  int steps = argc > 3 ? parse_positive(argv[3], 32) : 32;
  int locality_pct = argc > 4 ? parse_int(argv[4], 100) : 100;
  int side0_pct = argc > 5 ? parse_int(argv[5], 50) : 50;
  int runs = argc > 6 ? parse_positive(argv[6], 5) : 5;
  int seed = argc > 7 ? parse_positive(argv[7], 4242) : 4242;
  int ctas_per_sm = argc > 8 ? parse_positive(argv[8], 1) : 1;
  int bounces = argc > 9 ? parse_positive(argv[9], 1) : 1;
  int frames = argc > 10 ? parse_positive(argv[10], 1) : 1;

  total_nodes = std::max(total_nodes, 4096);
  rays = std::max(rays, 1024);
  steps = std::max(steps, 1);
  locality_pct = clamp_int(locality_pct, 0, 100);
  side0_pct = clamp_int(side0_pct, 1, 99);
  runs = std::max(runs, 1);
  ctas_per_sm = clamp_int(ctas_per_sm, 1, 16);
  bounces = clamp_int(bounces, 1, 16);
  frames = clamp_int(frames, 1, 16);

  int const nodes0 = total_nodes / 2;
  int const nodes1 = total_nodes - nodes0;
  int const roots0_count = std::max(
      1, static_cast<int>((static_cast<long long>(rays) * side0_pct + 50) /
                          100));
  int const roots1_count = std::max(1, rays - roots0_count);

  auto nodes0_alloc = sided_alloc_nodes(static_cast<size_t>(nodes0), 0);
  auto nodes1_alloc = sided_alloc_nodes(static_cast<size_t>(nodes1), 1);

  void *pool_base = nullptr;
  CHECK_CUDA(cudaGetSidedPoolBase(&pool_base));
  auto *pool = static_cast<unsigned char *>(pool_base);

  auto slots0 = make_node_slots(nodes0_alloc.pages, nodes0);
  auto slots1 = make_node_slots(nodes1_alloc.pages, nodes1);

  uint32_t *d_slots0 = nullptr;
  uint32_t *d_slots1 = nullptr;
  copy_u32(&d_slots0, slots0);
  copy_u32(&d_slots1, slots1);

  init_nodes_kernel<<<(nodes0 + kThreads - 1) / kThreads, kThreads>>>(
      pool, d_slots0, d_slots1, 0, nodes0, nodes1, locality_pct,
      static_cast<uint32_t>(seed));
  init_nodes_kernel<<<(nodes1 + kThreads - 1) / kThreads, kThreads>>>(
      pool, d_slots1, d_slots0, 1, nodes1, nodes0, locality_pct,
      static_cast<uint32_t>(seed) ^ 0xda3e39cbu);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  auto roots0 =
      make_roots(roots0_count, slots0, static_cast<uint32_t>(seed) ^ 0x1234u);
  auto roots1 =
      make_roots(roots1_count, slots1, static_cast<uint32_t>(seed) ^ 0x5678u);

  uint32_t max_slot = 0;
  for (uint32_t slot : slots0) {
    max_slot = std::max(max_slot, slot);
  }
  for (uint32_t slot : slots1) {
    max_slot = std::max(max_slot, slot);
  }
  std::vector<uint8_t> node_side(static_cast<size_t>(max_slot) + 1, 0);
  for (uint32_t slot : slots0) {
    node_side[slot] = 0;
  }
  for (uint32_t slot : slots1) {
    node_side[slot] = 1;
  }

  uint32_t *d_roots0 = nullptr;
  uint32_t *d_roots1 = nullptr;
  uint32_t *d_next0 = nullptr;
  uint32_t *d_next1 = nullptr;
  uint32_t *d_init0 = nullptr;
  uint32_t *d_init1 = nullptr;
  uint8_t *d_node_side = nullptr;
  int *d_work_counts = nullptr;
  CHECK_CUDA(cudaMalloc(&d_roots0, static_cast<size_t>(rays) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_roots1, static_cast<size_t>(rays) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next0, static_cast<size_t>(rays) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next1, static_cast<size_t>(rays) * sizeof(uint32_t)));
  copy_u32(&d_init0, roots0);
  copy_u32(&d_init1, roots1);
  CHECK_CUDA(cudaMemcpy(d_roots0, roots0.data(),
                        static_cast<size_t>(roots0_count) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_roots1, roots1.data(),
                        static_cast<size_t>(roots1_count) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&d_node_side, node_side.size() * sizeof(uint8_t)));
  CHECK_CUDA(cudaMemcpy(d_node_side, node_side.data(),
                        node_side.size() * sizeof(uint8_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&d_work_counts, 2 * sizeof(int)));

  uint64_t *d_checksum = nullptr;
  CHECK_CUDA(cudaMalloc(&d_checksum, 4 * sizeof(uint64_t)));

  cudaDeviceProp prop{};
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  size_t const l2_flush_bytes = static_cast<size_t>(prop.l2CacheSize) * 2;
  unsigned char *l2_flush = nullptr;
  CHECK_CUDA(cudaMalloc(&l2_flush, l2_flush_bytes));
  CHECK_CUDA(cudaMemset(l2_flush, 1, l2_flush_bytes));

  TimingObjects timing;
  CHECK_CUDA(cudaStreamCreateWithFlags(&timing.regular, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&timing.timing, cudaStreamNonBlocking));
  CHECK_CUDA(SidedGreenContext::create_stream(0, &timing.side0));
  CHECK_CUDA(SidedGreenContext::create_stream(1, &timing.side1));
  CHECK_CUDA(cudaEventCreate(&timing.start));
  CHECK_CUDA(cudaEventCreate(&timing.stop));
  CHECK_CUDA(cudaEventCreateWithFlags(&timing.done0, cudaEventDisableTiming));
  CHECK_CUDA(cudaEventCreateWithFlags(&timing.done1, cudaEventDisableTiming));

  DeviceData data;
  data.pool = pool;
  data.roots0 = d_roots0;
  data.roots1 = d_roots1;
  data.next0 = d_next0;
  data.next1 = d_next1;
  data.init0 = d_init0;
  data.init1 = d_init1;
  data.slots0 = d_slots0;
  data.slots1 = d_slots1;
  data.node_side = d_node_side;
  data.nodes0 = nodes0;
  data.nodes1 = nodes1;
  data.count0 = roots0_count;
  data.count1 = roots1_count;
  data.total_rays = rays;
  data.checksum = d_checksum;
  data.work_counts = d_work_counts;
  data.l2_flush = l2_flush;
  data.l2_flush_bytes = l2_flush_bytes;

  int const regular_ctas =
      std::max(1, prop.multiProcessorCount * ctas_per_sm);
  int const side_ctas =
      std::max(1, (prop.multiProcessorCount / 2) * ctas_per_sm);
  int const warmups = std::min(2, runs);

  auto time_mode = [&](int mode) {
    return run_bounced_mode(mode, data, timing, regular_ctas, side_ctas, steps,
                            bounces, frames, seed);
  };

  constexpr int mode_orders[4][4] = {
      {0, 1, 2, 3}, {1, 3, 0, 2}, {2, 0, 3, 1}, {3, 2, 1, 0}};

  for (int i = 0; i < warmups; ++i) {
    for (int j = 0; j < 4; ++j) {
      time_mode(mode_orders[i % 4][j]);
    }
  }

  std::vector<PhaseSample> regular_samples;
  std::vector<PhaseSample> aware_samples;
  std::vector<PhaseSample> adaptive_samples;
  std::vector<PhaseSample> swapped_samples;
  regular_samples.reserve(static_cast<size_t>(runs));
  aware_samples.reserve(static_cast<size_t>(runs));
  adaptive_samples.reserve(static_cast<size_t>(runs));
  swapped_samples.reserve(static_cast<size_t>(runs));

  auto record_sample = [&](int mode, PhaseSample sample) {
    switch (mode) {
    case 0:
      regular_samples.push_back(sample);
      break;
    case 1:
      aware_samples.push_back(sample);
      break;
    case 2:
      adaptive_samples.push_back(sample);
      break;
    default:
      swapped_samples.push_back(sample);
      break;
    }
  };

  for (int r = 0; r < runs; ++r) {
    for (int j = 0; j < 4; ++j) {
      int const mode = mode_orders[r % 4][j];
      record_sample(mode, time_mode(mode));
    }
  }

  TimingStats regular = summarize_phase(regular_samples);
  TimingStats aware = summarize_phase(aware_samples);
  TimingStats adaptive = summarize_phase(adaptive_samples);
  TimingStats swapped = summarize_phase(swapped_samples);

  run_bounced_mode(0, data, timing, regular_ctas, side_ctas, steps, bounces,
                   frames, seed);
  uint64_t const regular_sum = checksum_total(data);
  uint64_t const regular_steps = step_total(data);
  run_bounced_mode(1, data, timing, regular_ctas, side_ctas, steps, bounces,
                   frames, seed);
  uint64_t const aware_sum = checksum_total(data);
  uint64_t const aware_steps = step_total(data);
  run_bounced_mode(2, data, timing, regular_ctas, side_ctas, steps, bounces,
                   frames, seed);
  uint64_t const adaptive_sum = checksum_total(data);
  uint64_t const adaptive_steps = step_total(data);
  run_bounced_mode(3, data, timing, regular_ctas, side_ctas, steps, bounces,
                   frames, seed);
  uint64_t const swapped_sum = checksum_total(data);
  uint64_t const swapped_steps = step_total(data);

  bool const correct = regular_sum == aware_sum &&
                       regular_sum == adaptive_sum &&
                       regular_sum == swapped_sum &&
                       regular_steps == aware_steps &&
                       regular_steps == adaptive_steps &&
                       regular_steps == swapped_steps &&
                       regular_steps == static_cast<uint64_t>(rays) *
                                            static_cast<uint64_t>(steps) *
                                            static_cast<uint64_t>(bounces) *
                                            static_cast<uint64_t>(frames);

  std::printf("bvh sided allocator: gpu=%s nodes=%d rays=%d steps=%d "
              "locality=%d%% side0_rays=%d%% runs=%d seed=%d "
              "ctas_per_sm=%d threads=%d frames=%d bounces=%d\n",
              prop.name, total_nodes, rays, steps, locality_pct, side0_pct,
              runs, seed, ctas_per_sm, kThreads, frames, bounces);
  std::printf("node_pages=%zu/%zu worklist_counts=%d/%d checksum=%llu "
              "steps=%llu correct=%s\n",
              nodes0_alloc.pages.size(), nodes1_alloc.pages.size(),
              roots0_count, roots1_count,
              static_cast<unsigned long long>(regular_sum),
              static_cast<unsigned long long>(regular_steps),
              correct ? "PASS" : "FAIL");
  std::printf("| mode | mean_ms | speedup_vs_regular |\n");
  std::printf("|---|---:|---:|\n");
  auto print_mode = [&](const char *name, const TimingStats &stats) {
    std::printf("| %s | %.6f | %.6f |\n", name, stats.mean,
                regular.mean / stats.mean);
  };
  print_mode("regular", regular);
  print_mode("aware", aware);
  print_mode("adaptive", adaptive);
  print_mode("swapped", swapped);

  destroy_timing(timing);
  CHECK_CUDA(cudaFree(l2_flush));
  CHECK_CUDA(cudaFree(d_checksum));
  CHECK_CUDA(cudaFree(d_work_counts));
  CHECK_CUDA(cudaFree(d_node_side));
  CHECK_CUDA(cudaFree(d_init1));
  CHECK_CUDA(cudaFree(d_init0));
  CHECK_CUDA(cudaFree(d_next1));
  CHECK_CUDA(cudaFree(d_next0));
  CHECK_CUDA(cudaFree(d_roots1));
  CHECK_CUDA(cudaFree(d_roots0));
  CHECK_CUDA(cudaFree(d_slots1));
  CHECK_CUDA(cudaFree(d_slots0));
  CHECK_CUDA(cudaFreeSided(nodes1_alloc.pages));
  CHECK_CUDA(cudaFreeSided(nodes0_alloc.pages));

  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
