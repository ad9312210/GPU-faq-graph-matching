#include "gpu_topk.hpp"
#include <cub/block/block_merge_sort.cuh>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
using Item = GPUScoredCandidate;
static_assert(sizeof(Item) == 8, "Update workspace arithmetic for this layout");
constexpr std::size_t TILE = GPU_TOPK_CHUNK_SIZE;

std::size_t bytes_for(std::size_t n, std::size_t size) {
    if (n > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error("Top-K byte count overflow");
    return n * size;
}
struct AllocationError : std::runtime_error {
    cudaError_t code;
    AllocationError(cudaError_t e) : std::runtime_error(
        std::string("Top-K allocation failed: ") + cudaGetErrorString(e)), code(e) {}
};
// Device ownership and immediate RAII cover partial allocation failures.
template<class T> struct Buffer {
    T* p = nullptr;
    int device = -1;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(std::size_t n) {
        CUDA_CHECK(cudaGetDevice(&device));
        auto e = cudaMalloc(reinterpret_cast<void**>(&p), bytes_for(n, sizeof(T)));
        if (e != cudaSuccess) throw AllocationError(e);
    }
    ~Buffer() {
        if (!p) return;
        int previous = -1;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(device) == cudaSuccess) cudaFree(p);
        if (restore && previous != device) cudaSetDevice(previous);
    }
};
struct Events {
    cudaEvent_t begin = nullptr, end = nullptr;
    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        auto e = cudaEventCreate(&end);
        if (e != cudaSuccess) { cudaEventDestroy(begin); begin = nullptr; CUDA_CHECK(e); }
    }
    ~Events() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(begin, s)); }
    void stop(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(end, s)); }
    double elapsed() {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        return ms;
    }
};
// On every error path, drain outstanding work before caller-owned host staging
// can be destroyed. A failed pool cannot subsequently return partial results.
struct PoolOperation {
    const GPUContext& context;
    bool& active;
    bool completed = false;
    ~PoolOperation() {
        if (!completed) {
            cudaSetDevice(context.device_id);
            cudaStreamSynchronize(context.stream);
            active = false;
        }
    }
};
struct Better {
    __host__ __device__ bool operator()(Item a, Item b) const {
        if ((a.target >= 0) != (b.target >= 0)) return a.target >= 0;
        if (a.target < 0) return false;
        return a.score > b.score || (a.score == b.score && a.target < b.target);
    }
};
using Sort = cub::BlockMergeSort<Item, GPU_TOPK_CHUNK_SIZE, 1>;

__global__ void initialize(Item* heap, std::size_t n) {
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
        heap[i] = Item{0.0f, -1};
}
__global__ void sort_chunks(Item* workspace, std::size_t count,
                            int k, int targets, int* error) {
    const auto lane = static_cast<std::size_t>(threadIdx.x);
    const auto position = static_cast<std::size_t>(blockIdx.x) * TILE + lane;
    const auto base = (static_cast<std::size_t>(blockIdx.x) + 1) * TILE;
    Item item[1];
    item[0] = Item{0.0f, -1};
    if (position < count) {
        item[0] = workspace[base + lane];
        if (item[0].target < -1 || item[0].target >= targets ||
            (item[0].target >= 0 && !isfinite(item[0].score))) {
            atomicExch(error, 1);
            item[0] = Item{0.0f, -1};
        }
    }
    __shared__ typename Sort::TempStorage temp;
    // All lanes participate, including padding lanes.
    Sort(temp).Sort(item, Better{});
    if (threadIdx.x < k) workspace[base + lane] = item[0];
}
__device__ int lower_bound_item(const Item* list, int k, Item value) {
    int low = 0, high = k;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (Better{}(list[middle], value)) low = middle + 1;
        else high = middle;
    }
    return low;
}
__global__ void merge_pairs(const Item* input, Item* output,
                            unsigned lists, int k) {
    const unsigned left = blockIdx.x * 2;
    const bool has_right = left + 1 < lists;
    Item* destination = output + static_cast<std::size_t>(blockIdx.x) * TILE;
    if (threadIdx.x < k) destination[threadIdx.x] = Item{0.0f, -1};
    __syncthreads();
    for (int i = threadIdx.x; i < 2 * k; i += blockDim.x) {
        const bool right = i >= k;
        if (right && !has_right) continue;
        const int rank_in_list = right ? i - k : i;
        const Item* own = input + static_cast<std::size_t>(left + right) * TILE;
        const Item value = own[rank_in_list];
        // Never scatter equivalent padding values: they do not have unique ranks.
        if (value.target < 0) continue;
        const Item* other = input + static_cast<std::size_t>(left + !right) * TILE;
        const int rank = rank_in_list +
            ((right || has_right) ? lower_bound_item(other, k, value) : 0);
        if (rank < k) destination[rank] = value;
    }
}
__device__ void sift_down(Item* heap, int root, int k) {
    while (true) {
        const auto left_index = static_cast<std::size_t>(root) * 2 + 1;
        if (left_index >= static_cast<std::size_t>(k)) return;
        const int left = static_cast<int>(left_index);
        if (left >= k) return;
        int worst = left;
        if (left + 1 < k && Better{}(heap[left], heap[left + 1])) worst = left + 1;
        if (!Better{}(heap[root], heap[worst])) return;
        const Item old = heap[root]; heap[root] = heap[worst]; heap[worst] = old;
        root = worst;
    }
}
__global__ void serial_consume(Item* heap, int k, const Item* batch,
                               std::size_t n, int targets, int* error) {
    for (std::size_t i = 0; i < n; ++i) {
        const Item value = batch[i];
        if (value.target < -1 || value.target >= targets ||
            (value.target >= 0 && !isfinite(value.score))) {
            *error = 1; continue;
        }
        if (value.target >= 0 && Better{}(value, heap[0])) {
            heap[0] = value;
            sift_down(heap, 0, k);
        }
    }
}
__global__ void serial_finish(Item* heap, int k) {
    for (int end = k - 1; end > 0; --end) {
        const Item old = heap[0]; heap[0] = heap[end]; heap[end] = old;
        sift_down(heap, 0, end);
    }
}


// Export the sorted result without downloading it.
//
// Both selector paths place valid entries before padding:
//   parallel: sorted carry list
//   serial:   heap sorted by serial_finish()
__global__ void export_selected_device(
    const Item* selected,
    int k,
    int* target_ids,
    float* scores,
    std::size_t* valid_count)
{
    for (std::size_t i = static_cast<std::size_t>(threadIdx.x);
         i < static_cast<std::size_t>(k);
         i += blockDim.x) {
        const Item value = selected[i];
        target_ids[i] = value.target;
        scores[i] = value.target >= 0 ? value.score : 0.0f;
    }

    if (threadIdx.x == 0) {
        std::size_t count = 0;

        while (count < static_cast<std::size_t>(k) &&
               selected[count].target >= 0) {
            ++count;
        }

        *valid_count = count;
    }
}


void validate(const GPUScoredPools& p, int targets, int k) {
    if (k <= 0 || targets < 0 || p.offsets.empty() || p.offsets.front() != 0 ||
        p.target_ids.size() != p.scores.size() || p.offsets.back() != p.scores.size())
        throw std::invalid_argument("Invalid Top-K pool layout, targets, or K");
    if (!std::is_sorted(p.offsets.begin(), p.offsets.end()))
        throw std::invalid_argument("Top-K offsets must be nondecreasing");
    for (std::size_t i = 0; i < p.scores.size(); ++i)
        if (!std::isfinite(p.scores[i]) || p.target_ids[i] < 0 || p.target_ids[i] >= targets)
            throw std::invalid_argument("Invalid Top-K target ID or score");
}
} // namespace

struct GPUStreamingTopK::Impl {
    int targets, k;
    GPUContext context;
    GPUSelectionInfo status;
    Buffer<Item> a, b;
    Buffer<int> error;
    Events selection_events, transfer_events;
    bool active = false;

    Impl(int targets_, int k_, const GPUContext& ctx, GPUSelectionInfo status_,
         const GPUTopKOptions& options)
        : targets(targets_), k(k_), context(ctx), status(status_) {
        const auto budget = options.scratch_budget_bytes;
        if (budget < sizeof(int)) throw std::invalid_argument("Top-K scratch budget too small");
        std::size_t count_a = 0, count_b = 0;
        if (status.used == GPUSelector::Parallel) {
            const auto slots = (budget - sizeof(int)) / (2 * TILE * sizeof(Item));
            // At least one carry slot and one scoring/chunk slot in each buffer.
            if (slots < 2 || options.batch_candidates < TILE)
                throw std::invalid_argument("Top-K parallel scratch budget too small");
            const auto chunks = std::min<std::size_t>({
                slots - 1, options.batch_candidates / TILE, 65534});
            count_a = count_b = (chunks + 1) * TILE;
            status.batch_capacity = chunks * TILE;
            status.shared_bytes_per_block = sizeof(typename Sort::TempStorage);
        } else {
            count_a = static_cast<std::size_t>(k);
            const auto heap_bytes = bytes_for(count_a, sizeof(Item));
            if (heap_bytes > budget - sizeof(int))
                throw std::invalid_argument("Top-K serial heap exceeds scratch budget");
            count_b = std::min<std::size_t>({options.batch_candidates,
                (budget - sizeof(int) - heap_bytes) / sizeof(Item), 65534 * TILE});
            if (count_b == 0)
                throw std::invalid_argument("Top-K scratch budget leaves no scoring batch");
            status.batch_capacity = count_b;
        }
        const auto bytes_a = bytes_for(count_a, sizeof(Item));
        const auto bytes_b = bytes_for(count_b, sizeof(Item));
        if (bytes_a > budget - sizeof(int) || bytes_b > budget - sizeof(int) - bytes_a)
            throw std::overflow_error("Top-K workspace budget calculation overflow");
        status.scratch_peak_bytes = bytes_a + bytes_b + sizeof(int);
        a.allocate(count_a); b.allocate(count_b); error.allocate(1);
    }
};

GPUStreamingTopK::GPUStreamingTopK(int targets, int k, const GPUContext& ctx,
    GPUSelector selector, const GPUTopKOptions& options) {
    if (k <= 0 || targets < 0 || options.batch_candidates == 0)
        throw std::invalid_argument("Invalid streaming Top-K configuration");
    if (selector != GPUSelector::SerialHeap && selector != GPUSelector::Parallel)
        throw std::invalid_argument("Invalid Top-K selector");
    CUDA_CHECK(cudaSetDevice(ctx.device_id));
    GPUSelectionInfo status;
    status.requested = status.used = selector;
    if (selector == GPUSelector::Parallel && k > GPU_TOPK_PARALLEL_MAX_K) {
        status.used = GPUSelector::SerialHeap;
        status.fallback = GPUFallbackReason::UnsupportedK;
    } else if (selector == GPUSelector::Parallel &&
               (options.scratch_budget_bytes < 4 * TILE * sizeof(Item) + sizeof(int) ||
                options.batch_candidates < TILE)) {
        status.used = GPUSelector::SerialHeap;
        status.fallback = GPUFallbackReason::ScratchBudgetTooSmall;
    }
    try {
        impl_ = std::make_unique<Impl>(targets, k, ctx, status, options);
    } catch (const AllocationError& e) {
        if (status.used != GPUSelector::Parallel || e.code != cudaErrorMemoryAllocation) throw;
        // All partially constructed device buffers have already unwound.
        (void)cudaGetLastError(); // Clear only the allocation error being handled.
        status.used = GPUSelector::SerialHeap;
        status.fallback = GPUFallbackReason::DeviceAllocationFailure;
        impl_ = std::make_unique<Impl>(targets, k, ctx, status, options);
    }
}
GPUStreamingTopK::~GPUStreamingTopK() = default;
std::size_t GPUStreamingTopK::capacity() const { return impl_->status.batch_capacity; }
const GPUSelectionInfo& GPUStreamingTopK::info() const { return impl_->status; }
void GPUStreamingTopK::begin_pool() {
    auto& w = *impl_;
    if (w.active) throw std::logic_error("Finish the active Top-K pool first");
    CUDA_CHECK(cudaSetDevice(w.context.device_id));
    initialize<<<1, 256, 0, w.context.stream>>>(w.a.p,
        w.status.used == GPUSelector::Parallel ? TILE : static_cast<std::size_t>(w.k));
    CUDA_CHECK(cudaGetLastError());
    w.active = true;
}
GPUScoredCandidate* GPUStreamingTopK::input() {
    auto& w = *impl_;
    if (!w.active) throw std::logic_error("Top-K pool is not active");
    return w.status.used == GPUSelector::Parallel ? w.a.p + TILE : w.b.p;
}
void GPUStreamingTopK::consume(std::size_t count) {
    auto& w = *impl_;
    if (!w.active || count == 0 || count > capacity())
        throw std::invalid_argument("Invalid Top-K batch count/state");
    if (w.status.batches == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Top-K batch counter overflow");
    PoolOperation operation{w.context, w.active};
    CUDA_CHECK(cudaSetDevice(w.context.device_id));
    CUDA_CHECK(cudaMemsetAsync(w.error.p, 0, sizeof(int), w.context.stream));
    w.selection_events.start(w.context.stream);
    if (w.status.used == GPUSelector::Parallel) {
        const unsigned chunks = static_cast<unsigned>((count + TILE - 1) / TILE);
        sort_chunks<<<chunks, 256, 0, w.context.stream>>>(w.a.p, count, w.k, w.targets, w.error.p);
        CUDA_CHECK(cudaGetLastError());
        // List zero is the previous batches' Top-K; other lists are new chunks.
        unsigned lists = chunks + 1;
        while (lists > 1) {
            const unsigned pairs = (lists + 1) / 2;
            merge_pairs<<<pairs, 256, 0, w.context.stream>>>(w.a.p, w.b.p, lists, w.k);
            CUDA_CHECK(cudaGetLastError());
            std::swap(w.a.p, w.b.p);
            lists = pairs;
        }
    } else {
        serial_consume<<<1, 1, 0, w.context.stream>>>(
            w.a.p, w.k, w.b.p, count, w.targets, w.error.p);
        CUDA_CHECK(cudaGetLastError());
    }
    w.selection_events.stop(w.context.stream);
    int error = 0;
    try {
        CUDA_CHECK(cudaMemcpyAsync(&error, w.error.p, sizeof(int),
                                   cudaMemcpyDeviceToHost, w.context.stream));
        CUDA_CHECK(cudaStreamSynchronize(w.context.stream));
    } catch (...) {
        cudaStreamSynchronize(w.context.stream);
        w.active = false;
        throw;
    }
    w.status.selection_ms += w.selection_events.elapsed();
    ++w.status.batches;
    if (error) {
        w.active = false;
        throw std::invalid_argument("Nonfinite score or invalid ID in GPU scoring batch");
    }
    operation.completed = true;
}
void GPUStreamingTopK::push_host(const GPUScoredCandidate* candidates, std::size_t count) {
    auto& w = *impl_;
    if (!candidates || count == 0 || count > capacity())
        throw std::invalid_argument("Invalid host Top-K batch");
    PoolOperation operation{w.context, w.active};
    CUDA_CHECK(cudaSetDevice(w.context.device_id));
    w.transfer_events.start(w.context.stream);
    CUDA_CHECK(cudaMemcpyAsync(input(), candidates, bytes_for(count, sizeof(Item)),
                               cudaMemcpyHostToDevice, w.context.stream));
    w.transfer_events.stop(w.context.stream);
    consume(count);
    w.status.transfer_ms += w.transfer_events.elapsed();
    operation.completed = true;
}
std::vector<GPUScoredCandidate> GPUStreamingTopK::finish_pool() {
    auto& w = *impl_;
    if (!w.active) throw std::logic_error("Top-K pool is not active");
    PoolOperation operation{w.context, w.active};
    CUDA_CHECK(cudaSetDevice(w.context.device_id));
    std::vector<Item> output(static_cast<std::size_t>(w.k));
    if (w.status.used == GPUSelector::SerialHeap) {
        w.selection_events.start(w.context.stream);
        serial_finish<<<1, 1, 0, w.context.stream>>>(w.a.p, w.k);
        CUDA_CHECK(cudaGetLastError());
        w.selection_events.stop(w.context.stream);
    }
    w.transfer_events.start(w.context.stream);
    try {
        CUDA_CHECK(cudaMemcpyAsync(output.data(), w.a.p, bytes_for(output.size(), sizeof(Item)),
                                   cudaMemcpyDeviceToHost, w.context.stream));
        w.transfer_events.stop(w.context.stream);
        CUDA_CHECK(cudaStreamSynchronize(w.context.stream));
    } catch (...) {
        cudaStreamSynchronize(w.context.stream);
        w.active = false;
        throw;
    }
    if (w.status.used == GPUSelector::SerialHeap)
        w.status.selection_ms += w.selection_events.elapsed();
    w.status.transfer_ms += w.transfer_events.elapsed();
    w.active = false;
    output.erase(std::remove_if(output.begin(), output.end(),
        [](Item value) { return value.target < 0; }), output.end());
    operation.completed = true;
    return output;
}

void GPUStreamingTopK::finish_pool_to_device(
    int* device_target_ids,
    float* device_scores,
    std::size_t output_capacity,
    std::size_t* device_count)
{
    auto& w = *impl_;

    if (!w.active) {
        throw std::logic_error("Top-K pool is not active");
    }

    if (device_target_ids == nullptr ||
        device_scores == nullptr ||
        device_count == nullptr) {
        throw std::invalid_argument(
            "Top-K device output pointers must not be null");
    }

    if (output_capacity < static_cast<std::size_t>(w.k)) {
        throw std::invalid_argument(
            "Top-K device output capacity must be at least K");
    }

    // On failure, drain outstanding work and invalidate this pool.
    PoolOperation operation{w.context, w.active};

    CUDA_CHECK(cudaSetDevice(w.context.device_id));

    w.selection_events.start(w.context.stream);

    if (w.status.used == GPUSelector::SerialHeap) {
        serial_finish<<<1, 1, 0, w.context.stream>>>(
            w.a.p, w.k);

        CUDA_CHECK(cudaGetLastError());
    }

    export_selected_device<<<1, 256, 0, w.context.stream>>>(
        w.a.p,
        w.k,
        device_target_ids,
        device_scores,
        device_count);

    CUDA_CHECK(cudaGetLastError());

    w.selection_events.stop(w.context.stream);

    // Preserve the existing synchronous completion contract.
    // Caller-owned outputs are ready when this method returns.
    CUDA_CHECK(cudaStreamSynchronize(w.context.stream));

    w.status.selection_ms += w.selection_events.elapsed();

    // No H2D/D2H transfer occurred, so transfer_ms is unchanged.
    w.active = false;
    operation.completed = true;
}

GPUSelectedPools gpu_select_topk(const GPUScoredPools& pools, int targets, int k,
    const GPUContext& ctx, GPUSelector selector, GPUSelectionInfo* info,
    const GPUTopKOptions& options) {
    validate(pools, targets, k);
    if ((selector != GPUSelector::SerialHeap && selector != GPUSelector::Parallel) ||
        options.batch_candidates == 0)
        throw std::invalid_argument("Invalid Top-K configuration");
    if (pools.scores.empty()) {
        GPUSelectedPools empty;
        empty.offsets.assign(pools.offsets.size(), 0);
        if (info) {
            *info = GPUSelectionInfo{};
            info->requested = info->used = selector;
            if (selector == GPUSelector::Parallel && k > GPU_TOPK_PARALLEL_MAX_K) {
                info->used = GPUSelector::SerialHeap;
                info->fallback = GPUFallbackReason::UnsupportedK;
            }
        }
        return empty;
    }
    GPUStreamingTopK workspace(targets, k, ctx, selector, options);
    GPUSelectedPools result;
    std::vector<Item> batch;
    for (std::size_t source = 0; source + 1 < pools.offsets.size(); ++source) {
        workspace.begin_pool();
        for (auto start = pools.offsets[source]; start < pools.offsets[source + 1];) {
            const auto count = std::min(workspace.capacity(), pools.offsets[source + 1] - start);
            batch.resize(count);
            for (std::size_t i = 0; i < count; ++i)
                batch[i] = Item{pools.scores[start + i], pools.target_ids[start + i]};
            workspace.push_host(batch.data(), count);
            start += count;
        }
        for (auto value : workspace.finish_pool()) {
            result.target_ids.push_back(value.target); result.scores.push_back(value.score);
        }
        result.offsets.push_back(result.target_ids.size());
    }
    if (info) *info = workspace.info();
    return result;
}
GPUChunkTopK gpu_select_chunk_topk(const GPUScoredPools& pools, int targets, int k,
                                  const GPUContext& ctx) {
    validate(pools, targets, k);
    // A chunk never needs more than 256 retained entries, even for K=512.
    GPUStreamingTopK workspace(targets, std::min(k, 256), ctx, GPUSelector::Parallel);
    GPUChunkTopK result;
    std::vector<Item> batch;
    for (std::size_t source = 0; source + 1 < pools.offsets.size(); ++source) {
        for (auto start = pools.offsets[source]; start < pools.offsets[source + 1];) {
            const auto count = std::min<std::size_t>(TILE, pools.offsets[source + 1] - start);
            batch.resize(count);
            for (std::size_t i = 0; i < count; ++i)
                batch[i] = Item{pools.scores[start + i], pools.target_ids[start + i]};
            workspace.begin_pool(); workspace.push_host(batch.data(), count);
            for (auto value : workspace.finish_pool()) {
                result.chunks.target_ids.push_back(value.target);
                result.chunks.scores.push_back(value.score);
            }
            result.chunks.offsets.push_back(result.chunks.target_ids.size());
            start += count;
        }
        result.source_chunk_offsets.push_back(result.chunks.offsets.size() - 1);
    }
    return result;
}
