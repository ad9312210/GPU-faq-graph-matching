
#include "gpu_matching.hpp"
#include "cuda_kernels.cuh"
#include "cosine_similarity.hpp"
#include <limits>
#include <chrono>

namespace {

size_t active_device_allocation_bytes = 0;
size_t peak_device_application_allocation_bytes = 0;

template <typename T>
void tracked_cuda_malloc(T** pointer, size_t bytes) {
    CUDA_CHECK(cudaMalloc(pointer, bytes));
    active_device_allocation_bytes += bytes;
    peak_device_application_allocation_bytes = std::max(
        peak_device_application_allocation_bytes,
        active_device_allocation_bytes);
}

template <typename T>
void tracked_cuda_free(T* pointer, size_t bytes) {
    if (!pointer) return;
    CUDA_CHECK(cudaFree(pointer));
    active_device_allocation_bytes -= bytes;
}

} // namespace

/*
 * Feature computation kernel.
 * One thread per vertex.
 */
__global__ void kernel_compute_features(
    const int* row_ptr,
    const int* col_idx,
    int num_vertices,
    NodeFeatures* features)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    const int start = row_ptr[v];
    const int end = row_ptr[v + 1];
    const int deg = end - start;

    NodeFeatures feature{};

    feature.degree = static_cast<float>(deg);
    feature.degree_squared = static_cast<float>(deg * deg);

    float neighbor_degree_sum = 0.0f;
    for (int idx = start; idx < end; ++idx) {
        const int neighbor = col_idx[idx];
        const int neighbor_degree =
            row_ptr[neighbor + 1] - row_ptr[neighbor];
        neighbor_degree_sum += static_cast<float>(neighbor_degree);
    }

    const float neighbor_mean = deg > 0
        ? neighbor_degree_sum / static_cast<float>(deg)
        : 0.0f;

    feature.avg_neighbor_degree = neighbor_mean;

    float squared_deviation_sum = 0.0f;
    for (int idx = start; idx < end; ++idx) {
        const int neighbor = col_idx[idx];
        const int neighbor_degree =
            row_ptr[neighbor + 1] - row_ptr[neighbor];
        const float delta =
            static_cast<float>(neighbor_degree) - neighbor_mean;

        squared_deviation_sum += delta * delta;
    }

    feature.neighbor_degree_std = deg > 1
        ? sqrtf(squared_deviation_sum / static_cast<float>(deg - 1))
        : 0.0f;

    int triangle_count = 0;

    for (int i = start; i < end; ++i) {
        const int first_neighbor = col_idx[i];

        for (int j = i + 1; j < end; ++j) {
            const int second_neighbor = col_idx[j];

            // Adjacency lists are sorted.
            int low = row_ptr[first_neighbor];
            int high = row_ptr[first_neighbor + 1] - 1;

            while (low <= high) {
                const int middle = low + (high - low) / 2;
                const int candidate = col_idx[middle];

                if (candidate == second_neighbor) {
                    ++triangle_count;
                    break;
                }

                if (candidate < second_neighbor) {
                    low = middle + 1;
                } else {
                    high = middle - 1;
                }
            }
        }
    }

    feature.triangle_count = static_cast<float>(triangle_count);

    if (deg >= 2) {
        const float possible_triangles =
            static_cast<float>(deg) *
            static_cast<float>(deg - 1) / 2.0f;

        feature.clustering =
            static_cast<float>(triangle_count) / possible_triangles;
    }

    features[v] = feature;
}

__global__ void kernel_feature_stats(
    const float* d_features,
    int num_vertices,
    int dim,
    float* d_mean,
    float* d_var)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= dim) return;

    float sum = 0.0f;
    for (int i = 0; i < num_vertices; i++) {
        sum += d_features[i * dim + f];
    }

    float mean = sum / (float)num_vertices;
    d_mean[f] = mean;

    float var_sum = 0.0f;
    for (int i = 0; i < num_vertices; i++) {
        float diff = d_features[i * dim + f] - mean;
        var_sum += diff * diff;
    }

    d_var[f] = var_sum / (float)num_vertices;
}

__global__ void kernel_normalize_features(
    float* d_features,
    const float* d_mean,
    const float* d_std,
    int num_vertices,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_vertices * dim;
    if (idx >= total) return;

    int f = idx % dim;
    float mean = d_mean[f];
    float std_val = d_std[f];
    if (std_val < 1e-12f) std_val = 1.0f;

    d_features[idx] = (d_features[idx] - mean) / std_val;
}

// Each CTA computes one source row against a bounded tile of targets in
// parallel. Similarities are written to O(N * tile_width) scratch and a
// separate merge kernel updates an O(N * K) min-heap.
// No N*M matrix is formed.
static constexpr int TOPK_TILE_WIDTH = 256;
static constexpr int TOPK_THREADS = 256;

__global__ void kernel_similarity_tile(
    const float* d_feats_src,
    const float* d_feats_tgt,
    int M,
    int tile_start,
    int tile_count,
    int dim,
    float* d_tile_scores)
{
    const int src = blockIdx.x;
    const int lane = threadIdx.x;
    if (lane >= TOPK_TILE_WIDTH) return;

    const int tgt = tile_start + lane;
    float score = -2.0f;

    if (lane < tile_count && tgt < M) {
        score = cosine_similarity_device(
            d_feats_src + src * dim,
            d_feats_tgt + tgt * dim,
            dim);
    }

    d_tile_scores[src * TOPK_TILE_WIDTH + lane] = score;
}

// The heap root is the worst retained candidate:
// lower scores are worse; equal scores prefer lower target IDs.
__device__ inline bool candidate_is_worse(
    float score_a,
    int target_a,
    float score_b,
    int target_b)
{
    return score_a < score_b ||
           (score_a == score_b && target_a > target_b);
}

__device__ inline void sift_down_min_heap(
    float* scores,
    int* targets,
    int root,
    int heap_size,
    int base)
{
    while (true) {
        int left = root * 2 + 1;
        if (left >= heap_size) return;

        int worse_child = left;
        int right = left + 1;

        if (right < heap_size &&
            candidate_is_worse(
                scores[base + right],
                targets[base + right],
                scores[base + left],
                targets[base + left])) {
            worse_child = right;
        }

        if (!candidate_is_worse(
                scores[base + worse_child],
                targets[base + worse_child],
                scores[base + root],
                targets[base + root])) {
            return;
        }

        float score = scores[base + root];
        scores[base + root] = scores[base + worse_child];
        scores[base + worse_child] = score;

        int target = targets[base + root];
        targets[base + root] = targets[base + worse_child];
        targets[base + worse_child] = target;

        root = worse_child;
    }
}

// Preserved serial merge kernel.
// One thread per source updates the heap across successive target tiles.
__global__ void kernel_merge_tile_topk(
    const float* d_tile_scores,
    int tile_start,
    int tile_count,
    int K,
    bool finalize,
    int* d_candidate_target,
    float* d_candidate_score)
{
    const int src = blockIdx.x;
    if (threadIdx.x != 0) return;

    const int base = src * K;
    int heap_size = tile_start < K ? tile_start : K;

    if (tile_start == 0) {
        for (int k = 0; k < K; ++k) {
            d_candidate_target[base + k] = -1;
            d_candidate_score[base + k] = -2.0f;
        }
    }

    for (int lane = 0; lane < tile_count; ++lane) {
        const int target = tile_start + lane;
        const float score =
            d_tile_scores[src * TOPK_TILE_WIDTH + lane];

        if (heap_size < K) {
            int position = heap_size++;
            d_candidate_target[base + position] = target;
            d_candidate_score[base + position] = score;

            while (position > 0) {
                int parent = (position - 1) / 2;

                if (!candidate_is_worse(
                        d_candidate_score[base + position],
                        d_candidate_target[base + position],
                        d_candidate_score[base + parent],
                        d_candidate_target[base + parent])) {
                    break;
                }

                float old_score = d_candidate_score[base + parent];
                d_candidate_score[base + parent] =
                    d_candidate_score[base + position];
                d_candidate_score[base + position] = old_score;

                int old_target = d_candidate_target[base + parent];
                d_candidate_target[base + parent] =
                    d_candidate_target[base + position];
                d_candidate_target[base + position] = old_target;

                position = parent;
            }
        } else if (
            score > d_candidate_score[base] ||
            (score == d_candidate_score[base] &&
             target < d_candidate_target[base])) {
            d_candidate_target[base] = target;
            d_candidate_score[base] = score;

            sift_down_min_heap(
                d_candidate_score,
                d_candidate_target,
                0,
                K,
                base);
        }
    }

    if (finalize) {
        // Heap sort using the same ordering.
        // Output is best-first, with equal scores ordered by target ID.
        for (int end = K - 1; end > 0; --end) {
            float score = d_candidate_score[base];
            d_candidate_score[base] = d_candidate_score[base + end];
            d_candidate_score[base + end] = score;

            int target = d_candidate_target[base];
            d_candidate_target[base] = d_candidate_target[base + end];
            d_candidate_target[base + end] = target;

            sift_down_min_heap(
                d_candidate_score,
                d_candidate_target,
                0,
                end,
                base);
        }
    }
}
namespace {

// Uploads adjacency once. The returned object owns both allocations.
DeviceGraph upload_device_graph(
    const Graph& graph,
    const GPUContext& context,
    double* transfer_ms)
{
    validate_graph(graph);

    if (graph.col_idx.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "GPU graph adjacency exceeds int CSR capacity");
    }

    DeviceGraph device_graph;
    device_graph.num_vertices = graph.num_vertices;
    device_graph.adjacency_entries = graph.col_idx.size();

    device_graph.row_ptr = DeviceBuffer<int>(
        static_cast<std::size_t>(graph.num_vertices) + 1,
        context.device_id);

    device_graph.col_idx = DeviceBuffer<int>(
        graph.col_idx.size(),
        context.device_id);

    const auto start = std::chrono::steady_clock::now();

    device_graph.row_ptr.upload(
        graph.row_ptr.data(),
        graph.row_ptr.size(),
        context.stream);

    device_graph.col_idx.upload(
        graph.col_idx.data(),
        graph.col_idx.size(),
        context.stream);

    if (transfer_ms != nullptr) {
        *transfer_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }

    return device_graph;
}

// Borrows adjacency and returns owned raw features.
// No feature download or normalization occurs here.
DeviceFeatureSet compute_device_raw_features(
    const DeviceGraph& graph,
    const GPUContext& context)
{
    if (graph.row_ptr.device_id() != context.device_id ||
        graph.col_idx.device_id() != context.device_id) {
        throw std::invalid_argument(
            "GPU feature extraction: graph belongs to another device");
    }

    CUDA_CHECK(cudaSetDevice(context.device_id));

    DeviceFeatureSet features;
    features.num_vertices = graph.num_vertices;
    features.raw = DeviceBuffer<NodeFeatures>(
        static_cast<std::size_t>(graph.num_vertices),
        context.device_id);

    // Normalized storage is allocated by the later normalization stage.
    if (graph.num_vertices == 0) {
        return features;
    }

    constexpr int threads = 256;
    const int blocks = 1 + (graph.num_vertices - 1) / threads;

    kernel_compute_features<<<blocks, threads, 0, context.stream>>>(
        graph.row_ptr.data(),
        graph.col_idx.data(),
        graph.num_vertices,
        features.raw.data());

    CUDA_CHECK(cudaGetLastError());

    // Simple completion contract for this first implementation:
    // returned features are ready, and borrowed adjacency is no longer in use.
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    return features;
}

std::size_t checked_feature_allocation_sum(
    std::size_t first,
    std::size_t second)
{
    if (second > std::numeric_limits<std::size_t>::max() - first) {
        throw std::overflow_error(
            "GPU feature allocation accounting overflow");
    }
    return first + second;
}

} // namespace


// Helper: GPU feature computation.


namespace {

// These statistics stay on the GPU.
struct DeviceNormalizationStats {
    float mean;
    float variance;
    float divisor;
};

// NodeFeatures::as_array() is currently a host method.
// Use explicit typed device access instead of casting to float*.
__device__ float normalization_read_feature(
    const NodeFeatures& value, int feature)
{
    switch (feature) {
        case 0: return value.degree;
        case 1: return value.clustering;
        case 2: return value.avg_neighbor_degree;
        case 3: return value.neighbor_degree_std;
        case 4: return value.triangle_count;
        case 5: return value.degree_squared;
        default: return 0.0f;
    }
}

__device__ void normalization_write_feature(
    NodeFeatures& value, int feature, float result)
{
    switch (feature) {
        case 0: value.degree = result; break;
        case 1: value.clustering = result; break;
        case 2: value.avg_neighbor_degree = result; break;
        case 3: value.neighbor_degree_std = result; break;
        case 4: value.triangle_count = result; break;
        case 5: value.degree_squared = result; break;
    }
}

// Treat source and target as one logical sequence without concatenating them.
__device__ float normalization_read_joint(
    const NodeFeatures* source,
    std::size_t source_count,
    const NodeFeatures* target,
    std::size_t index,
    int feature,
    int* invalid)
{
    const NodeFeatures& value = index < source_count
        ? source[index]
        : target[index - source_count];

    const float result = normalization_read_feature(value, feature);

    if (!isfinite(result)) {
        atomicExch(invalid, 1);

        // Keep reductions numerically usable; the host rejects the result.
        return 0.0f;
    }

    return result;
}

constexpr int NORMALIZATION_THREADS = 256;

/*
 * One block per feature, with bounded shared memory.
 *
 * Pass 1: reduce values to compute the mean.
 * Pass 2: reduce squared deviations to compute population variance.
 *
 * Different reduction order can produce small differences from CPU results.
 */
__global__ void kernel_joint_normalization_stats(
    const NodeFeatures* source,
    std::size_t source_count,
    const NodeFeatures* target,
    std::size_t target_count,
    DeviceNormalizationStats* statistics,
    int* invalid)
{
    const int feature = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t count = source_count + target_count;

    __shared__ float partial[NORMALIZATION_THREADS];

    if (count == 0) {
        if (lane == 0) {
            statistics[feature] = {0.0f, 0.0f, 1.0f};
        }
        return;
    }

    float sum = 0.0f;

    for (std::size_t i = static_cast<std::size_t>(lane);
         i < count;
         i += NORMALIZATION_THREADS) {
        sum += normalization_read_joint(
            source, source_count, target, i, feature, invalid);
    }

    partial[lane] = sum;
    __syncthreads();

    for (int stride = NORMALIZATION_THREADS / 2;
         stride > 0;
         stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    const float mean = partial[0] / static_cast<float>(count);

    // Every thread must read the mean before shared storage is reused.
    __syncthreads();

    float squared_deviation_sum = 0.0f;

    for (std::size_t i = static_cast<std::size_t>(lane);
         i < count;
         i += NORMALIZATION_THREADS) {
        const float value = normalization_read_joint(
            source, source_count, target, i, feature, invalid);

        const float delta = value - mean;
        squared_deviation_sum += delta * delta;
    }

    partial[lane] = squared_deviation_sum;
    __syncthreads();

    for (int stride = NORMALIZATION_THREADS / 2;
         stride > 0;
         stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        const float variance =
            partial[0] / static_cast<float>(count);
        const float stddev = sqrtf(variance);

        if (!isfinite(mean) ||
            !isfinite(variance) ||
            !isfinite(stddev)) {
            atomicExch(invalid, 1);
            statistics[feature] = {0.0f, 0.0f, 1.0f};
        } else {
            statistics[feature] = {
                mean,
                variance,
                stddev < SCGM_EPS ? 1.0f : stddev
            };
        }
    }
}

// Raw descriptors remain unchanged.
__global__ void kernel_apply_device_normalization(
    const NodeFeatures* raw,
    NodeFeatures* normalized,
    std::size_t count,
    const DeviceNormalizationStats* statistics,
    int* invalid)
{
    const std::size_t vertex =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (vertex >= count) return;

    NodeFeatures output{};

    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        const float value = normalization_read_feature(
            raw[vertex], feature);

        const float result =
            (value - statistics[feature].mean) /
            statistics[feature].divisor;

        if (!isfinite(value) || !isfinite(result)) {
            atomicExch(invalid, 1);
            normalization_write_feature(output, feature, 0.0f);
        } else {
            normalization_write_feature(output, feature, result);
        }
    }

    normalized[vertex] = output;
}

void validate_device_feature_set(
    const DeviceFeatureSet& features,
    const GPUContext& context)
{
    if (features.num_vertices < 0 ||
        features.raw.size() !=
            static_cast<std::size_t>(features.num_vertices)) {
        throw std::invalid_argument(
            "GPU normalization: invalid feature count");
    }

    if (!features.raw.empty() &&
        features.raw.device_id() != context.device_id) {
        throw std::invalid_argument(
            "GPU normalization: features belong to another device");
    }
}

// Normalize one logical population, which may contain two feature arrays.
// Output buffers must already be allocated.
void normalize_device_population(
    DeviceFeatureSet& first,
    DeviceFeatureSet* second,
    const GPUContext& context)
{
    const std::size_t first_count = first.raw.size();
    const std::size_t second_count =
        second != nullptr ? second->raw.size() : 0;

    if (first_count == 0 && second_count == 0) {
        return;
    }

    DeviceBuffer<DeviceNormalizationStats> statistics(
        NUM_FEATURES, context.device_id);
    DeviceBuffer<int> invalid(1, context.device_id);

    CUDA_CHECK(cudaSetDevice(context.device_id));

    try {
        CUDA_CHECK(cudaMemsetAsync(
            invalid.data(), 0, sizeof(int), context.stream));

        kernel_joint_normalization_stats
            <<<NUM_FEATURES, NORMALIZATION_THREADS, 0, context.stream>>>(
                first.raw.data(),
                first_count,
                second != nullptr ? second->raw.data() : nullptr,
                second_count,
                statistics.data(),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());

        if (first_count != 0) {
            const unsigned int blocks = static_cast<unsigned int>(
                1 + (first_count - 1) / NORMALIZATION_THREADS);

            kernel_apply_device_normalization
                <<<blocks, NORMALIZATION_THREADS, 0, context.stream>>>(
                    first.raw.data(),
                    first.normalized.data(),
                    first_count,
                    statistics.data(),
                    invalid.data());

            CUDA_CHECK(cudaGetLastError());
        }

        if (second_count != 0) {
            const unsigned int blocks = static_cast<unsigned int>(
                1 + (second_count - 1) / NORMALIZATION_THREADS);

            kernel_apply_device_normalization
                <<<blocks, NORMALIZATION_THREADS, 0, context.stream>>>(
                    second->raw.data(),
                    second->normalized.data(),
                    second_count,
                    statistics.data(),
                    invalid.data());

            CUDA_CHECK(cudaGetLastError());
        }

        // Download only a validation flag, never normalization statistics.
        // download() also waits for all preceding work on this stream.
        int host_invalid = 0;
        invalid.download(&host_invalid, 1, context.stream);

        if (host_invalid != 0) {
            throw std::runtime_error(
                "GPU normalization: nonfinite input or arithmetic result");
        }
    } catch (...) {
        // Complete pending work before local workspace owners are destroyed.
        // Preserve the original exception.
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }
}

// Produces normalized device arrays while preserving both raw arrays.
// Successful return means normalization has completed.
void normalize_device_feature_pair(
    DeviceFeatureSet& source,
    DeviceFeatureSet& target,
    bool joint_normalize,
    const GPUContext& context)
{
    if (&source == &target) {
        throw std::invalid_argument(
            "GPU normalization: source and target must be distinct owners");
    }

    validate_device_feature_set(source, context);
    validate_device_feature_set(target, context);

    source.normalized = DeviceBuffer<NodeFeatures>(
        source.raw.size(), context.device_id);

    target.normalized = DeviceBuffer<NodeFeatures>(
        target.raw.size(), context.device_id);

    if (joint_normalize) {
        normalize_device_population(source, &target, context);
    } else {
        normalize_device_population(source, nullptr, context);
        normalize_device_population(target, nullptr, context);
    }
}

} // namespace

std::vector<NodeFeatures> gpu_compute_features(
    const Graph& graph,
    const GPUContext& context,
    double* transfer_ms,
    bool normalize,
    size_t* allocated_bytes)
{
    auto device_graph = upload_device_graph(
        graph, context, transfer_ms);

    auto device_features = compute_device_raw_features(
        device_graph, context);

    if (normalize) {
        // Single-graph normalization is the separate-population case.
        DeviceFeatureSet empty_target;

        const auto start = std::chrono::steady_clock::now();

        normalize_device_feature_pair(
            device_features, empty_target, false, context);

        // Validation includes one completed four-byte D2H flag copy.
        // Its exact transfer duration is not separately measured here.
        (void)start;
    }

    std::size_t resident_bytes = checked_feature_allocation_sum(
        device_graph.row_ptr.bytes(),
        device_graph.col_idx.bytes());

    resident_bytes = checked_feature_allocation_sum(
        resident_bytes, device_features.raw.bytes());

    resident_bytes = checked_feature_allocation_sum(
        resident_bytes, device_features.normalized.bytes());

    // Include temporary normalization allocations in this wrapper's peak.
    const std::size_t normalization_workspace_bytes =
        normalize && !device_features.raw.empty()
        ? NUM_FEATURES * sizeof(DeviceNormalizationStats) + sizeof(int)
        : 0;

    const std::size_t wrapper_peak_bytes =
        checked_feature_allocation_sum(
            resident_bytes, normalization_workspace_bytes);

    if (allocated_bytes != nullptr) {
        *allocated_bytes = wrapper_peak_bytes;
    }

    peak_device_application_allocation_bytes = std::max(
        peak_device_application_allocation_bytes,
        checked_feature_allocation_sum(
            active_device_allocation_bytes,
            wrapper_peak_bytes));

    const DeviceBuffer<NodeFeatures>& requested_features =
        normalize ? device_features.normalized : device_features.raw;

    std::vector<NodeFeatures> host_features(
        static_cast<std::size_t>(graph.num_vertices));

    const auto start = std::chrono::steady_clock::now();

    requested_features.download(
        host_features.data(),
        host_features.size(),
        context.stream);

    if (transfer_ms != nullptr) {
        *transfer_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }

    return host_features;
}

// Preserved reference implementation:
// tiled cosine scoring followed by a serial heap merge per source.
// Keep its algorithm unchanged so future selectors have a stable baseline.
static std::vector<std::vector<Candidate>> gpu_compute_topk_serial(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K,
    const GPUContext& ctx,
    double* transfer_ms,
    size_t* allocated_bytes,
    double* similarity_kernel_ms,
    double* topk_kernel_ms)
{
    CUDA_CHECK(cudaSetDevice(ctx.device_id));

    int N = (int)feats_source.size();
    int M = (int)feats_target.size();

    if (K > M) {
        throw std::invalid_argument("gpu_compute_topk: K exceeds M");
    }

    // Flatten features.
    std::vector<float> flat_src, flat_tgt;
    features_to_flat_array(feats_source, flat_src);
    features_to_flat_array(feats_target, flat_tgt);

    if (allocated_bytes) {
        *allocated_bytes =
            static_cast<size_t>(N + M) * NUM_FEATURES * sizeof(float) +
            static_cast<size_t>(N) * K * (sizeof(int) + sizeof(float)) +
            static_cast<size_t>(N) * TOPK_TILE_WIDTH * sizeof(float);
    }

    // Allocate GPU memory.
    float* d_feats_src = nullptr;
    float* d_feats_tgt = nullptr;
    int* d_candidate_target = nullptr;
    float* d_candidate_score = nullptr;
    float* d_tile_scores = nullptr;

    tracked_cuda_malloc(
        &d_feats_src, N * NUM_FEATURES * sizeof(float));
    tracked_cuda_malloc(
        &d_feats_tgt, M * NUM_FEATURES * sizeof(float));
    tracked_cuda_malloc(
        &d_candidate_target, N * K * sizeof(int));
    tracked_cuda_malloc(
        &d_candidate_score, N * K * sizeof(float));
    tracked_cuda_malloc(
        &d_tile_scores, N * TOPK_TILE_WIDTH * sizeof(float));

    auto transfer_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        d_feats_src,
        flat_src.data(),
        N * NUM_FEATURES * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_feats_tgt,
        flat_tgt.data(),
        M * NUM_FEATURES * sizeof(float),
        cudaMemcpyHostToDevice));

    auto transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_total_ms =
        std::chrono::duration<double, std::milli>(
            transfer_end - transfer_start).count();

    const int tile_count =
        (M + TOPK_TILE_WIDTH - 1) / TOPK_TILE_WIDTH;

    struct TileEvents {
        cudaEvent_t sim_begin;
        cudaEvent_t sim_end;
        cudaEvent_t topk_begin;
        cudaEvent_t topk_end;
    };

    std::vector<TileEvents> tile_events(tile_count);

    for (auto& events : tile_events) {
        CUDA_CHECK(cudaEventCreate(&events.sim_begin));
        CUDA_CHECK(cudaEventCreate(&events.sim_end));
        CUDA_CHECK(cudaEventCreate(&events.topk_begin));
        CUDA_CHECK(cudaEventCreate(&events.topk_end));
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        const int tile_start = tile * TOPK_TILE_WIDTH;
        const int count = std::min(TOPK_TILE_WIDTH, M - tile_start);
        auto& events = tile_events[tile];

        CUDA_CHECK(cudaEventRecord(events.sim_begin, ctx.stream));

        kernel_similarity_tile<<<N, TOPK_THREADS, 0, ctx.stream>>>(
            d_feats_src,
            d_feats_tgt,
            M,
            tile_start,
            count,
            NUM_FEATURES,
            d_tile_scores);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(events.sim_end, ctx.stream));
        CUDA_CHECK(cudaEventRecord(events.topk_begin, ctx.stream));

        kernel_merge_tile_topk<<<N, 1, 0, ctx.stream>>>(
            d_tile_scores,
            tile_start,
            count,
            K,
            tile == tile_count - 1,
            d_candidate_target,
            d_candidate_score);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(events.topk_end, ctx.stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    double sim_elapsed = 0.0;
    double topk_elapsed = 0.0;

    for (auto& events : tile_events) {
        float elapsed = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed, events.sim_begin, events.sim_end));
        sim_elapsed += elapsed;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed, events.topk_begin, events.topk_end));
        topk_elapsed += elapsed;

        CUDA_CHECK(cudaEventDestroy(events.sim_begin));
        CUDA_CHECK(cudaEventDestroy(events.sim_end));
        CUDA_CHECK(cudaEventDestroy(events.topk_begin));
        CUDA_CHECK(cudaEventDestroy(events.topk_end));
    }

    if (similarity_kernel_ms) {
        *similarity_kernel_ms = sim_elapsed;
    }
    if (topk_kernel_ms) {
        *topk_kernel_ms = topk_elapsed;
    }

    // Copy back.
    std::vector<int> h_targets(N * K);
    std::vector<float> h_scores(N * K);
    transfer_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        h_targets.data(),
        d_candidate_target,
        N * K * sizeof(int),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_scores.data(),
        d_candidate_score,
        N * K * sizeof(float),
        cudaMemcpyDeviceToHost));

    transfer_end = std::chrono::high_resolution_clock::now();
    transfer_total_ms +=
        std::chrono::duration<double, std::milli>(
            transfer_end - transfer_start).count();

    if (transfer_ms) *transfer_ms += transfer_total_ms;

    // Build candidates.
    std::vector<std::vector<Candidate>> candidates(N);

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            int tgt = h_targets[i * K + k];
            float score = h_scores[i * K + k];

            if (tgt >= 0) {
                candidates[i].push_back({tgt, score});
            }
        }
    }

    // Cleanup.
    tracked_cuda_free(
        d_feats_src, N * NUM_FEATURES * sizeof(float));
    tracked_cuda_free(
        d_feats_tgt, M * NUM_FEATURES * sizeof(float));
    tracked_cuda_free(
        d_candidate_target, N * K * sizeof(int));
    tracked_cuda_free(
        d_candidate_score, N * K * sizeof(float));
    tracked_cuda_free(
        d_tile_scores, N * TOPK_TILE_WIDTH * sizeof(float));

    return candidates;
}

// Existing calls retain the original serial reference by default.
// Explicit Parallel requests use bounded scoring and selection, including fallback.
std::vector<std::vector<Candidate>> gpu_compute_topk(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K, const GPUContext& ctx, double* transfer_ms,
    size_t* allocated_bytes, double* similarity_kernel_ms,
    double* topk_kernel_ms, GPUSelector selector,
    const GPUTopKOptions& options, GPUSelectionInfo* info)
{
    Timer candidate_wall("GPU candidate stage");
    if (K <= 0 || (!feats_source.empty() &&
        static_cast<size_t>(K) > feats_target.size()))
        throw std::invalid_argument("gpu_compute_topk: positive K and enough targets required");
    if (selector == GPUSelector::SerialHeap && !options.force_streaming_serial) {
        double measured_selection_ms = 0.0;
        auto candidates = gpu_compute_topk_serial(feats_source, feats_target, K, ctx,
            transfer_ms, allocated_bytes, similarity_kernel_ms,
            topk_kernel_ms ? topk_kernel_ms : &measured_selection_ms);
        if (info) {
            *info = GPUSelectionInfo{};
            info->batch_capacity = TOPK_TILE_WIDTH;
            info->selection_ms = topk_kernel_ms ? *topk_kernel_ms : measured_selection_ms;
            info->candidate_stage_wall_ms = candidate_wall.elapsed_ms();
        }
        return candidates;
    }
    if (selector != GPUSelector::Parallel && selector != GPUSelector::SerialHeap)
        throw std::invalid_argument("gpu_compute_topk: invalid selector");
    auto selected = gpu_score_exhaustive_pools(feats_source, feats_target, K, ctx, selector, options);
    if (selected.selection.scratch_peak_bytes >
        std::numeric_limits<size_t>::max() - selected.resident_feature_bytes)
        throw std::overflow_error("GPU candidate allocation accounting overflow");
    const auto peak = selected.resident_feature_bytes + selected.selection.scratch_peak_bytes;
    peak_device_application_allocation_bytes =
        std::max(peak_device_application_allocation_bytes, peak);
    if (allocated_bytes) *allocated_bytes = peak;
    if (transfer_ms) *transfer_ms += selected.transfer_ms;
    if (similarity_kernel_ms) *similarity_kernel_ms = selected.similarity_ms;
    if (topk_kernel_ms) *topk_kernel_ms = selected.selection.selection_ms;
    selected.selection.candidate_stage_wall_ms = candidate_wall.elapsed_ms();
    if (info) *info = selected.selection;
    return std::move(selected.candidates);
}

GPUMatchingResult gpu_compute_candidates_legacy(
    const Graph& source,
    const Graph& target,
    int K,
    const GPUContext& ctx,
    bool joint_normalize,
    GPUSelector selector,
    const GPUTopKOptions& options)
{
    GPUMatchingResult result;

    CUDA_CHECK(cudaSetDevice(ctx.device_id));
    active_device_allocation_bytes = 0;
    peak_device_application_allocation_bytes = 0;

    // Step 1: Feature extraction on GPU.
    const double transfer_before_features =
        result.timings.host_device_transfer_ms;

    Timer t_feat("GPU feature extraction");

    size_t feature_alloc_bytes = 0;
    size_t target_feature_alloc_bytes = 0;
    size_t topk_alloc_bytes = 0;

    result.feats_source = gpu_compute_features(
        source,
        ctx,
        &result.timings.host_device_transfer_ms,
        false,
        &feature_alloc_bytes);

    result.feats_target = gpu_compute_features(
        target,
        ctx,
        &result.timings.host_device_transfer_ms,
        false,
        &target_feature_alloc_bytes);

    if (joint_normalize) {
        normalize_features_joint_zscore(
            result.feats_source, result.feats_target);
    } else {
        normalize_features_zscore(result.feats_source);
        normalize_features_zscore(result.feats_target);
    }

    const double feature_stage_wall_ms = t_feat.elapsed_ms();
    const double feature_transfer_ms =
        result.timings.host_device_transfer_ms -
        transfer_before_features;

    result.timings.feature_extraction_ms = std::max(
        0.0, feature_stage_wall_ms - feature_transfer_ms);

    // Step 2: Tiled similarity and Top-K kernels run as separate stages.
    // Default calls retain SerialHeap; explicit Parallel uses bounded streaming.
    result.candidates = gpu_compute_topk(
        result.feats_source,
        result.feats_target,
        K,
        ctx,
        &result.timings.host_device_transfer_ms,
        &topk_alloc_bytes,
        &result.timings.similarity_ms,
        &result.timings.topk_selection_ms,
        selector, options, &result.selection);

    result.timings.candidate_generation_ms =
        result.timings.similarity_ms +
        result.timings.topk_selection_ms;

    result.timings.peak_device_application_allocated_bytes =
        peak_device_application_allocation_bytes;

    result.timings.total_ms =
        result.timings.feature_extraction_ms +
        result.timings.candidate_generation_ms +
        result.timings.host_device_transfer_ms;

    return result;
}

size_t gpu_feature_memory_bytes(int num_vertices) {
    return num_vertices * NUM_FEATURES * sizeof(float);
}

namespace device_cost_detail {

/*
 * One block per source row.
 *
 * Each candidate computes its target-ID rank within the row.
 * Unique target IDs give unique destination positions.
 *
 * Work: O(sum of squared row lengths).
 * Temporary global workspace: one validation flag.
 */
__global__ void build_cost_rows(
    const std::size_t* candidate_offsets,
    const int* candidate_targets,
    const float* candidate_scores,
    int num_sources,
    int num_targets,
    std::size_t entry_count,
    int* cost_offsets,
    int* cost_targets,
    float* cost_values,
    int* invalid)
{
    const int row = static_cast<int>(blockIdx.x);

    // Validate boundary offsets even when there are no source rows.
    if (row == 0 && threadIdx.x == 0) {
        if (candidate_offsets[0] != 0 ||
            candidate_offsets[num_sources] != entry_count) {
            atomicExch(invalid, 1);
        }

        if (num_sources == 0) {
            cost_offsets[0] = 0;
        }
    }

    if (row >= num_sources) {
        return;
    }

    const std::size_t begin = candidate_offsets[row];
    const std::size_t end = candidate_offsets[row + 1];

    // All threads in this block take the same branch.
    // Never dereference candidate arrays using invalid offsets.
    if (begin > end || end > entry_count) {
        if (threadIdx.x == 0) {
            atomicExch(invalid, 1);
        }
        return;
    }

    if (threadIdx.x == 0) {
        // Host validation guarantees entry_count <= INT_MAX.
        cost_offsets[row] = static_cast<int>(begin);

        if (row == num_sources - 1) {
            cost_offsets[num_sources] = static_cast<int>(end);
        }
    }

    for (std::size_t position =
             begin + static_cast<std::size_t>(threadIdx.x);
         position < end;
         position += blockDim.x) {
        const int target = candidate_targets[position];
        const float similarity = candidate_scores[position];

        if (target < 0 ||
            target >= num_targets ||
            !isfinite(similarity)) {
            atomicExch(invalid, 1);
            continue;
        }

        std::size_t rank = 0;
        bool duplicate = false;

        for (std::size_t other = begin; other < end; ++other) {
            const int other_target = candidate_targets[other];

            if (other_target < target) {
                ++rank;
            } else if (other != position && other_target == target) {
                duplicate = true;
            }
        }

        // Duplicate targets would have identical ranks.
        // Skip their writes to avoid racing on the same output position.
        if (duplicate) {
            atomicExch(invalid, 1);
            continue;
        }

        const float cost = 1.0f - similarity;

        if (!isfinite(cost)) {
            atomicExch(invalid, 1);
            continue;
        }

        const std::size_t destination = begin + rank;

        cost_targets[destination] = target;
        cost_values[destination] = cost;
    }
}

} // namespace device_cost_detail


namespace resident_detail {

std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("Resident pipeline byte-count overflow");
    }
    return a + b;
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b != 0 &&
        a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("Resident pipeline byte-count overflow");
    }
    return a * b;
}

std::size_t graph_bytes(const DeviceGraph& graph) {
    return add(graph.row_ptr.bytes(), graph.col_idx.bytes());
}

void note_download(
    DeviceMatchingResult& result,
    std::size_t bytes,
    double elapsed_ms)
{
    result.residency.known_d2h_bytes = add(
        result.residency.known_d2h_bytes, bytes);

    result.residency.cpu_handoff_ms += elapsed_ms;
    result.timings.host_device_transfer_ms += elapsed_ms;
}

} // namespace resident_detail

void gpu_normalize_device_features(
    DeviceFeatureSet& source,
    DeviceFeatureSet& target,
    bool joint_normalize,
    const GPUContext& context)
{
    normalize_device_feature_pair(
        source, target, joint_normalize, context);
}

DeviceMatchingResult gpu_compute_device_candidates(
    const Graph& source,
    const Graph& target,
    int k,
    const GPUContext& context,
    bool joint_normalize,
    GPUSelector selector,
    const GPUTopKOptions& options,
    const StructuralIndexConfig* index_config)
{
    validate_graph(source);
    validate_graph(target);

    if (k <= 0 ||
        (source.num_vertices > 0 && k > target.num_vertices)) {
        throw std::invalid_argument(
            "Resident matching requires positive K and enough targets");
    }

    if (index_config != nullptr) {
        validate_structural_index_config(*index_config);
    }

    CUDA_CHECK(cudaSetDevice(context.device_id));

    DeviceMatchingResult result;
    auto& evidence = result.residency;

    // Overall peak remains unavailable until every allocator is instrumented.
    result.device_allocation_tracking_complete = false;

    Timer total("Resident GPU pipeline");

    {
        Timer stage("Resident feature extraction");

        result.source_graph = upload_device_graph(
            source, context, &result.timings.host_device_transfer_ms);

        result.target_graph = upload_device_graph(
            target, context, &result.timings.host_device_transfer_ms);

        result.source_features = compute_device_raw_features(
            result.source_graph, context);

        result.target_features = compute_device_raw_features(
            result.target_graph, context);

        result.timings.feature_extraction_ms = stage.elapsed_ms();
    }

    evidence.adjacency_bytes = resident_detail::add(
        resident_detail::graph_bytes(result.source_graph),
        resident_detail::graph_bytes(result.target_graph));

    evidence.raw_feature_bytes = resident_detail::add(
        result.source_features.raw.bytes(),
        result.target_features.raw.bytes());

    // Graph adjacency is uploaded exactly once.
    evidence.known_h2d_bytes = evidence.adjacency_bytes;

    result.indexed = index_config != nullptr;

    if (result.indexed) {
        Timer build("Resident structural index");

        result.structural_index = build_gpu_structural_index(
            result.target_features.raw.data(),
            result.target_features.raw.size(),
            *index_config,
            context);

        evidence.index_build_ms = build.elapsed_ms();

        Timer retrieve("Resident structural retrieval");

        result.structural_queries = retrieve_gpu_structural_candidates(
            result.structural_index,
            result.source_features.raw.data(),
            result.source_features.raw.size(),
            k,
            context);

        evidence.retrieval_ms = retrieve.elapsed_ms();

        const auto& index = result.structural_index;

        evidence.index_payload_bytes = resident_detail::add(
            resident_detail::multiply(
                index.bucket_count, sizeof(StructuralBucketKey)),
            resident_detail::add(
                resident_detail::multiply(
                    resident_detail::add(index.bucket_count, 1),
                    sizeof(std::size_t)),
                resident_detail::multiply(
                    index.target_count, sizeof(int))));

        evidence.query_payload_bytes = resident_detail::multiply(
            result.structural_queries.source_count,
            sizeof(GPUStructuralQuery));

        // Index construction/retrieval library internals have not been
        // audited for every metadata transfer or temporary allocation.
        evidence.transfer_tracking_complete = false;
    }

    {
        Timer normalization("Resident GPU normalization");

        gpu_normalize_device_features(
            result.source_features,
            result.target_features,
            joint_normalize,
            context);

        evidence.normalization_ms = normalization.elapsed_ms();
    }

    evidence.normalized_feature_bytes = resident_detail::add(
        result.source_features.normalized.bytes(),
        result.target_features.normalized.bytes());

    const std::size_t normalization_populations = joint_normalize
        ? static_cast<std::size_t>(
            source.num_vertices > 0 || target.num_vertices > 0)
        : static_cast<std::size_t>(source.num_vertices > 0) +
          static_cast<std::size_t>(target.num_vertices > 0);

    if (normalization_populations != 0) {
        evidence.normalization_workspace_peak_bytes =
            NUM_FEATURES * sizeof(DeviceNormalizationStats) + sizeof(int);
    }

    // One validation flag per nonempty normalization population.
    evidence.known_d2h_bytes = resident_detail::multiply(
        normalization_populations, sizeof(int));

    {
        auto selected = result.indexed
            ? gpu_score_indexed_device_pools(
                result.source_features,
                result.target_features,
                result.structural_index,
                result.structural_queries,
                k, context, selector, options)
            : gpu_score_exhaustive_device_pools(
                result.source_features,
                result.target_features,
                k, context, selector, options);

        result.candidates = std::move(selected.candidates);
        result.selection = selected.selection;

        result.timings.similarity_ms = selected.similarity_ms;
        result.timings.topk_selection_ms = selected.selection.selection_ms;
        result.timings.host_device_transfer_ms += selected.transfer_ms;

        evidence.scoring_selection_scratch_peak_bytes =
            selected.selection.scratch_peak_bytes;

        /*
         * Current device scorer:
         *   - two scoring-validation flag downloads;
         *   - one validation flag download per consume() batch;
         *   - finish_pool_to_device() performs no host transfer.
         *
         * Keep this accounting synchronized with those implementations.
         */
        const std::size_t scoring_flag_count = resident_detail::add(
            selected.selection.batches, 2);

        evidence.known_d2h_bytes = resident_detail::add(
            evidence.known_d2h_bytes,
            resident_detail::multiply(scoring_flag_count, sizeof(int)));
    }

    evidence.candidate_bytes = resident_detail::add(
        result.candidates.row_offsets.bytes(),
        resident_detail::add(
            result.candidates.target_ids.bytes(),
            result.candidates.similarities.bytes()));

    {
        Timer costs("Resident GPU sparse costs");

        result.costs = gpu_build_device_costs(
            result.candidates, context);

        evidence.cost_construction_ms = costs.elapsed_ms();
    }

    evidence.cost_workspace_peak_bytes = sizeof(int);

    
    // Compact CSR construction downloads two validation flags
    evidence.known_d2h_bytes = resident_detail::add(
        evidence.known_d2h_bytes, 2 * sizeof(int));

    evidence.cost_bytes = resident_detail::add(
        result.costs.row_ptr.bytes(),
        resident_detail::add(
            result.costs.col_idx.bytes(),
            result.costs.values.bytes()));

    result.selection.candidate_stage_wall_ms += evidence.retrieval_ms;

    result.timings.candidate_generation_ms =
        evidence.retrieval_ms +
        result.timings.similarity_ms +
        result.timings.topk_selection_ms;

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    result.timings.total_ms = total.elapsed_ms();

    return result;
}

std::vector<std::vector<Candidate>> gpu_download_candidates(
    DeviceMatchingResult& result,
    const GPUContext& context)
{
    Timer handoff("Candidate CPU handoff");

    const auto& device = result.candidates;

    std::vector<std::size_t> offsets(device.row_offsets.size());
    std::vector<int> targets(device.target_ids.size());
    std::vector<float> scores(device.similarities.size());

    device.row_offsets.download(
        offsets.data(), offsets.size(), context.stream);
    device.target_ids.download(
        targets.data(), targets.size(), context.stream);
    device.similarities.download(
        scores.data(), scores.size(), context.stream);

    if (offsets.size() !=
            static_cast<std::size_t>(device.num_sources) + 1 ||
        offsets.front() != 0 ||
        offsets.back() != device.entry_count ||
        !std::is_sorted(offsets.begin(), offsets.end())) {
        throw std::runtime_error("Invalid downloaded candidate offsets");
    }

    std::vector<std::vector<Candidate>> host(
        static_cast<std::size_t>(device.num_sources));

    for (std::size_t row = 0; row < host.size(); ++row) {
        host[row].reserve(offsets[row + 1] - offsets[row]);

        for (std::size_t i = offsets[row]; i < offsets[row + 1]; ++i) {
            host[row].push_back(Candidate{targets[i], scores[i]});
        }
    }

    resident_detail::note_download(
        result,
        resident_detail::add(
            device.row_offsets.bytes(),
            resident_detail::add(
                device.target_ids.bytes(), device.similarities.bytes())),
        handoff.elapsed_ms());

    return host;
}

CSRMatrix gpu_download_costs(
    DeviceMatchingResult& result,
    const GPUContext& context)
{
    Timer handoff("Cost CSR CPU handoff");

    const auto& device = result.costs;

    CSRMatrix host;
    host.rows = device.num_rows;
    host.cols = device.num_cols;
    host.row_ptr.resize(device.row_ptr.size());
    host.col_idx.resize(device.col_idx.size());
    host.values.resize(device.values.size());

    device.row_ptr.download(
        host.row_ptr.data(), host.row_ptr.size(), context.stream);
    device.col_idx.download(
        host.col_idx.data(), host.col_idx.size(), context.stream);
    device.values.download(
        host.values.data(), host.values.size(), context.stream);

    resident_detail::note_download(
        result,
        resident_detail::add(
            device.row_ptr.bytes(),
            resident_detail::add(
                device.col_idx.bytes(), device.values.bytes())),
        handoff.elapsed_ms());

    return host;
}

void gpu_download_normalized_features(
    DeviceMatchingResult& result,
    std::vector<NodeFeatures>& source,
    std::vector<NodeFeatures>& target,
    const GPUContext& context)
{
    Timer handoff("Feature CPU handoff");

    source.resize(result.source_features.normalized.size());
    target.resize(result.target_features.normalized.size());

    result.source_features.normalized.download(
        source.data(), source.size(), context.stream);
    result.target_features.normalized.download(
        target.data(), target.size(), context.stream);

    resident_detail::note_download(
        result,
        resident_detail::add(
            result.source_features.normalized.bytes(),
            result.target_features.normalized.bytes()),
        handoff.elapsed_ms());
}

GPUMatchingResult gpu_download_matching_result(
    DeviceMatchingResult& result,
    const GPUContext& context)
{
    GPUMatchingResult host;

    gpu_download_normalized_features(
        result, host.feats_source, host.feats_target, context);

    host.candidates = gpu_download_candidates(result, context);

    host.timings = result.timings;
    host.timings.total_ms += result.residency.cpu_handoff_ms;
    host.selection = result.selection;
    host.device_allocation_tracking_complete =
        result.device_allocation_tracking_complete;
    host.device_resident = true;
    host.residency = result.residency;

    return host;
}

GPUMatchingResult gpu_compute_candidates(
    const Graph& source,
    const Graph& target,
    int k,
    const GPUContext& context,
    bool joint_normalize,
    GPUSelector selector,
    const GPUTopKOptions& options)
{
    // Preserve the original default serial reference.
    if (selector == GPUSelector::SerialHeap &&
        !options.force_streaming_serial) {
        return gpu_compute_candidates_legacy(
            source, target, k, context,
            joint_normalize, selector, options);
    }

    auto device = gpu_compute_device_candidates(
        source, target, k, context,
        joint_normalize, selector, options);

    return gpu_download_matching_result(device, context);
}

// DeviceCosts gpu_build_device_costs(
//     const DeviceCandidates& candidates,
//     const GPUContext& context)
// {
//     if (candidates.num_sources < 0 ||
//         candidates.num_targets < 0) {
//         throw std::invalid_argument(
//             "GPU costs: negative matrix dimensions");
//     }

//     if (candidates.entry_count >
//         static_cast<std::size_t>(std::numeric_limits<int>::max())) {
//         throw std::overflow_error(
//             "GPU costs: entry count exceeds host CSR int capacity");
//     }

//     const std::size_t offset_count =
//         static_cast<std::size_t>(candidates.num_sources) + 1;

//     if (candidates.row_offsets.size() != offset_count ||
//         candidates.target_ids.size() != candidates.entry_count ||
//         candidates.similarities.size() != candidates.entry_count) {
//         throw std::invalid_argument(
//             "GPU costs: candidate buffer sizes disagree");
//     }

//     if (candidates.row_offsets.device_id() != context.device_id ||
//         (!candidates.target_ids.empty() &&
//          candidates.target_ids.device_id() != context.device_id) ||
//         (!candidates.similarities.empty() &&
//          candidates.similarities.device_id() != context.device_id)) {
//         throw std::invalid_argument(
//             "GPU costs: candidates belong to another device");
//     }

//     CUDA_CHECK(cudaSetDevice(context.device_id));

//     DeviceCosts costs;
//     costs.num_rows = candidates.num_sources;
//     costs.num_cols = candidates.num_targets;
//     costs.entry_count = candidates.entry_count;

//     costs.row_ptr = DeviceBuffer<int>(
//         offset_count, context.device_id);

//     costs.col_idx = DeviceBuffer<int>(
//         candidates.entry_count, context.device_id);

//     costs.values = DeviceBuffer<float>(
//         candidates.entry_count, context.device_id);

//     DeviceBuffer<int> invalid(1, context.device_id);

//     try {
//         CUDA_CHECK(cudaMemsetAsync(
//             invalid.data(), 0, sizeof(int), context.stream));

//         const unsigned int blocks = candidates.num_sources > 0
//             ? static_cast<unsigned int>(candidates.num_sources)
//             : 1u;

//         device_cost_detail::build_cost_rows
//             <<<blocks, 256, 0, context.stream>>>(
//                 candidates.row_offsets.data(),
//                 candidates.target_ids.data(),
//                 candidates.similarities.data(),
//                 candidates.num_sources,
//                 candidates.num_targets,
//                 candidates.entry_count,
//                 costs.row_ptr.data(),
//                 costs.col_idx.data(),
//                 costs.values.data(),
//                 invalid.data());

//         CUDA_CHECK(cudaGetLastError());

//         int host_invalid = 0;
//         invalid.download(&host_invalid, 1, context.stream);

//         if (host_invalid != 0) {
//             throw std::invalid_argument(
//                 "GPU costs: invalid offsets, target IDs, "
//                 "duplicate targets, or nonfinite scores/costs");
//         }
//     } catch (...) {
//         // Finish pending work before owned output/workspace is released.
//         (void)cudaStreamSynchronize(context.stream);
//         throw;
//     }

//     return costs;
// }
DeviceCosts gpu_build_device_costs(
    const DeviceCandidates& candidates,
    const GPUContext& context)
{
    if (candidates.num_sources < 0 ||
        candidates.num_targets < 0) {
        throw std::invalid_argument(
            "GPU costs: negative matrix dimensions");
    }

    const std::size_t offset_count =
        static_cast<std::size_t>(candidates.num_sources) + 1;

    if (candidates.row_offsets.size() != offset_count ||
        candidates.target_ids.size() != candidates.entry_count ||
        candidates.similarities.size() != candidates.entry_count) {
        throw std::invalid_argument(
            "GPU costs: candidate buffer sizes disagree");
    }

    if (candidates.row_offsets.device_id() != context.device_id ||
        (!candidates.target_ids.empty() &&
         candidates.target_ids.device_id() != context.device_id) ||
        (!candidates.similarities.empty() &&
         candidates.similarities.device_id() != context.device_id)) {
        throw std::invalid_argument(
            "GPU costs: candidates belong to another device");
    }

    DeviceCandidateRowsView input;
    input.num_rows = candidates.num_sources;
    input.num_cols = candidates.num_targets;
    input.device_id = context.device_id;
    input.storage_entries = candidates.entry_count;
    input.storage_offsets = candidates.row_offsets.data();
    input.row_counts = nullptr; // Existing candidates are already compact.
    input.target_ids = candidates.target_ids.data();
    input.similarities = candidates.similarities.data();

    return gpu_build_candidate_csr(input, context);
}



size_t gpu_candidate_memory_bytes(int N, int K) {
    return N * K * (sizeof(int) + sizeof(float));
}