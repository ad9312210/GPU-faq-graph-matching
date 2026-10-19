#include "gpu_matching.hpp"
#include <math_constants.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
std::size_t checked_bytes(std::size_t n, std::size_t size) {
    if (size == 0) return 0;
    if (n > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error("GPU scoring allocation overflow");
    return n * size;
}
std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("GPU scoring counter overflow");
    return a + b;
}
template<class T> struct DeviceArray {
    T* p = nullptr;
    int device;
    explicit DeviceArray(std::size_t n) {
        CUDA_CHECK(cudaGetDevice(&device));
        if (n) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), checked_bytes(n, sizeof(T))));
    }
    ~DeviceArray() {
        if (!p) return;
        int previous = -1;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(device) == cudaSuccess) cudaFree(p);
        if (restore && previous != device) cudaSetDevice(previous);
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
struct Events {
    cudaEvent_t begin = nullptr, end = nullptr;
    Events() {
        CUDA_CHECK(cudaEventCreate(&begin));
        auto e = cudaEventCreate(&end);
        if (e != cudaSuccess) { cudaEventDestroy(begin); begin = nullptr; CUDA_CHECK(e); }
    }
    ~Events() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
    void start(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(begin, s)); }
    double finish(cudaStream_t s) {
        CUDA_CHECK(cudaEventRecord(end, s));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end)); return ms;
    }
};
double copy_timed(void* destination, const void* source, std::size_t bytes,
                  cudaMemcpyKind kind, cudaStream_t stream) {
    if (!bytes) return 0;
    Events events;
    events.start(stream);
    // Complete the copy before local host metadata can leave scope.
    auto error = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (error != cudaSuccess) { cudaStreamSynchronize(stream); CUDA_CHECK(error); }
    return events.finish(stream);
}
void validate_features(const std::vector<NodeFeatures>& features) {
    if (features.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("GPU scoring vertex count exceeds int IDs");
    for (const auto& feature : features)
        for (int f = 0; f < NUM_FEATURES; ++f)
            if (!std::isfinite(feature.as_array(f)))
                throw std::invalid_argument("Nonfinite GPU scoring feature");
}
void validate_request(const std::vector<NodeFeatures>& sources,
                      const std::vector<NodeFeatures>& targets, int k) {
    validate_features(sources); validate_features(targets);
    if (k <= 0 || (!sources.empty() && static_cast<std::size_t>(k) > targets.size()))
        throw std::invalid_argument("Matching requires positive K and enough targets");
}

// Preserve the existing float32 cosine operations, while checking nonfinite
// intermediates before fminf/fmaxf can conceal a NaN.
__device__ float checked_cosine(const float* source, const float* target) {
    float dot = 0, norm_source = 0, norm_target = 0;
    for (int f = 0; f < NUM_FEATURES; ++f) {
        dot += source[f] * target[f];
        norm_source += source[f] * source[f];
        norm_target += target[f] * target[f];
    }
    const float denominator = sqrtf(norm_source) * sqrtf(norm_target);
    if (!isfinite(dot) || !isfinite(norm_source) || !isfinite(norm_target) ||
        !isfinite(denominator)) return CUDART_NAN_F;
    if (denominator < 1e-12f) return 0.0f;
    const float score = dot / denominator;
    if (!isfinite(score)) return CUDART_NAN_F;
    return fminf(1.0f, fmaxf(-1.0f, score));
}
__global__ void score_span(const float* sources, const float* targets,
    std::size_t source, const int* grouped_targets, std::size_t start,
    std::size_t count, GPUScoredCandidate* output) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int target = grouped_targets ? grouped_targets[start + i] : static_cast<int>(start + i);
    output[i] = GPUScoredCandidate{
        checked_cosine(sources + source * NUM_FEATURES,
                       targets + static_cast<std::size_t>(target) * NUM_FEATURES), target};
}

// Host-side scheduling metadata only. IDs and scored candidates remain on GPU.
bool compatible(StructuralBucketKey source, StructuralBucketKey target, int radius) {
    const auto dx = source.degree_bin >= target.degree_bin
        ? source.degree_bin - target.degree_bin : target.degree_bin - source.degree_bin;
    const auto dy = source.neighbor_bin >= target.neighbor_bin
        ? source.neighbor_bin - target.neighbor_bin : target.neighbor_bin - source.neighbor_bin;
    return radius >= 0 && std::max(dx, dy) <= radius;
}

GPUCandidateSelectionResult score_pools(
    const std::vector<NodeFeatures>& sources, const std::vector<NodeFeatures>& targets,
    int k, const GPUContext& context, GPUSelector selector,
    const GPUTopKOptions& options, const GPUStructuralIndex* index,
    const GPUStructuralQueries* queries) {
    Timer candidate_wall("GPU scoring and selection");
    validate_request(sources, targets, k);
    CUDA_CHECK(cudaSetDevice(context.device_id));
    if (index && (index->device_id != context.device_id ||
        queries->device_id != context.device_id || queries->source_count != sources.size() ||
        index->target_count != targets.size() || !index->d_bucket_offsets ||
        (!sources.empty() && !queries->d_queries)))
        throw std::invalid_argument("Indexed scoring dimensions/device disagree");

    GPUCandidateSelectionResult result;
    result.candidates.resize(sources.size());
    GPUStreamingTopK workspace(static_cast<int>(targets.size()), k, context, selector, options);
    if (sources.empty()) { result.selection = workspace.info(); result.selection.candidate_stage_wall_ms = candidate_wall.elapsed_ms(); return result; }

    auto flatten = [](const std::vector<NodeFeatures>& features) {
        const auto count = checked_bytes(features.size(), static_cast<std::size_t>(NUM_FEATURES));
        (void)checked_bytes(count, sizeof(float));
        std::vector<float> flat(count);
        for (std::size_t i = 0; i < features.size(); ++i)
            for (int f = 0; f < NUM_FEATURES; ++f)
                flat[i * NUM_FEATURES + static_cast<std::size_t>(f)] = features[i].as_array(f);
        return flat;
    };
    auto flat_sources = flatten(sources);
    auto flat_targets = flatten(targets);
    DeviceArray<float> device_sources(flat_sources.size()), device_targets(flat_targets.size());
    const auto source_bytes = checked_bytes(flat_sources.size(), sizeof(float));
    const auto target_bytes = checked_bytes(flat_targets.size(), sizeof(float));
    result.resident_feature_bytes = checked_add(source_bytes, target_bytes);
    result.transfer_ms += copy_timed(device_sources.p, flat_sources.data(), source_bytes,
                                     cudaMemcpyHostToDevice, context.stream);
    result.transfer_ms += copy_timed(device_targets.p, flat_targets.data(), target_bytes,
                                     cudaMemcpyHostToDevice, context.stream);

    std::vector<StructuralBucketKey> keys;
    std::vector<std::size_t> offsets;
    std::vector<GPUStructuralQuery> decisions;
    if (index) {
        keys.resize(index->bucket_count); offsets.resize(checked_add(index->bucket_count, 1));
        decisions.resize(sources.size());
        result.transfer_ms += copy_timed(keys.data(), index->d_bucket_keys,
            checked_bytes(keys.size(), sizeof(StructuralBucketKey)), cudaMemcpyDeviceToHost, context.stream);
        result.transfer_ms += copy_timed(offsets.data(), index->d_bucket_offsets,
            checked_bytes(offsets.size(), sizeof(std::size_t)), cudaMemcpyDeviceToHost, context.stream);
        result.transfer_ms += copy_timed(decisions.data(), queries->d_queries,
            checked_bytes(decisions.size(), sizeof(GPUStructuralQuery)), cudaMemcpyDeviceToHost, context.stream);
    }
    Events score_events;
    for (std::size_t source = 0; source < sources.size(); ++source) {
        workspace.begin_pool();
        std::size_t scored = 0;
        auto feed_span = [&](const int* ids, std::size_t begin, std::size_t end) {
            for (auto start = begin; start < end;) {
                const auto count = std::min(workspace.capacity(), end - start);
                const auto blocks = static_cast<unsigned>((count + 255) / 256);
                score_events.start(context.stream);
                score_span<<<blocks, 256, 0, context.stream>>>(device_sources.p, device_targets.p,
                    source, ids, start, count, workspace.input());
                CUDA_CHECK(cudaGetLastError());
                result.similarity_ms += score_events.finish(context.stream);
                workspace.consume(count);
                scored = checked_add(scored, count);
                start += count;
            }
        };
        if (!index || decisions[source].used_exhaustive_fallback) {
            feed_span(nullptr, 0, targets.size());
        } else {
            for (std::size_t bucket = 0; bucket < keys.size(); ++bucket)
                if (compatible(decisions[source].source_key, keys[bucket], decisions[source].final_radius))
                    feed_span(index->d_target_ids, offsets[bucket], offsets[bucket + 1]);
        }
        const auto expected = index ? decisions[source].effective_pool_size : targets.size();
        if (scored != expected) throw std::runtime_error("Indexed scoring pool size disagrees with retrieval");
        for (auto value : workspace.finish_pool())
            result.candidates[source].push_back(Candidate{value.target, value.score});
        if (result.candidates[source].size() != static_cast<std::size_t>(k))
            throw std::runtime_error("Matching source has fewer than K candidates");
    }
    result.selection = workspace.info();
    result.transfer_ms += result.selection.transfer_ms;
    result.selection.candidate_stage_wall_ms = candidate_wall.elapsed_ms();
    return result;
}
} // namespace

GPUCandidateSelectionResult gpu_score_exhaustive_pools(
    const std::vector<NodeFeatures>& sources, const std::vector<NodeFeatures>& targets,
    int k, const GPUContext& context, GPUSelector selector, const GPUTopKOptions& options) {
    Timer wall("Complete GPU exhaustive scoring/selection call");
    auto result = score_pools(sources, targets, k, context, selector, options, nullptr, nullptr);
    result.selection.candidate_stage_wall_ms = wall.elapsed_ms();
    return result;
}
GPUCandidateSelectionResult gpu_score_indexed_pools(
    const std::vector<NodeFeatures>& sources, const std::vector<NodeFeatures>& targets,
    const GPUStructuralIndex& index, const GPUStructuralQueries& queries, int k,
    const GPUContext& context, GPUSelector selector, const GPUTopKOptions& options) {
    Timer wall("Complete GPU indexed scoring/selection call");
    auto result = score_pools(sources, targets, k, context, selector, options, &index, &queries);
    result.selection.candidate_stage_wall_ms = wall.elapsed_ms();
    return result;
}

// GPUIndexedMatchingResult gpu_compute_indexed_candidates(
//     const Graph& source, const Graph& target, int k, const StructuralIndexConfig& config,
//     const GPUContext& context, bool joint_normalize, GPUSelector selector,
//     const GPUTopKOptions& options) {
//     validate_structural_index_config(config);
//     if (source.num_vertices < 0 || target.num_vertices < 0 ||
//         (joint_normalize && checked_add(static_cast<std::size_t>(source.num_vertices),
//             static_cast<std::size_t>(target.num_vertices)) > static_cast<std::size_t>(std::numeric_limits<int>::max())))
//         throw std::overflow_error("Indexed graph dimensions exceed normalization range");
//     if (k <= 0 || (source.num_vertices > 0 && k > target.num_vertices))
//         throw std::invalid_argument("Indexed matching requires positive K and enough targets");
//     CUDA_CHECK(cudaSetDevice(context.device_id));
//     GPUIndexedMatchingResult result;
//     auto& matching = result.matching;
//     auto& timing = matching.timings;
//     matching.device_allocation_tracking_complete = false;
//     Timer feature_timer("Indexed GPU features");
//     if (source.num_vertices > 0)
//         matching.feats_source = gpu_compute_features(source, context, &timing.host_device_transfer_ms, false);
//     if (target.num_vertices > 0)
//         matching.feats_target = gpu_compute_features(target, context, &timing.host_device_transfer_ms, false);
//     const double raw_feature_wall = feature_timer.elapsed_ms();
//     const double raw_feature_transfer = timing.host_device_transfer_ms;

//     GPUStructuralIndex index;
//     GPUStructuralQueries queries;
//     {
//         DeviceArray<NodeFeatures> raw_sources(matching.feats_source.size());
//         DeviceArray<NodeFeatures> raw_targets(matching.feats_target.size());
//         timing.host_device_transfer_ms += copy_timed(raw_sources.p, matching.feats_source.data(),
//             checked_bytes(matching.feats_source.size(), sizeof(NodeFeatures)), cudaMemcpyHostToDevice, context.stream);
//         timing.host_device_transfer_ms += copy_timed(raw_targets.p, matching.feats_target.data(),
//             checked_bytes(matching.feats_target.size(), sizeof(NodeFeatures)), cudaMemcpyHostToDevice, context.stream);
//         Timer index_timer("GPU index build");
//         index = build_gpu_structural_index(raw_targets.p, matching.feats_target.size(), config, context);
//         result.index_build_ms = index_timer.elapsed_ms();
//         Timer query_timer("GPU retrieval");
//         queries = retrieve_gpu_structural_candidates(index, raw_sources.p, matching.feats_source.size(), k, context);
//         result.retrieval_ms = query_timer.elapsed_ms();
//     }
//     Timer normalization_timer("Normalization");
//     if (joint_normalize) normalize_features_joint_zscore(matching.feats_source, matching.feats_target);
//     else { normalize_features_zscore(matching.feats_source); normalize_features_zscore(matching.feats_target); }
//     timing.feature_extraction_ms = std::max(0.0, raw_feature_wall - raw_feature_transfer) +
//                                    normalization_timer.elapsed_ms();
//     auto scored = gpu_score_indexed_pools(matching.feats_source, matching.feats_target,
//         index, queries, k, context, selector, options);
//     matching.candidates = std::move(scored.candidates);
//     matching.selection = scored.selection;
//     matching.selection.candidate_stage_wall_ms += result.retrieval_ms;
//     timing.similarity_ms = scored.similarity_ms;
//     timing.topk_selection_ms = scored.selection.selection_ms;
//     timing.host_device_transfer_ms += scored.transfer_ms;
//     timing.candidate_generation_ms = result.retrieval_ms + timing.similarity_ms + timing.topk_selection_ms;

//     // Reporting downloads O(M + buckets + sources) diagnostics, never N*M pools.
//     Timer diagnostic_transfer("Diagnostic download");
//     result.index_diagnostics = download_structural_index(index);
//     const auto decisions = download_structural_queries(queries);
//     timing.host_device_transfer_ms += diagnostic_transfer.elapsed_ms();
//     for (const auto& decision : decisions) {
//         StructuralRetrievalStats stats;
//         stats.desired_pool_size = decision.desired_pool_size;
//         stats.bucket_pool_size = decision.bucket_pool_size;
//         stats.final_radius = decision.final_radius;
//         stats.used_exhaustive_fallback = decision.used_exhaustive_fallback;
//         const auto shells = static_cast<std::size_t>(static_cast<std::int64_t>(decision.final_radius) + 1);
//         stats.bucket_keys_examined = checked_bytes(shells, index.bucket_count);
//         for (const auto key : result.index_diagnostics.bucket_keys)
//             if (compatible(decision.source_key, key, decision.final_radius)) ++stats.occupied_buckets_found;
//         result.retrieval_stats.push_back(stats);
//     }
//     timing.total_ms = timing.feature_extraction_ms + result.index_build_ms +
//                       timing.candidate_generation_ms + timing.host_device_transfer_ms;
//     return result;
// }

void gpu_download_index_diagnostics(
    DeviceMatchingResult& result,
    StructuralIndex& host_index,
    std::vector<StructuralRetrievalStats>& retrieval,
    const GPUContext& context)
{
    if (!result.indexed) {
        throw std::invalid_argument(
            "Index diagnostics requested for exhaustive pipeline");
    }

    CUDA_CHECK(cudaSetDevice(context.device_id));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    Timer handoff("Index diagnostic CPU handoff");

    host_index = download_structural_index(result.structural_index);

    const auto decisions =
        download_structural_queries(result.structural_queries);

    retrieval.clear();
    retrieval.reserve(decisions.size());

    for (const auto& decision : decisions) {
        StructuralRetrievalStats stats;
        stats.desired_pool_size = decision.desired_pool_size;
        stats.bucket_pool_size = decision.bucket_pool_size;
        stats.final_radius = decision.final_radius;
        stats.used_exhaustive_fallback =
            decision.used_exhaustive_fallback;

        const auto shells = static_cast<std::size_t>(
            static_cast<std::int64_t>(decision.final_radius) + 1);

        stats.bucket_keys_examined = checked_bytes(
            shells, result.structural_index.bucket_count);

        for (const auto key : host_index.bucket_keys) {
            if (compatible(
                    decision.source_key, key, decision.final_radius)) {
                ++stats.occupied_buckets_found;
            }
        }

        retrieval.push_back(stats);
    }

    std::size_t bytes = checked_bytes(
        host_index.bucket_keys.size(), sizeof(StructuralBucketKey));

    bytes = checked_add(bytes, checked_bytes(
        host_index.bucket_offsets.size(), sizeof(std::size_t)));

    bytes = checked_add(bytes, checked_bytes(
        host_index.target_ids.size(), sizeof(int)));

    bytes = checked_add(bytes, checked_bytes(
        decisions.size(), sizeof(GPUStructuralQuery)));

    result.residency.known_d2h_bytes = checked_add(
        result.residency.known_d2h_bytes, bytes);

    const double elapsed = handoff.elapsed_ms();
    result.residency.cpu_handoff_ms += elapsed;
    result.timings.host_device_transfer_ms += elapsed;
}

GPUIndexedMatchingResult gpu_compute_indexed_candidates(
    const Graph& source,
    const Graph& target,
    int k,
    const StructuralIndexConfig& config,
    const GPUContext& context,
    bool joint_normalize,
    GPUSelector selector,
    const GPUTopKOptions& options)
{
    auto device = gpu_compute_device_candidates(
        source, target, k, context,
        joint_normalize, selector, options, &config);

    GPUIndexedMatchingResult host;

    // Existing return type requires these host diagnostics.
    gpu_download_index_diagnostics(
        device, host.index_diagnostics, host.retrieval_stats, context);

    host.index_build_ms = device.residency.index_build_ms;
    host.retrieval_ms = device.residency.retrieval_ms;
    host.matching = gpu_download_matching_result(device, context);

    return host;
}


namespace {

__device__ float device_scoring_feature(
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

__device__ float device_scoring_cosine(
    const NodeFeatures& source,
    const NodeFeatures& target)
{
    float dot = 0.0f;
    float source_norm = 0.0f;
    float target_norm = 0.0f;

    for (int f = 0; f < NUM_FEATURES; ++f) {
        const float a = device_scoring_feature(source, f);
        const float b = device_scoring_feature(target, f);

        dot += a * b;
        source_norm += a * a;
        target_norm += b * b;
    }

    const float denominator =
        sqrtf(source_norm) * sqrtf(target_norm);

    // Validate before clamping, which could otherwise conceal NaNs.
    if (!isfinite(dot) ||
        !isfinite(source_norm) ||
        !isfinite(target_norm) ||
        !isfinite(denominator)) {
        return CUDART_NAN_F;
    }

    if (denominator < 1e-12f) {
        return 0.0f;
    }

    const float score = dot / denominator;

    if (!isfinite(score)) {
        return CUDART_NAN_F;
    }

    return fminf(1.0f, fmaxf(-1.0f, score));
}

__device__ bool device_bucket_compatible(
    StructuralBucketKey source,
    StructuralBucketKey target,
    int radius)
{
    if (radius < 0) return false;

    // Index coordinates are nonnegative bins produced from log1p features.
    const auto dx = source.degree_bin >= target.degree_bin
        ? source.degree_bin - target.degree_bin
        : target.degree_bin - source.degree_bin;

    const auto dy = source.neighbor_bin >= target.neighbor_bin
        ? source.neighbor_bin - target.neighbor_bin
        : target.neighbor_bin - source.neighbor_bin;

    return dx <= radius && dy <= radius;
}

__global__ void validate_device_scoring_features(
    const NodeFeatures* features,
    std::size_t count,
    int* invalid)
{
    const std::size_t vertex =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (vertex >= count) return;

    for (int f = 0; f < NUM_FEATURES; ++f) {
        if (!isfinite(device_scoring_feature(features[vertex], f))) {
            atomicExch(invalid, 1);
        }
    }
}

// Validate retrieval metadata on GPU instead of downloading it for scheduling.
__global__ void validate_device_scoring_queries(
    const StructuralBucketKey* keys,
    const std::size_t* offsets,
    std::size_t bucket_count,
    const GPUStructuralQuery* queries,
    std::size_t source_count,
    std::size_t target_count,
    int k,
    int* invalid)
{
    const std::size_t source =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (source >= source_count) return;

    if (offsets[0] != 0 || offsets[bucket_count] != target_count) {
        atomicExch(invalid, 1);
        return;
    }

    const GPUStructuralQuery query = queries[source];
    std::size_t selected_count = 0;

    for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
        const std::size_t begin = offsets[bucket];
        const std::size_t end = offsets[bucket + 1];

        if (begin > end || end > target_count) {
            atomicExch(invalid, 1);
            return;
        }

        if (device_bucket_compatible(
                query.source_key, keys[bucket], query.final_radius)) {
            selected_count += end - begin;
        }
    }

    const std::size_t expected = query.used_exhaustive_fallback
        ? target_count
        : selected_count;

    if (query.effective_pool_size != expected ||
        expected < static_cast<std::size_t>(k)) {
        atomicExch(invalid, 1);
    }
}

/*
 * Exhaustive mode traverses target IDs directly.
 *
 * Indexed mode traverses the index's grouped target positions, scoring only
 * compatible buckets. Nonselected positions become internal padding.
 *
 * This avoids host metadata downloads and extra pool allocations.
 * It still examines all grouped positions; optimizing sparse scheduling
 * is a later performance change.
 */
__global__ void score_device_candidate_batch(
    const NodeFeatures* sources,
    const NodeFeatures* targets,
    std::size_t source,
    std::size_t target_count,
    const StructuralBucketKey* keys,
    const std::size_t* offsets,
    const int* grouped_targets,
    std::size_t bucket_count,
    const GPUStructuralQuery* queries,
    std::size_t start,
    std::size_t count,
    GPUScoredCandidate* output)
{
    const std::size_t lane =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (lane >= count) return;

    const std::size_t position = start + lane;

    if (queries != nullptr &&
        !queries[source].used_exhaustive_fallback) {
        // Find the occupied bucket containing this grouped target position.
        std::size_t low = 0;
        std::size_t high = bucket_count;

        while (low < high) {
            const std::size_t middle = low + (high - low) / 2;

            if (offsets[middle + 1] <= position) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }

        if (low == bucket_count ||
            !device_bucket_compatible(
                queries[source].source_key,
                keys[low],
                queries[source].final_radius)) {
            output[lane] = GPUScoredCandidate{0.0f, -1};
            return;
        }
    }

    const bool indexed_pool =
        queries != nullptr &&
        !queries[source].used_exhaustive_fallback;

    const int target = indexed_pool
        ? grouped_targets[position]
        : static_cast<int>(position);

    if (target < 0 ||
        static_cast<std::size_t>(target) >= target_count) {
        // consume() rejects this invalid ID before any feature dereference.
        output[lane] = GPUScoredCandidate{CUDART_NAN_F, -2};
        return;
    }

    output[lane] = GPUScoredCandidate{
        device_scoring_cosine(sources[source], targets[target]),
        target
    };
}

// finish_pool_to_device() initially stores one valid-count value per row.
// Convert those counts to compact fixed-K offsets after validating them.
__global__ void finalize_matching_candidate_offsets(
    std::size_t* offsets,
    std::size_t source_count,
    std::size_t k,
    int* invalid)
{
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (row < source_count) {
        if (offsets[row] != k) {
            atomicExch(invalid, 1);
        }

        offsets[row] = row * k;
    }

    if (row == 0) {
        offsets[source_count] = source_count * k;
    }
}

void validate_device_scoring_layout(
    const DeviceFeatureSet& features,
    const GPUContext& context)
{
    if (features.num_vertices < 0 ||
        features.normalized.size() !=
            static_cast<std::size_t>(features.num_vertices)) {
        throw std::invalid_argument(
            "Device scoring: normalized feature dimensions disagree");
    }

    if (!features.normalized.empty() &&
        features.normalized.device_id() != context.device_id) {
        throw std::invalid_argument(
            "Device scoring: normalized features belong to another GPU");
    }
}

GPUDeviceCandidateSelectionResult score_device_pools_impl(
    const DeviceFeatureSet& sources,
    const DeviceFeatureSet& targets,
    int k,
    const GPUContext& context,
    GPUSelector selector,
    const GPUTopKOptions& options,
    const GPUStructuralIndex* index,
    const GPUStructuralQueries* queries)
{
    validate_device_scoring_layout(sources, context);
    validate_device_scoring_layout(targets, context);

    if (k <= 0 ||
        (sources.num_vertices > 0 && k > targets.num_vertices)) {
        throw std::invalid_argument(
            "Device matching requires positive K and enough targets");
    }

    if ((index == nullptr) != (queries == nullptr)) {
        throw std::invalid_argument(
            "Device scoring requires both index and queries");
    }

    const std::size_t source_count = sources.normalized.size();
    const std::size_t target_count = targets.normalized.size();

    if (index != nullptr) {
        if (index->device_id != context.device_id ||
            queries->device_id != context.device_id ||
            index->target_count != target_count ||
            queries->source_count != source_count ||
            index->d_bucket_offsets == nullptr ||
            (index->bucket_count != 0 &&
             index->d_bucket_keys == nullptr) ||
            (target_count != 0 &&
             index->d_target_ids == nullptr) ||
            (source_count != 0 &&
             queries->d_queries == nullptr)) {
            throw std::invalid_argument(
                "Device indexed scoring dimensions or allocations disagree");
        }
    }

    // checked_bytes also provides checked multiplication here.
    const std::size_t entry_count = checked_bytes(
        source_count, static_cast<std::size_t>(k));

    if (entry_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Device candidates exceed host cost CSR int capacity");
    }

    // Reserve the validation flag within the same scoring/selection budget.
    if (options.scratch_budget_bytes <= sizeof(int)) {
        throw std::invalid_argument(
            "Device scoring scratch budget is too small");
    }

    GPUTopKOptions workspace_options = options;
    workspace_options.scratch_budget_bytes -= sizeof(int);

    CUDA_CHECK(cudaSetDevice(context.device_id));

    GPUDeviceCandidateSelectionResult result;
    result.resident_feature_bytes = checked_add(
        sources.normalized.bytes(), targets.normalized.bytes());

    auto& candidates = result.candidates;
    candidates.num_sources = sources.num_vertices;
    candidates.num_targets = targets.num_vertices;
    candidates.requested_k = k;
    candidates.entry_count = entry_count;

    candidates.row_offsets = DeviceBuffer<std::size_t>(
        checked_add(source_count, 1), context.device_id);
    candidates.target_ids = DeviceBuffer<int>(
        entry_count, context.device_id);
    candidates.similarities = DeviceBuffer<float>(
        entry_count, context.device_id);

    DeviceBuffer<int> invalid(1, context.device_id);

    GPUStreamingTopK workspace(
        targets.num_vertices, k, context, selector, workspace_options);

    Events score_events;

    auto check_validation = [&]() {
        int host_invalid = 0;
        Timer transfer("Device scoring validation transfer");

        invalid.download(&host_invalid, 1, context.stream);
        result.transfer_ms += transfer.elapsed_ms();

        if (host_invalid != 0) {
            throw std::invalid_argument(
                "Device scoring: invalid features, retrieval metadata, "
                "or insufficient selected candidates");
        }
    };

    try {
        CUDA_CHECK(cudaMemsetAsync(
            invalid.data(), 0, sizeof(int), context.stream));

        auto validate_features_on_device =
            [&](const DeviceFeatureSet& features) {
                const std::size_t count = features.normalized.size();
                if (count == 0) return;

                const unsigned int blocks =
                    static_cast<unsigned int>(1 + (count - 1) / 256);

                validate_device_scoring_features
                    <<<blocks, 256, 0, context.stream>>>(
                        features.normalized.data(),
                        count,
                        invalid.data());

                CUDA_CHECK(cudaGetLastError());
            };

        validate_features_on_device(sources);
        validate_features_on_device(targets);

        if (index != nullptr && source_count != 0) {
            const unsigned int blocks =
                static_cast<unsigned int>(1 + (source_count - 1) / 256);

            validate_device_scoring_queries
                <<<blocks, 256, 0, context.stream>>>(
                    index->d_bucket_keys,
                    index->d_bucket_offsets,
                    index->bucket_count,
                    queries->d_queries,
                    source_count,
                    target_count,
                    k,
                    invalid.data());

            CUDA_CHECK(cudaGetLastError());
        }

        check_validation();

        for (std::size_t source = 0; source < source_count; ++source) {
            workspace.begin_pool();

            for (std::size_t start = 0; start < target_count;) {
                const std::size_t count =
                    std::min(workspace.capacity(), target_count - start);

                const unsigned int blocks =
                    static_cast<unsigned int>(1 + (count - 1) / 256);

                score_events.start(context.stream);

                score_device_candidate_batch
                    <<<blocks, 256, 0, context.stream>>>(
                        sources.normalized.data(),
                        targets.normalized.data(),
                        source,
                        target_count,
                        index != nullptr ? index->d_bucket_keys : nullptr,
                        index != nullptr ? index->d_bucket_offsets : nullptr,
                        index != nullptr ? index->d_target_ids : nullptr,
                        index != nullptr ? index->bucket_count : 0,
                        queries != nullptr ? queries->d_queries : nullptr,
                        start,
                        count,
                        workspace.input());

                CUDA_CHECK(cudaGetLastError());

                result.similarity_ms +=
                    score_events.finish(context.stream);

                workspace.consume(count);
                start += count;
            }

            const std::size_t destination =
                source * static_cast<std::size_t>(k);

            workspace.finish_pool_to_device(
                candidates.target_ids.data() + destination,
                candidates.similarities.data() + destination,
                static_cast<std::size_t>(k),
                candidates.row_offsets.data() + source);
        }

        const unsigned int blocks = source_count != 0
            ? static_cast<unsigned int>(1 + (source_count - 1) / 256)
            : 1u;

        finalize_matching_candidate_offsets
            <<<blocks, 256, 0, context.stream>>>(
                candidates.row_offsets.data(),
                source_count,
                static_cast<std::size_t>(k),
                invalid.data());

        CUDA_CHECK(cudaGetLastError());
        check_validation();

        result.selection = workspace.info();
        result.selection.scratch_peak_bytes = checked_add(
            result.selection.scratch_peak_bytes, sizeof(int));

        result.transfer_ms += result.selection.transfer_ms;
    } catch (...) {
        (void)cudaStreamSynchronize(context.stream);
        throw;
    }

    return result;
}

} // namespace

GPUDeviceCandidateSelectionResult gpu_score_exhaustive_device_pools(
    const DeviceFeatureSet& sources,
    const DeviceFeatureSet& targets,
    int k,
    const GPUContext& context,
    GPUSelector selector,
    const GPUTopKOptions& options)
{
    Timer wall("Device exhaustive scoring and selection");

    auto result = score_device_pools_impl(
        sources, targets, k, context, selector, options,
        nullptr, nullptr);

    result.selection.candidate_stage_wall_ms = wall.elapsed_ms();
    return result;
}

GPUDeviceCandidateSelectionResult gpu_score_indexed_device_pools(
    const DeviceFeatureSet& sources,
    const DeviceFeatureSet& targets,
    const GPUStructuralIndex& index,
    const GPUStructuralQueries& queries,
    int k,
    const GPUContext& context,
    GPUSelector selector,
    const GPUTopKOptions& options)
{
    Timer wall("Device indexed scoring and selection");

    auto result = score_device_pools_impl(
        sources, targets, k, context, selector, options,
        &index, &queries);

    result.selection.candidate_stage_wall_ms = wall.elapsed_ms();
    return result;
}