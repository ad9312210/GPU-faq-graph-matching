#include "gpu_matching.hpp"
#include "cuda_kernels.cuh"
#include "cosine_similarity.hpp"


namespace {
size_t active_device_allocation_bytes = 0;
size_t peak_device_application_allocation_bytes = 0;

template <typename T>
void tracked_cuda_malloc(T** pointer, size_t bytes) {
    CUDA_CHECK(cudaMalloc(pointer, bytes));
    active_device_allocation_bytes += bytes;
    peak_device_application_allocation_bytes = std::max(
        peak_device_application_allocation_bytes, active_device_allocation_bytes);
}

template <typename T>
void tracked_cuda_free(T* pointer, size_t bytes) {
    if (!pointer) return;
    CUDA_CHECK(cudaFree(pointer));
    active_device_allocation_bytes -= bytes;
}
}

/*
 * Feature computation kernel.
 * One thread per vertex.
 */
__global__ void kernel_compute_features(
    const int* d_row_ptr,
    const int* d_col_idx,
    int num_vertices,
    float* d_features)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = d_row_ptr[v];
    int end = d_row_ptr[v + 1];
    int deg = end - start;

    // Feature 0: degree
    d_features[v * 6 + 0] = (float)deg;

    // Feature 5: degree squared
    d_features[v * 6 + 5] = (float)(deg * deg);

    // Precompute neighbor degrees for features 2,3
    // We need to read neighbor degrees from d_row_ptr
    float sum_nbr_deg = 0.0f;
    for (int idx = start; idx < end; idx++) {
        int nb = d_col_idx[idx];
        int nb_deg = d_row_ptr[nb + 1] - d_row_ptr[nb];
        sum_nbr_deg += (float)nb_deg;
    }

    // Feature 2: average neighbor degree
    float avg_nbr_deg = (deg > 0) ? (sum_nbr_deg / (float)deg) : 0.0f;
    d_features[v * 6 + 2] = avg_nbr_deg;

    // Feature 3: neighbor degree std
    float var_sum = 0.0f;
    for (int idx = start; idx < end; idx++) {
        int nb = d_col_idx[idx];
        int nb_deg = d_row_ptr[nb + 1] - d_row_ptr[nb];
        float diff = (float)nb_deg - avg_nbr_deg;
        var_sum += diff * diff;
    }
    float nbr_std = (deg > 1) ? sqrtf(var_sum / (float)(deg - 1)) : 0.0f;
    d_features[v * 6 + 3] = nbr_std;

    // Feature 4: triangle count
    // Count triangles: for each pair of neighbors, check if they share an edge
    int tri_count = 0;
    for (int i_idx = start; i_idx < end; i_idx++) {
        int ni = d_col_idx[i_idx];
        for (int j_idx = i_idx + 1; j_idx < end; j_idx++) {
            int nj = d_col_idx[j_idx];
            // Check if ni and nj are connected: binary search in ni's adjacency
            int ni_start = d_row_ptr[ni];
            int ni_end = d_row_ptr[ni + 1];
            // Linear scan (col_idx is sorted)
            bool found = false;
            int lo = ni_start, hi = ni_end - 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (d_col_idx[mid] == nj) { found = true; break; }
                else if (d_col_idx[mid] < nj) lo = mid + 1;
                else hi = mid - 1;
            }
            if (found) tri_count++;
        }
    }
    d_features[v * 6 + 4] = (float)tri_count;

    // Feature 1: clustering coefficient
    float clustering = 0.0f;
    if (deg >= 2) {
        float max_tri = (float)deg * (float)(deg - 1) / 2.0f;
        clustering = (float)tri_count / max_tri;
    }
    d_features[v * 6 + 1] = clustering;
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
// separate merge kernel updates an O(N * K) min-heap. No N*M matrix is formed.
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
            d_feats_src + src * dim, d_feats_tgt + tgt * dim, dim);
    }
    d_tile_scores[src * TOPK_TILE_WIDTH + lane] = score;
}

__device__ inline bool candidate_is_worse(float score_a, int target_a,
                                          float score_b, int target_b) {
    return score_a < score_b || (score_a == score_b && target_a > target_b);
}

__device__ inline void sift_down_min_heap(float* scores, int* targets,
                                         int root, int heap_size, int base) {
    while (true) {
        int left = root * 2 + 1;
        if (left >= heap_size) return;
        int worse_child = left;
        int right = left + 1;
        if (right < heap_size && candidate_is_worse(
                scores[base + right], targets[base + right],
                scores[base + left], targets[base + left])) {
            worse_child = right;
        }
        if (!candidate_is_worse(scores[base + worse_child], targets[base + worse_child],
                                scores[base + root], targets[base + root])) return;
        float score = scores[base + root];
        scores[base + root] = scores[base + worse_child];
        scores[base + worse_child] = score;
        int target = targets[base + root];
        targets[base + root] = targets[base + worse_child];
        targets[base + worse_child] = target;
        root = worse_child;
    }
}

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
        const float score = d_tile_scores[src * TOPK_TILE_WIDTH + lane];
        if (heap_size < K) {
            int position = heap_size++;
            d_candidate_target[base + position] = target;
            d_candidate_score[base + position] = score;
            while (position > 0) {
                int parent = (position - 1) / 2;
                if (!candidate_is_worse(d_candidate_score[base + position],
                                        d_candidate_target[base + position],
                                        d_candidate_score[base + parent],
                                        d_candidate_target[base + parent])) break;
                float old_score = d_candidate_score[base + parent];
                d_candidate_score[base + parent] = d_candidate_score[base + position];
                d_candidate_score[base + position] = old_score;
                int old_target = d_candidate_target[base + parent];
                d_candidate_target[base + parent] = d_candidate_target[base + position];
                d_candidate_target[base + position] = old_target;
                position = parent;
            }
        } else if (score > d_candidate_score[base] ||
                   (score == d_candidate_score[base] && target < d_candidate_target[base])) {
            d_candidate_target[base] = target;
            d_candidate_score[base] = score;
            sift_down_min_heap(d_candidate_score, d_candidate_target, 0, K, base);
        }
    }

    if (finalize) {
        // Heap sort using the same total ordering; output is best-first and
        // ties are stable by target ID.
        for (int end = K - 1; end > 0; --end) {
            float score = d_candidate_score[base];
            d_candidate_score[base] = d_candidate_score[base + end];
            d_candidate_score[base + end] = score;
            int target = d_candidate_target[base];
            d_candidate_target[base] = d_candidate_target[base + end];
            d_candidate_target[base + end] = target;
            sift_down_min_heap(d_candidate_score, d_candidate_target, 0, end, base);
        }
    }
}

// Helper: GPU feature computation
std::vector<NodeFeatures> gpu_compute_features(
    const Graph& g,
    const GPUContext& ctx,
    double* transfer_ms,
    bool normalize,
    size_t* allocated_bytes)
{
    CUDA_CHECK(cudaSetDevice(ctx.device_id));

    int N = g.num_vertices;
    int nnz_adj = (int)g.col_idx.size();
    if (allocated_bytes) {
        *allocated_bytes = static_cast<size_t>(N + 1) * sizeof(int) +
            static_cast<size_t>(std::max(nnz_adj, 1)) * sizeof(int) +
            static_cast<size_t>(N) * NUM_FEATURES * sizeof(float) +
            (normalize ? 3 * NUM_FEATURES * sizeof(float) : 0);
    }

    // Allocate and copy graph
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_features = nullptr;

    tracked_cuda_malloc(&d_row_ptr, (N + 1) * sizeof(int));
    tracked_cuda_malloc(&d_col_idx, std::max(nnz_adj, 1) * sizeof(int));
    tracked_cuda_malloc(&d_features, N * 6 * sizeof(float));

    auto transfer_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (nnz_adj > 0) {
        CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), nnz_adj * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    auto transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_total_ms = std::chrono::duration<double, std::milli>(
        transfer_end - transfer_start).count();

    // Launch feature computation kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    kernel_compute_features<<<grid_size, block_size, 0, ctx.stream>>>(
        d_row_ptr, d_col_idx, N, d_features);
    CUDA_CHECK(cudaGetLastError());

    float* d_mean = nullptr;
    float* d_var = nullptr;
    float* d_std = nullptr;
    if (normalize) {
        tracked_cuda_malloc(&d_mean, 6 * sizeof(float));
        tracked_cuda_malloc(&d_var, 6 * sizeof(float));
        tracked_cuda_malloc(&d_std, 6 * sizeof(float));

        kernel_feature_stats<<<1, 6, 0, ctx.stream>>>(d_features, N, 6, d_mean, d_var);
        CUDA_CHECK(cudaGetLastError());

        // Compute std from var on host (small array)
        float h_var[6], h_std[6];
        transfer_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(h_var, d_var, 6 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int f = 0; f < 6; f++) {
            h_std[f] = std::sqrt(h_var[f]);
        }
        CUDA_CHECK(cudaMemcpy(d_std, h_std, 6 * sizeof(float), cudaMemcpyHostToDevice));
        transfer_end = std::chrono::high_resolution_clock::now();
        transfer_total_ms += std::chrono::duration<double, std::milli>(
            transfer_end - transfer_start).count();

        int total_elems = N * 6;
        int norm_grid = (total_elems + block_size - 1) / block_size;
        kernel_normalize_features<<<norm_grid, block_size, 0, ctx.stream>>>(
            d_features, d_mean, d_std, N, 6);
        CUDA_CHECK(cudaGetLastError());

    }

    // Copy back
    std::vector<float> h_features(N * 6);
    transfer_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_features.data(), d_features, N * 6 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    transfer_end = std::chrono::high_resolution_clock::now();
    transfer_total_ms += std::chrono::duration<double, std::milli>(
        transfer_end - transfer_start).count();

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    if (transfer_ms) *transfer_ms += transfer_total_ms;

    // Convert to NodeFeatures
    std::vector<NodeFeatures> feats;
    flat_array_to_features(h_features, N, feats);

    // Cleanup
    tracked_cuda_free(d_row_ptr, (N + 1) * sizeof(int));
    tracked_cuda_free(d_col_idx, std::max(nnz_adj, 1) * sizeof(int));
    tracked_cuda_free(d_features, N * 6 * sizeof(float));
    if (d_mean) tracked_cuda_free(d_mean, 6 * sizeof(float));
    if (d_var) tracked_cuda_free(d_var, 6 * sizeof(float));
    if (d_std) tracked_cuda_free(d_std, 6 * sizeof(float));

    return feats;
}

std::vector<std::vector<Candidate>> gpu_compute_topk(
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

    // Flatten features
    std::vector<float> flat_src, flat_tgt;
    features_to_flat_array(feats_source, flat_src);
    features_to_flat_array(feats_target, flat_tgt);

    if (allocated_bytes) {
        *allocated_bytes =
            static_cast<size_t>(N + M) * NUM_FEATURES * sizeof(float) +
            static_cast<size_t>(N) * K * (sizeof(int) + sizeof(float)) +
            static_cast<size_t>(N) * TOPK_TILE_WIDTH * sizeof(float);
    }

    // Allocate GPU memory
    float* d_feats_src = nullptr;
    float* d_feats_tgt = nullptr;
    int* d_candidate_target = nullptr;
    float* d_candidate_score = nullptr;
    float* d_tile_scores = nullptr;

    tracked_cuda_malloc(&d_feats_src, N * NUM_FEATURES * sizeof(float));
    tracked_cuda_malloc(&d_feats_tgt, M * NUM_FEATURES * sizeof(float));
    tracked_cuda_malloc(&d_candidate_target, N * K * sizeof(int));
    tracked_cuda_malloc(&d_candidate_score, N * K * sizeof(float));
    tracked_cuda_malloc(&d_tile_scores, N * TOPK_TILE_WIDTH * sizeof(float));

    auto transfer_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_feats_src, flat_src.data(),
                          N * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_feats_tgt, flat_tgt.data(),
                          M * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice));
    auto transfer_end = std::chrono::high_resolution_clock::now();
    double transfer_total_ms = std::chrono::duration<double, std::milli>(
        transfer_end - transfer_start).count();

    const int tile_count = (M + TOPK_TILE_WIDTH - 1) / TOPK_TILE_WIDTH;
    struct TileEvents { cudaEvent_t sim_begin, sim_end, topk_begin, topk_end; };
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
            d_feats_src, d_feats_tgt, M, tile_start, count, NUM_FEATURES, d_tile_scores);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(events.sim_end, ctx.stream));
        CUDA_CHECK(cudaEventRecord(events.topk_begin, ctx.stream));
        kernel_merge_tile_topk<<<N, 1, 0, ctx.stream>>>(
            d_tile_scores, tile_start, count, K, tile == tile_count - 1,
            d_candidate_target, d_candidate_score);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(events.topk_end, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    double sim_elapsed = 0.0, topk_elapsed = 0.0;
    for (auto& events : tile_events) {
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.sim_begin, events.sim_end));
        sim_elapsed += elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.topk_begin, events.topk_end));
        topk_elapsed += elapsed;
        CUDA_CHECK(cudaEventDestroy(events.sim_begin));
        CUDA_CHECK(cudaEventDestroy(events.sim_end));
        CUDA_CHECK(cudaEventDestroy(events.topk_begin));
        CUDA_CHECK(cudaEventDestroy(events.topk_end));
    }
    if (similarity_kernel_ms) *similarity_kernel_ms = sim_elapsed;
    if (topk_kernel_ms) *topk_kernel_ms = topk_elapsed;

    // Copy back
    std::vector<int> h_targets(N * K);
    std::vector<float> h_scores(N * K);
    transfer_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_targets.data(), d_candidate_target,
                          N * K * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_candidate_score,
                          N * K * sizeof(float), cudaMemcpyDeviceToHost));
    transfer_end = std::chrono::high_resolution_clock::now();
    transfer_total_ms += std::chrono::duration<double, std::milli>(
        transfer_end - transfer_start).count();
    if (transfer_ms) *transfer_ms += transfer_total_ms;

    // Build candidates
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

    // Cleanup
    tracked_cuda_free(d_feats_src, N * NUM_FEATURES * sizeof(float));
    tracked_cuda_free(d_feats_tgt, M * NUM_FEATURES * sizeof(float));
    tracked_cuda_free(d_candidate_target, N * K * sizeof(int));
    tracked_cuda_free(d_candidate_score, N * K * sizeof(float));
    tracked_cuda_free(d_tile_scores, N * TOPK_TILE_WIDTH * sizeof(float));

    return candidates;
}

GPUMatchingResult gpu_compute_candidates(
    const Graph& source,
    const Graph& target,
    int K,
    const GPUContext& ctx,
    bool joint_normalize)
{
    GPUMatchingResult result;

    CUDA_CHECK(cudaSetDevice(ctx.device_id));
    active_device_allocation_bytes = 0;
    peak_device_application_allocation_bytes = 0;

    // Step 1: Feature extraction on GPU
    const double transfer_before_features = result.timings.host_device_transfer_ms;
    Timer t_feat("GPU feature extraction");
    size_t feature_alloc_bytes = 0;
    size_t target_feature_alloc_bytes = 0;
    size_t topk_alloc_bytes = 0;
    result.feats_source = gpu_compute_features(
        source, ctx, &result.timings.host_device_transfer_ms, false, &feature_alloc_bytes);
    result.feats_target = gpu_compute_features(
        target, ctx, &result.timings.host_device_transfer_ms, false, &target_feature_alloc_bytes);
    if (joint_normalize) {
        normalize_features_joint_zscore(result.feats_source, result.feats_target);
    } else {
        normalize_features_zscore(result.feats_source);
        normalize_features_zscore(result.feats_target);
    }
    const double feature_stage_wall_ms = t_feat.elapsed_ms();
    const double feature_transfer_ms = result.timings.host_device_transfer_ms -
                                      transfer_before_features;
    result.timings.feature_extraction_ms = std::max(
        0.0, feature_stage_wall_ms - feature_transfer_ms);

    // Step 2: Tiled similarity and Top-K kernels run as separate stages.
    result.candidates = gpu_compute_topk(
        result.feats_source, result.feats_target, K, ctx,
        &result.timings.host_device_transfer_ms, &topk_alloc_bytes,
        &result.timings.similarity_ms, &result.timings.topk_selection_ms);
    result.timings.candidate_generation_ms = result.timings.similarity_ms +
                                             result.timings.topk_selection_ms;
    result.timings.peak_device_application_allocated_bytes =
        peak_device_application_allocation_bytes;

    result.timings.total_ms = result.timings.feature_extraction_ms +
                              result.timings.candidate_generation_ms +
                              result.timings.host_device_transfer_ms;

    return result;
}

size_t gpu_feature_memory_bytes(int num_vertices) {
    return num_vertices * NUM_FEATURES * sizeof(float);
}

size_t gpu_candidate_memory_bytes(int N, int K) {
    return N * K * (sizeof(int) + sizeof(float));
}