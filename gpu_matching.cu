#include "gpu_matching.hpp"
#include "cuda_kernels.cuh"
#include "cosine_similarity.hpp"

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
    float nbr_std = (deg > 0) ? sqrtf(var_sum / (float)deg) : 0.0f;
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

/*
 * Top-K candidate selection kernel.
 *
 * Each block handles one source node.
 * Threads iterate over target nodes, maintaining a local min-heap of K
 * candidates (best = highest similarity).
 *
 * This does NOT allocate a dense N*M matrix.
 */
__global__ void kernel_topk_candidates(
    const float* d_feats_src,
    int N,
    const float* d_feats_tgt,
    int M,
    int K,
    int dim,
    int* d_candidate_target,
    float* d_candidate_score)
{
    int src = blockIdx.x;
    if (src >= N) return;

    // Use thread 0 to do the work (simple but correct approach)
    // For production: can parallelize across threads with reduction
    if (threadIdx.x != 0) return;

    const float* feat_src = d_feats_src + src * dim;

    // We maintain a min-heap (sorted array) of K best candidates
    // Simple approach: maintain sorted array of K elements
    // For small K this is efficient enough

    // Initialize candidates with -INF similarity
    int base = src * K;
    for (int k = 0; k < K; k++) {
        d_candidate_target[base + k] = -1;
        d_candidate_score[base + k] = -1e30f;
    }

    // Track the minimum similarity in our top-K set
    float min_score = -1e30f;
    int min_pos = 0;

    for (int tgt = 0; tgt < M; tgt++) {
        const float* feat_tgt = d_feats_tgt + tgt * dim;
        float sim = cosine_similarity_device(feat_src, feat_tgt, dim);

        if (sim > min_score) {
            // Replace the minimum element
            d_candidate_target[base + min_pos] = tgt;
            d_candidate_score[base + min_pos] = sim;

            // Find new minimum
            min_score = d_candidate_score[base + 0];
            min_pos = 0;
            for (int k = 1; k < K; k++) {
                if (d_candidate_score[base + k] < min_score) {
                    min_score = d_candidate_score[base + k];
                    min_pos = k;
                }
            }
        }
    }

    // Sort candidates by similarity descending for consistency
    for (int a = 0; a < K - 1; a++) {
        for (int b = a + 1; b < K; b++) {
            if (d_candidate_score[base + b] > d_candidate_score[base + a]) {
                float tmp_s = d_candidate_score[base + a];
                d_candidate_score[base + a] = d_candidate_score[base + b];
                d_candidate_score[base + b] = tmp_s;
                int tmp_t = d_candidate_target[base + a];
                d_candidate_target[base + a] = d_candidate_target[base + b];
                d_candidate_target[base + b] = tmp_t;
            }
        }
    }
}

// Helper: GPU feature computation
std::vector<NodeFeatures> gpu_compute_features(
    const Graph& g,
    const GPUContext& ctx)
{
    CUDA_CHECK(cudaSetDevice(ctx.device_id));

    int N = g.num_vertices;
    int nnz_adj = (int)g.col_idx.size();

    // Allocate and copy graph
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_features = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, std::max(nnz_adj, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_features, N * 6 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (N + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (nnz_adj > 0) {
        CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), nnz_adj * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // Launch feature computation kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    kernel_compute_features<<<grid_size, block_size, 0, ctx.stream>>>(
        d_row_ptr, d_col_idx, N, d_features);
    CUDA_CHECK(cudaGetLastError());

    // Normalize features on GPU
    float* d_mean = nullptr;
    float* d_var = nullptr;
    float* d_std = nullptr;

    CUDA_CHECK(cudaMalloc(&d_mean, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std, 6 * sizeof(float)));

    kernel_feature_stats<<<1, 6, 0, ctx.stream>>>(d_features, N, 6, d_mean, d_var);
    CUDA_CHECK(cudaGetLastError());

    // Compute std from var on host (small array)
    float h_var[6], h_std[6];
    CUDA_CHECK(cudaMemcpy(h_var, d_var, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int f = 0; f < 6; f++) {
        h_std[f] = std::sqrt(h_var[f]);
    }
    CUDA_CHECK(cudaMemcpy(d_std, h_std, 6 * sizeof(float), cudaMemcpyHostToDevice));

    int total_elems = N * 6;
    int norm_grid = (total_elems + block_size - 1) / block_size;
    kernel_normalize_features<<<norm_grid, block_size, 0, ctx.stream>>>(
        d_features, d_mean, d_std, N, 6);
    CUDA_CHECK(cudaGetLastError());

    // Copy back
    std::vector<float> h_features(N * 6);
    CUDA_CHECK(cudaMemcpy(h_features.data(), d_features, N * 6 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Convert to NodeFeatures
    std::vector<NodeFeatures> feats;
    flat_array_to_features(h_features, N, feats);

    // Cleanup
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_features));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_var));
    CUDA_CHECK(cudaFree(d_std));

    return feats;
}

std::vector<std::vector<Candidate>> gpu_compute_topk(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K,
    const GPUContext& ctx)
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

    // Allocate GPU memory
    float* d_feats_src = nullptr;
    float* d_feats_tgt = nullptr;
    int* d_candidate_target = nullptr;
    float* d_candidate_score = nullptr;

    CUDA_CHECK(cudaMalloc(&d_feats_src, N * NUM_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_feats_tgt, M * NUM_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candidate_target, N * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_score, N * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_feats_src, flat_src.data(),
                          N * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_feats_tgt, flat_tgt.data(),
                          M * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice));

    // Launch top-K kernel: one block per source node, one thread per block
    // (simple but correct)
    kernel_topk_candidates<<<N, 1, 0, ctx.stream>>>(
        d_feats_src, N, d_feats_tgt, M, K, NUM_FEATURES,
        d_candidate_target, d_candidate_score);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // Copy back
    std::vector<int> h_targets(N * K);
    std::vector<float> h_scores(N * K);
    CUDA_CHECK(cudaMemcpy(h_targets.data(), d_candidate_target,
                          N * K * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_candidate_score,
                          N * K * sizeof(float), cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaFree(d_feats_src));
    CUDA_CHECK(cudaFree(d_feats_tgt));
    CUDA_CHECK(cudaFree(d_candidate_target));
    CUDA_CHECK(cudaFree(d_candidate_score));

    return candidates;
}

GPUMatchingResult gpu_compute_candidates(
    const Graph& source,
    const Graph& target,
    int K,
    const GPUContext& ctx)
{
    GPUMatchingResult result;

    CUDA_CHECK(cudaSetDevice(ctx.device_id));

    // Step 1: Feature extraction on GPU
    Timer t_feat("GPU feature extraction");
    result.feats_source = gpu_compute_features(source, ctx);
    result.feats_target = gpu_compute_features(target, ctx);
    result.timings.feature_extraction_ms = t_feat.elapsed_ms();

    // Step 2: Top-K candidate generation on GPU
    Timer t_cand("GPU candidate generation");
    result.candidates = gpu_compute_topk(
        result.feats_source, result.feats_target, K, ctx);
    result.timings.candidate_generation_ms = t_cand.elapsed_ms();

    result.timings.total_ms = result.timings.feature_extraction_ms +
                              result.timings.candidate_generation_ms;

    return result;
}

size_t gpu_feature_memory_bytes(int num_vertices) {
    return num_vertices * NUM_FEATURES * sizeof(float);
}

size_t gpu_candidate_memory_bytes(int N, int K) {
    return N * K * (sizeof(int) + sizeof(float));
}