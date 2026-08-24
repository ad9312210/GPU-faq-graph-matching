#ifndef SCGM_CUDA_KERNELS_CUH
#define SCGM_CUDA_KERNELS_CUH

#include <cuda_runtime.h>

/*
 * Device-side cosine similarity for six-dimensional feature vectors.
 *
 *   S(a,b) = dot(a,b) / (||a|| * ||b||)
 *
 * Returns 0 if either norm is near zero.
 */

__device__ inline float cosine_similarity_device(
    const float* a, const float* b, int dim)
{
    float dot_val = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int d = 0; d < dim; d++) {
        dot_val += a[d] * b[d];
        norm_a += a[d] * a[d];
        norm_b += b[d] * b[d];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-12f) return 0.0f;
    return dot_val / denom;
}

/*
 * Kernel: compute features for each node of a graph.
 *
 * The graph is passed via CSR adjacency: d_row_ptr, d_col_idx, num_vertices.
 * Output: d_features[v * 6 + f] for v in [0, num_vertices), f in [0, 6).
 *
 * Features:
 *   [0] degree
 *   [1] clustering coefficient
 *   [2] average neighbour degree
 *   [3] neighbour degree std
 *   [4] triangle count
 *   [5] degree squared
 */
__global__ void kernel_compute_features(
    const int* d_row_ptr,
    const int* d_col_idx,
    int num_vertices,
    float* d_features);

/*
 * Kernel: z-score normalize features in-place.
 *
 * d_features: [num_vertices * 6], row-major.
 * d_mean, d_std: precomputed [6] arrays.
 */
__global__ void kernel_normalize_features(
    float* d_features,
    const float* d_mean,
    const float* d_std,
    int num_vertices,
    int dim);

/*
 * Kernel: for each source node, compute cosine similarity to all target
 * nodes and write top-K candidates.
 *
 * d_feats_src: [N * 6]
 * d_feats_tgt: [M * 6]
 * d_candidate_target: [N * K]   output target indices
 * d_candidate_score:  [N * K]   output similarity scores
 *
 * This does NOT create a dense N*M matrix; each thread block handles
 * one source node, iterates over M targets, and maintains a local top-K
 * heap in shared memory or registers.
 */
__global__ void kernel_topk_candidates(
    const float* d_feats_src,
    int N,
    const float* d_feats_tgt,
    int M,
    int K,
    int dim,
    int* d_candidate_target,
    float* d_candidate_score);

/*
 * Kernel: compute mean and variance for feature normalization.
 */
__global__ void kernel_feature_stats(
    const float* d_features,
    int num_vertices,
    int dim,
    float* d_mean,
    float* d_var);

#endif // SCGM_CUDA_KERNELS_CUH