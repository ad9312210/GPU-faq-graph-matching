#include "cosine_similarity.hpp"

float cosine_similarity(const NodeFeatures& a, const NodeFeatures& b,
                        unsigned int feature_mask) {
    return cosine_similarity(a, b, feature_mask, {});
}

float cosine_similarity(const NodeFeatures& a, const NodeFeatures& b,
                        unsigned int feature_mask,
                        const std::vector<float>& feature_weights) {
    float dot_val = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;

    for (int f = 0; f < NUM_FEATURES; f++) {
        if ((feature_mask & (1u << f)) == 0) continue;
        float weight = feature_weights.empty() ? 1.0f : feature_weights[f];
        if (weight < 0.0f) {
            throw std::invalid_argument("feature weights must be non-negative");
        }
        float va = a.as_array(f);
        float vb = b.as_array(f);
        dot_val += weight * va * vb;
        norm_a_sq += weight * va * va;
        norm_b_sq += weight * vb * vb;
    }

    float denom = std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq);
    if (denom < SCGM_EPS) return 0.0f;
    // Floating-point roundoff can put an identical-vector score slightly
    // above 1, which would create a negative 1 - cosine assignment cost.
    return std::clamp(dot_val / denom, -1.0f, 1.0f);
}

float cosine_cost(const NodeFeatures& a, const NodeFeatures& b) {
    return 1.0f - cosine_similarity(a, b);
}

void compute_all_pairwise_similarities_cpu(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    std::vector<float>& sim_flat)
{
    int N = (int)feats_source.size();
    int M = (int)feats_target.size();
    sim_flat.resize(N * M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            sim_flat[i * M + j] = cosine_similarity(feats_source[i], feats_target[j]);
        }
    }
}