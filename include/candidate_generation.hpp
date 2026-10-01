#ifndef SCGM_CANDIDATE_GENERATION_HPP
#define SCGM_CANDIDATE_GENERATION_HPP

#include "common.hpp"
#include "features.hpp"

struct Candidate {
    int target;
    float similarity;
};

using NeighborDegreeProfiles = std::vector<std::vector<float>>;

NeighborDegreeProfiles compute_neighbor_degree_profiles(const Graph& graph,
                                                        int bins = 16);

float neighbor_degree_profile_similarity(
    const std::vector<float>& source_profile,
    const std::vector<float>& target_profile);

/*
 * For each source vertex i, compute cosine similarity against all target
 * vertices, retain the K highest similarities.
 *
 * Returns candidates[i] = vector of up to K Candidates for source node i.
 */
std::vector<std::vector<Candidate>> generate_top_k_candidates(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K,
    unsigned int feature_mask = ALL_FEATURES_MASK,
    const std::vector<float>& feature_weights = {},
    const NeighborDegreeProfiles* source_profiles = nullptr,
    const NeighborDegreeProfiles* target_profiles = nullptr,
    float neighbor_profile_weight = 0.0f,
    double* similarity_ms = nullptr,
    double* topk_selection_ms = nullptr);

#endif // SCGM_CANDIDATE_GENERATION_HPP