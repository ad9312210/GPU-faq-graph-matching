#ifndef SCGM_CANDIDATE_GENERATION_HPP
#define SCGM_CANDIDATE_GENERATION_HPP

#include "common.hpp"
#include "features.hpp"

struct Candidate {
    int target;
    float similarity;
};

/*
 * For each source vertex i, compute cosine similarity against all target
 * vertices, retain the K highest similarities.
 *
 * Returns candidates[i] = vector of up to K Candidates for source node i.
 */
std::vector<std::vector<Candidate>> generate_top_k_candidates(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    int K);

#endif // SCGM_CANDIDATE_GENERATION_HPP