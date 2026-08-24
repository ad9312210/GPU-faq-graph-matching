#ifndef SCGM_COSINE_SIMILARITY_HPP
#define SCGM_COSINE_SIMILARITY_HPP

#include "common.hpp"
#include "features.hpp"

/*
 * Cosine similarity between two feature vectors:
 *
 *   S(a,b) = dot(a,b) / (||a|| * ||b||)
 *
 * If either vector has zero norm, returns 0.0 (safe handling).
 *
 * Cost is defined as:
 *   cost(i,j) = 1 - similarity(i,j)
 *
 * This is the mathematically correct formula.
 * We do NOT substitute weighted L1 distance.
 */

float cosine_similarity(const NodeFeatures& a, const NodeFeatures& b);

float cosine_cost(const NodeFeatures& a, const NodeFeatures& b);

void compute_all_pairwise_similarities_cpu(
    const std::vector<NodeFeatures>& feats_source,
    const std::vector<NodeFeatures>& feats_target,
    std::vector<float>& sim_flat);

#endif // SCGM_COSINE_SIMILARITY_HPP