#ifndef SCGM_EVALUATION_HPP
#define SCGM_EVALUATION_HPP

#include "common.hpp"
#include "graph.hpp"
#include "candidate_generation.hpp"

/*
 * PERMUTATION CONVENTION (used throughout the native implementation):
 *
 *   row_to_col[i] = j   means   source node i -> target node j
 *
 * ground_truth[i] = target node corresponding to source node i.
 *
 * This convention is NEVER reversed in the native backend.
 */

double mapping_accuracy(
    const std::vector<int>& predicted,
    const std::vector<int>& ground_truth);

bool is_one_to_one(const std::vector<int>& mapping, int num_targets);

/*
 * Candidate recall: fraction of ground-truth pairs (i, ground_truth[i])
 * that appear in the top-K candidate support.
 */
double candidate_recall(
    const std::vector<std::vector<Candidate>>& candidates,
    const std::vector<int>& ground_truth);

/*
 * Edge preservation: fraction of source edges (u,v) for which
 * (mapping[u], mapping[v]) is an edge in the target graph.
 *
 * For a perfect isomorphism, edge_preservation == 1.0.
 */
double edge_preservation(
    const Graph& source,
    const Graph& target,
    const std::vector<int>& mapping);

/*
 * Reverse edge check: fraction of target edges that are images of
 * source edges under the mapping. For a perfect isomorphism this is also 1.0.
 */
double reverse_edge_preservation(
    const Graph& source,
    const Graph& target,
    const std::vector<int>& mapping);

std::vector<int> inverse_permutation(const std::vector<int>& p);

bool validate_permutation(const std::vector<int>& p, int n);

#endif // SCGM_EVALUATION_HPP