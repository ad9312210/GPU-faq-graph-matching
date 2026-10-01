#ifndef SCGM_BIPARTITE_MATCHING_HPP
#define SCGM_BIPARTITE_MATCHING_HPP

#include "candidate_generation.hpp"

struct CandidateSupportStats {
    int maximum_matching_size = 0;
    std::vector<int> target_incoming_counts;
    std::vector<int> source_to_target;
    std::vector<int> target_to_source;
};

// Maximum-cardinality matching of source rows to candidate target columns.
CandidateSupportStats analyze_candidate_support(
    const std::vector<std::vector<Candidate>>& candidates,
    int num_targets);

#endif
