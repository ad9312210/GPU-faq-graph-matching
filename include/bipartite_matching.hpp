#ifndef SCGM_BIPARTITE_MATCHING_HPP
#define SCGM_BIPARTITE_MATCHING_HPP

#include "candidate_generation.hpp"

struct CandidateSupportStats {
    int maximum_matching_size = 0;
    std::vector<int> target_incoming_counts;
    std::vector<int> source_to_target;
    std::vector<int> target_to_source;
};

struct CandidateConflictRegion {
    // Ascending source IDs reachable through alternating paths.
    std::vector<int> sources;
    std::vector<int> unmatched_sources;
};

// Follow unmatched source-to-target edges, then matched target-to-source
// edges, starting at every unmatched source. No candidate edges are changed.
// Pass a matching computed from this exact candidate support.
CandidateConflictRegion candidate_conflict_region(
    const std::vector<std::vector<Candidate>>& candidates,
    const CandidateSupportStats& matching);

// Maximum-cardinality matching of source rows to candidate target columns.
CandidateSupportStats analyze_candidate_support(
    const std::vector<std::vector<Candidate>>& candidates,
    int num_targets);

#endif
