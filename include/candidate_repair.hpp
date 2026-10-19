#ifndef SCGM_CANDIDATE_REPAIR_HPP
#define SCGM_CANDIDATE_REPAIR_HPP

#include "bipartite_matching.hpp"
#include <functional>

enum class CandidateRepairStop { Feasible, PrefixLimit, RoundLimit,
                                 CandidateLimit, PayloadLimit };

// Stable diagnostic strings for CLI/JSON consumers.
const char* candidate_repair_stop_name(CandidateRepairStop stop) noexcept;

struct CandidateRepairOptions {
    int initial_k = 64;
    int k_limit = 512;
    int max_rounds = 3;
    std::size_t max_candidate_edges = 16u * 1024u * 1024u;
    // Bounds logical Candidate payload: retained support + staged additions
    // + one requested prefix. Allocator overhead, matching workspace, and
    // the prefix producer's own workspace are excluded, not a process cap.
    std::size_t max_payload_bytes = 256u * 1024u * 1024u;
};

struct CandidateRepairCheckpoint {
    int k = 0;
    int maximum_matching_size = 0;
    std::size_t candidate_edges = 0;
    std::size_t conflict_sources = 0;
    std::size_t edges_added = 0;
};

struct CandidateRepairResult {
    CandidateRepairStop stop = CandidateRepairStop::PrefixLimit;
    CandidateSupportStats support;
    std::vector<CandidateRepairCheckpoint> history;
    int rounds = 0;
    std::size_t edges_added = 0;
    std::size_t initial_candidate_edges = 0;
    std::size_t final_candidate_edges = 0;
};

// Return exactly k unique finite candidates for this source, ordered by score
// descending then target ID ascending. Producer must honor this workspace
// contract independently. No dense source-by-target matrix is required.
using CandidatePrefixProvider =
    std::function<std::vector<Candidate>(int source, int k)>;

// Mutates support only after an entire expansion has passed validation/limits.
// Existing edges and scores retain their order and value, matching Python.
// Exceptions leave the current expansion uncommitted; prior rounds persist.
CandidateRepairResult repair_candidate_support(
    std::vector<std::vector<Candidate>>& candidates, int num_targets,
    const CandidatePrefixProvider& prefix,
    const CandidateRepairOptions& options = {});

#endif
